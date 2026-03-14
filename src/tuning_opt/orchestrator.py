from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from experiments.ablation_config import ABLATION_SETTINGS, EXPERIMENT_NAME_ALIASES

from tuning_opt.evaluator import (
    ExperimentEvaluator,
    MultiSeedResult,
    parse_final_miou_from_md,
)
from tuning_opt.llm_client import TuningLLMClient
from tuning_opt.llm_config import TuningLLMConfig
from tuning_opt.pool_resume import PoolResumeManager
from tuning_opt.proposer import LLMProposer
from tuning_opt.space import ParameterSpace


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _load_final_miou(repo_root: Path, run_id: str, exp_name: str) -> Optional[float]:
    run_dir = repo_root / "results" / "runs" / run_id
    p1 = run_dir / f"result_{exp_name}.json"
    obj = _read_json(p1)
    if isinstance(obj, dict):
        section = obj.get(exp_name)
        if isinstance(section, dict):
            v = section.get("final_miou")
            if isinstance(v, (int, float)):
                return float(v)

    p2 = run_dir / "experiment_results.json"
    obj = _read_json(p2)
    if isinstance(obj, dict):
        section = obj.get(exp_name)
        if isinstance(section, dict):
            v = section.get("final_miou")
            if isinstance(v, (int, float)):
                return float(v)

    md = run_dir / f"{exp_name}.md"
    return parse_final_miou_from_md(md)


def _load_incumbent_cfg(
    repo_root: Path, run_id: str, exp_name: str
) -> Optional[Dict[str, Any]]:
    manifest = _read_json(repo_root / "results" / "runs" / run_id / "manifest.json")
    if isinstance(manifest, dict):
        exps = manifest.get("experiments")
        if isinstance(exps, dict):
            cfg = exps.get(exp_name)
            if isinstance(cfg, dict):
                return cfg
    canonical = str(EXPERIMENT_NAME_ALIASES.get(exp_name, exp_name))
    cfg = ABLATION_SETTINGS.get(canonical)
    return dict(cfg) if isinstance(cfg, dict) else None


def _write_sidecar(repo_root: Path, configs: Dict[str, Dict[str, Any]]) -> Path:
    sidecar = repo_root / "src" / "experiments" / "auto_tune_configs.json"
    tmp = sidecar.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(configs, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, sidecar)
    return sidecar


_PLATEAU_WINDOW = 5
_PLATEAU_EPS = 5e-4
_TR_COLLAPSE_PATIENCE = 3


@dataclass
class IterationResult:
    iteration: int
    run_id: str
    best_exp: str
    best_miou: float
    best_overrides: Dict[str, Any]
    stop_reason: str = ""


def _statistically_better(
    candidate: float,
    incumbent: float,
    candidate_std: Optional[float],
    n_seeds: int,
    z: float = 1.645,
) -> bool:
    if n_seeds < 2 or candidate_std is None:
        return candidate > incumbent + 1e-6
    se = candidate_std / math.sqrt(n_seeds)
    return (candidate - se * z) > incumbent


def _plateau_detected(history: List[IterationResult], window: int, eps: float) -> bool:
    if len(history) < window:
        return False
    recent = [h.best_miou for h in history[-window:]]
    mean_gain = (recent[-1] - recent[0]) / max(window - 1, 1)
    return mean_gain < eps


class MultiFidelityTuningOrchestrator:
    def __init__(
        self,
        *,
        repo_root: Path,
        results_dir: str,
        target_miou: float,
        max_iterations: int,
        seeds: List[int],
        max_concurrent: int,
        enable_llm: bool,
        llm_config_path: Optional[Path],
        objective: str = "val",
    ):
        self.repo_root = repo_root
        self.results_dir = str(results_dir)
        self.target_miou = float(target_miou)
        self.max_iterations = int(max_iterations)
        self.seeds = list(seeds) if seeds else [42]
        self.max_concurrent = int(max_concurrent)
        self.objective = str(objective or "val").strip().lower()

        self.space = ParameterSpace.default()
        self.pool_mgr = PoolResumeManager(results_dir=self.results_dir)
        self.evaluator = ExperimentEvaluator(repo_root=self.repo_root)

        self.enable_llm = bool(enable_llm)
        self.llm_config_path = llm_config_path
        self.llm_proposer: Optional[LLMProposer] = None
        if self.enable_llm:
            cfg_path = llm_config_path or TuningLLMConfig.default_path(self.repo_root)
            llm_cfg = TuningLLMConfig.load(cfg_path)
            self.llm_proposer = LLMProposer(TuningLLMClient(llm_cfg))

        self.history: List[IterationResult] = []
        self._radius = 0.10
        self._no_improve_streak = 0

    def run(self, *, initial_run_id: str, initial_exp: str) -> List[IterationResult]:
        incumbent_run = str(initial_run_id)
        incumbent_exp = str(initial_exp)
        incumbent_miou = (
            _load_final_miou(self.repo_root, incumbent_run, incumbent_exp) or 0.0
        )

        for it in range(self.max_iterations):
            if float(incumbent_miou) >= float(self.target_miou):
                break
            if _plateau_detected(self.history, _PLATEAU_WINDOW, _PLATEAU_EPS):
                break
            if (
                self._radius <= 0.02
                and self._no_improve_streak >= _TR_COLLAPSE_PATIENCE
            ):
                break

            incumbent_cfg = _load_incumbent_cfg(
                self.repo_root, incumbent_run, incumbent_exp
            )
            if not isinstance(incumbent_cfg, dict):
                raise RuntimeError(
                    f"Unable to load incumbent config: run_id={incumbent_run} exp={incumbent_exp}"
                )

            center = self.space.flatten_from_ablation_cfg(incumbent_cfg)
            rng = random.Random(1337 + it)
            llm_overrides = self._propose_with_llm(
                iteration=it,
                incumbent_run=incumbent_run,
                incumbent_exp=incumbent_exp,
                incumbent_miou=float(incumbent_miou),
                center=center,
            )
            tr_samples = self.space.trust_region_sample(
                center=center, radius=self._radius, rng=rng, n=6
            )
            candidates = self.space.deduplicate(llm_overrides + tr_samples)
            candidates = candidates[:8]
            if not candidates:
                break

            ts = datetime.now().strftime("%Y%m%d_%H%M")
            run_id = f"autotune_opt_iter{it:03d}_{ts}"
            exp_map: Dict[str, Dict[str, Any]] = {}
            exp_names: List[str] = []
            for j, ov in enumerate(candidates):
                direction = str(ov.get("_direction", f"cand{j}"))
                safe_dir = (
                    re.sub(r"[^a-zA-Z0-9_]+", "_", direction).strip("_") or f"cand{j}"
                )
                exp_name = f"auto_opt_iter{it:02d}_{safe_dir}_{j:02d}"
                exp_names.append(exp_name)
                cleaned = {k: v for k, v in ov.items() if not str(k).startswith("_")}
                cfg = self.space.apply_overrides(incumbent_cfg, cleaned)
                cfg["description"] = f"AutoTuneOpt: {safe_dir}"
                exp_map[exp_name] = cfg

            _write_sidecar(self.repo_root, exp_map)

            self.pool_mgr.branch_from_round(
                source_run_id=incumbent_run,
                source_exp=incumbent_exp,
                target_run_id=run_id,
                target_exps=exp_names,
                branch_round=7,
            )

            phase_a_seed = int(self.seeds[0])

            # F1: cheap screening
            self.evaluator.run_batch(
                run_id=run_id,
                exp_names=exp_names,
                seed=phase_a_seed,
                max_concurrent=self.max_concurrent,
                n_rounds=10,
                epochs_per_round=8,
                resume=True,
            )
            screen = self.evaluator.collect_results(
                run_id=run_id, exp_names=exp_names, objective=self.objective
            )
            ranked = sorted(
                [(name, r.miou or -1.0) for name, r in screen.items()],
                key=lambda x: x[1],
                reverse=True,
            )
            top = [name for name, _ in ranked[: max(1, min(2, len(ranked)))]]

            # F2: full confirmation on top-k
            self.evaluator.run_batch(
                run_id=run_id,
                exp_names=top,
                seed=phase_a_seed,
                max_concurrent=1,
                n_rounds=16,
                epochs_per_round=None,
                resume=True,
            )
            full = self.evaluator.collect_results(
                run_id=run_id, exp_names=top, objective=self.objective
            )
            best_exp, best_miou_f2 = max(
                ((name, r.miou or -1.0) for name, r in full.items()),
                key=lambda x: x[1],
            )

            # F3: multi-seed statistical review (only when multiple seeds configured)
            best_miou: float = float(best_miou_f2)
            best_std: Optional[float] = None
            if len(self.seeds) > 1:
                run_ids_by_seed: Dict[int, str] = {int(self.seeds[0]): str(run_id)}
                for seed in self.seeds[1:]:
                    seed_run_id = f"{run_id}_seed{int(seed)}"
                    run_ids_by_seed[int(seed)] = seed_run_id
                    self.pool_mgr.branch_from_round(
                        source_run_id=incumbent_run,
                        source_exp=incumbent_exp,
                        target_run_id=seed_run_id,
                        target_exps=[best_exp],
                        branch_round=7,
                    )
                    self.evaluator.run_batch(
                        run_id=seed_run_id,
                        exp_names=[best_exp],
                        seed=int(seed),
                        max_concurrent=1,
                        n_rounds=16,
                        epochs_per_round=None,
                        resume=True,
                    )

                ms_results = self.evaluator.collect_multi_seed_results(
                    run_ids_by_seed=run_ids_by_seed,
                    exp_names=[best_exp],
                    objective=self.objective,
                )
                ms = ms_results.get(best_exp)
                if ms is not None and ms.mean is not None:
                    best_miou = ms.mean
                    best_std = ms.std

            best_overrides = {
                k: v
                for k, v in candidates[exp_names.index(best_exp)].items()
                if not str(k).startswith("_")
            }

            improved = _statistically_better(
                candidate=best_miou,
                incumbent=float(incumbent_miou),
                candidate_std=best_std,
                n_seeds=len(self.seeds),
            )
            if improved:
                incumbent_run, incumbent_exp, incumbent_miou = (
                    run_id,
                    best_exp,
                    float(best_miou),
                )
                self._radius = min(0.25, self._radius * 1.25)
                self._no_improve_streak = 0
            else:
                self._radius = max(0.02, self._radius * 0.70)
                self._no_improve_streak += 1

            self.history.append(
                IterationResult(
                    iteration=int(it),
                    run_id=str(run_id),
                    best_exp=str(best_exp),
                    best_miou=float(best_miou),
                    best_overrides=dict(best_overrides),
                )
            )

        return list(self.history)

    def _propose_with_llm(
        self,
        *,
        iteration: int,
        incumbent_run: str,
        incumbent_exp: str,
        incumbent_miou: float,
        center: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        if not self.llm_proposer:
            return []
        ctx = {
            "iteration": int(iteration),
            "incumbent": {
                "run_id": incumbent_run,
                "exp": incumbent_exp,
                "miou": float(incumbent_miou),
            },
            "center": dict(center),
            "target_miou": float(self.target_miou),
        }
        proposals = self.llm_proposer.propose(context=ctx)
        out: List[Dict[str, Any]] = []
        for p in proposals:
            item = dict(p.parameter_changes)
            item["_direction"] = p.direction
            out.append(item)
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-run-id", required=True)
    parser.add_argument("--initial-exp", required=True)
    parser.add_argument("--target-miou", type=float, default=0.725)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--objective", type=str, default="val")
    parser.add_argument("--no-llm", action="store_true", default=False)
    parser.add_argument("--llm-config", type=str, default="")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    llm_cfg_path = (
        Path(args.llm_config).expanduser().resolve()
        if str(args.llm_config).strip()
        else None
    )

    orch = MultiFidelityTuningOrchestrator(
        repo_root=repo_root,
        results_dir=str(args.results_dir),
        target_miou=float(args.target_miou),
        max_iterations=int(args.max_iterations),
        seeds=seeds,
        max_concurrent=int(args.max_concurrent),
        enable_llm=not bool(args.no_llm),
        llm_config_path=llm_cfg_path,
        objective=str(args.objective),
    )
    history = orch.run(
        initial_run_id=str(args.initial_run_id), initial_exp=str(args.initial_exp)
    )
    print(json.dumps([h.__dict__ for h in history], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
