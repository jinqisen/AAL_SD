from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from experiments.ablation_config import ABLATION_SETTINGS, EXPERIMENT_NAME_ALIASES

from tuning_opt.evaluator import (
    ExperimentEvaluator,
    parse_final_miou_from_md,
)
from tuning_opt.llm_client import TuningLLMClient
from tuning_opt.llm_config import TuningLLMConfig
from tuning_opt.pool_resume import PoolResumeManager
from tuning_opt.proposer import LLMProposer
from tuning_opt.space import ParameterSpace
from utils import atomic_write_json, locked_update_json, read_json_dict


class _OrphanTorchShmCleaner:
    def __init__(self, *, interval_seconds: int, kill_timeout_seconds: float):
        self.interval_seconds = int(interval_seconds)
        self.kill_timeout_seconds = float(kill_timeout_seconds)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        t = threading.Thread(
            target=self._run, name="orphan_torch_shm_cleaner", daemon=True
        )
        self._thread = t
        t.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=3.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._cleanup_once()
            self._stop.wait(timeout=max(1, self.interval_seconds))

    def _ps_rows(self) -> List[Dict[str, Any]]:
        result = subprocess.run(
            ["ps", "-axo", "pid,ppid,user,command"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if result.returncode != 0:
            return []
        lines = [
            ln.rstrip("\n") for ln in str(result.stdout).splitlines() if ln.strip()
        ]
        rows: List[Dict[str, Any]] = []
        for ln in lines[1:]:
            parts = ln.strip().split(None, 3)
            if len(parts) < 4:
                continue
            pid_s, ppid_s, user_s, cmd = parts[0], parts[1], parts[2], parts[3]
            try:
                pid = int(pid_s)
                ppid = int(ppid_s)
            except Exception:
                continue
            rows.append(
                {"pid": pid, "ppid": ppid, "user": str(user_s), "cmd": str(cmd)}
            )
        return rows

    def _cleanup_once(self) -> None:
        user = os.environ.get("USER") or ""
        rows = self._ps_rows()
        pids = []
        for r in rows:
            try:
                pid = int(r.get("pid"))
                ppid = int(r.get("ppid"))
            except Exception:
                continue
            if user and str(r.get("user") or "") != user:
                continue
            cmd = str(r.get("cmd") or "")
            if "torch_shm_manager" not in cmd:
                continue
            if ppid != 1:
                continue
            pids.append(pid)

        if not pids:
            return

        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass

        deadline = time.time() + max(0.1, self.kill_timeout_seconds)
        while time.time() < deadline:
            alive = set()
            for r in self._ps_rows():
                if int(r.get("pid") or -1) in pids:
                    alive.add(int(r.get("pid")))
            if not alive:
                return
            time.sleep(0.1)

        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    return read_json_dict(path)


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
    merged: Dict[str, Dict[str, Any]] = {}
    if sidecar.exists():
        try:
            existing = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                for k, v in existing.items():
                    if isinstance(k, str) and k.strip() and isinstance(v, dict):
                        merged[k.strip()] = dict(v)
        except Exception:
            merged = {}
    for k, v in configs.items():
        if isinstance(k, str) and k.strip() and isinstance(v, dict):
            merged[k.strip()] = dict(v)
    atomic_write_json(sidecar, merged, indent=2)
    return sidecar


_PLATEAU_WINDOW = 5
_PLATEAU_EPS = 5e-4
_TR_COLLAPSE_PATIENCE = 3


_RE_OPT_RUN = re.compile(r"^autotune_opt_iter(\d+)_")


def _next_opt_iteration_index(repo_root: Path) -> int:
    runs_dir = repo_root / "results" / "runs"
    if not runs_dir.exists():
        return 0
    best = -1
    for p in runs_dir.iterdir():
        if not p.is_dir():
            continue
        m = _RE_OPT_RUN.match(p.name)
        if not m:
            continue
        try:
            it = int(m.group(1))
        except Exception:
            continue
        if it > best:
            best = it
    return best + 1


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
        screen_objective: Optional[str] = None,
        confirm_objective: Optional[str] = None,
        branch_round: int = 7,
        screen_end_round: int = 10,
        confirm_end_round: int = 16,
        screen_epochs_per_round: int = 8,
        dynamic_agent_workers: bool = False,
        agent_workers_min: int = 1,
        reserve_free_gb: float = 4.0,
        mem_per_agent_worker_gb: float = 2.0,
        dynamic_poll_seconds: float = 2.0,
        batch_timeout_seconds: Optional[int] = None,
        max_seed_workers: Optional[int] = None,
        auto_clean_orphan_shm: bool = False,
        orphan_shm_cleanup_interval_seconds: int = 300,
        orphan_shm_kill_timeout_seconds: float = 2.0,
    ):
        self.repo_root = repo_root
        self.results_dir = str(results_dir)
        self.target_miou = float(target_miou)
        self.max_iterations = int(max_iterations)
        self.seeds = list(seeds) if seeds else [42]
        self.max_concurrent = int(max_concurrent)
        obj = str(objective or "val").strip().lower()
        self.screen_objective = str(screen_objective or "alc").strip().lower()
        self.confirm_objective = str(confirm_objective or obj).strip().lower()
        self.branch_round = int(branch_round)
        self.screen_end_round = int(screen_end_round)
        self.confirm_end_round = int(confirm_end_round)
        self.screen_epochs_per_round = int(screen_epochs_per_round)
        self.dynamic_agent_workers = bool(dynamic_agent_workers)
        self.agent_workers_min = int(agent_workers_min)
        self.reserve_free_gb = float(reserve_free_gb)
        self.mem_per_agent_worker_gb = float(mem_per_agent_worker_gb)
        self.dynamic_poll_seconds = float(dynamic_poll_seconds)
        self.batch_timeout_seconds = (
            int(batch_timeout_seconds)
            if batch_timeout_seconds is not None and int(batch_timeout_seconds) > 0
            else None
        )
        self.max_seed_workers = (
            int(max_seed_workers) if max_seed_workers is not None else None
        )

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
        self.auto_clean_orphan_shm = bool(auto_clean_orphan_shm)
        self.orphan_shm_cleanup_interval_seconds = int(
            orphan_shm_cleanup_interval_seconds
        )
        self.orphan_shm_kill_timeout_seconds = float(orphan_shm_kill_timeout_seconds)

    def _state_path(self) -> Path:
        return self.repo_root / self.results_dir / "orchestrator_state.json"

    def _save_state(
        self,
        *,
        incumbent_run: str,
        incumbent_exp: str,
        incumbent_miou: float,
        it0: int,
        stop_reason: str = "",
    ) -> None:
        payload = {
            "updated_at": datetime.now().isoformat(),
            "incumbent_run": str(incumbent_run),
            "incumbent_exp": str(incumbent_exp),
            "incumbent_miou": float(incumbent_miou),
            "radius": float(self._radius),
            "no_improve_streak": int(self._no_improve_streak),
            "it0": int(it0),
            "history": [h.__dict__ for h in self.history],
            "stop_reason": str(stop_reason),
        }
        state_path = self._state_path()
        atomic_write_json(state_path, payload, indent=2)

    def _load_state(self) -> Optional[Dict[str, Any]]:
        state_path = self._state_path()
        obj = _read_json(state_path)
        if not isinstance(obj, dict) or "incumbent_run" not in obj:
            return None
        return obj

    def _restore_state(self, state: Dict[str, Any]) -> None:
        self._radius = float(state.get("radius", 0.10))
        self._no_improve_streak = int(state.get("no_improve_streak", 0))
        raw_history = state.get("history")
        if isinstance(raw_history, list):
            self.history = []
            for h in raw_history:
                if isinstance(h, dict):
                    self.history.append(
                        IterationResult(
                            iteration=int(h.get("iteration", 0)),
                            run_id=str(h.get("run_id", "")),
                            best_exp=str(h.get("best_exp", "")),
                            best_miou=float(h.get("best_miou", 0.0)),
                            best_overrides=dict(h.get("best_overrides") or {}),
                            stop_reason=str(h.get("stop_reason", "")),
                        )
                    )

    def _manifest_paths(self, *, run_id: str) -> tuple[Path, Path]:
        run_dir = self.repo_root / self.results_dir / "runs" / str(run_id)
        manifest_path = run_dir / "manifest.json"
        lock_path = run_dir / "manifest.json.lock"
        return manifest_path, lock_path

    def _update_run_manifest_tuning_progress(
        self, *, run_id: str, tuning_progress: Dict[str, Any]
    ) -> None:
        if not str(run_id).strip():
            return
        if not isinstance(tuning_progress, dict):
            return
        manifest_path, lock_path = self._manifest_paths(run_id=str(run_id))
        def _merge(existing: dict) -> dict:
            merged = dict(existing)
            cur = merged.get("tuning_progress")
            cur_dict = dict(cur) if isinstance(cur, dict) else {}
            cur_dict.update(tuning_progress)
            merged["tuning_progress"] = cur_dict
            merged["updated_at"] = datetime.now().isoformat()
            return merged

        locked_update_json(manifest_path, lock_path=lock_path, update=_merge)

    def run(self, *, initial_run_id: str, initial_exp: str) -> List[IterationResult]:
        incumbent_run = str(initial_run_id)
        incumbent_exp = str(initial_exp)
        incumbent_miou = (
            _load_final_miou(self.repo_root, incumbent_run, incumbent_exp) or 0.0
        )
        it0 = _next_opt_iteration_index(self.repo_root)

        saved = self._load_state()
        if saved is not None:
            self._restore_state(saved)
            incumbent_run = str(saved.get("incumbent_run", incumbent_run))
            incumbent_exp = str(saved.get("incumbent_exp", incumbent_exp))
            saved_miou = saved.get("incumbent_miou")
            if isinstance(saved_miou, (int, float)) and float(saved_miou) > float(
                incumbent_miou
            ):
                incumbent_miou = float(saved_miou)
            saved_it0 = saved.get("it0")
            if isinstance(saved_it0, int) and saved_it0 > it0:
                it0 = saved_it0

        cleaner: Optional[_OrphanTorchShmCleaner] = None
        if self.auto_clean_orphan_shm:
            cleaner = _OrphanTorchShmCleaner(
                interval_seconds=self.orphan_shm_cleanup_interval_seconds,
                kill_timeout_seconds=self.orphan_shm_kill_timeout_seconds,
            )
            cleaner.start()

        _shutdown_requested = threading.Event()

        def _signal_handler(signum, frame):
            _shutdown_requested.set()
            raise KeyboardInterrupt

        old_sigint = signal.signal(signal.SIGINT, _signal_handler)
        old_sigterm = signal.signal(signal.SIGTERM, _signal_handler)

        try:
            if self.branch_round < 2:
                raise ValueError(f"branch_round must be >= 2, got {self.branch_round}")
            if self.screen_end_round < self.branch_round:
                raise ValueError(
                    f"screen_end_round must be >= branch_round, got screen_end_round={self.screen_end_round} branch_round={self.branch_round}"
                )
            if self.confirm_end_round < self.screen_end_round:
                raise ValueError(
                    f"confirm_end_round must be >= screen_end_round, got confirm_end_round={self.confirm_end_round} screen_end_round={self.screen_end_round}"
                )

            stop_reason = ""
            last_iter_run_id: Optional[str] = None
            try:
                for local_it in range(self.max_iterations):
                    it = int(it0) + int(local_it)
                    if _shutdown_requested.is_set():
                        stop_reason = "interrupted"
                        break
                    if float(incumbent_miou) >= float(self.target_miou):
                        stop_reason = "target_reached"
                        break
                    if _plateau_detected(self.history, _PLATEAU_WINDOW, _PLATEAU_EPS):
                        stop_reason = "plateau_detected"
                        break
                    if (
                        self._radius <= 0.02
                        and self._no_improve_streak >= _TR_COLLAPSE_PATIENCE
                    ):
                        stop_reason = "trust_region_collapse"
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
                        stop_reason = "no_candidates"
                        break

                    ts = datetime.now().strftime("%Y%m%d_%H%M")
                    run_id = f"autotune_opt_iter{it:03d}_{ts}"
                    last_iter_run_id = str(run_id)
                    exp_map: Dict[str, Dict[str, Any]] = {}
                    exp_names: List[str] = []
                    for j, ov in enumerate(candidates):
                        direction = str(ov.get("_direction", f"cand{j}"))
                        safe_dir = (
                            re.sub(r"[^a-zA-Z0-9_]+", "_", direction).strip("_")
                            or f"cand{j}"
                        )
                        exp_name = f"auto_opt_iter{it:02d}_{safe_dir}_{j:02d}"
                        exp_names.append(exp_name)
                        cleaned = {
                            k: v for k, v in ov.items() if not str(k).startswith("_")
                        }
                        cfg = self.space.apply_overrides(incumbent_cfg, cleaned)
                        if cfg.get("use_agent") is True:
                            if not isinstance(cfg.get("grad_probe_source"), str):
                                cfg["grad_probe_source"] = "train_holdout"
                            if (
                                str(cfg.get("grad_probe_source")).strip().lower()
                                != "train_holdout"
                            ):
                                cfg["grad_probe_source"] = "train_holdout"
                            if cfg.get("grad_probe_holdout_frac") is None:
                                cfg["grad_probe_holdout_frac"] = 0.10
                            if cfg.get("grad_probe_holdout_min") is None:
                                cfg["grad_probe_holdout_min"] = 8
                        cfg["description"] = f"AutoTuneOpt: {safe_dir}"
                        exp_map[exp_name] = cfg

                    _write_sidecar(self.repo_root, exp_map)

                    self.pool_mgr.branch_from_round(
                        source_run_id=incumbent_run,
                        source_exp=incumbent_exp,
                        target_run_id=run_id,
                        target_exps=exp_names,
                        branch_round=int(self.branch_round),
                    )

                    phase_a_seed = int(self.seeds[0])

                    self.evaluator.run_batch(
                        run_id=run_id,
                        exp_names=exp_names,
                        seed=phase_a_seed,
                        max_concurrent=self.max_concurrent,
                        n_rounds=int(self.screen_end_round),
                        epochs_per_round=int(self.screen_epochs_per_round),
                        resume=True,
                        dynamic_agent_workers=self.dynamic_agent_workers,
                        agent_workers_min=self.agent_workers_min,
                        reserve_free_gb=self.reserve_free_gb,
                        mem_per_agent_worker_gb=self.mem_per_agent_worker_gb,
                        dynamic_poll_seconds=self.dynamic_poll_seconds,
                        timeout=self.batch_timeout_seconds,
                    )
                    screen = self.evaluator.collect_results(
                        run_id=run_id,
                        exp_names=exp_names,
                        objective=self.screen_objective,
                    )
                    ranked = sorted(
                        [(name, r.miou or -1.0) for name, r in screen.items()],
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    top = [name for name, _ in ranked[: max(1, min(2, len(ranked)))]]

                    self.evaluator.run_batch(
                        run_id=run_id,
                        exp_names=top,
                        seed=phase_a_seed,
                        max_concurrent=self.max_concurrent,
                        n_rounds=int(self.confirm_end_round),
                        epochs_per_round=None,
                        resume=True,
                        dynamic_agent_workers=self.dynamic_agent_workers,
                        agent_workers_min=self.agent_workers_min,
                        reserve_free_gb=self.reserve_free_gb,
                        mem_per_agent_worker_gb=self.mem_per_agent_worker_gb,
                        dynamic_poll_seconds=self.dynamic_poll_seconds,
                        timeout=self.batch_timeout_seconds,
                    )
                    full = self.evaluator.collect_results(
                        run_id=run_id, exp_names=top, objective=self.confirm_objective
                    )
                    best_exp, best_miou_f2 = max(
                        ((name, r.miou or -1.0) for name, r in full.items()),
                        key=lambda x: x[1],
                    )

                    best_miou: float = float(best_miou_f2)
                    best_std: Optional[float] = None
                    if len(self.seeds) > 1:
                        run_ids_by_seed: Dict[int, str] = {
                            int(self.seeds[0]): str(run_id)
                        }
                        seed_list = [int(s) for s in self.seeds[1:]]
                        for seed in seed_list:
                            seed_run_id = f"{run_id}_seed{int(seed)}"
                            run_ids_by_seed[int(seed)] = seed_run_id

                        def _run_seed(seed: int) -> None:
                            seed_run_id = str(run_ids_by_seed[int(seed)])
                            self.pool_mgr.branch_from_round(
                                source_run_id=incumbent_run,
                                source_exp=incumbent_exp,
                                target_run_id=seed_run_id,
                                target_exps=[best_exp],
                                branch_round=int(self.branch_round),
                            )
                            self.evaluator.run_batch(
                                run_id=seed_run_id,
                                exp_names=[best_exp],
                                seed=int(seed),
                                max_concurrent=self.max_concurrent,
                                n_rounds=int(self.confirm_end_round),
                                epochs_per_round=None,
                                resume=True,
                                dynamic_agent_workers=self.dynamic_agent_workers,
                                agent_workers_min=self.agent_workers_min,
                                reserve_free_gb=self.reserve_free_gb,
                                mem_per_agent_worker_gb=self.mem_per_agent_worker_gb,
                                dynamic_poll_seconds=self.dynamic_poll_seconds,
                                timeout=self.batch_timeout_seconds,
                            )

                        max_sw = (
                            int(self.max_seed_workers)
                            if self.max_seed_workers is not None
                            else max(1, int(self.max_concurrent))
                        )
                        seed_workers = min(max_sw, len(seed_list))
                        if seed_workers <= 1:
                            for seed in seed_list:
                                _run_seed(int(seed))
                        else:
                            with concurrent.futures.ThreadPoolExecutor(
                                max_workers=seed_workers
                            ) as ex:
                                futs = [
                                    ex.submit(_run_seed, int(seed))
                                    for seed in seed_list
                                ]
                                for fut in concurrent.futures.as_completed(futs):
                                    fut.result()

                        ms_results = self.evaluator.collect_multi_seed_results(
                            run_ids_by_seed=run_ids_by_seed,
                            exp_names=[best_exp],
                            objective=self.confirm_objective,
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

                    self._save_state(
                        incumbent_run=incumbent_run,
                        incumbent_exp=incumbent_exp,
                        incumbent_miou=float(incumbent_miou),
                        it0=int(it) + 1,
                    )
                    try:
                        self._update_run_manifest_tuning_progress(
                            run_id=str(run_id),
                            tuning_progress={
                                "iteration": int(it),
                                "it0": int(it) + 1,
                                "incumbent_run": str(incumbent_run),
                                "incumbent_exp": str(incumbent_exp),
                                "incumbent_miou": float(incumbent_miou),
                                "radius": float(self._radius),
                                "no_improve_streak": int(self._no_improve_streak),
                                "history_len": int(len(self.history)),
                                "state_path": str(
                                    (Path(self.results_dir) / "orchestrator_state.json")
                                ),
                            },
                        )
                    except Exception:
                        pass
            except KeyboardInterrupt:
                stop_reason = "interrupted"

            if not stop_reason:
                stop_reason = "max_iterations"

            self._save_state(
                incumbent_run=incumbent_run,
                incumbent_exp=incumbent_exp,
                incumbent_miou=float(incumbent_miou),
                it0=_next_opt_iteration_index(self.repo_root),
                stop_reason=stop_reason,
            )
            try:
                if last_iter_run_id is not None and str(last_iter_run_id).strip():
                    self._update_run_manifest_tuning_progress(
                        run_id=str(last_iter_run_id),
                        tuning_progress={
                            "stop_reason": str(stop_reason),
                            "stopped_at": datetime.now().isoformat(),
                            "incumbent_run": str(incumbent_run),
                            "incumbent_exp": str(incumbent_exp),
                            "incumbent_miou": float(incumbent_miou),
                            "radius": float(self._radius),
                            "no_improve_streak": int(self._no_improve_streak),
                            "history_len": int(len(self.history)),
                        },
                    )
            except Exception:
                pass

            result_path = self.repo_root / self.results_dir / "tuning_result.json"
            result_payload = {
                "completed_at": datetime.now().isoformat(),
                "stop_reason": str(stop_reason),
                "incumbent_run": str(incumbent_run),
                "incumbent_exp": str(incumbent_exp),
                "incumbent_miou": float(incumbent_miou),
                "total_iterations": int(len(self.history)),
                "radius": float(self._radius),
                "no_improve_streak": int(self._no_improve_streak),
                "history": [h.__dict__ for h in self.history],
            }
            atomic_write_json(result_path, result_payload, indent=2)

            return list(self.history)
        finally:
            signal.signal(signal.SIGINT, old_sigint)
            signal.signal(signal.SIGTERM, old_sigterm)
            if cleaner is not None:
                cleaner.stop()

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
    parser.add_argument("--screen-objective", type=str, default="")
    parser.add_argument("--confirm-objective", type=str, default="")
    parser.add_argument("--branch-round", type=int, default=7)
    parser.add_argument("--screen-end-round", type=int, default=10)
    parser.add_argument("--confirm-end-round", type=int, default=16)
    parser.add_argument("--screen-epochs-per-round", type=int, default=8)
    parser.add_argument("--dynamic-agent-workers", action="store_true", default=False)
    parser.add_argument("--agent-workers-min", type=int, default=1)
    parser.add_argument("--reserve-free-gb", type=float, default=4.0)
    parser.add_argument("--mem-per-agent-worker-gb", type=float, default=2.0)
    parser.add_argument("--dynamic-poll-seconds", type=float, default=2.0)
    parser.add_argument("--batch-timeout-seconds", type=int, default=0)
    parser.add_argument("--max-seed-workers", type=int, default=None)
    parser.add_argument("--auto-clean-orphan-shm", action="store_true", default=False)
    parser.add_argument("--orphan-shm-cleanup-interval-seconds", type=int, default=300)
    parser.add_argument("--orphan-shm-kill-timeout-seconds", type=float, default=2.0)
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
    screen_obj = str(args.screen_objective).strip() or None
    confirm_obj = str(args.confirm_objective).strip() or None

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
        screen_objective=screen_obj,
        confirm_objective=confirm_obj,
        branch_round=int(args.branch_round),
        screen_end_round=int(args.screen_end_round),
        confirm_end_round=int(args.confirm_end_round),
        screen_epochs_per_round=int(args.screen_epochs_per_round),
        dynamic_agent_workers=bool(args.dynamic_agent_workers),
        agent_workers_min=int(args.agent_workers_min),
        reserve_free_gb=float(args.reserve_free_gb),
        mem_per_agent_worker_gb=float(args.mem_per_agent_worker_gb),
        dynamic_poll_seconds=float(args.dynamic_poll_seconds),
        batch_timeout_seconds=int(args.batch_timeout_seconds)
        if int(args.batch_timeout_seconds) > 0
        else None,
        max_seed_workers=args.max_seed_workers,
        auto_clean_orphan_shm=bool(args.auto_clean_orphan_shm),
        orphan_shm_cleanup_interval_seconds=int(
            args.orphan_shm_cleanup_interval_seconds
        ),
        orphan_shm_kill_timeout_seconds=float(args.orphan_shm_kill_timeout_seconds),
    )
    history = orch.run(
        initial_run_id=str(args.initial_run_id), initial_exp=str(args.initial_exp)
    )
    print(json.dumps([h.__dict__ for h in history], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
