from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import Config
from experiments.ablation_config import MULTI_SEED_DEFAULTS
from experiments.run_all_experiments import ExperimentRunner
from utils.logger import logger
from utils.multi_seed import aggregate_multi_seed, render_markdown


@dataclass(frozen=True)
class MultiSeedRunSpec:
    base_run_id: str
    seeds: list[int]

    def run_ids(self) -> list[str]:
        return [f"{self.base_run_id}_seed{seed}" for seed in self.seeds]


def _parse_seeds(raw: str) -> list[int]:
    parts = [p.strip() for p in (raw or "").replace(",", " ").split() if p.strip()]
    seeds = []
    for p in parts:
        seeds.append(int(p))
    if not seeds:
        seeds = [42, 43, 44]
    return seeds


def _ensure_group_dir(results_dir: Path, base_run_id: str) -> Path:
    group_dir = results_dir / "runs" / base_run_id
    group_dir.mkdir(parents=True, exist_ok=True)
    return group_dir


def _run_single_seed_job(payload: dict[str, Any]) -> dict[str, Any]:
    results_dir = Path(payload["results_dir"])
    base_run_id = str(payload["base_run_id"])
    seed = int(payload["seed"])
    run_id = str(payload["run_id"])
    start = str(payload["start"])
    experiments = payload.get("experiments")
    n_rounds = payload.get("n_rounds")
    exp_workers = int(payload.get("exp_workers") or 1)
    seed_pool_workers = int(payload.get("seed_pool_workers") or 1)

    try:
        cfg = Config()
        cfg.RANDOM_SEED = int(seed)
        cfg.DETERMINISTIC = True
        cfg.RESEARCH_MODE = True
        cfg.FIX_EPOCHS_PER_ROUND = True
        cfg.EPOCHS_PER_ROUND = 10
        cfg.SHARING_STRATEGY = "file_descriptor"
        cfg.RESULTS_DIR = str(results_dir)
        if n_rounds is not None and int(n_rounds) > 0:
            cfg.N_ROUNDS = int(n_rounds)
        os.environ["AAL_SD_RANDOM_SEED"] = str(seed)
        os.environ["AAL_SD_DETERMINISTIC"] = "1"
        os.environ["AAL_SD_SHARING_STRATEGY"] = "file_descriptor"
        if str(os.getenv("AAL_SD_AUTOTUNE_WORKERS", "")).strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        ):
            total_concurrency = max(1, int(seed_pool_workers) * int(exp_workers))
            cpu_count = os.cpu_count() or 2
            if os.getenv("AAL_SD_NUM_WORKERS") is None:
                if total_concurrency >= 4:
                    os.environ["AAL_SD_NUM_WORKERS"] = "0"
                    os.environ["AAL_SD_PERSISTENT_WORKERS"] = "0"
                else:
                    per_job = max(1, int(cpu_count) // int(total_concurrency))
                    os.environ["AAL_SD_NUM_WORKERS"] = str(min(2, per_job // 2))
            if os.getenv("AAL_SD_FEATURE_NUM_WORKERS") is None:
                if total_concurrency >= 4:
                    os.environ["AAL_SD_FEATURE_NUM_WORKERS"] = "0"
                    os.environ["AAL_SD_FEATURE_PERSISTENT_WORKERS"] = "0"
                else:
                    per_job = max(1, int(cpu_count) // int(total_concurrency))
                    os.environ["AAL_SD_FEATURE_NUM_WORKERS"] = str(min(2, per_job // 2))
            if os.getenv("AAL_SD_PREFETCH_FACTOR") is None:
                os.environ["AAL_SD_PREFETCH_FACTOR"] = "2"
            if os.getenv("AAL_SD_FEATURE_PREFETCH_FACTOR") is None:
                os.environ["AAL_SD_FEATURE_PREFETCH_FACTOR"] = "2"

        logger.info(
            f"MULTI SEED RUN START | base_run_id={base_run_id} | run_id={run_id} | seed={seed} | start={start}"
        )
        runner = ExperimentRunner(cfg, str(results_dir), run_id=run_id, start_mode=start)
        runner.run_all_experiments(experiments, parallel_workers=exp_workers)
        logger.info(f"MULTI SEED RUN END   | base_run_id={base_run_id} | run_id={run_id} | seed={seed}")
        return {"run_id": run_id, "seed": seed, "status": "success"}
    except Exception as e:
        logger.error(
            f"MULTI SEED RUN FAIL  | base_run_id={base_run_id} | run_id={run_id} | seed={seed} | error={e}\n{traceback.format_exc()}"
        )
        return {"run_id": run_id, "seed": seed, "status": "failed", "error": str(e)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run multi-seed experiments and aggregate results")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--run_id", type=str, default=None, help="Base run id (group id)")
    parser.add_argument(
        "--seeds",
        type=str,
        default=" ".join(str(s) for s in MULTI_SEED_DEFAULTS["seeds"]),
    )
    parser.add_argument("--start", type=str, default="fresh", choices=["fresh", "resume"])
    parser.add_argument("--experiments", type=str, nargs="+", default=None)
    parser.add_argument("--n_rounds", type=int, default=None)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--exp_workers", type=int, default=1)
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    base_run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    seeds = _parse_seeds(args.seeds)
    spec = MultiSeedRunSpec(base_run_id=base_run_id, seeds=seeds)

    group_dir = _ensure_group_dir(results_dir, base_run_id)
    manifest_path = group_dir / "multi_seed_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "base_run_id": base_run_id,
                "seeds": seeds,
                "run_ids": spec.run_ids(),
                "created_at": datetime.now().isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    experiments = args.experiments or MULTI_SEED_DEFAULTS["paper_experiments"]
    jobs = []
    for seed, run_id in zip(seeds, spec.run_ids()):
        jobs.append(
            {
                "results_dir": str(results_dir),
                "base_run_id": base_run_id,
                "seed": int(seed),
                "run_id": run_id,
                "start": args.start,
                "experiments": experiments,
                "n_rounds": args.n_rounds,
                "exp_workers": int(args.exp_workers),
            }
        )

    if args.parallel:
        max_workers = int(args.workers) if int(args.workers) > 0 else min(len(jobs), os.cpu_count() or 2)
        max_workers = max(1, max_workers)
        for j in jobs:
            j["seed_pool_workers"] = int(max_workers)
        ctx = multiprocessing.get_context("spawn")
        logger.info(f"MULTI SEED PARALLEL | base_run_id={base_run_id} | workers={max_workers} | seeds={seeds}")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            max_tasks_per_child=1,
        ) as ex:
            futures = [ex.submit(_run_single_seed_job, j) for j in jobs]
            for fut in concurrent.futures.as_completed(futures):
                _ = fut.result()
    else:
        for j in jobs:
            j["seed_pool_workers"] = 1
        for j in jobs:
            _run_single_seed_job(j)

    summary = aggregate_multi_seed(results_dir, spec.run_ids())
    summary_path = group_dir / "multi_seed_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = group_dir / "multi_seed_report.md"
    md_path.write_text(render_markdown(summary), encoding="utf-8")

    logger.info(f"Multi-seed summary saved: {summary_path}")
    logger.info(f"Multi-seed report saved: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
