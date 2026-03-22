from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import multiprocessing
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import Config
from experiments.ablation_config import ABLATION_SETTINGS, MULTI_SEED_DEFAULTS
from experiments.run_all_experiments import ExperimentRunner
from utils.logger import logger


SWEEP_PARAMS = {
    "tau_risk": {
        "key": "OVERFIT_RISK_HI",
        "location": "agent_threshold_overrides",
        "values": [0.6, 0.9, 1.2, 1.5, 1.8],
    },
    "beta_ema": {
        "key": "OVERFIT_RISK_EMA_ALPHA",
        "location": "agent_threshold_overrides",
        "values": [0.2, 0.4, 0.6, 0.8, 1.0],
    },
    "lambda_max": {
        "key": "LAMBDA_CLAMP_MAX",
        "location": "agent_threshold_overrides",
        "values": [0.4, 0.6, 0.8, 0.95, 1.0],
    },
    "k_max": {
        "key": "lambda_max_step",
        "location": "lambda_policy",
        "values": [0.05, 0.10, 0.17, 0.25, 0.35],
    },
    "n_cool": {
        "key": "LAMBDA_DOWN_COOLING_ROUNDS",
        "location": "agent_threshold_overrides",
        "values": [0, 1, 2, 3, 4],
    },
}


def _parse_sensitivity_name(name: str) -> tuple[str, int, float]:
    m = re.match(r"sensitivity_(.+)_(\d+)$", name)
    if not m:
        return "", -1, math.nan
    param = m.group(1)
    idx = int(m.group(2))
    spec = SWEEP_PARAMS.get(param)
    if spec is None or idx >= len(spec["values"]):
        return param, idx, math.nan
    return param, idx, float(spec["values"][idx])


def _run_single_seed_job(payload: dict[str, Any]) -> dict[str, Any]:
    results_dir = Path(payload["results_dir"])
    base_run_id = str(payload["base_run_id"])
    seed = int(payload["seed"])
    run_id = str(payload["run_id"])
    start = str(payload["start"])
    experiments = payload.get("experiments")
    n_rounds = payload.get("n_rounds")
    exp_workers = int(payload.get("exp_workers") or 1)

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

        logger.info(
            f"SENSITIVITY RUN START | base={base_run_id} | run_id={run_id} | seed={seed}"
        )
        runner = ExperimentRunner(
            cfg, str(results_dir), run_id=run_id, start_mode=start
        )
        runner.run_all_experiments(experiments, parallel_workers=exp_workers)
        logger.info(
            f"SENSITIVITY RUN END   | base={base_run_id} | run_id={run_id} | seed={seed}"
        )
        return {"run_id": run_id, "seed": seed, "status": "success"}
    except Exception as e:
        logger.error(
            f"SENSITIVITY RUN FAIL  | base={base_run_id} | run_id={run_id} | seed={seed} | error={e}\n{traceback.format_exc()}"
        )
        return {"run_id": run_id, "seed": seed, "status": "failed", "error": str(e)}


def _load_status_metrics(run_dir: Path, experiment: str) -> dict[str, float]:
    status_path = run_dir / f"{experiment}_status.json"
    if not status_path.exists():
        return {}
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    result = payload.get("result", {}) if isinstance(payload, dict) else {}
    if not isinstance(result, dict):
        return {}
    return {
        "alc": float(result.get("alc", math.nan)),
        "final_miou": float(result.get("final_mIoU", math.nan)),
        "final_f1": float(result.get("final_f1", math.nan)),
    }


def _collect_summary(
    results_dir: Path,
    base_run_id: str,
    seeds: list[int],
    experiments: list[str],
    output_path: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        run_id = f"{base_run_id}_seed{seed}"
        run_dir = results_dir / "runs" / run_id
        for exp in experiments:
            metrics = _load_status_metrics(run_dir, exp)
            param_name, param_idx, param_value = _parse_sensitivity_name(exp)
            rows.append(
                {
                    "experiment": exp,
                    "seed": seed,
                    "alc": metrics.get("alc", math.nan),
                    "final_miou": metrics.get("final_miou", math.nan),
                    "final_f1": metrics.get("final_f1", math.nan),
                    "param_name": param_name,
                    "param_index": param_idx,
                    "param_value": param_value,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "seed",
        "alc",
        "final_miou",
        "final_f1",
        "param_name",
        "param_index",
        "param_value",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Sensitivity summary saved: {output_path} ({len(rows)} rows)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run sensitivity sweep experiments")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument(
        "--start", type=str, default="fresh", choices=["fresh", "resume"]
    )
    parser.add_argument("--experiments", type=str, nargs="+", default=None)
    parser.add_argument("--include_no_risk_control", action="store_true", default=True)
    parser.add_argument("--no_risk_control_only", action="store_true", default=False)
    parser.add_argument("--n_rounds", type=int, default=None)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--exp_workers", type=int, default=1)
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    base_run_id = (
        args.run_id or f"sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    seeds = [int(s.strip()) for s in args.seeds.replace(",", " ").split() if s.strip()]
    if not seeds:
        seeds = [42]

    if args.experiments:
        experiments = list(args.experiments)
    elif args.no_risk_control_only:
        experiments = ["no_risk_control"]
    else:
        experiments = list(MULTI_SEED_DEFAULTS.get("sensitivity_experiments", []))
        if args.include_no_risk_control:
            experiments = ["no_risk_control"] + experiments

    group_dir = results_dir / "runs" / base_run_id
    group_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "base_run_id": base_run_id,
        "seeds": seeds,
        "experiments": experiments,
        "created_at": datetime.now().isoformat(),
    }
    (group_dir / "sensitivity_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    jobs = []
    for seed in seeds:
        run_id = f"{base_run_id}_seed{seed}"
        jobs.append(
            {
                "results_dir": str(results_dir),
                "base_run_id": base_run_id,
                "seed": seed,
                "run_id": run_id,
                "start": args.start,
                "experiments": experiments,
                "n_rounds": args.n_rounds,
                "exp_workers": int(args.exp_workers),
            }
        )

    if args.parallel:
        max_workers = (
            int(args.workers)
            if int(args.workers) > 0
            else min(len(jobs), os.cpu_count() or 2)
        )
        max_workers = max(1, max_workers)
        ctx = multiprocessing.get_context("spawn")
        logger.info(
            f"SENSITIVITY PARALLEL | base={base_run_id} | workers={max_workers} | seeds={seeds}"
        )
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
            _run_single_seed_job(j)

    _collect_summary(
        results_dir,
        base_run_id,
        seeds,
        experiments,
        group_dir / "sensitivity_summary.csv",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
