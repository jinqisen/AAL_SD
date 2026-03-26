from __future__ import annotations

import argparse
import copy
import json
import shutil
from datetime import datetime
from pathlib import Path

from config import Config
from experiments.ablation_config import ABLATION_SETTINGS
from experiments.run_all_experiments import ExperimentRunner
from utils.logger import logger
from utils.multi_seed import aggregate_multi_seed, render_markdown


def _parse_seeds(raw: str) -> list[int]:
    tokens = [t.strip() for t in raw.replace(",", " ").split() if t.strip()]
    return [int(t) for t in tokens]


def _seed_run_ids(base_run_id: str, seeds: list[int]) -> list[str]:
    return [f"{base_run_id}_seed{seed}" for seed in seeds]


def _ensure_group_dir(results_dir: Path, base_run_id: str) -> Path:
    group_dir = results_dir / "runs" / base_run_id
    group_dir.mkdir(parents=True, exist_ok=True)
    return group_dir


def _copy_base_pools(
    results_dir: Path,
    source_base_run_id: str,
    target_base_run_id: str,
    seed: int,
    start_mode: str,
) -> None:
    pools_dir = results_dir / "pools"
    src = pools_dir / f"{source_base_run_id}_seed{seed}" / "_base"
    dst = pools_dir / f"{target_base_run_id}_seed{seed}" / "_base"
    if start_mode == "fresh":
        if not src.exists():
            raise FileNotFoundError(f"Missing source pools: {src}")
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
        return
    if dst.exists():
        return
    if not src.exists():
        raise FileNotFoundError(f"Missing source pools: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def _geometry_enabled_cfg(base_cfg: dict, *, description: str | None = None, lambda_override=None, use_agent=None) -> dict:
    cfg = copy.deepcopy(base_cfg)
    if description is not None:
        cfg["description"] = description
    if lambda_override is not None:
        cfg["lambda_override"] = float(lambda_override)
    if use_agent is not None:
        cfg["use_agent"] = bool(use_agent)
    cfg["enable_l3_selection_logging"] = True
    cfg["enable_score_snapshot_logging"] = True
    cfg.setdefault("l3_topk", 256)
    cfg.setdefault("l3_max_selected", 256)
    cfg.setdefault("score_snapshot_boundary_window", 64)
    cfg.setdefault("score_snapshot_max_pool_items", None)
    cfg.setdefault("geometry_boundary_delta_ratio", 0.2)
    cfg.setdefault("geometry_sensitivity_delta_lambda", 0.1)
    return cfg


def _register_geometry_probe_experiments() -> None:
    if "geometry_fixed_lambda_02" not in ABLATION_SETTINGS:
        base = ABLATION_SETTINGS.get("uncertainty_only", {})
        ABLATION_SETTINGS["geometry_fixed_lambda_02"] = _geometry_enabled_cfg(
            base,
            description="Geometry probe: fixed λ=0.2, no-agent",
            lambda_override=0.2,
            use_agent=False,
        )
    for name in ("uncertainty_only", "knowledge_only", "full_model_A_lambda_policy", "fixed_lambda"):
        if name in ABLATION_SETTINGS:
            ABLATION_SETTINGS[name] = _geometry_enabled_cfg(ABLATION_SETTINGS[name])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument(
        "--source_base_run_id",
        type=str,
        default="baselines_only_p3_20260313_212023",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=f"geometry_probe_p3_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--start", type=str, default="fresh", choices=["fresh", "resume"])
    parser.add_argument("--n_rounds", type=int, default=16)
    parser.add_argument("--parallel_workers", type=int, default=2)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["uncertainty_only", "geometry_fixed_lambda_02", "full_model_A_lambda_policy"],
    )
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_seeds(args.seeds)
    _register_geometry_probe_experiments()

    group_dir = _ensure_group_dir(results_dir, args.run_id)
    manifest_path = group_dir / "multi_seed_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "base_run_id": args.run_id,
                "source_base_run_id": args.source_base_run_id,
                "seeds": seeds,
                "run_ids": _seed_run_ids(args.run_id, seeds),
                "experiments": list(args.experiments),
                "created_at": datetime.now().isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    for seed, run_id in zip(seeds, _seed_run_ids(args.run_id, seeds)):
        _copy_base_pools(
            results_dir,
            args.source_base_run_id,
            args.run_id,
            int(seed),
            args.start,
        )
        cfg = Config()
        cfg.SEED = int(seed)
        cfg.START_MODE = str(args.start)
        cfg.N_ROUNDS = int(args.n_rounds)
        cfg.SUPPRESS_MANIFEST_UPDATE = True
        runner = ExperimentRunner(cfg, str(results_dir), run_id=run_id, start_mode=str(args.start))
        logger.info(
            "GEOMETRY PROBE RUN | source_base=%s | run_id=%s | seed=%s | parallel_workers=%s | experiments=%s",
            args.source_base_run_id,
            run_id,
            seed,
            args.parallel_workers,
            ",".join(args.experiments),
        )
        runner.run_all_experiments(
            list(args.experiments),
            parallel_workers=max(1, int(args.parallel_workers)),
        )

    summary = aggregate_multi_seed(results_dir, _seed_run_ids(args.run_id, seeds))
    summary_path = group_dir / "multi_seed_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = group_dir / "multi_seed_report.md"
    md_path.write_text(render_markdown(summary), encoding="utf-8")
    logger.info("Geometry probe summary saved: %s", summary_path)
    logger.info("Geometry probe report saved: %s", md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
