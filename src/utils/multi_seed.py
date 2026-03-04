from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Iterable


_T_CRIT_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.16,
    14: 2.145,
    15: 2.131,
    16: 2.12,
    17: 2.11,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.08,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.06,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _t_critical_95(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T_CRIT_95:
        return float(_T_CRIT_95[df])
    return 1.96


@dataclass(frozen=True)
class MetricSummary:
    n: int
    mean: float
    std: float
    ci95: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": int(self.n),
            "mean": float(self.mean),
            "std": float(self.std),
            "ci95": float(self.ci95),
        }


def summarize(values: Iterable[float]) -> MetricSummary:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(vals)
    if n == 0:
        return MetricSummary(n=0, mean=float("nan"), std=float("nan"), ci95=float("nan"))
    if n == 1:
        return MetricSummary(n=1, mean=float(vals[0]), std=0.0, ci95=float("nan"))
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals)
    sem = std / math.sqrt(n)
    t = _t_critical_95(n - 1)
    ci = float(t * sem)
    return MetricSummary(n=n, mean=float(mean), std=float(std), ci95=ci)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_run_results(run_dir: Path) -> dict[str, dict[str, Any]]:
    main_path = run_dir / "experiment_results.json"
    payload = _load_json(main_path)
    if isinstance(payload, dict) and payload:
        return {str(k): v for k, v in payload.items() if isinstance(v, dict)}

    merged: dict[str, dict[str, Any]] = {}
    for p in sorted(run_dir.glob("result_*.json")):
        item = _load_json(p)
        if not isinstance(item, dict):
            continue
        for exp_name, exp_payload in item.items():
            if isinstance(exp_payload, dict):
                merged[str(exp_name)] = exp_payload
    return merged


def aggregate_multi_seed(
    results_dir: Path,
    run_ids: list[str],
    *,
    metrics: tuple[str, ...] = ("alc", "final_miou", "final_f1"),
) -> dict[str, Any]:
    runs: dict[str, dict[str, dict[str, Any]]] = {}
    for run_id in run_ids:
        run_dir = results_dir / "runs" / run_id
        runs[run_id] = load_run_results(run_dir)

    experiments: set[str] = set()
    for run_payload in runs.values():
        experiments.update(run_payload.keys())

    by_experiment: dict[str, Any] = {}
    for exp in sorted(experiments):
        exp_block: dict[str, Any] = {"per_seed": {}, "summary": {}}
        for metric in metrics:
            exp_block["per_seed"][metric] = {}
        for run_id in run_ids:
            record = runs.get(run_id, {}).get(exp)
            if not isinstance(record, dict):
                continue
            for metric in metrics:
                value = record.get(metric)
                if value is None:
                    continue
                try:
                    exp_block["per_seed"][metric][run_id] = float(value)
                except Exception:
                    continue
        for metric in metrics:
            values = list(exp_block["per_seed"][metric].values())
            exp_block["summary"][metric] = summarize(values).as_dict()
        statuses: dict[str, Any] = {}
        for run_id in run_ids:
            record = runs.get(run_id, {}).get(exp)
            if isinstance(record, dict):
                statuses[run_id] = record.get("status")
        exp_block["statuses"] = statuses
        by_experiment[exp] = exp_block

    return {"run_ids": list(run_ids), "experiments": by_experiment}


def render_markdown(summary: Mapping[str, Any]) -> str:
    run_ids = list(summary.get("run_ids") or [])
    experiments = summary.get("experiments")
    if not isinstance(experiments, Mapping):
        experiments = {}

    lines: list[str] = []
    lines.append("# Multi-Seed Summary\n")
    lines.append(f"Run IDs: {', '.join(run_ids)}\n")
    lines.append("\n## 汇总表\n")
    lines.append("| experiment | metric | n | mean | std | ci95 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for exp_name in sorted(experiments.keys()):
        exp_payload = experiments.get(exp_name)
        if not isinstance(exp_payload, Mapping):
            continue
        summary_block = exp_payload.get("summary")
        if not isinstance(summary_block, Mapping):
            continue
        for metric, stats_payload in summary_block.items():
            if not isinstance(stats_payload, Mapping):
                continue
            n = stats_payload.get("n")
            mean = stats_payload.get("mean")
            std = stats_payload.get("std")
            ci95 = stats_payload.get("ci95")
            try:
                lines.append(
                    f"| {exp_name} | {metric} | {int(n)} | {float(mean):.6f} | {float(std):.6f} | {float(ci95):.6f} |\n"
                )
            except Exception:
                lines.append(f"| {exp_name} | {metric} | - | - | - | - |\n")
    return "".join(lines)

