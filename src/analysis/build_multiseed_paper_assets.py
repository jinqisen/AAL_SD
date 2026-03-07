from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from statistics import fmean, stdev
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = [
    "full_model_A_lambda_policy",
    "full_model_B_lambda_agent",
    "baseline_entropy",
    "baseline_bald",
    "baseline_dial_style",
    "baseline_wang_style",
    "baseline_coreset",
    "baseline_random",
]

LABELS = {
    "full_model_A_lambda_policy": "AAL-SD (policy full)",
    "full_model_B_lambda_agent": "AAL-SD (agent-lambda)",
    "baseline_entropy": "Entropy",
    "baseline_bald": "BALD",
    "baseline_dial_style": "DIAL-style",
    "baseline_wang_style": "Wang-style",
    "baseline_coreset": "Core-set",
    "baseline_random": "Random",
}

COLORS = {
    "full_model_A_lambda_policy": "#0b5d8b",
    "full_model_B_lambda_agent": "#1f8a70",
    "baseline_entropy": "#f28e2b",
    "baseline_bald": "#8c564b",
    "baseline_dial_style": "#b07aa1",
    "baseline_wang_style": "#e15759",
    "baseline_coreset": "#9c755f",
    "baseline_random": "#7f7f7f",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_dir", required=True)
    parser.add_argument("--run_dirs", nargs="+", required=True)
    return parser.parse_args()


def _seed_from_run_dir(run_dir: Path) -> int:
    match = re.search(r"seed(\d+)", run_dir.name)
    if not match:
        raise ValueError(f"Cannot parse seed from run dir: {run_dir}")
    return int(match.group(1))


def _t_critical_95(df: int) -> float:
    table = {
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
    }
    return table.get(df, 1.96)


def _summary(values: Iterable[float]) -> Tuple[int, float, float, float]:
    xs = [float(v) for v in values if pd.notna(v)]
    if not xs:
        return 0, math.nan, math.nan, math.nan
    if len(xs) == 1:
        return 1, xs[0], 0.0, math.nan
    mean = float(fmean(xs))
    std = float(stdev(xs))
    ci95 = float(_t_critical_95(len(xs) - 1) * std / math.sqrt(len(xs)))
    return len(xs), mean, std, ci95


def _category(name: str) -> str:
    if name.startswith("full_model"):
        return "proposed"
    return "baseline"


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
            if not isinstance(payload, dict):
                raise TypeError(f"Expected JSON object at {path}:{line_no}, got {type(payload).__name__}")
            yield payload


def _load_status_metrics(run_dir: Path, method: str) -> Dict[str, float]:
    status_path = run_dir / f"{method}_status.json"
    if not status_path.exists():
        return {}
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    result = payload.get("result", {}) if isinstance(payload, dict) else {}
    if not isinstance(result, dict):
        return {}
    return {
        "alc": float(result.get("alc", math.nan)),
        "final_miou": float(result.get("final_mIoU", math.nan)),
        "final_f1": float(result.get("final_f1", math.nan)),
    }


def _load_agent_round_metrics(run_dir: Path, method: str) -> pd.DataFrame:
    rows: Dict[int, Dict[str, float]] = {}
    for event in _iter_jsonl(run_dir / f"{method}_trace.jsonl"):
        if event.get("type") != "round_summary":
            continue
        round_idx = event.get("round")
        if round_idx is None:
            continue
        training_state = event.get("training_state", {})
        labeled_size = None
        if isinstance(training_state, dict):
            labeled_size = training_state.get("current_labeled_count")
        if labeled_size is None:
            labeled_size = event.get("labeled_size")
        rows[int(round_idx)] = {
            "round": int(round_idx),
            "labeled_size": int(labeled_size) if labeled_size is not None else math.nan,
            "final_miou": float(event.get("mIoU", math.nan)),
            "final_f1": float(event.get("f1", math.nan)),
        }
    return pd.DataFrame([rows[key] for key in sorted(rows.keys())])


def _infer_training_sizes(curve_df: pd.DataFrame) -> List[int]:
    labeled_sizes = [int(v) for v in curve_df["labeled_size"].tolist()]
    if not labeled_sizes:
        return []
    diffs = [b - a for a, b in zip(labeled_sizes[:-1], labeled_sizes[1:]) if b - a > 0]
    step = int(round(float(np.median(diffs)))) if diffs else 88
    initial = labeled_sizes[0] - step
    training_sizes = [initial] + labeled_sizes[:-1]
    return training_sizes


def _compute_alc(curve_df: pd.DataFrame, total_budget: int = 1519) -> float:
    curve_df = curve_df.sort_values("round").copy()
    if curve_df.empty:
        return math.nan
    sizes = [int(v) for v in curve_df["labeled_size"].tolist()]
    miou = [float(v) for v in curve_df["final_miou"].tolist()]
    if not sizes or len(sizes) != len(miou):
        return math.nan
    x = np.array(sizes, dtype=float) / float(total_budget)
    y = np.array(miou, dtype=float)
    if x[-1] < 1.0:
        x = np.append(x, 1.0)
        y = np.append(y, y[-1])
    return float(np.trapezoid(y, x))


def _load_seed_tables(run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = run_dir / "run_experiment_summary.csv"
    per_round_path = run_dir / "run_all_experiments_per_round_with_grad.csv"
    summary = pd.read_csv(summary_path)
    per_round = pd.read_csv(per_round_path)
    summary = summary[summary["experiment_name"].isin(METHOD_ORDER)].copy()
    per_round = per_round[per_round["experiment_name"].isin(METHOD_ORDER)].copy()
    return summary, per_round


def _build_seed_metric_rows(
    run_dir: Path, summary: pd.DataFrame, per_round: pd.DataFrame
) -> pd.DataFrame:
    seed = _seed_from_run_dir(run_dir)
    output_rows: List[Dict[str, object]] = []
    for method in METHOD_ORDER:
        srows = summary.loc[summary["experiment_name"] == method]
        prow = (
            per_round.loc[per_round["experiment_name"] == method]
            .sort_values("round")
            .copy()
        )
        if srows.empty or prow.empty:
            continue
        srow = srows.iloc[0]
        prow.loc[:, "labeled_size"] = _infer_training_sizes(prow)
        status_metrics = (
            _load_status_metrics(run_dir, method)
            if method.startswith("full_model")
            else {}
        )
        if status_metrics:
            alc = float(status_metrics.get("alc", math.nan))
            final_miou = float(status_metrics.get("final_miou", math.nan))
            final_f1 = float(status_metrics.get("final_f1", math.nan))
        else:
            last_row = prow.iloc[-1]
            alc = _compute_alc(prow)
            final_miou = float(
                last_row.get("final_miou", srow.get("last_miou", math.nan))
            )
            final_f1 = float(last_row.get("final_f1", math.nan))
        output_rows.append(
            {
                "seed": seed,
                "run_id": run_dir.name,
                "experiment": method,
                "label": LABELS[method],
                "category": _category(method),
                "alc": alc,
                "final_miou": final_miou,
                "final_f1": final_f1,
                "lambda_mean": float(srow.get("lambda_mean", math.nan))
                if pd.notna(srow.get("lambda_mean", math.nan))
                else math.nan,
                "lambda_last": float(srow.get("lambda_last", math.nan))
                if pd.notna(srow.get("lambda_last", math.nan))
                else math.nan,
                "overfit_risk_mean": float(srow.get("overfit_risk_mean", math.nan))
                if pd.notna(srow.get("overfit_risk_mean", math.nan))
                else math.nan,
                "tvc_neg_rate_mean": float(srow.get("tvc_neg_rate_mean", math.nan))
                if pd.notna(srow.get("tvc_neg_rate_mean", math.nan))
                else math.nan,
                "labeled_size": int(prow.iloc[-1].get("labeled_size", math.nan))
                if pd.notna(prow.iloc[-1].get("labeled_size", math.nan))
                else math.nan,
            }
        )
    return pd.DataFrame(output_rows)[
        [
            "seed",
            "run_id",
            "experiment",
            "label",
            "category",
            "alc",
            "final_miou",
            "final_f1",
            "lambda_mean",
            "lambda_last",
            "overfit_risk_mean",
            "tvc_neg_rate_mean",
            "labeled_size",
        ]
    ]


def _build_seed_curve_rows(run_dir: Path, per_round: pd.DataFrame) -> pd.DataFrame:
    seed = _seed_from_run_dir(run_dir)
    rows = []
    keep_cols = [
        "round",
        "labeled_size",
        "final_miou",
        "final_f1",
        "grad_train_val_cos_last",
        "overfit_risk",
        "lambda_eff",
        "lambda_policy_rule",
        "experiment_name",
    ]
    df = per_round[[c for c in keep_cols if c in per_round.columns]].copy()
    for method in METHOD_ORDER:
        method_df = df[df["experiment_name"] == method].sort_values("round").copy()
        if method_df.empty:
            continue
        method_df.loc[:, "labeled_size"] = _infer_training_sizes(method_df)
        if method.startswith("full_model"):
            agent_curve = _load_agent_round_metrics(run_dir, method)
            if not agent_curve.empty:
                method_df = method_df.merge(
                    agent_curve[["round", "labeled_size", "final_miou", "final_f1"]],
                    on="round",
                    how="left",
                    suffixes=("", "_trace"),
                )
                method_df["labeled_size"] = method_df["labeled_size_trace"].fillna(
                    method_df["labeled_size"]
                )
                method_df["final_miou"] = method_df["final_miou_trace"].fillna(
                    method_df["final_miou"]
                )
                method_df["final_f1"] = method_df["final_f1_trace"].fillna(
                    method_df["final_f1"]
                )
                method_df = method_df.drop(
                    columns=[
                        "labeled_size_trace",
                        "final_miou_trace",
                        "final_f1_trace",
                    ],
                    errors="ignore",
                )
        for _, row in method_df.iterrows():
            rows.append(
                {
                    "seed": seed,
                    "run_id": run_dir.name,
                    "experiment": method,
                    "label": LABELS[method],
                    "round": int(row["round"]),
                    "labeled_size": int(row["labeled_size"]),
                    "final_miou": float(row["final_miou"]),
                    "final_f1": float(row["final_f1"]),
                    "grad_train_val_cos_last": float(row["grad_train_val_cos_last"])
                    if pd.notna(row.get("grad_train_val_cos_last", math.nan))
                    else math.nan,
                    "overfit_risk": float(row["overfit_risk"])
                    if pd.notna(row.get("overfit_risk", math.nan))
                    else math.nan,
                    "lambda_eff": float(row["lambda_eff"])
                    if pd.notna(row.get("lambda_eff", math.nan))
                    else math.nan,
                    "lambda_policy_rule": str(row["lambda_policy_rule"])
                    if pd.notna(row.get("lambda_policy_rule", math.nan))
                    else "",
                }
            )
    return pd.DataFrame(rows)


def _aggregate_metrics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for method in METHOD_ORDER:
        df = seed_metrics[seed_metrics["experiment"] == method]
        if df.empty:
            continue
        n, alc_mean, alc_std, alc_ci = _summary(df["alc"])
        _, miou_mean, miou_std, miou_ci = _summary(df["final_miou"])
        _, f1_mean, f1_std, f1_ci = _summary(df["final_f1"])
        _, risk_mean, risk_std, risk_ci = _summary(df["overfit_risk_mean"])
        _, lam_mean, lam_std, lam_ci = _summary(df["lambda_mean"])
        rows.append(
            {
                "experiment": method,
                "label": LABELS[method],
                "n": n,
                "alc_mean": alc_mean,
                "alc_std": alc_std,
                "alc_ci95": alc_ci,
                "final_miou_mean": miou_mean,
                "final_miou_std": miou_std,
                "final_miou_ci95": miou_ci,
                "final_f1_mean": f1_mean,
                "final_f1_std": f1_std,
                "final_f1_ci95": f1_ci,
                "overfit_risk_mean": risk_mean,
                "overfit_risk_std": risk_std,
                "overfit_risk_ci95": risk_ci,
                "lambda_mean_mean": lam_mean,
                "lambda_mean_std": lam_std,
                "lambda_mean_ci95": lam_ci,
            }
        )
    return pd.DataFrame(rows)


def _aggregate_curves(seed_curves: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    grouped = seed_curves.groupby(
        ["experiment", "label", "round", "labeled_size"], as_index=False
    )
    for (experiment, label, round_idx, labeled_size), df in grouped:
        n, mean, std, ci95 = _summary(df["final_miou"])
        rows.append(
            {
                "experiment": experiment,
                "label": label,
                "round": int(round_idx),
                "labeled_size": int(labeled_size),
                "n": n,
                "miou_mean": mean,
                "miou_std": std,
                "miou_ci95": ci95,
            }
        )
    return pd.DataFrame(rows)


def _aggregate_gradient_summary(seed_curves: pd.DataFrame) -> pd.DataFrame:
    seed_level_rows: List[Dict[str, object]] = []
    for (seed, experiment, label), df in seed_curves.groupby(
        ["seed", "experiment", "label"]
    ):
        grad = df["grad_train_val_cos_last"].dropna()
        risk = df["overfit_risk"].dropna()
        lam = df["lambda_eff"].dropna()
        seed_level_rows.append(
            {
                "seed": int(seed),
                "experiment": experiment,
                "label": label,
                "mean_grad_cos_last": float(grad.mean())
                if not grad.empty
                else math.nan,
                "neg_grad_rate": float((grad < 0).mean())
                if not grad.empty
                else math.nan,
                "mean_overfit_risk": float(risk.mean()) if not risk.empty else math.nan,
                "mean_lambda_eff": float(lam.mean()) if not lam.empty else math.nan,
            }
        )
    seed_level = pd.DataFrame(seed_level_rows)
    rows: List[Dict[str, object]] = []
    for method in METHOD_ORDER:
        df = seed_level[seed_level["experiment"] == method]
        if df.empty:
            continue
        n, grad_mean, grad_std, grad_ci = _summary(df["mean_grad_cos_last"])
        _, neg_mean, neg_std, neg_ci = _summary(df["neg_grad_rate"])
        _, risk_mean, risk_std, risk_ci = _summary(df["mean_overfit_risk"])
        _, lam_mean, lam_std, lam_ci = _summary(df["mean_lambda_eff"])
        rows.append(
            {
                "experiment": method,
                "label": LABELS[method],
                "n": n,
                "mean_grad_cos_last": grad_mean,
                "mean_grad_cos_last_std": grad_std,
                "mean_grad_cos_last_ci95": grad_ci,
                "neg_grad_rate": neg_mean,
                "neg_grad_rate_std": neg_std,
                "neg_grad_rate_ci95": neg_ci,
                "mean_overfit_risk": risk_mean,
                "mean_overfit_risk_std": risk_std,
                "mean_overfit_risk_ci95": risk_ci,
                "mean_lambda_eff": lam_mean,
                "mean_lambda_eff_std": lam_std,
                "mean_lambda_eff_ci95": lam_ci,
            }
        )
    return pd.DataFrame(rows)


def _plot_learning_curves(curves: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for method in METHOD_ORDER:
        df = curves[curves["experiment"] == method].sort_values("labeled_size")
        if df.empty:
            continue
        color = COLORS[method]
        lw = 3.0 if method.startswith("full_model") else 2.0
        ax.plot(
            df["labeled_size"],
            df["miou_mean"],
            label=LABELS[method],
            color=color,
            linewidth=lw,
        )
        if df["miou_ci95"].notna().any():
            lower = df["miou_mean"] - df["miou_ci95"].fillna(0.0)
            upper = df["miou_mean"] + df["miou_ci95"].fillna(0.0)
            ax.fill_between(df["labeled_size"], lower, upper, color=color, alpha=0.12)
    ax.set_title("Multi-seed Learning Curves on Landslide4Sense")
    ax.set_xlabel("Labeled Pool Size")
    ax.set_ylabel("mIoU")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_bar(
    summary: pd.DataFrame, metric: str, output_path: Path, title: str
) -> None:
    mean_col = f"{metric}_mean"
    ci_col = f"{metric}_ci95"
    df = summary.sort_values(mean_col, ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    colors = [COLORS.get(exp, "#4c78a8") for exp in df["experiment"]]
    ax.bar(
        df["label"],
        df[mean_col],
        yerr=df[ci_col].fillna(0.0),
        color=colors,
        alpha=0.9,
        capsize=4,
    )
    ax.set_title(title)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.grid(axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_controller_case(seed_curves: pd.DataFrame, output_path: Path) -> None:
    focus = seed_curves[
        (seed_curves["seed"] == 42)
        & (
            seed_curves["experiment"].isin(
                ["full_model_A_lambda_policy", "full_model_B_lambda_agent"]
            )
        )
    ].copy()
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    panels = [
        ("final_miou", "mIoU", None),
        ("lambda_eff", "Effective lambda", None),
        ("grad_train_val_cos_last", "Train-val gradient cosine", 0.0),
        ("overfit_risk", "Overfit risk", None),
    ]
    for method in ["full_model_A_lambda_policy", "full_model_B_lambda_agent"]:
        df = focus[focus["experiment"] == method].sort_values("round")
        if df.empty:
            continue
        color = COLORS[method]
        for ax, (col, ylabel, ref) in zip(axes, panels):
            ax.plot(
                df["round"],
                df[col],
                marker="o",
                linewidth=2.2,
                color=color,
                label=LABELS[method],
            )
            if ref is not None:
                ax.axhline(ref, color="#555555", linestyle="--", linewidth=1)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
    axes[0].set_title("Seed-42 Controller Trajectory: Policy Full vs Agent-Lambda")
    axes[-1].set_xlabel("Round")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_gradient_diagnostics(summary: pd.DataFrame, output_path: Path) -> None:
    df = summary.sort_values("mean_overfit_risk", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
    axes[0].barh(
        y,
        df["mean_overfit_risk"],
        xerr=df["mean_overfit_risk_ci95"].fillna(0.0),
        color=[COLORS[e] for e in df["experiment"]],
        alpha=0.9,
    )
    axes[0].set_title("Mean overfit risk")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(
        y,
        df["neg_grad_rate"],
        xerr=df["neg_grad_rate_ci95"].fillna(0.0),
        color=[COLORS[e] for e in df["experiment"]],
        alpha=0.9,
    )
    axes[1].set_title("Negative gradient-cosine rate")
    axes[1].grid(axis="x", alpha=0.25)

    axes[2].barh(
        y,
        df["mean_grad_cos_last"],
        xerr=df["mean_grad_cos_last_ci95"].fillna(0.0),
        color=[COLORS[e] for e in df["experiment"]],
        alpha=0.9,
    )
    axes[2].axvline(0.0, color="#555555", linestyle="--", linewidth=1)
    axes[2].set_title("Mean train-val gradient cosine")
    axes[2].grid(axis="x", alpha=0.25)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df["label"])
    for ax in axes[1:]:
        ax.set_yticks(y)
        ax.set_yticklabels([])
    fig.suptitle("Optimization-Proxy Diagnostics Across Methods", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_manifest(output_path: Path, run_dirs: List[Path]) -> None:
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source_run_dirs": [str(path) for path in run_dirs],
        "methods": METHOD_ORDER,
        "notes": [
            "Figures and CSV files are regenerated from baseline_20260228_124857_seed42-45.",
            "Multi-seed comparison is restricted to methods present in all four seeds.",
            "full_model_A_lambda_policy is treated as the primary full model.",
        ],
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )


def main() -> None:
    args = _parse_args()
    paper_dir = Path(args.paper_dir).expanduser().resolve()
    run_dirs = [Path(path).expanduser().resolve() for path in args.run_dirs]
    paper_dir.mkdir(parents=True, exist_ok=True)

    all_seed_metrics = []
    all_seed_curves = []
    for run_dir in run_dirs:
        summary, per_round = _load_seed_tables(run_dir)
        all_seed_metrics.append(_build_seed_metric_rows(run_dir, summary, per_round))
        all_seed_curves.append(_build_seed_curve_rows(run_dir, per_round))

    seed_metrics = pd.concat(all_seed_metrics, ignore_index=True)
    seed_curves = pd.concat(all_seed_curves, ignore_index=True)

    seed42_metrics = seed_metrics[seed_metrics["seed"] == 42].copy()
    metrics_summary = _aggregate_metrics(seed_metrics)
    multiseed_curves = _aggregate_curves(seed_curves)
    gradient_summary = _aggregate_gradient_summary(seed_curves)
    controller_trajectories = seed_curves[
        (seed_curves["seed"] == 42)
        & (
            seed_curves["experiment"].isin(
                ["full_model_A_lambda_policy", "full_model_B_lambda_agent"]
            )
        )
    ].copy()

    seed42_metrics.to_csv(
        paper_dir / "metrics_summary.csv", index=False, encoding="utf-8"
    )
    metrics_summary.to_csv(
        paper_dir / "multiseed_metrics_summary.csv", index=False, encoding="utf-8"
    )
    multiseed_curves.to_csv(
        paper_dir / "multiseed_round_curves.csv", index=False, encoding="utf-8"
    )
    gradient_summary.to_csv(
        paper_dir / "multiseed_gradient_summary.csv", index=False, encoding="utf-8"
    )
    controller_trajectories.to_csv(
        paper_dir / "controller_trajectories.csv", index=False, encoding="utf-8"
    )
    _write_manifest(paper_dir / "paper_assets_manifest.json", run_dirs)

    _plot_learning_curves(
        multiseed_curves, paper_dir / "Figure2_Learning_Curves_generated.png"
    )
    _plot_controller_case(
        seed_curves, paper_dir / "Figure3_Controller_Trajectory_Full_Model.png"
    )
    _plot_metric_bar(
        metrics_summary,
        "alc",
        paper_dir / "Figure4_ALC_Bar.png",
        "Multi-seed ALC (mean ± 95% CI)",
    )
    _plot_metric_bar(
        metrics_summary,
        "final_miou",
        paper_dir / "Figure5_Final_mIoU_Bar.png",
        "Multi-seed Final mIoU (mean ± 95% CI)",
    )
    _plot_gradient_diagnostics(
        gradient_summary, paper_dir / "Figure6_Gradient_Diagnostics.png"
    )


if __name__ == "__main__":
    main()
