"""Generate all figures needed by IEEE_JSTARS_Paper_Draft.md.

Figures produced (saved into paper_draft/figures/):
  Fig1  – architecture placeholder (copies existing concept figure)
  Fig2  – multi-seed mean learning curves with 95% CI bands
  Fig3  – seed-42 controller trajectory (A vs B): mIoU, lambda, gradient cosine, overfit risk
  Fig4  – multi-seed ALC bar chart (mean +/- 95% CI)
  Fig5  – multi-seed final-mIoU bar chart (mean +/- 95% CI)
  Fig6  – seed-42 ablation learning curves (full model vs component ablations)
  Fig7  – multi-seed gradient diagnostics (horizontal bars)
  Fig8  – seed-42 lambda-gradient coupling scatter (supports closed-loop claim)
  Fig9  – four-seed ALC & final-mIoU box plots (supports robustness claim)

Data sources:
  - 4 seeds: results/runs/baseline_20260228_124857_seed{42,43,44,45}
  - pre-aggregated CSVs: paper/multiseed_metrics_summary.csv, paper/multiseed_round_curves.csv
  - seed-42 full per-round CSV for ablation curves and controller traces
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent  # AAL_SD/
DRAFT = REPO / "paper_draft"
FIG_DIR = DRAFT / "figures"
PAPER = REPO / "paper"
RUNS = REPO / "results" / "runs"
SEEDS = [42, 43, 44, 45]
SEED_DIRS = [RUNS / f"baseline_20260228_124857_seed{s}" for s in SEEDS]

# ── style ────────────────────────────────────────────────────────────────
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
    "full_model_A_lambda_policy": "AAL-SD (A, policy)",
    "full_model_B_lambda_agent": "AAL-SD (B, agent-λ)",
    "baseline_entropy": "Entropy",
    "baseline_bald": "BALD",
    "baseline_dial_style": "DIAL-style",
    "baseline_wang_style": "Wang-style",
    "baseline_coreset": "Core-Set",
    "baseline_random": "Random",
}
COLORS = {
    "full_model_A_lambda_policy": "#d62728",
    "full_model_B_lambda_agent": "#1f77b4",
    "baseline_entropy": "#ff7f0e",
    "baseline_bald": "#8c564b",
    "baseline_dial_style": "#9467bd",
    "baseline_wang_style": "#2ca02c",
    "baseline_coreset": "#bcbd22",
    "baseline_random": "#7f7f7f",
}
ABLATION_ORDER = [
    "full_model_A_lambda_policy",
    "no_agent",
    "fixed_lambda",
    "uncertainty_only",
    "knowledge_only",
]
ABLATION_LABELS = {
    "full_model_A_lambda_policy": "AAL-SD (full)",
    "no_agent": "w/o Agent",
    "fixed_lambda": "Fixed λ=0.5",
    "uncertainty_only": "Uncertainty only (λ=0)",
    "knowledge_only": "Knowledge only (λ=1)",
}
ABLATION_COLORS = {
    "full_model_A_lambda_policy": "#d62728",
    "no_agent": "#1f77b4",
    "fixed_lambda": "#2ca02c",
    "uncertainty_only": "#ff7f0e",
    "knowledge_only": "#9467bd",
}

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "figure.dpi": 200,
    }
)


def t_critical_95(n: int) -> float:
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
    return table.get(n - 1, 1.96)


# ── data loaders ─────────────────────────────────────────────────────────
def load_multiseed_curves() -> pd.DataFrame:
    """Load per-round data from all seeds for the 8 shared methods."""
    frames = []
    for seed, sd in zip(SEEDS, SEED_DIRS):
        df = pd.read_csv(sd / "run_all_experiments_per_round_with_grad.csv")
        df["seed"] = seed
        frames.append(df[df["experiment_name"].isin(METHOD_ORDER)])
    return pd.concat(frames, ignore_index=True)


def load_seed42_ablation_curves() -> pd.DataFrame:
    """Load per-round data from seed-42 for ablation methods."""
    sd = SEED_DIRS[0]
    df = pd.read_csv(sd / "run_all_experiments_per_round_with_grad.csv")
    return df[df["experiment_name"].isin(ABLATION_ORDER)]


def load_metrics_summary() -> pd.DataFrame:
    return pd.read_csv(PAPER / "multiseed_metrics_summary.csv")


def load_gradient_summary() -> pd.DataFrame:
    return pd.read_csv(PAPER / "multiseed_gradient_summary.csv")


# ── Fig 1: architecture ─────────────────────────────────────────────────
def fig1_architecture():
    src = PAPER / "Figure1_AAL_SD_Architecture.png"
    dst = FIG_DIR / "Fig1_Architecture.png"
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  Fig1: copied from {src}")
    else:
        print(f"  Fig1: WARNING – source not found at {src}")


# ── Fig 2: multi-seed learning curves +末段 inset 放大 ─────────────────
def fig2_multiseed_learning_curves():
    raw = load_multiseed_curves()
    fig, ax = plt.subplots(figsize=(10, 6))

    # collect data for inset
    all_means = {}
    all_cis = {}
    for method in METHOD_ORDER:
        mdf = raw[raw["experiment_name"] == method]
        grouped = mdf.groupby("round")["final_miou"]
        mean = grouped.mean()
        std = grouped.std()
        n = grouped.count()
        ci = std * t_critical_95(4) / np.sqrt(n)
        rounds = mean.index.values
        all_means[method] = (rounds, mean.values, ci.values)
        color = COLORS[method]
        lw = 2.8 if method.startswith("full_model") else 1.8
        ls = "-" if method.startswith("full_model") else "--"
        ax.plot(
            rounds,
            mean.values,
            label=LABELS[method],
            color=color,
            linewidth=lw,
            linestyle=ls,
        )
        ax.fill_between(
            rounds, (mean - ci).values, (mean + ci).values, color=color, alpha=0.10
        )
    ax.set_xlabel("Active Learning Round")
    ax.set_ylabel("mIoU")
    ax.set_title("Multi-Seed Mean Learning Curves (Seeds 42–45, 95% CI)")
    ax.legend(ncol=2, frameon=False, loc="lower right", fontsize=8)
    ax.grid(alpha=0.2)

    # inset: last 5 rounds zoomed
    axins = ax.inset_axes(
        [0.42, 0.08, 0.55, 0.40]
    )  # [x, y, width, height] in axes coords
    for method in METHOD_ORDER:
        rounds, mean_vals, ci_vals = all_means[method]
        mask = rounds >= 11
        color = COLORS[method]
        lw = 2.4 if method.startswith("full_model") else 1.4
        ls = "-" if method.startswith("full_model") else "--"
        axins.plot(
            rounds[mask], mean_vals[mask], color=color, linewidth=lw, linestyle=ls
        )
        axins.fill_between(
            rounds[mask],
            mean_vals[mask] - ci_vals[mask],
            mean_vals[mask] + ci_vals[mask],
            color=color,
            alpha=0.12,
        )
    axins.set_title("Rounds 11–15 (zoom)", fontsize=8)
    axins.tick_params(labelsize=7)
    axins.grid(alpha=0.2)
    ax.indicate_inset_zoom(axins, edgecolor="#555", linewidth=1)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig2_MultiSeed_Learning_Curves.png", bbox_inches="tight")
    plt.close(fig)
    print("  Fig2: multi-seed learning curves + inset")


# ── Fig 3: seed-42 controller trajectory (A vs B) ────────────────────────
def fig3_controller_trajectory():
    sd = SEED_DIRS[0]
    df = pd.read_csv(sd / "run_all_experiments_per_round_with_grad.csv")
    focus = df[
        df["experiment_name"].isin(
            ["full_model_A_lambda_policy", "full_model_B_lambda_agent"]
        )
    ]
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    panels = [
        ("final_miou", "mIoU"),
        ("lambda_eff", "Effective λ"),
        ("grad_train_val_cos_last", "Grad Cosine (train-val)"),
        ("overfit_risk", "Overfit Risk"),
    ]
    for method in ["full_model_A_lambda_policy", "full_model_B_lambda_agent"]:
        mdf = focus[focus["experiment_name"] == method].sort_values("round")
        color = COLORS[method]
        for ax, (col, ylabel) in zip(axes, panels):
            if col in mdf.columns:
                ax.plot(
                    mdf["round"],
                    mdf[col],
                    marker="o",
                    linewidth=2,
                    color=color,
                    label=LABELS[method],
                    markersize=5,
                )
    for ax, (col, ylabel) in zip(axes, panels):
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        if col == "grad_train_val_cos_last":
            ax.axhline(0, color="#555", linestyle="--", linewidth=0.8)
    axes[0].set_title(
        "Seed-42 Controller Trajectory: Policy-Full (A) vs Agent-Lambda (B)"
    )
    axes[0].legend(loc="lower right", frameon=False)
    axes[-1].set_xlabel("Round")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig3_Controller_Trajectory_Seed42.png", bbox_inches="tight")
    plt.close(fig)
    print("  Fig3: seed-42 controller trajectory")


# ── Fig 4: multi-seed ALC bar ────────────────────────────────────────────
def fig4_alc_bar():
    ms = load_metrics_summary()
    ms = (
        ms.set_index("experiment")
        .loc[[m for m in METHOD_ORDER if m in ms["experiment"].values]]
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [COLORS[e] for e in ms["experiment"]]
    ax.bar(
        ms["label"],
        ms["alc_mean"],
        yerr=ms["alc_ci95"].fillna(0),
        color=colors,
        alpha=0.88,
        capsize=5,
    )
    ax.set_ylabel("ALC")
    ax.set_title("Multi-Seed ALC Comparison (Mean ± 95% CI)")
    ax.grid(axis="y", alpha=0.2)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig4_MultiSeed_ALC_Bar.png", bbox_inches="tight")
    plt.close(fig)
    print("  Fig4: multi-seed ALC bar")


# ── Fig 5: multi-seed final mIoU bar ─────────────────────────────────────
def fig5_miou_bar():
    ms = load_metrics_summary()
    ms = (
        ms.set_index("experiment")
        .loc[[m for m in METHOD_ORDER if m in ms["experiment"].values]]
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [COLORS[e] for e in ms["experiment"]]
    ax.bar(
        ms["label"],
        ms["final_miou_mean"],
        yerr=ms["final_miou_ci95"].fillna(0),
        color=colors,
        alpha=0.88,
        capsize=5,
    )
    ax.set_ylabel("Final mIoU")
    ax.set_title("Multi-Seed Final mIoU Comparison (Mean ± 95% CI)")
    ax.grid(axis="y", alpha=0.2)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig5_MultiSeed_Final_mIoU_Bar.png", bbox_inches="tight")
    plt.close(fig)
    print("  Fig5: multi-seed final mIoU bar")


# ── Fig 6: seed-42 ablation learning curves ──────────────────────────────
def fig6_ablation_curves():
    df = load_seed42_ablation_curves()
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in ABLATION_ORDER:
        mdf = df[df["experiment_name"] == method].sort_values("round")
        if mdf.empty:
            continue
        color = ABLATION_COLORS[method]
        lw = 2.8 if method == "full_model_A_lambda_policy" else 1.8
        ax.plot(
            mdf["round"],
            mdf["final_miou"],
            label=ABLATION_LABELS[method],
            color=color,
            linewidth=lw,
            marker="o",
            markersize=4,
        )
    ax.set_xlabel("Active Learning Round")
    ax.set_ylabel("mIoU")
    ax.set_title("Seed-42 Component Ablation: Learning Curves")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig6_Ablation_Curves_Seed42.png", bbox_inches="tight")
    plt.close(fig)
    print("  Fig6: seed-42 ablation curves")


# ── Fig 7: gradient diagnostics ──────────────────────────────────────────
def fig7_gradient_diagnostics():
    gs = load_gradient_summary()
    gs = (
        gs.set_index("experiment")
        .loc[[m for m in METHOD_ORDER if m in gs["experiment"].values]]
        .reset_index()
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
    y = np.arange(len(gs))
    colors = [COLORS[e] for e in gs["experiment"]]

    axes[0].barh(
        y,
        gs["mean_overfit_risk"],
        xerr=gs["mean_overfit_risk_ci95"].fillna(0),
        color=colors,
        alpha=0.88,
        capsize=3,
    )
    axes[0].set_title("Mean Overfit Risk")
    axes[0].grid(axis="x", alpha=0.2)

    axes[1].barh(
        y,
        gs["neg_grad_rate"],
        xerr=gs["neg_grad_rate_ci95"].fillna(0),
        color=colors,
        alpha=0.88,
        capsize=3,
    )
    axes[1].set_title("Negative Gradient-Cosine Rate")
    axes[1].grid(axis="x", alpha=0.2)

    axes[2].barh(
        y,
        gs["mean_grad_cos_last"],
        xerr=gs["mean_grad_cos_last_ci95"].fillna(0),
        color=colors,
        alpha=0.88,
        capsize=3,
    )
    axes[2].axvline(0, color="#555", linestyle="--", linewidth=0.8)
    axes[2].set_title("Mean Train-Val Grad Cosine")
    axes[2].grid(axis="x", alpha=0.2)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(gs["label"])
    fig.suptitle("Optimization-Proxy Diagnostics Across Methods (4 Seeds)", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig7_Gradient_Diagnostics.png", bbox_inches="tight")
    plt.close(fig)
    print("  Fig7: gradient diagnostics")


# ── Fig 8: lambda-gradient coupling scatter (seed-42) ────────────────────
def fig8_lambda_gradient_coupling():
    sd = SEED_DIRS[0]
    df = pd.read_csv(sd / "run_all_experiments_per_round_with_grad.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for method, ax in zip(
        ["full_model_A_lambda_policy", "full_model_B_lambda_agent"], axes
    ):
        mdf = df[df["experiment_name"] == method].sort_values("round").copy()
        if mdf.empty or "lambda_eff" not in mdf.columns:
            continue
        mdf = mdf.dropna(subset=["lambda_eff", "grad_train_val_cos_last"])
        x = mdf["grad_train_val_cos_last"].values
        y = mdf["lambda_eff"].values
        rounds = mdf["round"].values
        color = COLORS[method]
        ax.scatter(x, y, c=color, s=80, edgecolors="white", linewidth=0.8, zorder=3)
        for i, r in enumerate(rounds):
            ax.annotate(
                f"R{int(r)}",
                (x[i], y[i]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=7.5,
                color="#333",
            )
        # trend line
        if len(x) >= 3:
            z = np.polyfit(x, y, 1)
            xline = np.linspace(x.min() - 0.05, x.max() + 0.05, 50)
            ax.plot(
                xline, np.polyval(z, xline), "--", color=color, alpha=0.5, linewidth=1.5
            )
        ax.axvline(0, color="#555", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Train-Val Gradient Cosine")
        ax.set_ylabel("Effective λ")
        ax.set_title(LABELS[method])
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Seed-42: Coupling Between Gradient Signal and Acquisition Weight", y=1.02
    )
    fig.tight_layout()
    fig.savefig(
        FIG_DIR / "Fig8_Lambda_Gradient_Coupling_Seed42.png", bbox_inches="tight"
    )
    plt.close(fig)
    print("  Fig8: lambda-gradient coupling scatter")


# ── Fig 9: four-seed box plots (ALC + final mIoU) ───────────────────────
def fig9_multiseed_boxplots():
    frames = []
    for seed, sd in zip(SEEDS, SEED_DIRS):
        summary = pd.read_csv(sd / "run_experiment_summary.csv")
        summary["seed"] = seed
        frames.append(summary[summary["experiment_name"].isin(METHOD_ORDER)])
    all_seeds = pd.concat(frames, ignore_index=True)

    # compute per-seed ALC from status.json for proposed methods, else use last_miou as proxy
    # For simplicity, merge with the per-seed metrics from paper/metrics_summary if available
    per_seed_path = REPO / "paper" / "metrics_summary.csv"
    if per_seed_path.exists():
        # This only has seed-42; use the full multiseed per-seed data instead
        pass

    # Build per-seed metrics from run_experiment_summary + status.json
    import json

    rows = []
    for seed, sd in zip(SEEDS, SEED_DIRS):
        summary = pd.read_csv(sd / "run_experiment_summary.csv")
        for method in METHOD_ORDER:
            srows = summary[summary["experiment_name"] == method]
            if srows.empty:
                continue
            srow = srows.iloc[0]
            # try to get ALC from status.json for proposed methods
            status_path = sd / f"{method}_status.json"
            if status_path.exists():
                with open(status_path) as f:
                    sj = json.load(f)
                result = sj.get("result", {})
                alc = result.get("alc", float("nan"))
                final_miou = result.get("final_mIoU", float("nan"))
            else:
                alc = float(srow.get("last_miou", float("nan")))
                final_miou = float(srow.get("last_miou", float("nan")))
            rows.append(
                {
                    "seed": seed,
                    "experiment": method,
                    "label": LABELS[method],
                    "alc": float(alc),
                    "final_miou": float(final_miou),
                }
            )
    per_seed = pd.DataFrame(rows)

    # Order for plotting
    order = [LABELS[m] for m in METHOD_ORDER if m in per_seed["experiment"].values]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ALC box plot
    box_data_alc = [per_seed[per_seed["label"] == lab]["alc"].values for lab in order]
    bp1 = axes[0].boxplot(box_data_alc, labels=order, patch_artist=True, widths=0.55)
    for patch, method in zip(
        bp1["boxes"], [m for m in METHOD_ORDER if m in per_seed["experiment"].values]
    ):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.7)
    # overlay individual seed points
    for i, lab in enumerate(order):
        vals = per_seed[per_seed["label"] == lab]["alc"].values
        axes[0].scatter(
            [i + 1] * len(vals),
            vals,
            color="#333",
            s=30,
            zorder=5,
            edgecolors="white",
            linewidth=0.5,
        )
    axes[0].set_ylabel("ALC")
    axes[0].set_title("ALC Distribution (4 Seeds)")
    axes[0].grid(axis="y", alpha=0.2)
    plt.setp(axes[0].get_xticklabels(), rotation=25, ha="right")

    # Final mIoU box plot
    box_data_miou = [
        per_seed[per_seed["label"] == lab]["final_miou"].values for lab in order
    ]
    bp2 = axes[1].boxplot(box_data_miou, labels=order, patch_artist=True, widths=0.55)
    for patch, method in zip(
        bp2["boxes"], [m for m in METHOD_ORDER if m in per_seed["experiment"].values]
    ):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.7)
    for i, lab in enumerate(order):
        vals = per_seed[per_seed["label"] == lab]["final_miou"].values
        axes[1].scatter(
            [i + 1] * len(vals),
            vals,
            color="#333",
            s=30,
            zorder=5,
            edgecolors="white",
            linewidth=0.5,
        )
    axes[1].set_ylabel("Final mIoU")
    axes[1].set_title("Final mIoU Distribution (4 Seeds)")
    axes[1].grid(axis="y", alpha=0.2)
    plt.setp(axes[1].get_xticklabels(), rotation=25, ha="right")

    fig.suptitle("Per-Seed Metric Distributions Across Methods", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig9_MultiSeed_Boxplots.png", bbox_inches="tight")
    plt.close(fig)
    print("  Fig9: multi-seed box plots")


# ── Fig 10: ΔmIoU difference curves (AAL-SD A minus baselines) ───────────
def fig10_delta_miou_curves():
    raw = load_multiseed_curves()
    ref_method = "full_model_A_lambda_policy"
    compare_methods = [m for m in METHOD_ORDER if m != ref_method]

    # compute per-seed, per-round paired difference
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for method in compare_methods:
        deltas_by_round = {}
        for seed in SEEDS:
            ref = raw[
                (raw["experiment_name"] == ref_method) & (raw["seed"] == seed)
            ].sort_values("round")
            comp = raw[
                (raw["experiment_name"] == method) & (raw["seed"] == seed)
            ].sort_values("round")
            if ref.empty or comp.empty:
                continue
            merged = ref[["round", "final_miou"]].merge(
                comp[["round", "final_miou"]], on="round", suffixes=("_ref", "_comp")
            )
            for _, row in merged.iterrows():
                r = int(row["round"])
                deltas_by_round.setdefault(r, []).append(
                    row["final_miou_ref"] - row["final_miou_comp"]
                )

        rounds = sorted(deltas_by_round.keys())
        means = [np.mean(deltas_by_round[r]) for r in rounds]
        if len(SEEDS) > 1:
            cis = [
                t_critical_95(len(deltas_by_round[r]))
                * np.std(deltas_by_round[r], ddof=1)
                / np.sqrt(len(deltas_by_round[r]))
                for r in rounds
            ]
        else:
            cis = [0] * len(rounds)

        color = COLORS[method]
        label = LABELS[method]
        ax.plot(
            rounds,
            means,
            label=f"vs {label}",
            color=color,
            linewidth=1.8,
            marker="s",
            markersize=4,
        )
        lower = [m - c for m, c in zip(means, cis)]
        upper = [m + c for m, c in zip(means, cis)]
        ax.fill_between(rounds, lower, upper, color=color, alpha=0.10)

    ax.axhline(0, color="#333", linewidth=1, linestyle="-")
    ax.set_xlabel("Active Learning Round")
    ax.set_ylabel("ΔmIoU  (AAL-SD A − Baseline)")
    ax.set_title(
        "Paired mIoU Difference: AAL-SD (A) vs Each Baseline (4-Seed Mean ± 95% CI)"
    )
    ax.legend(ncol=2, frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.2)

    # shade positive region
    ax.axhspan(
        0,
        ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.05,
        color="#d4edda",
        alpha=0.15,
        zorder=0,
    )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig10_DeltamIoU_Curves.png", bbox_inches="tight")
    plt.close(fig)
    print("  Fig10: ΔmIoU difference curves")


# ── Fig 11: U/K/Score distribution violin per round (seed-42, full_model_A) ──
def fig11_score_distribution():
    import json as _json

    l3_path = SEED_DIRS[0] / "figures" / "l3_selection.csv"
    if not l3_path.exists():
        print("  Fig11: SKIP – l3_selection.csv not found")
        return

    l3 = pd.read_csv(l3_path)
    fma = l3[l3["experiment"] == "full_model_A_lambda_policy"].copy()
    if fma.empty:
        print("  Fig11: SKIP – no full_model_A data in l3_selection")
        return

    # parse JSON blobs into per-sample rows
    rows = []
    for _, record in fma.iterrows():
        rnd = int(record["round"])
        top_items = _json.loads(record["top_items"])
        sel_items = (
            _json.loads(record["selected_items"])
            if pd.notna(record["selected_items"])
            else []
        )
        sel_ids = set()
        if isinstance(sel_items, list):
            for item in sel_items:
                if isinstance(item, dict):
                    sel_ids.add(item.get("sample_id"))
                else:
                    sel_ids.add(item)
        for item in top_items:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "round": rnd,
                    "sample_id": item.get("sample_id"),
                    "uncertainty": item.get("uncertainty", float("nan")),
                    "knowledge_gain": item.get("knowledge_gain", float("nan")),
                    "final_score": item.get("final_score", float("nan")),
                    "selected": item.get("sample_id") in sel_ids,
                }
            )
    samples = pd.DataFrame(rows)

    # select representative rounds: early (2), mid (7), late (12, 14)
    show_rounds = [2, 5, 8, 11, 14]
    show_rounds = [r for r in show_rounds if r in samples["round"].unique()]
    if len(show_rounds) < 3:
        show_rounds = sorted(samples["round"].unique())[:5]

    sub = samples[samples["round"].isin(show_rounds)].copy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [
        ("uncertainty", "Uncertainty U(x)", "#ff7f0e"),
        ("knowledge_gain", "Knowledge Gain K(x)", "#2ca02c"),
        ("final_score", "Final Score", "#d62728"),
    ]

    for ax, (col, title, base_color) in zip(axes, metrics):
        positions = []
        data_all = []
        data_sel = []
        labels = []
        for i, rnd in enumerate(show_rounds):
            rdf = sub[sub["round"] == rnd]
            all_vals = rdf[col].dropna().values
            sel_vals = rdf[rdf["selected"]][col].dropna().values
            data_all.append(all_vals)
            data_sel.append(sel_vals)
            positions.append(i)
            labels.append(f"R{rnd}")

        # violin for all candidates (pool)
        parts = ax.violinplot(
            data_all, positions=positions, showmeans=True, showmedians=False, widths=0.7
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(base_color)
            pc.set_alpha(0.3)
        for pn in ["cmeans", "cmins", "cmaxes", "cbars"]:
            if pn in parts:
                parts[pn].set_color(base_color)
                parts[pn].set_alpha(0.5)

        # overlay selected samples as colored dots
        for i, rnd in enumerate(show_rounds):
            sel_vals = data_sel[i]
            if len(sel_vals) > 0:
                jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(sel_vals))
                ax.scatter(
                    positions[i] + jitter,
                    sel_vals,
                    c="#d62728",
                    s=12,
                    alpha=0.7,
                    edgecolors="white",
                    linewidth=0.3,
                    zorder=5,
                    label="Selected" if i == 0 else None,
                )

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("Score Value")
    # add shared legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [
        Patch(facecolor="#ff7f0e", alpha=0.3, label="Candidate pool (violin)"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#d62728",
            markersize=6,
            label="Selected samples",
        ),
    ]
    axes[2].legend(
        handles=legend_elements, loc="upper right", frameon=False, fontsize=8
    )

    fig.suptitle(
        "Seed-42 AAL-SD (A): Score Distributions per Round with Selected Samples Highlighted",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig11_Score_Distribution_Seed42.png", bbox_inches="tight")
    plt.close(fig)
    print("  Fig11: U/K/Score distribution violins")


# ── main ─────────────────────────────────────────────────────────────────
def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating paper_draft figures …")
    fig1_architecture()
    fig2_multiseed_learning_curves()
    fig3_controller_trajectory()
    fig4_alc_bar()
    fig5_miou_bar()
    fig6_ablation_curves()
    fig7_gradient_diagnostics()
    fig8_lambda_gradient_coupling()
    fig9_multiseed_boxplots()
    fig10_delta_miou_curves()
    fig11_score_distribution()
    print(f"Done. All figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
