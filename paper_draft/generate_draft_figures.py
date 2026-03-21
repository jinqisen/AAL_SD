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

import json
import sys
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
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

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
    "no_risk_control",
    "no_agent",
    "fixed_lambda",
    "uncertainty_only",
    "knowledge_only",
]
ABLATION_LABELS = {
    "full_model_A_lambda_policy": "AAL-SD (full)",
    "no_risk_control": "w/o Risk Control",
    "no_agent": "w/o Agent",
    "fixed_lambda": "Fixed λ=0.5",
    "uncertainty_only": "Uncertainty only (λ=0)",
    "knowledge_only": "Knowledge only (λ=1)",
}
ABLATION_COLORS = {
    "full_model_A_lambda_policy": "#d62728",
    "no_risk_control": "#17becf",
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

def _safe_float(x: object) -> float | None:
    try:
        v = float(x)
        if np.isfinite(v):
            return float(v)
        return None
    except Exception:
        return None

def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}

def _discover_latest_seed_run_dir(seed: int, prefix: str = "baseline_") -> Path | None:
    candidates = []
    for p in RUNS.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if not name.startswith(prefix) or not name.endswith(f"_seed{int(seed)}"):
            continue
        man = p / "manifest.json"
        if not man.exists():
            continue
        created_at = None
        try:
            created_at = (_read_json(man).get("created_at") or "").strip()
        except Exception:
            created_at = None
        candidates.append((created_at or "", name, p))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][2]

def _load_paper_round_curves() -> pd.DataFrame:
    path = PAPER / "multiseed_round_curves.csv"
    if not path.exists():
        raise RuntimeError(f"Missing paper asset: {path}")
    df = pd.read_csv(path)
    for col in ["round", "labeled_size", "n", "miou_mean", "miou_std", "miou_ci95"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def _load_controller_trajectories() -> pd.DataFrame:
    path = PAPER / "controller_trajectories.csv"
    if not path.exists():
        raise RuntimeError(f"Missing paper asset: {path}")
    df = pd.read_csv(path)
    for col in [
        "seed",
        "round",
        "labeled_size",
        "final_miou",
        "grad_train_val_cos_last",
        "overfit_risk",
        "lambda_eff",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── data loaders ─────────────────────────────────────────────────────────
def load_multiseed_curves() -> pd.DataFrame:
    """Load pre-aggregated multi-seed per-round curves (paper assets)."""
    df = _load_paper_round_curves()
    return df[df["experiment"].isin(METHOD_ORDER)].copy()


def load_seed42_ablation_curves() -> pd.DataFrame:
    """Load per-round data from seed-42 for ablation methods."""
    for p in RUNS.iterdir():
        if not p.is_dir() or not p.name.endswith("_seed42"):
            continue
        csv_path = p / "run_all_experiments_per_round_with_grad.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "experiment_name" not in df.columns:
            continue
        sub = df[df["experiment_name"].isin(ABLATION_ORDER)]
        if not sub.empty:
            return sub
    return pd.DataFrame()


def load_metrics_summary() -> pd.DataFrame:
    return pd.read_csv(PAPER / "multiseed_metrics_summary.csv")


def load_gradient_summary() -> pd.DataFrame:
    return pd.read_csv(PAPER / "multiseed_gradient_summary.csv")


# ── Fig 1: architecture ─────────────────────────────────────────────────
def fig1_architecture():
    try:
        import importlib.util

        script_path = DRAFT / "generate_architecture_figure.py"
        if script_path.exists():
            spec = importlib.util.spec_from_file_location(
                "paper_draft.generate_architecture_figure", script_path
            )
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "main"):
                    mod.main()
                    print("  Fig1: generated by generate_architecture_figure.py")
                    return
    except Exception:
        pass

    src = PAPER / "Figure1_AAL_SD_Architecture.png"
    dst = FIG_DIR / "Fig1_Architecture.png"
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  Fig1: copied from {src}")
        return
    print(f"  Fig1: SKIP – source not found at {src}")


# ── Fig 2: multi-seed learning curves +末段 inset 放大 ─────────────────
def fig2_multiseed_learning_curves():
    raw = load_multiseed_curves()
    fig, ax = plt.subplots(figsize=(10, 6))

    # collect data for inset
    all_means = {}
    all_cis = {}
    for method in METHOD_ORDER:
        mdf = raw[raw["experiment"] == method].sort_values("round")
        rounds = mdf["round"].to_numpy()
        mean_vals = mdf["miou_mean"].to_numpy()
        ci_vals = mdf["miou_ci95"].to_numpy()
        all_means[method] = (rounds, mean_vals, ci_vals)
        color = COLORS[method]
        lw = 2.8 if method.startswith("full_model") else 1.8
        ls = "-" if method.startswith("full_model") else "--"
        ax.plot(
            rounds,
            mean_vals,
            label=LABELS[method],
            color=color,
            linewidth=lw,
            linestyle=ls,
        )
        ax.fill_between(
            rounds, mean_vals - ci_vals, mean_vals + ci_vals, color=color, alpha=0.10
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
    df = _load_controller_trajectories()
    focus = df[df["experiment"].isin(["full_model_A_lambda_policy", "full_model_B_lambda_agent"])]
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    panels = [
        ("final_miou", "mIoU"),
        ("lambda_eff", "Effective λ"),
        ("grad_train_val_cos_last", "Grad Cosine (train-val)"),
        ("overfit_risk", "Overfit Risk"),
    ]
    for method in ["full_model_A_lambda_policy", "full_model_B_lambda_agent"]:
        mdf = focus[focus["experiment"] == method].sort_values("round")
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
    plotted = 0

    if not df.empty:
        for method in ABLATION_ORDER:
            mdf = df[df["experiment_name"] == method].sort_values("round")
            if mdf.empty:
                continue
            color = ABLATION_COLORS.get(method, "#333333")
            lw = 2.8 if method == "full_model_A_lambda_policy" else 1.8
            ax.plot(
                mdf["round"],
                mdf["final_miou"],
                label=ABLATION_LABELS.get(method, method),
                color=color,
                linewidth=lw,
                marker="o",
                markersize=4,
            )
            plotted += 1
    else:
        def _parse_round_summaries(trace_path: Path) -> pd.DataFrame:
            rows = []
            if not trace_path.exists():
                return pd.DataFrame()
            with open(trace_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(ev, dict):
                        continue
                    if ev.get("type") != "round_summary":
                        continue
                    r = ev.get("round")
                    try:
                        rr = int(r)
                    except Exception:
                        continue
                    rows.append(
                        {
                            "round": rr,
                            "labeled_size": _safe_float(ev.get("labeled_size")),
                            "miou": _safe_float(ev.get("mIoU")),
                        }
                    )
            out = pd.DataFrame(rows)
            if out.empty:
                return out
            out = out.dropna(subset=["round", "miou"]).sort_values("round")
            return out

        seed42_runs = [p for p in RUNS.iterdir() if p.is_dir() and p.name.startswith("baseline_") and p.name.endswith("_seed42")]
        best_dir = None
        best_count = -1
        for p in seed42_runs:
            traces = list(p.glob("full_model_A_lambda_policy*trace.jsonl"))
            if len(traces) > best_count:
                best_dir = p
                best_count = len(traces)

        if best_dir is None or best_count <= 0:
            plt.close(fig)
            print("  Fig6: SKIP – ablation traces not found")
            return

        available = []
        for tpath in sorted(best_dir.glob("*_trace.jsonl")):
            name = tpath.name[: -len("_trace.jsonl")]
            if name in ABLATION_ORDER or name.startswith("full_model_A_lambda_policy"):
                available.append(name)
        if "full_model_A_lambda_policy" in available:
            available.remove("full_model_A_lambda_policy")
            available = ["full_model_A_lambda_policy"] + available

        palette = plt.get_cmap("tab10")
        for idx, name in enumerate(available[:8]):
            curve = _parse_round_summaries(best_dir / f"{name}_trace.jsonl")
            if curve.empty:
                continue
            color = ABLATION_COLORS.get(name, palette(idx % 10))
            lw = 2.8 if name == "full_model_A_lambda_policy" else 1.8
            ax.plot(
                curve["round"],
                curve["miou"],
                label=ABLATION_LABELS.get(name, name),
                color=color,
                linewidth=lw,
                marker="o",
                markersize=4,
            )
            plotted += 1

    if plotted <= 0:
        plt.close(fig)
        print("  Fig6: SKIP – no ablation curves available")
        return
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
    df = _load_controller_trajectories()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for method, ax in zip(
        ["full_model_A_lambda_policy", "full_model_B_lambda_agent"], axes
    ):
        mdf = df[df["experiment"] == method].sort_values("round").copy()
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
    rows = []
    for seed in SEEDS:
        sd = _discover_latest_seed_run_dir(seed, prefix="baseline_")
        if sd is None:
            continue
        for method in METHOD_ORDER:
            status_path = sd / f"{method}_status.json"
            if status_path.exists():
                with open(status_path) as f:
                    sj = json.load(f)
                result = sj.get("result", {})
                alc = result.get("alc", float("nan"))
                final_miou = result.get("final_mIoU", float("nan"))
            else:
                continue
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

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for method in compare_methods:
        ref = raw[raw["experiment"] == ref_method].sort_values("round")
        comp = raw[raw["experiment"] == method].sort_values("round")
        merged = ref[["round", "miou_mean", "miou_std", "n"]].merge(
            comp[["round", "miou_mean", "miou_std", "n"]],
            on="round",
            suffixes=("_ref", "_comp"),
        )
        if merged.empty:
            continue
        rounds = merged["round"].to_numpy()
        means = (merged["miou_mean_ref"] - merged["miou_mean_comp"]).to_numpy()
        n = merged["n_ref"].fillna(merged["n_comp"]).fillna(4).astype(float).to_numpy()
        std_ref = merged["miou_std_ref"].to_numpy()
        std_comp = merged["miou_std_comp"].to_numpy()
        std_delta = np.sqrt(np.square(std_ref) + np.square(std_comp))
        cis = np.array([t_critical_95(int(nn)) for nn in n]) * (std_delta / np.sqrt(n))

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

    run_dir = _discover_latest_seed_run_dir(42, prefix="baseline_")
    l3_path = None
    if run_dir is not None:
        cand = run_dir / "figures" / "l3_selection.csv"
        if cand.exists():
            l3_path = cand
    if l3_path is None or (not l3_path.exists()):
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

def fig12_segmentation_visual_comparison():
    try:
        import torch
    except Exception:
        print("  Fig12: SKIP – torch not available")
        return

    from src.core.dataset import Landslide4SenseDataset
    from src.core.model import LandslideDeepLabV3

    run_dir = _discover_latest_seed_run_dir(42, prefix="baseline_")
    if run_dir is None:
        print("  Fig12: SKIP – baseline run dir not found for seed=42")
        return

    man_path = run_dir / "manifest.json"
    data_root = None
    if man_path.exists():
        try:
            data_root = (_read_json(man_path).get("data_dir") or "").strip()
        except Exception:
            data_root = None
    if not data_root:
        data_root = str(Path.home() / "AAL_SD" / "data" / "Landslide4Sense")

    methods = ["full_model_A_lambda_policy", "baseline_entropy", "baseline_random"]

    def _find_latest_round_model(exp_name: str) -> Path | None:
        d = run_dir / f"{exp_name}_round_models"
        if not d.exists():
            return None
        pts = sorted([p for p in d.glob("round_*_best_val.pt") if p.is_file()])
        if not pts:
            return None
        best = None
        best_r = -1
        for p in pts:
            try:
                r = int(p.name.split("_", 2)[1])
            except Exception:
                continue
            if r > best_r:
                best_r = r
                best = p
        return best

    ckpts = {m: _find_latest_round_model(m) for m in methods}
    ckpts = {k: v for k, v in ckpts.items() if v is not None}
    if not ckpts:
        print("  Fig12: SKIP – round_models not found in baseline run dir")
        return

    ds = Landslide4SenseDataset(data_root, split="test", transform=None, with_mask=True)
    want_pos = 2
    want_neg = 2
    chosen = []
    for i in range(len(ds)):
        sample = ds[i]
        mask = sample["mask"]
        if mask is None or (hasattr(mask, "numel") and int(mask.numel()) == 0):
            continue
        m = mask.detach().cpu().numpy()
        pos = bool(np.any(m > 0))
        if pos and want_pos > 0:
            chosen.append(sample)
            want_pos -= 1
        elif (not pos) and want_neg > 0:
            chosen.append(sample)
            want_neg -= 1
        if want_pos <= 0 and want_neg <= 0:
            break
    if not chosen:
        print("  Fig12: SKIP – no test samples with masks found")
        return

    device = "cpu"

    def _load_model(ckpt_path: Path):
        payload = torch.load(ckpt_path, map_location="cpu")
        state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        model = LandslideDeepLabV3(in_channels=14, classes=2, encoder_weights=None, mc_dropout=0.0)
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
        return model

    models = {m: _load_model(p) for m, p in ckpts.items()}

    def _to_rgb(image_chw: np.ndarray) -> np.ndarray:
        c = int(image_chw.shape[0])
        idx = [0, 1, 2] if c >= 3 else list(range(c)) + [0] * (3 - c)
        rgb = image_chw[idx, :, :].transpose(1, 2, 0).astype(np.float32, copy=False)
        lo = np.quantile(rgb, 0.02)
        hi = np.quantile(rgb, 0.98)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.min(rgb))
            hi = float(np.max(rgb)) if float(np.max(rgb)) > float(np.min(rgb)) else float(np.min(rgb) + 1.0)
        rgb = (rgb - lo) / (hi - lo + 1e-6)
        return np.clip(rgb, 0.0, 1.0)

    n_rows = len(chosen)
    n_cols = 2 + len(models)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.0 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    headers = ["Image", "GT"] + [LABELS.get(m, m) for m in models.keys()]
    for j, h in enumerate(headers):
        axes[0, j].set_title(h, fontsize=12)

    with torch.no_grad():
        for i, sample in enumerate(chosen):
            x = sample["image"].unsqueeze(0).to(device)
            x_np = sample["image"].detach().cpu().numpy()
            gt = sample["mask"].detach().cpu().numpy().astype(np.uint8)
            rgb = _to_rgb(x_np)

            axes[i, 0].imshow(rgb)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(rgb)
            axes[i, 1].imshow(gt > 0, cmap="Reds", alpha=0.45, vmin=0, vmax=1)
            axes[i, 1].axis("off")

            for j, (mname, model) in enumerate(models.items(), start=2):
                logits = model(x)
                pred = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
                axes[i, j].imshow(rgb)
                axes[i, j].imshow(pred > 0, cmap="Reds", alpha=0.45, vmin=0, vmax=1)
                axes[i, j].axis("off")

            sid = str(sample.get("sample_id") or sample.get("image_name") or f"sample_{i}")
            axes[i, 0].text(0.02, 0.02, sid, transform=axes[i, 0].transAxes, fontsize=9, color="white",
                            bbox=dict(facecolor="black", alpha=0.35, pad=2, edgecolor="none"))

    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig12_Segmentation_Visual_Comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  Fig12: segmentation visual comparison")

def fig13_hyperparam_sensitivity():
    rows = []
    for run_dir in sorted([p for p in RUNS.iterdir() if p.is_dir() and p.name.startswith("autotune_opt_iter")]):
        man_path = run_dir / "manifest.json"
        if not man_path.exists():
            continue
        man = _read_json(man_path)
        exps = man.get("experiments")
        if not isinstance(exps, dict):
            continue
        for exp_name, exp_cfg in exps.items():
            if not isinstance(exp_cfg, dict):
                continue
            status_path = run_dir / f"{exp_name}_status.json"
            if not status_path.exists():
                continue
            st = _read_json(status_path)
            res = st.get("result") if isinstance(st.get("result"), dict) else {}
            final_miou = _safe_float(res.get("final_mIoU"))
            alc = _safe_float(res.get("alc"))
            lp = exp_cfg.get("lambda_policy") if isinstance(exp_cfg.get("lambda_policy"), dict) else {}
            guard = lp.get("selection_guardrail") if isinstance(lp.get("selection_guardrail"), dict) else {}
            thr = exp_cfg.get("agent_threshold_overrides") if isinstance(exp_cfg.get("agent_threshold_overrides"), dict) else {}

            rows.append(
                {
                    "run_id": run_dir.name,
                    "experiment": exp_name,
                    "final_miou": final_miou,
                    "alc": alc,
                    "tau_risk": _safe_float(lp.get("risk_ci_quantile")),
                    "beta_ema": _safe_float(lp.get("lambda_smoothing_alpha")),
                    "lambda_max": _safe_float(thr.get("LAMBDA_CLAMP_MAX")),
                    "lambda_max_step": _safe_float(lp.get("lambda_max_step")),
                    "k_max": _safe_float(guard.get("max_steps")),
                    "n_cool": _safe_float(thr.get("LAMBDA_DOWN_COOLING_ROUNDS")),
                }
            )

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["final_miou"])
    if df.empty:
        print("  Fig13: SKIP – no autotune status results found")
        return

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.1])

    ax_hm = fig.add_subplot(gs[0, :2])
    sub = df.dropna(subset=["tau_risk", "beta_ema"]).copy()
    if sub.empty:
        ax_hm.text(0.5, 0.5, "No (τ_risk, β) pairs found", ha="center", va="center")
        ax_hm.set_axis_off()
    else:
        sub["tau_risk_r"] = sub["tau_risk"].round(3)
        sub["beta_ema_r"] = sub["beta_ema"].round(3)
        pivot = (
            sub.pivot_table(
                index="beta_ema_r",
                columns="tau_risk_r",
                values="final_miou",
                aggfunc="mean",
            )
            .sort_index()
        )
        im = ax_hm.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
        ax_hm.set_xticks(np.arange(pivot.shape[1]))
        ax_hm.set_xticklabels([str(x) for x in pivot.columns.tolist()], rotation=45, ha="right")
        ax_hm.set_yticks(np.arange(pivot.shape[0]))
        ax_hm.set_yticklabels([str(x) for x in pivot.index.tolist()])
        ax_hm.set_xlabel("τ_risk (risk_ci_quantile)")
        ax_hm.set_ylabel("β (EMA alpha)")
        ax_hm.set_title("Hyperparameter Sensitivity Heatmap (mean final mIoU)")
        cax = fig.add_subplot(gs[0, 2])
        fig.colorbar(im, cax=cax, label="final mIoU")

    axs = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
    ]
    panels = [
        ("lambda_max", "λ_max (LAMBDA_CLAMP_MAX)"),
        ("lambda_max_step", "λ_max_step"),
        ("n_cool", "n_cool (cooling rounds)"),
    ]
    for ax, (col, title) in zip(axs, panels):
        s = df.dropna(subset=[col]).copy()
        if s.empty:
            ax.text(0.5, 0.5, f"No data for {col}", ha="center", va="center")
            ax.set_axis_off()
            continue
        x = s[col].to_numpy()
        y = s["final_miou"].to_numpy()
        ax.scatter(x, y, s=32, alpha=0.55, edgecolors="white", linewidth=0.4)
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        if len(x_sorted) >= 8:
            bins = np.unique(np.quantile(x_sorted, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).round(6))
            if len(bins) >= 3:
                bx = []
                by = []
                for lo, hi in zip(bins[:-1], bins[1:]):
                    mask = (x_sorted >= lo) & (x_sorted <= hi)
                    if not np.any(mask):
                        continue
                    bx.append((lo + hi) / 2.0)
                    by.append(float(np.mean(y_sorted[mask])))
                if len(bx) >= 2:
                    ax.plot(bx, by, "-o", linewidth=2.0, markersize=5, color="#d62728")
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("final mIoU")
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig13_Hyperparam_Sensitivity.png", bbox_inches="tight")
    plt.close(fig)
    print("  Fig13: hyperparameter sensitivity (heatmap + trends)")


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
    fig12_segmentation_visual_comparison()
    fig13_hyperparam_sensitivity()
    print(f"Done. All figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
