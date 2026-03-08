"""Generate the AAL-SD architecture figure (Fig 1) with dual-line design.

Two visual lines:
  - ACTIVE LEARNING LINE (top, blue tones): data flow through the AL loop
  - LEARNING CONTROL LINE (bottom, orange/red tones): gradient-risk-policy control flow

They converge at the Agent Decision block where the policy-determined lambda
constrains the agent's candidate scoring and selection.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent / "figures"


def rounded_box(
    ax,
    xy,
    w,
    h,
    text,
    color,
    text_color="white",
    fontsize=9,
    fontweight="bold",
    alpha=0.92,
    lw=1.5,
    edgecolor=None,
    zorder=3,
):
    x, y = xy
    ec = edgecolor or color
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.12",
        facecolor=color,
        edgecolor=ec,
        linewidth=lw,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=text_color,
        zorder=zorder + 1,
    )
    return box


def arrow(
    ax,
    start,
    end,
    color="#333",
    lw=1.8,
    style="-|>",
    zorder=2,
    connectionstyle="arc3,rad=0",
):
    a = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        color=color,
        linewidth=lw,
        zorder=zorder,
        connectionstyle=connectionstyle,
        mutation_scale=14,
    )
    ax.add_patch(a)
    return a


def label_text(
    ax, xy, text, fontsize=7.5, color="#333", ha="center", va="bottom", style="italic"
):
    ax.text(
        xy[0],
        xy[1],
        text,
        ha=ha,
        va=va,
        fontsize=fontsize,
        color=color,
        style=style,
        zorder=10,
    )


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 9.5))
    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(-1.0, 10.0)
    ax.axis("off")

    # ── Color palette ─────────────────────────────────────────────
    C_BLUE_DARK = "#1a5276"
    C_BLUE_MED = "#2980b9"
    C_BLUE_LIGHT = "#5dade2"
    C_GREEN = "#1e8449"
    C_GREEN_LIGHT = "#27ae60"
    C_ORANGE = "#e67e22"
    C_RED = "#c0392b"
    C_RED_LIGHT = "#e74c3c"
    C_PURPLE = "#8e44ad"
    C_GRAY = "#7f8c8d"
    C_GRAY_LIGHT = "#bdc3c7"
    C_YELLOW = "#f39c12"

    # ══════════════════════════════════════════════════════════════
    # TITLE
    # ══════════════════════════════════════════════════════════════
    ax.text(
        8.0,
        9.5,
        "AAL-SD: Dual-Line Architecture",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#2c3e50",
    )
    ax.text(
        8.0,
        9.1,
        "Active Learning Loop  ×  Learning Control Loop",
        ha="center",
        va="center",
        fontsize=11,
        color="#555",
    )

    # ══════════════════════════════════════════════════════════════
    # LINE LABELS (left side)
    # ══════════════════════════════════════════════════════════════
    ax.text(
        -0.3,
        7.2,
        "ACTIVE\nLEARNING\nLINE",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=C_BLUE_DARK,
        rotation=0,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#d6eaf8",
            edgecolor=C_BLUE_DARK,
            linewidth=1.5,
        ),
    )

    ax.text(
        -0.3,
        3.0,
        "LEARNING\nCONTROL\nLINE",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=C_RED,
        rotation=0,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#fdedec",
            edgecolor=C_RED,
            linewidth=1.5,
        ),
    )

    # ══════════════════════════════════════════════════════════════
    # ACTIVE LEARNING LINE (top row, y≈6.5-8.5)
    # ══════════════════════════════════════════════════════════════
    bw, bh = 2.2, 1.2  # box width, height

    # Box 1: Data Pools
    rounded_box(
        ax,
        (0.8, 7.0),
        bw,
        bh,
        "Data Pools\n─────────\nL₀ (151)\nU (2888)\nT (760)",
        C_BLUE_DARK,
        fontsize=8,
    )

    # Box 2: Model Training
    rounded_box(
        ax,
        (3.8, 7.0),
        bw,
        bh,
        "DeepLabV3+\nTraining\n─────────\n10 epochs/round",
        C_BLUE_MED,
        fontsize=8,
    )

    # Box 3: Unlabeled Inference
    rounded_box(
        ax,
        (6.8, 7.0),
        bw,
        bh,
        "Unlabeled\nInference\n─────────\nP(x), f(x)",
        C_BLUE_LIGHT,
        fontsize=8,
    )

    # Box 4: AD-KUCS Scoring
    rounded_box(
        ax,
        (9.8, 7.0),
        bw + 0.3,
        bh,
        "AD-KUCS Scoring\n─────────────\nU(x): pixel entropy\nK(x): coreset dist\nScore=(1-λ)U+λK",
        C_GREEN,
        fontsize=7.5,
    )

    # Box 5: LLM Agent
    rounded_box(
        ax,
        (13.0, 6.6),
        2.8,
        1.8,
        "LLM Agent\n(ReAct Loop)\n──────────\n① get_system_status\n② get_top_k_samples\n③ finalize_selection\n+ reasoning trace",
        C_PURPLE,
        fontsize=7.5,
    )

    # Arrows: AL line flow
    arrow(ax, (3.0, 7.6), (3.8, 7.6), C_BLUE_DARK)
    label_text(ax, (3.4, 7.75), "train on Lₜ", fontsize=7)

    arrow(ax, (6.0, 7.6), (6.8, 7.6), C_BLUE_MED)
    label_text(ax, (6.4, 7.75), "model fθ", fontsize=7)

    arrow(ax, (9.0, 7.6), (9.8, 7.6), C_BLUE_LIGHT)
    label_text(ax, (9.4, 7.75), "probmaps\n+ features", fontsize=6.5)

    arrow(ax, (12.3, 7.6), (13.0, 7.6), C_GREEN)
    label_text(ax, (12.65, 7.75), "ranked\ncandidates", fontsize=6.5)

    # Return arrow: Agent -> Pool Update -> Data Pools
    # Agent output goes down then left
    rounded_box(
        ax,
        (13.3, 5.0),
        2.2,
        0.9,
        "Pool Update\n─────────\nQt → Lₜ₊₁\nU ← U\\Qt",
        C_GREEN_LIGHT,
        fontsize=8,
    )

    arrow(ax, (14.4, 6.6), (14.4, 5.9), C_PURPLE)
    label_text(ax, (14.65, 6.25), "selected\nQt + reason", fontsize=6.5)

    # Return arrow back to Data Pools (curved)
    arrow(
        ax,
        (13.3, 5.45),
        (1.9, 7.0),
        C_GREEN_LIGHT,
        lw=2.0,
        connectionstyle="arc3,rad=0.25",
    )
    label_text(
        ax,
        (7.5, 5.7),
        "← update labeled pool → next round",
        fontsize=7.5,
        color=C_GREEN,
    )

    # ══════════════════════════════════════════════════════════════
    # LEARNING CONTROL LINE (bottom row, y≈1.5-4.5)
    # ══════════════════════════════════════════════════════════════

    # Box C1: Gradient Diagnostics
    rounded_box(
        ax,
        (3.0, 2.5),
        2.5,
        1.3,
        "Gradient Diagnostics\n──────────────\ngrad_train_val_cos\ntvc_neg_rate\nper-epoch tracking",
        C_ORANGE,
        fontsize=7.5,
    )

    # Box C2: Risk Assessment
    rounded_box(
        ax,
        (6.3, 2.5),
        2.5,
        1.3,
        "Risk Assessment\n──────────────\noverfit_risk =\n neg_rate+½|cos⁻|+½|last⁻|\nrollback_flag",
        C_RED_LIGHT,
        fontsize=7.5,
    )

    # Box C3: Lambda Policy (warmup+risk closed-loop)
    rounded_box(
        ax,
        (9.6, 2.2),
        2.8,
        1.8,
        "λ Policy Engine\n(warmup + risk CL)\n───────────────\nR1-2: λ=0 (uncertainty)\nR3:   λ=0.2 (warmup)\nR4+:  closed-loop\n  ↑risk→λ↓  ↓risk→λ↑\n  EMA smooth + clamp",
        C_RED,
        fontsize=7,
        text_color="white",
    )

    # Arrows: Control line flow
    arrow(
        ax, (5.0, 7.0), (4.25, 3.8), C_ORANGE, lw=1.5, connectionstyle="arc3,rad=-0.15"
    )
    label_text(
        ax,
        (4.0, 5.4),
        "epoch gradients\n(train-val cosine)",
        fontsize=6.5,
        color=C_ORANGE,
    )

    arrow(ax, (5.5, 3.15), (6.3, 3.15), C_ORANGE)

    arrow(ax, (8.8, 3.15), (9.6, 3.15), C_RED_LIGHT)
    label_text(ax, (9.2, 3.35), "risk signals", fontsize=6.5, color=C_RED)

    # Lambda policy -> AD-KUCS (upward)
    arrow(ax, (11.0, 4.0), (11.0, 7.0), C_RED, lw=2.2)
    label_text(
        ax,
        (11.25, 5.5),
        "λₜ (policy-\ndetermined)",
        fontsize=7.5,
        color=C_RED,
        ha="left",
    )

    # Lambda policy -> Agent (upward-right)
    arrow(ax, (12.1, 4.0), (13.8, 6.6), C_RED, lw=1.8, connectionstyle="arc3,rad=-0.15")
    label_text(
        ax, (13.3, 5.4), "overfit signals\n+ constraints", fontsize=6.5, color=C_RED
    )

    # ══════════════════════════════════════════════════════════════
    # CHECKPOINT / TRACE (bottom-right)
    # ══════════════════════════════════════════════════════════════
    rounded_box(
        ax,
        (13.3, 1.5),
        2.2,
        1.0,
        "Checkpoint &\nTrace Store\n─────────\nstate.json\ntrace.jsonl",
        C_GRAY,
        fontsize=7.5,
    )

    arrow(ax, (14.4, 5.0), (14.4, 2.5), C_GRAY, lw=1.2)
    label_text(ax, (14.65, 3.8), "save state", fontsize=6.5, color=C_GRAY)

    arrow(ax, (13.3, 2.0), (12.4, 3.1), C_GRAY, lw=1.2, connectionstyle="arc3,rad=0.2")
    label_text(ax, (12.5, 2.3), "recovery\nsupport", fontsize=6.5, color=C_GRAY)

    # ══════════════════════════════════════════════════════════════
    # ROUND INDICATOR (top-left)
    # ══════════════════════════════════════════════════════════════
    ax.text(
        1.9,
        8.7,
        "Round t = 1, 2, …, 15",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=C_BLUE_DARK,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#eaf2f8", edgecolor=C_BLUE_DARK),
    )

    # ══════════════════════════════════════════════════════════════
    # CONVERGENCE ANNOTATION (center)
    # ══════════════════════════════════════════════════════════════
    ax.annotate("", xy=(11.0, 5.5), xytext=(11.0, 5.5), fontsize=0)

    # Dashed box around the convergence zone
    conv_box = FancyBboxPatch(
        (9.5, 4.3),
        3.0,
        2.5,
        boxstyle="round,pad=0.15",
        facecolor="none",
        edgecolor=C_YELLOW,
        linewidth=2.0,
        linestyle="--",
        alpha=0.8,
        zorder=1,
    )
    ax.add_patch(conv_box)
    ax.text(
        11.0,
        4.45,
        "← convergence zone: policy λ feeds into scoring & agent context",
        ha="center",
        va="center",
        fontsize=7,
        color=C_YELLOW,
        fontweight="bold",
    )

    # ══════════════════════════════════════════════════════════════
    # LEGEND
    # ══════════════════════════════════════════════════════════════
    legend_elements = [
        mpatches.Patch(facecolor=C_BLUE_MED, label="Active Learning Line (data flow)"),
        mpatches.Patch(facecolor=C_RED, label="Learning Control Line (risk→λ policy)"),
        mpatches.Patch(
            facecolor=C_PURPLE, label="LLM Agent (observe + explain + finalize)"
        ),
        mpatches.Patch(facecolor=C_GREEN, label="AD-KUCS Acquisition Scoring"),
        mpatches.Patch(facecolor=C_GRAY, label="Checkpoint / Trace / Recovery"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower left",
        fontsize=8,
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor="#ccc",
        bbox_to_anchor=(0.0, -0.02),
    )

    # ══════════════════════════════════════════════════════════════
    # KEY INSIGHT annotation (bottom center)
    # ══════════════════════════════════════════════════════════════
    ax.text(
        7.0,
        0.8,
        "Key: The Learning Control Line uses gradient-derived risk signals to regulate λₜ,\n"
        "which indirectly shapes the labeled-data distribution and downstream optimization trajectory.\n"
        "The LLM Agent observes the policy-determined λₜ, inspects candidates, and provides interpretable selection.",
        ha="center",
        va="center",
        fontsize=8,
        color="#555",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#fef9e7",
            edgecolor="#f0e68c",
            linewidth=1,
        ),
    )

    # ── finalize ──────────────────────────────────────────────────
    fig.tight_layout()
    out_path = FIG_DIR / "Fig1_Architecture.png"
    fig.savefig(out_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
