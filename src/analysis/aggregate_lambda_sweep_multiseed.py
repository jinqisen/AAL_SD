"""
Aggregate multi-seed lambda sweep results and generate averaged curves with error bars.

Validates flat landscape hypothesis by checking if averaged curves smooth out noise.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def parse_trace(trace_path: Path) -> Dict[int, Dict[str, float]]:
    round_metrics = {}
    with open(trace_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except:
                continue

            if event.get("type") == "epoch_end":
                r = event.get("round")
                if r not in round_metrics:
                    round_metrics[r] = []

                metrics = {
                    "mIoU": event.get("mIoU", 0.0),
                    "f1": event.get("f1", 0.0),
                    "per_class_iou": event.get("per_class_iou", []),
                }
                round_metrics[r].append(metrics)

    best_per_round = {}
    for r, metrics_list in round_metrics.items():
        best_epoch = max(metrics_list, key=lambda x: x["mIoU"])
        best_per_round[r] = best_epoch

    return best_per_round


def lambda_from_exp_name(exp_name: str) -> float:
    if "lam_00" in exp_name:
        return 0.0
    elif "lam_02" in exp_name:
        return 0.2
    elif "lam_04" in exp_name:
        return 0.4
    elif "lam_06" in exp_name:
        return 0.6
    elif "lam_08" in exp_name:
        return 0.8
    elif "lam_10" in exp_name:
        return 1.0
    else:
        raise ValueError(f"Cannot parse lambda from exp_name: {exp_name}")


def collect_multiseed_data(
    base_run_id: str, seeds: List[int], sweep_round: int, results_dir: Path
) -> Dict[float, List[float]]:
    lambda_to_mious = {}

    for seed in seeds:
        run_id = f"{base_run_id}_seed{seed}"
        run_dir = results_dir / "runs" / run_id

        if not run_dir.exists():
            print(f"WARNING: Run directory not found: {run_dir}")
            continue

        for trace_file in run_dir.glob(f"sweep_r{sweep_round}_lam_*_trace.jsonl"):
            exp_name = trace_file.stem.replace("_trace", "")
            lam = lambda_from_exp_name(exp_name)

            best_per_round = parse_trace(trace_file)
            if sweep_round in best_per_round:
                miou = best_per_round[sweep_round]["mIoU"]
                if lam not in lambda_to_mious:
                    lambda_to_mious[lam] = []
                lambda_to_mious[lam].append(miou)

    return lambda_to_mious


def compute_statistics(
    lambda_to_mious: Dict[float, List[float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lambdas = sorted(lambda_to_mious.keys())
    means = []
    stds = []
    mins = []
    maxs = []

    for lam in lambdas:
        values = lambda_to_mious[lam]
        means.append(np.mean(values))
        stds.append(np.std(values, ddof=1) if len(values) > 1 else 0.0)
        mins.append(np.min(values))
        maxs.append(np.max(values))

    return (
        np.array(lambdas),
        np.array(means),
        np.array(stds),
        np.array(mins),
        np.array(maxs),
    )


def test_unimodality(lambdas: np.ndarray, means: np.ndarray) -> Dict[str, any]:
    peaks = []
    for i in range(1, len(means) - 1):
        if means[i] > means[i - 1] and means[i] > means[i + 1]:
            peaks.append((lambdas[i], means[i]))

    if means[0] > means[1]:
        peaks.insert(0, (lambdas[0], means[0]))
    if means[-1] > means[-2]:
        peaks.append((lambdas[-1], means[-1]))

    coeffs = np.polyfit(lambdas, means, 2)
    fitted = np.polyval(coeffs, lambdas)
    ss_res = np.sum((means - fitted) ** 2)
    ss_tot = np.sum((means - means.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    vertex = -coeffs[1] / (2 * coeffs[0]) if coeffs[0] != 0 else float("nan")
    concave = "concave" if coeffs[0] < 0 else "convex"

    rho, p = stats.spearmanr(lambdas, means)

    return {
        "num_peaks": len(peaks),
        "peaks": peaks,
        "quadratic_r2": r2,
        "quadratic_a": coeffs[0],
        "concavity": concave,
        "vertex": vertex,
        "spearman_rho": rho,
        "spearman_p": p,
    }


def plot_multiseed_curves(
    sweep_round: int,
    lambdas: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    unimodality: Dict[str, any],
    output_path: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].errorbar(
        lambdas, means, yerr=stds, marker="o", capsize=5, label="Mean ± Std"
    )
    axes[0].fill_between(lambdas, mins, maxs, alpha=0.2, label="Min-Max Range")
    axes[0].set_title(f"Round {sweep_round}: Multi-Seed Averaged mIoU vs λ")
    axes[0].set_xlabel("λ")
    axes[0].set_ylabel("mIoU")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].text(
        0.05,
        0.95,
        f"Unimodality Analysis:\n"
        f"  Peaks: {unimodality['num_peaks']}\n"
        f"  Quadratic R²: {unimodality['quadratic_r2']:.3f}\n"
        f"  Concavity: {unimodality['concavity']}\n"
        f"  Vertex: {unimodality['vertex']:.2f}\n"
        f"  Spearman ρ: {unimodality['spearman_rho']:.3f}\n"
        f"  Spearman p: {unimodality['spearman_p']:.3f}\n\n"
        f"Effect Size:\n"
        f"  Range: {means.max() - means.min():.4f}\n"
        f"  Mean Std: {stds.mean():.4f}\n"
        f"  SNR: {(means.max() - means.min()) / stds.mean():.2f}",
        transform=axes[1].transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed lambda sweep results"
    )
    parser.add_argument("--base-run-id", required=True, help="Base run ID")
    parser.add_argument(
        "--seeds", type=str, default="42,43,44,45", help="Comma-separated seeds"
    )
    parser.add_argument(
        "--sweep-rounds",
        type=str,
        default="3,6,9",
        help="Comma-separated rounds to analyze",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Results directory (default: repo_root/results)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: results/runs/base_run_id)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    results_dir = Path(args.results_dir) if args.results_dir else repo_root / "results"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else results_dir / "runs" / args.base_run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    sweep_rounds = [int(r.strip()) for r in args.sweep_rounds.split(",")]

    print(f"=== Multi-Seed Lambda Sweep Aggregation ===")
    print(f"Base run ID: {args.base_run_id}")
    print(f"Seeds: {seeds}")
    print(f"Sweep rounds: {sweep_rounds}")
    print(f"Results dir: {results_dir}")
    print(f"Output dir: {output_dir}")
    print()

    all_results = {}

    for sweep_round in sweep_rounds:
        print(f"\n--- Analyzing Round {sweep_round} ---")

        lambda_to_mious = collect_multiseed_data(
            args.base_run_id, seeds, sweep_round, results_dir
        )

        if not lambda_to_mious:
            print(f"WARNING: No data found for round {sweep_round}")
            continue

        print(f"Collected data for {len(lambda_to_mious)} lambda values:")
        for lam in sorted(lambda_to_mious.keys()):
            values = lambda_to_mious[lam]
            print(
                f"  λ={lam:.1f}: n={len(values)}, mean={np.mean(values):.4f}, std={np.std(values, ddof=1):.4f}"
            )

        lambdas, means, stds, mins, maxs = compute_statistics(lambda_to_mious)

        unimodality = test_unimodality(lambdas, means)

        print(f"\nUnimodality test:")
        print(f"  Peaks: {unimodality['num_peaks']}")
        print(f"  Quadratic R²: {unimodality['quadratic_r2']:.3f}")
        print(f"  Concavity: {unimodality['concavity']}")
        print(
            f"  Spearman ρ: {unimodality['spearman_rho']:.3f} (p={unimodality['spearman_p']:.3f})"
        )

        print(f"\nEffect size:")
        print(f"  Range: {means.max() - means.min():.4f}")
        print(f"  Mean Std: {stds.mean():.4f}")
        print(f"  SNR: {(means.max() - means.min()) / stds.mean():.2f}")

        output_path = output_dir / f"multiseed_lambda_sweep_r{sweep_round}.png"
        plot_multiseed_curves(
            sweep_round, lambdas, means, stds, mins, maxs, unimodality, output_path
        )

        all_results[sweep_round] = {
            "lambdas": lambdas.tolist(),
            "means": means.tolist(),
            "stds": stds.tolist(),
            "mins": mins.tolist(),
            "maxs": maxs.tolist(),
            "unimodality": {k: v for k, v in unimodality.items() if k != "peaks"},
        }

    summary_path = output_dir / "multiseed_lambda_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    print(f"\n{'=' * 72}")
    print("  Multi-Seed Aggregation Complete")
    print(f"{'=' * 72}")
    print(f"\nConclusion:")
    print(f"  If averaged curves are smooth and unimodal → noise hypothesis confirmed")
    print(f"  If still non-monotonic → genuine multi-modality")
    print(f"  Compare SNR with single-seed analysis (expected: ~2.5)")


if __name__ == "__main__":
    main()
