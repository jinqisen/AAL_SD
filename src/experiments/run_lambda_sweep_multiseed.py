"""
Multi-seed lambda sweep runner for validating flat landscape hypothesis.

Usage:
    python src/experiments/run_lambda_sweep_multiseed.py \\
        --base-run-id lambda_sweep_validation \\
        --trunk-exp fixed_lambda \\
        --sweep-rounds 3,6,9 \\
        --lambdas 0.0,0.2,0.4,0.6,0.8,1.0 \\
        --seeds 42,43,44,45 \\
        --workers 2
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-seed lambda sweep for landscape validation"
    )
    parser.add_argument(
        "--base-run-id",
        required=True,
        help="Base run ID (will append _seedXX for each seed)",
    )
    parser.add_argument(
        "--trunk-exp", required=True, help="Name of trunk experiment to branch from"
    )
    parser.add_argument(
        "--sweep-rounds",
        type=str,
        default="3,6,9",
        help="Comma-separated rounds to sweep (e.g., 3,6,9)",
    )
    parser.add_argument(
        "--lambdas",
        type=str,
        default="0.0,0.2,0.4,0.6,0.8,1.0",
        help="Comma-separated lambda values",
    )
    parser.add_argument(
        "--seeds", type=str, default="42,43,44,45", help="Comma-separated seeds"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of parallel workers per seed"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[2]
    sweep_script = base_dir / "src/experiments/run_lambda_sweep.py"

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    sweep_rounds = [int(r.strip()) for r in args.sweep_rounds.split(",")]

    print(f"=== Multi-Seed Lambda Sweep Configuration ===")
    print(f"Base run ID: {args.base_run_id}")
    print(f"Trunk experiment: {args.trunk_exp}")
    print(f"Sweep rounds: {sweep_rounds}")
    print(f"Lambda values: {args.lambdas}")
    print(f"Seeds: {seeds}")
    print(f"Workers per seed: {args.workers}")
    print(f"Dry run: {args.dry_run}")
    print()

    for seed in seeds:
        run_id = f"{args.base_run_id}_seed{seed}"
        print(f"\n{'=' * 72}")
        print(f"  Processing seed {seed} (run_id: {run_id})")
        print(f"{'=' * 72}\n")

        for sweep_round in sweep_rounds:
            print(f"\n--- Sweep at Round {sweep_round} for seed {seed} ---")

            cmd = [
                sys.executable,
                str(sweep_script),
                "--run-id",
                run_id,
                "--trunk-exp",
                args.trunk_exp,
                "--sweep-round",
                str(sweep_round),
                "--lambdas",
                args.lambdas,
                "--workers",
                str(args.workers),
            ]

            if args.dry_run:
                print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
            else:
                print(f"Executing: {' '.join(cmd)}")
                result = subprocess.run(cmd)
                if result.returncode != 0:
                    print(f"ERROR: Sweep failed for seed {seed}, round {sweep_round}")
                    sys.exit(1)

    print(f"\n{'=' * 72}")
    print("  Multi-Seed Lambda Sweep Completed")
    print(f"{'=' * 72}")
    print(f"\nResults saved in:")
    for seed in seeds:
        run_id = f"{args.base_run_id}_seed{seed}"
        print(f"  - results/runs/{run_id}/")

    print(f"\nNext steps:")
    print(f"  1. Run aggregation analysis:")
    print(
        f"     python src/analysis/aggregate_lambda_sweep_multiseed.py --base-run-id {args.base_run_id} --seeds {args.seeds}"
    )
    print(f"  2. Generate averaged curves with error bars")
    print(f"  3. Compare with single-seed results")


if __name__ == "__main__":
    main()
