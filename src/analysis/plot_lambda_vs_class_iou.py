import argparse
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def parse_trace(trace_path):
    round_metrics = {}
    with open(trace_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
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
                    "per_class_iou": event.get("per_class_iou", [])
                }
                round_metrics[r].append(metrics)
                
    # Aggregate by best epoch per round
    best_per_round = {}
    for r, metrics_list in round_metrics.items():
        # Using mIoU to select best epoch
        best_epoch = max(metrics_list, key=lambda x: x["mIoU"])
        best_per_round[r] = best_epoch
        
    return best_per_round

def _lambda_from_code(code: str) -> float:
    code = str(code or "").strip()
    mapping = {
        "00": 0.0,
        "02": 0.2,
        "04": 0.4,
        "06": 0.6,
        "08": 0.8,
        "10": 1.0,
    }
    if code in mapping:
        return mapping[code]
    if re.fullmatch(r"\d+(\.\d+)?", code):
        return float(code)
    raise ValueError(f"Unrecognized lambda code: {code}")


def _extract_round_metric(best_per_round: dict, round_idx: int):
    if not best_per_round:
        return None
    if round_idx not in best_per_round:
        return None
    return best_per_round.get(round_idx)


def _ensure_list(x):
    if isinstance(x, list):
        return x
    return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default="lambda_sweep_experiment_01")
    parser.add_argument("--sweep-round", type=int, default=3, choices=[3, 6, 9])
    parser.add_argument("--trunk-exp", type=str, default="fixed_lambda")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    base_dir = repo_root / "results" / "runs" / str(args.run_id)

    if not base_dir.is_dir():
        raise FileNotFoundError(f"Run dir not found: {base_dir}")

    trunk_trace = base_dir / f"{args.trunk_exp}_trace.jsonl"
    if not trunk_trace.exists():
        raise FileNotFoundError(f"Trunk trace not found: {trunk_trace}")

    trunk_best = parse_trace(str(trunk_trace))
    trunk_metric = _extract_round_metric(trunk_best, int(args.sweep_round))
    trunk_miou = float(trunk_metric.get("mIoU")) if isinstance(trunk_metric, dict) else None
    trunk_pci = _ensure_list(trunk_metric.get("per_class_iou")) if isinstance(trunk_metric, dict) else []

    exp_pattern = re.compile(rf"^sweep_r{int(args.sweep_round)}_lam_(.+)$")
    exp_names = []
    for p in base_dir.glob("*_trace.jsonl"):
        name = p.name.replace("_trace.jsonl", "")
        m = exp_pattern.match(name)
        if not m:
            continue
        exp_names.append((name, m.group(1)))

    if not exp_names:
        raise RuntimeError(f"No sweep traces found for round={args.sweep_round} in {base_dir}")

    rows = []
    for name, code in sorted(exp_names, key=lambda x: _lambda_from_code(x[1])):
        lam = _lambda_from_code(code)
        best = parse_trace(str(base_dir / f"{name}_trace.jsonl"))
        metric = _extract_round_metric(best, int(args.sweep_round))
        if not isinstance(metric, dict):
            continue
        miou = float(metric.get("mIoU", 0.0))
        pci = _ensure_list(metric.get("per_class_iou", []))
        rows.append((lam, miou, pci, name))

    if not rows:
        raise RuntimeError("No valid metrics found in sweep traces")

    lambdas = [r[0] for r in rows]
    mious = [r[1] for r in rows]
    pcis = [r[2] for r in rows]

    max_c = max((len(p) for p in pcis), default=0)
    class_curves = []
    for c in range(max_c):
        class_curves.append([float(p[c]) if len(p) > c else 0.0 for p in pcis])

    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].grid(True, alpha=0.3)
    axes[0].plot(lambdas, mious, marker="o", label="mIoU")
    for c, ys in enumerate(class_curves):
        axes[0].plot(lambdas, ys, marker=".", label=f"class_{c}_IoU")
    axes[0].set_title(f"Round {args.sweep_round} IoU vs. λ (absolute)")
    axes[0].set_xlabel("λ")
    axes[0].set_ylabel("IoU")
    axes[0].legend()

    axes[1].grid(True, alpha=0.3)
    if trunk_miou is None:
        axes[1].set_title(f"Round {args.sweep_round} ΔIoU vs. λ (no trunk ref)")
        axes[1].plot(lambdas, [0.0 for _ in lambdas], marker="o", label="ΔmIoU")
    else:
        axes[1].set_title(f"Round {args.sweep_round} ΔIoU vs. λ (vs {args.trunk_exp})")
        axes[1].plot(lambdas, [v - trunk_miou for v in mious], marker="o", label="ΔmIoU")
        for c, ys in enumerate(class_curves):
            if len(trunk_pci) > c:
                axes[1].plot(lambdas, [v - float(trunk_pci[c]) for v in ys], marker=".", label=f"Δclass_{c}_IoU")
    axes[1].set_xlabel("λ")
    axes[1].set_ylabel("ΔIoU")
    axes[1].legend()

    plt.tight_layout()

    if args.out:
        out_path = Path(args.out).expanduser()
    else:
        out_path = base_dir / f"lambda_sweep_r{int(args.sweep_round)}_vs_class_iou.png"

    plt.savefig(out_path, dpi=300)
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    main()
