import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _find_trace_files(path: str) -> List[str]:
    p = os.path.abspath(str(path))
    if os.path.isfile(p) and p.endswith(".jsonl"):
        return [p]
    if not os.path.isdir(p):
        raise FileNotFoundError(p)
    files = []
    files.extend(glob.glob(os.path.join(p, "*_trace.jsonl")))
    files.extend(glob.glob(os.path.join(p, "**", "*_trace.jsonl"), recursive=True))
    uniq = sorted(set(os.path.abspath(x) for x in files))
    if not uniq:
        raise FileNotFoundError(f"no *_trace.jsonl under: {p}")
    return uniq


def _load_trace(trace_path: str) -> List[Dict[str, Any]]:
    out = []
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except Exception:
                continue
            if isinstance(evt, dict):
                out.append(evt)
    return out


def _extract_series(events: List[Dict[str, Any]]) -> pd.DataFrame:
    u_by_round: Dict[int, float] = {}
    lam_by_round: Dict[int, float] = {}
    meta = {"run_id": None, "experiment_name": None}

    for evt in events:
        if meta["run_id"] is None and evt.get("run_id") is not None:
            meta["run_id"] = str(evt.get("run_id"))
        if meta["experiment_name"] is None and evt.get("experiment_name") is not None:
            meta["experiment_name"] = str(evt.get("experiment_name"))
        t = evt.get("type")
        r = evt.get("round")
        try:
            r = int(r)
        except Exception:
            continue
        if t == "l3_selection_stats":
            u = _parse_float(evt.get("u_median_selected"))
            if u is not None:
                u_by_round[r] = float(u)
        elif t == "lambda_policy_apply":
            lam = _parse_float(evt.get("applied"))
            if lam is not None:
                lam_by_round[r] = float(lam)

    rounds = sorted(set(u_by_round.keys()) | set(lam_by_round.keys()))
    rows = []
    prev_u = None
    for r in rounds:
        u = u_by_round.get(r)
        lam = lam_by_round.get(r)
        u_delta = None
        if u is not None and prev_u is not None:
            u_delta = float(u) - float(prev_u)
        if u is not None:
            prev_u = float(u)
        rows.append(
            {
                "run_id": meta["run_id"],
                "experiment_name": meta["experiment_name"],
                "round": int(r),
                "u_median_selected": u,
                "u_delta": u_delta,
                "lambda_applied": lam,
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frames = []
    for p in args.runs:
        for trace_path in _find_trace_files(p):
            events = _load_trace(trace_path)
            df = _extract_series(events)
            df["trace_path"] = os.path.abspath(trace_path)
            frames.append(df)

    if not frames:
        raise RuntimeError("no data")
    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["u_delta", "lambda_applied"])
    if data.empty:
        raise RuntimeError("no overlapping u_delta and lambda_applied points to plot")

    plt.figure(figsize=(8, 6))
    x = data["u_delta"].to_numpy(dtype=float)
    y = data["lambda_applied"].to_numpy(dtype=float)
    plt.scatter(x, y, s=18, alpha=0.7)
    plt.axvline(0.0, color="gray", linewidth=1.0, alpha=0.5)
    plt.xlabel("ΔU (median_selected, round t - t-1)")
    plt.ylabel("λ applied")
    plt.title("U median delta vs λ applied")
    plt.grid(True, alpha=0.2)

    out_path = os.path.abspath(str(args.output))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(out_path)


if __name__ == "__main__":
    main()
