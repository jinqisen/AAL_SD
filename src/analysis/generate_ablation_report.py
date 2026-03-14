import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np


def _parse_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _status_files(root: str) -> List[str]:
    files = []
    files.extend(glob.glob(os.path.join(root, "*_status.json")))
    files.extend(glob.glob(os.path.join(root, "**", "*_status.json"), recursive=True))
    return sorted(set(os.path.abspath(x) for x in files))


def _extract(status_path: str) -> Dict[str, Any]:
    try:
        payload = json.loads(open(status_path, "r", encoding="utf-8").read())
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    return {
        "status_path": status_path,
        "experiment_name": payload.get("experiment_name"),
        "run_id": payload.get("run_id"),
        "final_mIoU": _parse_float(result.get("final_mIoU")),
        "final_f1": _parse_float(result.get("final_f1")),
        "alc": _parse_float(result.get("alc")),
        "status": payload.get("status"),
    }


def _summarize(dir_path: str) -> Dict[str, Any]:
    rows = []
    for sp in _status_files(dir_path):
        r = _extract(sp)
        if not r:
            continue
        rows.append(r)
    def _stats(key: str):
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0:
            return {"n": 0, "mean": None, "std": None}
        std = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
        return {"n": int(arr.size), "mean": float(np.mean(arr)), "std": std}
    return {
        "dir": os.path.abspath(dir_path),
        "final_mIoU": _stats("final_mIoU"),
        "final_f1": _stats("final_f1"),
        "alc": _stats("alc"),
        "status_files": len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablations", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ablation_dirs = [os.path.abspath(x) for x in args.ablations]
    summaries = [_summarize(d) for d in ablation_dirs]

    lines = []
    lines.append("# Ablation Report\n")
    lines.append("| Ablation Dir | final_mIoU (mean±std, n) | ALC (mean±std, n) | final_f1 (mean±std, n) |\n")
    lines.append("|---|---:|---:|---:|\n")
    for s in summaries:
        m = s["final_mIoU"]
        a = s["alc"]
        f = s["final_f1"]
        def _fmt(x):
            if x["n"] <= 0 or x["mean"] is None:
                return "n=0"
            return f"{x['mean']:.4f}±{(x['std'] or 0.0):.4f} (n={x['n']})"
        lines.append(
            f"| {s['dir']} | {_fmt(m)} | {_fmt(a)} | {_fmt(f)} |\n"
        )

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(out_path)


if __name__ == "__main__":
    main()
