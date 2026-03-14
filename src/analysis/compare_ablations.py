import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _parse_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _resolve_dir(path: str) -> str:
    p = os.path.abspath(str(path))
    if os.path.isdir(p):
        return p
    raise FileNotFoundError(p)


def _status_files(root: str) -> List[str]:
    files = []
    files.extend(glob.glob(os.path.join(root, "*_status.json")))
    files.extend(glob.glob(os.path.join(root, "**", "*_status.json"), recursive=True))
    uniq = sorted(set(os.path.abspath(x) for x in files))
    return uniq


def _extract_metric(status_path: str, metric: str) -> Optional[float]:
    try:
        payload = json.loads(open(status_path, "r", encoding="utf-8").read())
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    result = payload.get("result")
    if isinstance(result, dict):
        if metric in result:
            return _parse_float(result.get(metric))
    if metric in payload:
        return _parse_float(payload.get(metric))
    alt_map = {
        "final_miou": "final_mIoU",
        "final_mIoU": "final_miou",
    }
    alt = alt_map.get(metric)
    if alt:
        if isinstance(result, dict) and alt in result:
            return _parse_float(result.get(alt))
        if alt in payload:
            return _parse_float(payload.get(alt))
    return None


def _summarize(dir_path: str, metric: str) -> Dict[str, Any]:
    vals = []
    for sp in _status_files(dir_path):
        v = _extract_metric(sp, metric)
        if v is None:
            continue
        vals.append(float(v))
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return {"dir": dir_path, "metric": metric, "n": 0, "mean": None, "std": None}
    std = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
    return {
        "dir": dir_path,
        "metric": metric,
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": std,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--metric", default="final_mIoU")
    args = parser.parse_args()

    baseline_dir = _resolve_dir(args.baseline)
    variant_dirs = [_resolve_dir(x) for x in (args.variants or [])]
    metric = str(args.metric)

    out = {"baseline": _summarize(baseline_dir, metric), "variants": []}
    for d in variant_dirs:
        out["variants"].append(_summarize(d, metric))
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
