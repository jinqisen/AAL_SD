import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np


@dataclass(frozen=True)
class MaskStat:
    has_positive: bool
    positive_frac: float


def _safe_float(x: object) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


def _summarize(values: List[float]) -> Dict[str, Optional[float]]:
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not xs:
        return {"n": 0, "mean": None, "p10": None, "p50": None, "p90": None}
    arr = np.asarray(xs, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _load_initialized_meta(trace_path: str) -> Dict[str, object]:
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("type") == "initialized":
                return ev
    return {}


def _resolve_pools_base_dir(trace_path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    meta = _load_initialized_meta(trace_path)
    run_id = meta.get("run_id")
    exp_name = meta.get("experiment_name")
    pools_dir = meta.get("pools_dir")
    if not isinstance(pools_dir, str) or not pools_dir:
        return None, (str(run_id) if run_id is not None else None), (str(exp_name) if exp_name is not None else None)

    parent = os.path.dirname(pools_dir.rstrip("/"))
    base_dir = os.path.join(parent, "_base")
    if os.path.isdir(base_dir):
        return base_dir, (str(run_id) if run_id is not None else None), (str(exp_name) if exp_name is not None else None)
    if os.path.isdir(pools_dir):
        return pools_dir, (str(run_id) if run_id is not None else None), (str(exp_name) if exp_name is not None else None)
    return None, (str(run_id) if run_id is not None else None), (str(exp_name) if exp_name is not None else None)


def _load_pool_csv(pool_csv_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(pool_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                continue
            rows.append({str(k): ("" if v is None else str(v)) for k, v in row.items()})
    return rows


def _build_sample_to_mask_path(pools_base_dir: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for name in ("labeled_pool.csv", "unlabeled_pool.csv"):
        path = os.path.join(pools_base_dir, name)
        if not os.path.isfile(path):
            continue
        for row in _load_pool_csv(path):
            sid = (row.get("sample_id") or "").strip()
            mp = (row.get("mask_path") or "").strip()
            if sid and mp:
                mapping[sid] = mp
    return mapping


def _read_mask_stat(mask_path: str) -> MaskStat:
    with h5py.File(mask_path, "r") as f:
        if "mask" not in f:
            raise RuntimeError(f"Missing dataset key 'mask' in {mask_path} (keys={list(f.keys())})")
        mask = f["mask"][()]
    arr = np.asarray(mask)
    has_pos = bool(np.any(arr > 0))
    pos_frac = float(np.mean(arr > 0))
    return MaskStat(has_positive=has_pos, positive_frac=pos_frac)


def _load_cache(cache_path: str) -> Dict[str, MaskStat]:
    if not os.path.isfile(cache_path):
        return {}
    out: Dict[str, MaskStat] = {}
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            sid = obj.get("sample_id")
            mp = obj.get("mask_path")
            hp = obj.get("has_positive")
            pf = obj.get("positive_frac")
            if not isinstance(sid, str) or not isinstance(mp, str):
                continue
            if not isinstance(hp, bool):
                continue
            pv = _safe_float(pf)
            if pv is None:
                continue
            out[sid] = MaskStat(has_positive=hp, positive_frac=float(pv))
    return out


def _append_cache(cache_path: str, sample_id: str, mask_path: str, stat: MaskStat) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "mask_path": mask_path,
                    "has_positive": bool(stat.has_positive),
                    "positive_frac": float(stat.positive_frac),
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def _iter_selected_ids_by_round(trace_path: str) -> Iterable[Tuple[int, List[int]]]:
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("type") != "selection":
                continue
            r = ev.get("round")
            try:
                rr = int(r)
            except Exception:
                continue
            ids = ev.get("selected_ids")
            if not isinstance(ids, list):
                continue
            out: List[int] = []
            for it in ids:
                try:
                    out.append(int(it))
                except Exception:
                    continue
            yield rr, out


def _sample_id_from_int(sample_int_id: int) -> str:
    return f"image_{int(sample_int_id)}"


def analyze(trace_path: str, output_dir: str) -> str:
    pools_base_dir, run_id, exp_name = _resolve_pools_base_dir(trace_path)
    if pools_base_dir is None:
        raise RuntimeError(f"Cannot resolve pools base dir from trace: {trace_path}")

    sample_to_mask = _build_sample_to_mask_path(pools_base_dir)
    if not sample_to_mask:
        raise RuntimeError(f"No mask paths found in pools base dir: {pools_base_dir}")

    pools_manifest = os.path.join(pools_base_dir, "pools_manifest.json")
    dataset_fingerprint = None
    data_root = None
    if os.path.isfile(pools_manifest):
        try:
            with open(pools_manifest, "r", encoding="utf-8") as f:
                man = json.load(f)
            if isinstance(man, dict):
                data_root = man.get("data_root")
                dataset_fingerprint = man.get("splits", {}).get("train", {}).get("masks", {}).get("sha256")
        except Exception:
            pass

    os.makedirs(output_dir, exist_ok=True)
    cache_path = os.path.join(output_dir, f"{(run_id or 'run')}_{(exp_name or 'exp')}_mask_stat_cache.jsonl")
    cache = _load_cache(cache_path)

    seen_cache = set(cache.keys())
    out_rows: List[Dict[str, object]] = []
    labeled_set: List[str] = []

    init_labeled_csv = os.path.join(pools_base_dir, "labeled_pool.csv")
    if os.path.isfile(init_labeled_csv):
        for row in _load_pool_csv(init_labeled_csv):
            sid = (row.get("sample_id") or "").strip()
            if sid:
                labeled_set.append(sid)

    for round_idx, sel_ids in sorted(_iter_selected_ids_by_round(trace_path), key=lambda x: x[0]):
        selected_sids = [_sample_id_from_int(i) for i in sel_ids]

        selected_pos_fracs: List[float] = []
        selected_has_pos: List[float] = []

        for sid in selected_sids:
            mp = sample_to_mask.get(sid)
            if not mp:
                continue
            if sid in cache:
                stat = cache[sid]
            else:
                stat = _read_mask_stat(mp)
                cache[sid] = stat
                _append_cache(cache_path, sid, mp, stat)
                seen_cache.add(sid)
            selected_pos_fracs.append(stat.positive_frac)
            selected_has_pos.append(1.0 if stat.has_positive else 0.0)

        labeled_set.extend([sid for sid in selected_sids if sid not in set(labeled_set)])
        labeled_pos_fracs: List[float] = []
        labeled_has_pos: List[float] = []
        for sid in labeled_set:
            mp = sample_to_mask.get(sid)
            if not mp:
                continue
            if sid in cache:
                stat = cache[sid]
            else:
                stat = _read_mask_stat(mp)
                cache[sid] = stat
                _append_cache(cache_path, sid, mp, stat)
                seen_cache.add(sid)
            labeled_pos_fracs.append(stat.positive_frac)
            labeled_has_pos.append(1.0 if stat.has_positive else 0.0)

        ssum = _summarize(selected_pos_fracs)
        lsum = _summarize(labeled_pos_fracs)

        out_rows.append(
            {
                "run_id": run_id or "",
                "experiment": exp_name or "",
                "round": int(round_idx),
                "selected_n": int(ssum["n"] or 0),
                "selected_has_positive_rate": float(np.mean(selected_has_pos)) if selected_has_pos else None,
                "selected_positive_frac_mean": ssum["mean"],
                "selected_positive_frac_p50": ssum["p50"],
                "selected_positive_frac_p10": ssum["p10"],
                "selected_positive_frac_p90": ssum["p90"],
                "labeled_n": int(lsum["n"] or 0),
                "labeled_has_positive_rate": float(np.mean(labeled_has_pos)) if labeled_has_pos else None,
                "labeled_positive_frac_mean": lsum["mean"],
                "labeled_positive_frac_p50": lsum["p50"],
                "data_root": data_root or "",
                "train_masks_sha256": dataset_fingerprint or "",
            }
        )

    out_path = os.path.join(output_dir, f"{(run_id or 'run')}_{(exp_name or 'exp')}_mask_distribution_by_round.csv")
    if out_rows:
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("run_id,experiment,round\n")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze mask distribution of selected samples per round from trace + pools")
    parser.add_argument("trace_path", type=str, help="Path to *_trace.jsonl")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: trace directory)")
    args = parser.parse_args()

    trace_path = str(args.trace_path)
    output_dir = str(args.output) if args.output else os.path.dirname(trace_path)
    out = analyze(trace_path, output_dir)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

