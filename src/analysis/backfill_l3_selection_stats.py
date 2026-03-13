import os
import json
import argparse
import csv
import math
import statistics
from typing import Any, Dict, List, Optional, Tuple


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _median(xs: List[float]) -> Optional[float]:
    vals = [float(v) for v in xs if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    try:
        return float(statistics.median(vals))
    except Exception:
        vals = sorted(vals)
        n = len(vals)
        mid = n // 2
        if n % 2 == 1:
            return float(vals[mid])
        return float((vals[mid - 1] + vals[mid]) / 2.0)


def _extract_vals(items: Any, key: str) -> List[float]:
    if not isinstance(items, list) or not items:
        return []
    out: List[float] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        v = _safe_float(it.get(key))
        if v is not None:
            out.append(v)
    return out


def _parse_trace(trace_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    per_round: Dict[int, Dict[str, Any]] = {}
    meta: Dict[str, Any] = {"trace_path": trace_path}

    def _row(r: Any) -> Dict[str, Any]:
        rr = int(r)
        if rr not in per_round:
            per_round[rr] = {"round": rr}
        return per_round[rr]

    with open(trace_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {trace_path}:{line_no} ({e})") from e

            etype = entry.get("type")
            if etype == "selection":
                r = entry.get("round")
                if r is None:
                    continue
                row = _row(r)
                row["lambda_effective"] = _safe_float(entry.get("lambda_effective"))
                row["lambda_source"] = entry.get("lambda_source")
                ctx = entry.get("context") if isinstance(entry.get("context"), dict) else {}
                sel_stats = None
                if isinstance(ctx, dict):
                    sel_stats = ctx.get("selected_score_stats")
                if isinstance(sel_stats, dict):
                    row["selected_score_stats"] = sel_stats

            if etype == "l3_selection_stats":
                r = entry.get("round")
                if r is None:
                    continue
                row = _row(r)
                row["source"] = entry.get("source")
                row["topk"] = entry.get("topk")
                row["selected_limit"] = entry.get("selected_limit")
                row["u_median_selected"] = _safe_float(entry.get("u_median_selected"))
                row["k_median_selected"] = _safe_float(entry.get("k_median_selected"))
                row["u_median_top"] = _safe_float(entry.get("u_median_top"))
                row["k_median_top"] = _safe_float(entry.get("k_median_top"))
                row["stats_method"] = "l3_selection_stats"

            if etype == "l3_selection":
                r = entry.get("round")
                if r is None:
                    continue
                row = _row(r)
                if row.get("stats_method") == "l3_selection_stats":
                    continue
                row["source"] = entry.get("source")
                row["topk"] = entry.get("topk")
                row["selected_limit"] = entry.get("selected_limit")
                top_items = entry.get("top_items")
                selected_items = entry.get("selected_items")
                row["u_median_selected"] = _median(_extract_vals(selected_items, "uncertainty"))
                row["k_median_selected"] = _median(_extract_vals(selected_items, "knowledge_gain"))
                row["u_median_top"] = _median(_extract_vals(top_items, "uncertainty"))
                row["k_median_top"] = _median(_extract_vals(top_items, "knowledge_gain"))
                row["stats_method"] = "l3_selection"

    rows = [per_round[k] for k in sorted(per_round.keys())]
    return rows, meta


def _resolve_run_dir(base_runs_dir: str, run_id: str) -> str:
    p = os.path.join(base_runs_dir, run_id)
    if not os.path.isdir(p):
        raise FileNotFoundError(f"Run directory not found: {p}")
    return p


def _find_traces(run_dir: str) -> List[str]:
    out: List[str] = []
    for name in os.listdir(run_dir):
        if name.endswith("_trace.jsonl"):
            out.append(os.path.join(run_dir, name))
    out.sort()
    return out


def _infer_lambda_from_selected_stats(sel_stats: Any, key: str) -> Optional[float]:
    if not isinstance(sel_stats, dict):
        return None
    st = sel_stats.get(key)
    if not isinstance(st, dict):
        return None
    v = st.get("p50")
    return _safe_float(v)


def _write_csv(csv_path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    keys = [
        "run_id",
        "experiment_name",
        "round",
        "source",
        "topk",
        "selected_limit",
        "u_median_selected",
        "k_median_selected",
        "u_median_top",
        "k_median_top",
        "lambda_effective",
        "lambda_source",
        "stats_method",
        "fallback_selected_stats",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_runs_dir",
        type=str,
        default=os.path.join("results", "runs"),
    )
    parser.add_argument(
        "--run_ids",
        type=str,
        default="baseline_20260311_201728_seed42,baseline_20260311_201728_seed43,baseline_20260309_211601_seed42,baseline_20260309_211601_seed43",
    )
    args = parser.parse_args()

    base_runs_dir = os.path.abspath(args.base_runs_dir)
    run_ids = [x.strip() for x in str(args.run_ids).split(",") if x.strip()]
    if not run_ids:
        raise ValueError("Empty run_ids")

    summary_rows: List[Dict[str, Any]] = []

    for run_id in run_ids:
        run_dir = _resolve_run_dir(base_runs_dir, run_id)
        traces = _find_traces(run_dir)
        if not traces:
            print(f"[WARN] no trace files found: run_id={run_id} dir={run_dir}")
            continue

        reports_dir = os.path.join(run_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        for trace_path in traces:
            exp = os.path.basename(trace_path).replace("_trace.jsonl", "")
            rows, _ = _parse_trace(trace_path)
            out_rows: List[Dict[str, Any]] = []
            with_stats = 0
            for row in rows:
                rr = {
                    "run_id": run_id,
                    "experiment_name": exp,
                    "round": row.get("round"),
                    "source": row.get("source"),
                    "topk": row.get("topk"),
                    "selected_limit": row.get("selected_limit"),
                    "u_median_selected": row.get("u_median_selected"),
                    "k_median_selected": row.get("k_median_selected"),
                    "u_median_top": row.get("u_median_top"),
                    "k_median_top": row.get("k_median_top"),
                    "lambda_effective": row.get("lambda_effective"),
                    "lambda_source": row.get("lambda_source"),
                    "stats_method": row.get("stats_method"),
                    "fallback_selected_stats": False,
                }
                if out_rows and (out_rows[-1].get("round") == rr.get("round")):
                    pass
                if rr.get("u_median_selected") is not None or rr.get("k_median_selected") is not None:
                    with_stats += 1
                else:
                    sel_stats = row.get("selected_score_stats")
                    u50 = _infer_lambda_from_selected_stats(sel_stats, "uncertainty")
                    k50 = _infer_lambda_from_selected_stats(sel_stats, "knowledge_gain")
                    if u50 is not None or k50 is not None:
                        rr["u_median_selected"] = u50
                        rr["k_median_selected"] = k50
                        rr["stats_method"] = rr.get("stats_method") or "selection_context_stats"
                        rr["fallback_selected_stats"] = True
                        with_stats += 1
                out_rows.append(rr)

            csv_path = os.path.join(
                reports_dir, f"backfill_l3_selection_stats_{exp}.csv"
            )
            _write_csv(csv_path, out_rows)
            summary_rows.append(
                {
                    "run_id": run_id,
                    "experiment_name": exp,
                    "trace_path": trace_path,
                    "csv_path": csv_path,
                    "rounds_total": len(out_rows),
                    "rounds_with_stats": with_stats,
                }
            )
            print(
                f"[OK] run_id={run_id} exp={exp} rounds={len(out_rows)} "
                f"with_stats={with_stats} -> {csv_path}"
            )

        summary_csv = os.path.join(reports_dir, "backfill_l3_selection_stats_summary.csv")
        os.makedirs(os.path.dirname(summary_csv), exist_ok=True)
        with open(summary_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "run_id",
                    "experiment_name",
                    "trace_path",
                    "csv_path",
                    "rounds_total",
                    "rounds_with_stats",
                ],
            )
            w.writeheader()
            for r in summary_rows:
                if r.get("run_id") == run_id:
                    w.writerow(r)
        print(f"[OK] run_id={run_id} summary -> {summary_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

