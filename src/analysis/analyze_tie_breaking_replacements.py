import argparse
import json
from pathlib import Path


def iter_trace_entries(trace_path: Path):
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def collect_score_snapshots(trace_path: Path):
    out = {}
    for entry in iter_trace_entries(trace_path):
        if entry.get("type") == "score_snapshot":
            round_idx = entry.get("round")
            if round_idx is not None:
                out[int(round_idx)] = entry
    return out


def collect_query_size(trace_path: Path):
    for entry in iter_trace_entries(trace_path):
        if entry.get("type") == "selection":
            expected = entry.get("expected")
            if expected is not None:
                return int(expected)
    return None


def build_row_index(rows):
    index = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if sample_id is not None:
            index[int(sample_id)] = row
    return index


def _float_or_none(v):
    if v is None:
        return None
    return float(v)


def _summary(values):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {}
    vals = sorted(vals)
    n = len(vals)

    def _pick(q):
        idx = min(n - 1, max(0, int(round((n - 1) * q))))
        return vals[idx]

    return {
        "n": n,
        "min": vals[0],
        "p50": _pick(0.5),
        "p90": _pick(0.9),
        "max": vals[-1],
        "mean": sum(vals) / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_lambda_0", type=Path)
    parser.add_argument("trace_lambda_02", type=Path)
    parser.add_argument("--round", dest="round_idx", type=int, required=True)
    parser.add_argument("--epsilon", type=float, default=0.01)
    args = parser.parse_args()

    snap0 = collect_score_snapshots(args.trace_lambda_0)
    snap2 = collect_score_snapshots(args.trace_lambda_02)
    query_size = collect_query_size(args.trace_lambda_0) or collect_query_size(
        args.trace_lambda_02
    )

    if args.round_idx not in snap0 or args.round_idx not in snap2:
        print("Missing score_snapshot for the requested round.")
        print("This script requires reruns generated with the new geometry logging.")
        return 1
    if not query_size:
        print("Could not infer query size from selection trace.")
        return 1

    rows0 = snap0[args.round_idx].get("rows") or []
    rows2 = snap2[args.round_idx].get("rows") or []
    if not rows0 or not rows2:
        print("Score snapshot rows are empty.")
        return 1

    idx0 = build_row_index(rows0)

    top0 = [int(r["sample_id"]) for r in rows0[:query_size]]
    top2 = [int(r["sample_id"]) for r in rows2[:query_size]]
    top0_set = set(top0)
    top2_set = set(top2)
    replaced_out = sorted(top0_set - top2_set)
    replaced_in = sorted(top2_set - top0_set)

    boundary_start = max(0, query_size - max(1, int(query_size * 0.2)))
    boundary_rows0 = rows0[boundary_start:query_size]
    boundary_u = [float(r["uncertainty"]) for r in boundary_rows0 if r.get("uncertainty") is not None]
    boundary_span = (
        max(boundary_u) - min(boundary_u) if boundary_u else None
    )

    out_rows = [idx0[sid] for sid in replaced_out if sid in idx0]
    cutoff_row = rows0[query_size - 1]
    cutoff_u = _float_or_none(cutoff_row.get("uncertainty"))
    lambda_value = _float_or_none((rows2[0] if rows2 else {}).get("lambda_t"))
    near_tie_out = [
        row
        for row in out_rows
        if row.get("uncertainty") is not None
        and cutoff_u is not None
        and abs(float(row["uncertainty"]) - float(cutoff_u))
        <= float(args.epsilon)
    ]
    boundary_ku_diff = [
        float(r["knowledge_gain"]) - float(r["uncertainty"])
        for r in boundary_rows0
        if r.get("knowledge_gain") is not None and r.get("uncertainty") is not None
    ]
    ku_sigma = None
    ku_abs_mean = None
    if boundary_ku_diff:
        mean_ku = sum(boundary_ku_diff) / len(boundary_ku_diff)
        ku_sigma = (
            sum((x - mean_ku) ** 2 for x in boundary_ku_diff) / len(boundary_ku_diff)
        ) ** 0.5
        ku_abs_mean = sum(abs(x) for x in boundary_ku_diff) / len(boundary_ku_diff)
    top0_rows = rows0[:query_size]
    top0_margins = [
        abs(float(r["uncertainty"]) - float(cutoff_u))
        for r in top0_rows
        if r.get("uncertainty") is not None and cutoff_u is not None
    ]
    replaced_out_margins = [
        abs(float(r["uncertainty"]) - float(cutoff_u))
        for r in out_rows
        if r.get("uncertainty") is not None and cutoff_u is not None
    ]
    retained_top0_margins = [
        abs(float(r["uncertainty"]) - float(cutoff_u))
        for r in top0_rows
        if r.get("sample_id") is not None
        and int(r["sample_id"]) not in set(replaced_out)
        and r.get("uncertainty") is not None
        and cutoff_u is not None
    ]
    margin_band = (
        float(lambda_value) * float(ku_sigma)
        if lambda_value is not None and ku_sigma is not None
        else None
    )
    top0_in_band = (
        [
            m
            for m in top0_margins
            if margin_band is not None and float(m) < float(margin_band)
        ]
        if margin_band is not None
        else []
    )
    replaced_out_in_band = (
        [
            m
            for m in replaced_out_margins
            if margin_band is not None and float(m) < float(margin_band)
        ]
        if margin_band is not None
        else []
    )

    ks_stat = None
    ks_pvalue = None
    mw_stat = None
    mw_pvalue = None
    if replaced_out_margins and retained_top0_margins:
        try:
            from scipy.stats import ks_2samp, mannwhitneyu

            ks_res = ks_2samp(replaced_out_margins, retained_top0_margins)
            ks_stat = float(ks_res.statistic)
            ks_pvalue = float(ks_res.pvalue)
            mw_res = mannwhitneyu(
                replaced_out_margins,
                retained_top0_margins,
                alternative="less",
            )
            mw_stat = float(mw_res.statistic)
            mw_pvalue = float(mw_res.pvalue)
        except Exception:
            pass

    print(f"round={args.round_idx}")
    print(f"query_size={query_size}")
    print(f"boundary_start_rank={boundary_start + 1}")
    print(f"lambda_probe={lambda_value}")
    print(f"cutoff_u={cutoff_u}")
    print(f"lambda0_top_size={len(top0_set)} lambda02_top_size={len(top2_set)}")
    print(f"replaced_out_count={len(replaced_out)}")
    print(f"replaced_in_count={len(replaced_in)}")
    print(f"boundary_u_span={boundary_span}")
    print(f"near_tie_out_count={len(near_tie_out)}")
    print(f"near_tie_out_ratio={len(near_tie_out) / len(out_rows) if out_rows else 0.0}")
    print(f"boundary_ku_sigma={ku_sigma}")
    print(f"boundary_ku_abs_mean={ku_abs_mean}")
    print(f"predicted_margin_band={margin_band}")
    print(f"top0_in_margin_band_count={len(top0_in_band)}")
    print(
        f"top0_in_margin_band_ratio={len(top0_in_band) / len(top0_margins) if top0_margins else 0.0}"
    )
    print(f"replaced_out_in_margin_band_count={len(replaced_out_in_band)}")
    print(
        "replaced_out_in_margin_band_ratio="
        + str(
            len(replaced_out_in_band) / len(replaced_out_margins)
            if replaced_out_margins
            else 0.0
        )
    )
    print("replaced_out_margin_summary=" + json.dumps(_summary(replaced_out_margins), ensure_ascii=False))
    print("retained_top0_margin_summary=" + json.dumps(_summary(retained_top0_margins), ensure_ascii=False))
    print("top0_margin_summary=" + json.dumps(_summary(top0_margins), ensure_ascii=False))
    print(f"ks_stat={ks_stat}")
    print(f"ks_pvalue={ks_pvalue}")
    print(f"mannwhitney_less_stat={mw_stat}")
    print(f"mannwhitney_less_pvalue={mw_pvalue}")
    print("replaced_out_sample_ids=" + ",".join(str(x) for x in replaced_out[:50]))
    print("replaced_in_sample_ids=" + ",".join(str(x) for x in replaced_in[:50]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
