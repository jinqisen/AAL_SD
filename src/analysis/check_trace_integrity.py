import argparse
import glob
import json
import os


def _iter_trace_events(trace_path: str):
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except Exception:
                yield {"_parse_error": True, "_raw": line}
                continue
            if isinstance(evt, dict):
                yield evt


def _resolve_run_dir(run_id_or_dir: str) -> str:
    v = str(run_id_or_dir or "").strip()
    if not v:
        raise ValueError("run_id is required")
    if os.path.isdir(v):
        return v
    guess = os.path.join("results", "runs", v)
    if os.path.isdir(guess):
        return guess
    raise FileNotFoundError(f"run dir not found: {v}")


def _collect_trace_paths(run_dir: str):
    paths = []
    paths.extend(glob.glob(os.path.join(run_dir, "*_trace.jsonl")))
    paths.extend(glob.glob(os.path.join(run_dir, "**", "*_trace.jsonl"), recursive=True))
    uniq = []
    seen = set()
    for p in paths:
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        seen.add(ap)
        uniq.append(ap)
    uniq.sort()
    return uniq


def _check_data_source(events):
    total = 0
    missing = 0
    bad = 0
    allowed = {"official_val", "train_holdout"}
    for evt in events:
        if not isinstance(evt, dict):
            continue
        if evt.get("type") != "overfit_signal":
            continue
        total += 1
        src = evt.get("grad_probe_source")
        if src is None:
            missing += 1
        elif str(src) not in allowed:
            bad += 1
    ok = (total > 0) and (missing == 0) and (bad == 0)
    return ok, {"total": total, "missing": missing, "bad": bad, "allowed": sorted(allowed)}


def _check_lambda_decisions(events):
    total = 0
    missing_diag = 0
    for evt in events:
        if not isinstance(evt, dict):
            continue
        if evt.get("type") != "lambda_policy_apply":
            continue
        total += 1
        diag = evt.get("diagnostics")
        if diag is None or not isinstance(diag, dict):
            missing_diag += 1
    ok = (total > 0) and (missing_diag == 0)
    return ok, {"total": total, "missing_diagnostics": missing_diag}


def _check_u_history(events):
    total = 0
    missing = 0
    for evt in events:
        if not isinstance(evt, dict):
            continue
        if evt.get("type") != "lambda_policy_apply":
            continue
        diag = evt.get("diagnostics")
        if not isinstance(diag, dict):
            continue
        total += 1
        u_ad = diag.get("u_adaptive")
        if u_ad is None:
            missing += 1
            continue
        if not isinstance(u_ad, dict):
            missing += 1
            continue
        ts = evt.get("training_state")
        if not isinstance(ts, dict):
            missing += 1
            continue
        u_hist = ts.get("train_u_median_history")
        k_hist = ts.get("train_k_median_history")
        if not isinstance(u_hist, list) or not isinstance(k_hist, list):
            missing += 1
            continue
        if u_ad.get("enabled") is True and u_ad.get("history_len") is None:
            missing += 1
            continue
        if u_ad.get("enabled") is True:
            try:
                if int(u_ad.get("history_len")) != int(len(u_hist)):
                    missing += 1
                    continue
            except Exception:
                missing += 1
                continue
    ok = (total > 0) and (missing == 0)
    return ok, {"total": total, "missing_u_history_snapshot": missing}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument(
        "--checks",
        nargs="+",
        default=["data_source", "lambda_decisions", "u_history"],
    )
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_id)
    trace_paths = _collect_trace_paths(run_dir)
    if not trace_paths:
        raise FileNotFoundError(f"no trace files under: {run_dir}")

    check_map = {
        "data_source": _check_data_source,
        "lambda_decisions": _check_lambda_decisions,
        "u_history": _check_u_history,
    }
    requested = [str(x).strip() for x in (args.checks or []) if str(x).strip()]
    unknown = [x for x in requested if x not in check_map]
    if unknown:
        raise ValueError(f"unknown checks: {unknown}. supported={sorted(check_map.keys())}")

    overall_ok = True
    summary = {"run_dir": run_dir, "trace_files": trace_paths, "checks": {}}
    for trace_path in trace_paths:
        events = list(_iter_trace_events(trace_path))
        per_file = {}
        for name in requested:
            ok, meta = check_map[name](events)
            per_file[name] = {"ok": bool(ok), "meta": meta}
            overall_ok = overall_ok and bool(ok)
        summary["checks"][trace_path] = per_file

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not overall_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
