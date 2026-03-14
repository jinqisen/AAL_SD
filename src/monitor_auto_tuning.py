from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_RE_RUN = re.compile(r"^(autotune_iter\d+_|autotune_opt_iter\d+_|autotune_opt_iter\d{3}_)", re.I)
_RE_MIOU_TEST = re.compile(r"最终报告 mIoU\(test\):\s*([0-9.]+)")
_RE_MIOU_OUT = re.compile(r"最终输出 mIoU:\s*([0-9.]+)")
_RE_MIOU_LAST_VAL = re.compile(r"最后一轮选模 mIoU\(val\):\s*([0-9.]+)")
_RE_ROUND_RESULT = re.compile(r"本轮结果:\s*Round=(\d+).*?mIoU=([0-9.]+)")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _tail_lines(path: Path, *, max_bytes: int = 1_500_000, max_lines: int = 3000) -> List[str]:
    if not path.exists():
        return []
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            read_size = min(max_bytes, size)
            f.seek(max(0, size - read_size))
            data = f.read(read_size)
        text = data.decode("utf-8", errors="ignore")
        lines = text.splitlines()
        return lines[-max_lines:]
    except Exception:
        return []


def _parse_epoch_end_events(trace_path: Path) -> List[Dict[str, Any]]:
    lines = _tail_lines(trace_path)
    out: List[Dict[str, Any]] = []
    for ln in lines:
        s = ln.strip()
        if not s or not s.startswith("{"):
            continue
        try:
            e = json.loads(s)
        except Exception:
            continue
        if not isinstance(e, dict):
            continue
        if e.get("type") != "epoch_end":
            continue
        out.append(e)
    return out


def _parse_objective_miou_from_md(md_path: Path, objective: str) -> Optional[float]:
    if not md_path.exists():
        return None
    try:
        text = md_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    obj = str(objective or "").strip().lower()
    if obj in ("val", "best_val", "last_val"):
        m = _RE_MIOU_LAST_VAL.search(text)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
    m = _RE_MIOU_TEST.search(text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    m = _RE_MIOU_OUT.search(text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def _parse_round_curve_from_md(md_path: Path) -> List[Tuple[int, float]]:
    if not md_path.exists():
        return []
    try:
        text = md_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    best_by_round: Dict[int, float] = {}
    for m in _RE_ROUND_RESULT.finditer(text):
        try:
            r = int(m.group(1))
            v = float(m.group(2))
        except Exception:
            continue
        best_by_round[r] = v
    return sorted(best_by_round.items(), key=lambda x: x[0])


def _mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return sum(xs) / len(xs)


def _std(xs: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    m = _mean(xs)
    if m is None:
        return None
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


@dataclass(frozen=True)
class TrainingCurveSummary:
    run_id: str
    exp_name: str
    status: str
    round_idx: Optional[int]
    epoch_idx: Optional[int]
    loss: Optional[float]
    miou: Optional[float]
    loss_down_ratio: Optional[float]
    loss_first_last: Optional[Tuple[float, float]]
    recent_round_mious: List[Tuple[int, float]]
    recent_round_std: Optional[float]


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    created_at: str
    updated_at: str
    objective: str
    best_exp: Optional[str]
    best_miou: Optional[float]
    counts: Dict[str, int]


def _list_runs(runs_dir: Path, *, max_runs: int) -> List[Path]:
    dirs: List[Path] = []
    for p in runs_dir.iterdir():
        if not p.is_dir():
            continue
        if not _RE_RUN.search(p.name):
            continue
        dirs.append(p)
    dirs.sort(key=lambda x: x.stat().st_mtime if x.exists() else 0.0, reverse=True)
    return dirs[: max_runs]


def _summarize_run(run_dir: Path, objective: str) -> RunSummary:
    run_id = run_dir.name
    manifest = _read_json(run_dir / "manifest.json") or {}
    created_at = str(manifest.get("created_at") or "-")
    updated_at = str(manifest.get("updated_at") or "-")

    status_files = sorted(run_dir.glob("*_status.json"))
    counts: Dict[str, int] = {"completed": 0, "running": 0, "failed": 0, "unknown": 0}
    best_exp: Optional[str] = None
    best_miou: Optional[float] = None

    for sp in status_files:
        payload = _read_json(sp) or {}
        st = str(payload.get("status") or "unknown").lower()
        if st not in counts:
            st = "unknown"
        counts[st] += 1
        exp_name = str(payload.get("experiment_name") or sp.name.replace("_status.json", ""))
        md_path = run_dir / f"{exp_name}.md"
        v = _parse_objective_miou_from_md(md_path, objective)
        if v is None and isinstance(payload.get("result"), dict):
            vv = payload["result"].get("final_mIoU")
            if isinstance(vv, (int, float)):
                v = float(vv)
        if v is None:
            continue
        if best_miou is None or v > best_miou:
            best_miou = v
            best_exp = exp_name

    return RunSummary(
        run_id=run_id,
        created_at=created_at,
        updated_at=updated_at,
        objective=str(objective),
        best_exp=best_exp,
        best_miou=best_miou,
        counts=counts,
    )


def _summarize_training_curve(run_dir: Path, exp_name: str, objective: str, window_rounds: int) -> TrainingCurveSummary:
    run_id = run_dir.name
    status_path = run_dir / f"{exp_name}_status.json"
    status = _read_json(status_path) or {}
    st = str(status.get("status") or "unknown").lower()

    progress = status.get("progress") if isinstance(status.get("progress"), dict) else {}
    round_idx = progress.get("round")
    epoch_idx = progress.get("epoch")
    loss = progress.get("loss")
    miou = progress.get("mIoU")
    try:
        round_idx_i = int(round_idx) if isinstance(round_idx, (int, float, str)) and str(round_idx).isdigit() else None
    except Exception:
        round_idx_i = None
    try:
        epoch_idx_i = int(epoch_idx) if isinstance(epoch_idx, (int, float, str)) and str(epoch_idx).isdigit() else None
    except Exception:
        epoch_idx_i = None
    loss_f = float(loss) if isinstance(loss, (int, float)) else None
    miou_f = float(miou) if isinstance(miou, (int, float)) else None

    trace_path = run_dir / f"{exp_name}_trace.jsonl"
    events = _parse_epoch_end_events(trace_path)
    current_round = None
    if round_idx_i is not None:
        current_round = round_idx_i
    else:
        for e in reversed(events):
            r = e.get("round")
            try:
                current_round = int(r)
                break
            except Exception:
                continue

    losses: List[float] = []
    if current_round is not None:
        per_epoch = {}
        for e in events:
            try:
                r = int(e.get("round"))
                ep = int(e.get("epoch"))
            except Exception:
                continue
            if r != int(current_round):
                continue
            lv = e.get("loss")
            if not isinstance(lv, (int, float)):
                continue
            per_epoch[ep] = float(lv)
        for ep in sorted(per_epoch):
            losses.append(per_epoch[ep])

    loss_down_ratio: Optional[float] = None
    loss_first_last: Optional[Tuple[float, float]] = None
    if len(losses) >= 2:
        downs = sum(1 for i in range(1, len(losses)) if losses[i] <= losses[i - 1])
        loss_down_ratio = downs / (len(losses) - 1)
        loss_first_last = (losses[0], losses[-1])

    md_path = run_dir / f"{exp_name}.md"
    curve = _parse_round_curve_from_md(md_path)
    recent_curve = curve[-window_rounds:] if window_rounds > 0 else curve
    recent_vals = [v for _, v in recent_curve]
    recent_std = _std(recent_vals)

    return TrainingCurveSummary(
        run_id=run_id,
        exp_name=exp_name,
        status=st,
        round_idx=round_idx_i,
        epoch_idx=epoch_idx_i,
        loss=loss_f,
        miou=miou_f,
        loss_down_ratio=loss_down_ratio,
        loss_first_last=loss_first_last,
        recent_round_mious=recent_curve,
        recent_round_std=recent_std,
    )


def _iter_order_key(run_id: str) -> Tuple[int, str]:
    m = re.search(r"iter(\d+)", run_id)
    if m:
        try:
            return int(m.group(1)), run_id
        except Exception:
            return 10**9, run_id
    return 10**9, run_id


def _print_overview(
    runs: List[RunSummary],
    *,
    objective: str,
    show_limit: int,
) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Auto-Tuning Overview (objective={objective})")
    print("=" * 110)
    print(f"{'Run':<40} | {'Done':<4} | {'Run':<4} | {'Fail':<4} | {'Best':<6} | {'BestExp'}")
    print("-" * 110)
    for r in runs[:show_limit]:
        done = int(r.counts.get("completed", 0))
        running = int(r.counts.get("running", 0))
        failed = int(r.counts.get("failed", 0))
        best = "-" if r.best_miou is None else f"{r.best_miou:.4f}"
        be = str(r.best_exp or "-")
        print(f"{r.run_id:<40} | {done:<4} | {running:<4} | {failed:<4} | {best:<6} | {be}")
    print()


def _print_tuning_curve(runs: List[RunSummary]) -> None:
    ordered = sorted(runs, key=lambda r: _iter_order_key(r.run_id))
    best_so_far: Optional[float] = None
    print("Tuning Curve (best-per-iter vs best-so-far)")
    print("-" * 110)
    print(f"{'IterRun':<40} | {'Best':<8} | {'BestSoFar':<9} | {'Improved'}")
    for r in ordered:
        b = r.best_miou
        if b is None:
            print(f"{r.run_id:<40} | {'-':<8} | {('-' if best_so_far is None else f'{best_so_far:.4f}'):<9} | -")
            continue
        if best_so_far is None or b > best_so_far + 1e-12:
            best_so_far = b
            improved = "Y"
        else:
            improved = "N"
        print(f"{r.run_id:<40} | {b:.4f}   | {best_so_far:.4f}     | {improved}")
    print()


def _print_training_curves(curves: List[TrainingCurveSummary]) -> None:
    if not curves:
        return
    print("Training Curve (per running experiment)")
    print("-" * 110)
    for c in curves:
        r = "-" if c.round_idx is None else str(c.round_idx)
        e = "-" if c.epoch_idx is None else str(c.epoch_idx)
        loss = "-" if c.loss is None else f"{c.loss:.4f}"
        miou = "-" if c.miou is None else f"{c.miou:.4f}"
        if c.loss_down_ratio is None:
            trend = "loss_trend=-"
        else:
            f0, f1 = c.loss_first_last or (None, None)
            trend = f"loss_down={c.loss_down_ratio:.2f}"
            if f0 is not None and f1 is not None:
                trend += f" ({f0:.4f}->{f1:.4f})"
        recent = ", ".join([f"R{rr}:{vv:.4f}" for rr, vv in c.recent_round_mious[-5:]])
        std = "-" if c.recent_round_std is None else f"{c.recent_round_std:.4f}"
        print(f"{c.run_id}/{c.exp_name}")
        print(f"  status={c.status} round={r} epoch={e} loss={loss} miou={miou} {trend}")
        print(f"  recent_best_val=[{recent}] std≈{std}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default="results/runs")
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--max-runs", type=int, default=30)
    parser.add_argument("--objective", type=str, default="val")
    parser.add_argument("--window-rounds", type=int, default=5)
    parser.add_argument("--show", type=int, default=12)
    parser.add_argument("--run-ids", type=str, default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = (repo_root / runs_dir).resolve()

    run_ids = [x.strip() for x in str(args.run_ids).split(",") if x.strip()]

    while True:
        if run_ids:
            run_dirs = [runs_dir / rid for rid in run_ids if (runs_dir / rid).is_dir()]
        else:
            run_dirs = _list_runs(runs_dir, max_runs=int(args.max_runs))

        summaries = [_summarize_run(d, str(args.objective)) for d in run_dirs]
        summaries.sort(key=lambda r: r.run_id)
        summaries.sort(key=lambda r: _iter_order_key(r.run_id))

        os.system("clear")
        _print_overview(summaries, objective=str(args.objective), show_limit=int(args.show))
        _print_tuning_curve(summaries)

        running_curves: List[TrainingCurveSummary] = []
        for d in run_dirs:
            for sp in d.glob("*_status.json"):
                payload = _read_json(sp) or {}
                if str(payload.get("status") or "").lower() != "running":
                    continue
                exp_name = str(payload.get("experiment_name") or sp.name.replace("_status.json", ""))
                running_curves.append(
                    _summarize_training_curve(
                        d,
                        exp_name,
                        str(args.objective),
                        int(args.window_rounds),
                    )
                )
        running_curves.sort(key=lambda x: (x.run_id, x.exp_name))
        _print_training_curves(running_curves)

        time.sleep(int(args.interval))


if __name__ == "__main__":
    main()

