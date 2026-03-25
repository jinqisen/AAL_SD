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

try:
    from tuning_opt.evaluator import parse_objective_from_status
except Exception:
    parse_objective_from_status = None


_RE_RUN = re.compile(r"^(autotune_iter\d+_|autotune_opt_iter\d+_|autotune_opt_iter\d{3}_)", re.I)
_RE_MIOU_TEST = re.compile(r"最终报告 mIoU\(test\):\s*([0-9.]+)")
_RE_MIOU_OUT = re.compile(r"最终输出 mIoU:\s*([0-9.]+)")
_RE_MIOU_LAST_VAL = re.compile(r"最后一轮选模 mIoU\(val\):\s*([0-9.]+)")
_RE_ROUND_RESULT = re.compile(r"本轮结果:\s*Round=(\d+).*?mIoU=([0-9.]+)")
_RE_BASE_RUN = re.compile(r"^autotune_opt_iter(\d+)_")
_RE_SEED_RUN = re.compile(r"^(autotune_opt_iter\d+_[0-9]{8}_[0-9]{4})_seed(\d+)$")


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
        matches = _RE_MIOU_LAST_VAL.findall(text)
        if matches:
            try:
                return float(matches[-1])
            except Exception:
                return None
    matches = _RE_MIOU_TEST.findall(text)
    if matches:
        try:
            return float(matches[-1])
        except Exception:
            return None
    matches = _RE_MIOU_OUT.findall(text)
    if matches:
        try:
            return float(matches[-1])
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


def _best_round_miou_from_md(md_path: Path) -> Tuple[Optional[int], Optional[float]]:
    curve = _parse_round_curve_from_md(md_path)
    if not curve:
        return None, None
    best_r, best_v = max(curve, key=lambda x: x[1])
    return int(best_r), float(best_v)


def _latest_epoch_end_from_trace(trace_path: Path) -> Optional[Dict[str, Any]]:
    events = _parse_epoch_end_events(trace_path)
    if not events:
        return None
    return events[-1]


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
    best_rd: Optional[int]
    counts: Dict[str, int]

@dataclass(frozen=True)
class AutoTuningIterSummary:
    iteration: int
    run_id: str
    updated_at: str
    n_exps: int
    n_running: int
    n_done: int
    f1_done: bool
    f2_done: bool
    f3_done: bool
    screen_best_exp: Optional[str]
    screen_best: Optional[float]
    confirm_best_exp: Optional[str]
    confirm_best: Optional[float]
    topk: List[str]
    topk_round: Dict[str, Optional[int]]
    topk_status: Dict[str, str]
    seed_done: Dict[int, bool]
    seed_round: Dict[int, Optional[int]]
    seed_score: Dict[int, Optional[float]]
    f3_mean: Optional[float]
    f3_std: Optional[float]
    f3_min: Optional[float]
    running_experiments: List[Dict[str, Any]]


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
    best_rd: Optional[int] = None

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
            rd_val, _ = _best_round_miou_from_md(md_path)
            best_rd = rd_val

    return RunSummary(
        run_id=run_id,
        created_at=created_at,
        updated_at=updated_at,
        objective=str(objective),
        best_exp=best_exp,
        best_miou=best_miou,
        best_rd=best_rd,
        counts=counts,
    )

def _list_autotune_base_runs(runs_dir: Path, *, max_runs: int) -> List[Tuple[int, Path]]:
    out: List[Tuple[int, Path]] = []
    for p in runs_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if _RE_SEED_RUN.match(name):
            continue
        m = _RE_BASE_RUN.match(name)
        if not m:
            continue
        try:
            it = int(m.group(1))
        except Exception:
            continue
        out.append((it, p))
    out.sort(key=lambda x: (x[0], x[1].name))
    return out[-max_runs:] if max_runs > 0 else out


def _find_seed_runs(runs_dir: Path, base_run_id: str) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for p in runs_dir.iterdir():
        if not p.is_dir():
            continue
        m = _RE_SEED_RUN.match(p.name)
        if not m:
            continue
        if str(m.group(1)) != str(base_run_id):
            continue
        out[int(m.group(2))] = p
    return dict(sorted(out.items(), key=lambda x: x[0]))


def _objective_from_status(run_dir: Path, exp: str, objective: str) -> Optional[float]:
    if parse_objective_from_status is not None:
        v = parse_objective_from_status(run_dir / f"{exp}_status.json", objective)
        if isinstance(v, (int, float)):
            return float(v)
    status = _read_json(run_dir / f"{exp}_status.json") or {}
    res = status.get("result") if isinstance(status.get("result"), dict) else {}
    obj = str(objective or "").strip().lower()
    if obj in ("alc", "learning_curve", "learning_curve_area"):
        v = res.get("alc")
        return float(v) if isinstance(v, (int, float)) else None
    if obj in ("val", "best_val", "last_val", "miou", "final_miou", "final_val"):
        v = res.get("final_mIoU")
        return float(v) if isinstance(v, (int, float)) else None
    if obj in ("f1", "final_f1"):
        v = res.get("final_f1")
        return float(v) if isinstance(v, (int, float)) else None
    return None


def _objective_from_status_or_md(run_dir: Path, exp: str, objective: str) -> Optional[float]:
    v = _objective_from_status(run_dir, exp, objective)
    if v is not None:
        return v
    return _parse_objective_miou_from_md(run_dir / f"{exp}.md", objective)


def _autotune_iter_summary(
    run_dir: Path,
    *,
    iteration: int,
    screen_objective: str,
    confirm_objective: str,
    screen_end_round: int,
    confirm_end_round: int,
    screen_topk: int,
    expected_seeds: List[int],
) -> AutoTuningIterSummary:
    manifest = _read_json(run_dir / "manifest.json") or {}
    updated_at = str(manifest.get("updated_at") or "-")
    status_files = sorted(run_dir.glob("*_status.json"))
    exps = []
    progress_round: Dict[str, int] = {}
    status_map: Dict[str, str] = {}
    running_experiments: List[Dict[str, Any]] = []
    running = 0
    done = 0
    for sp in status_files:
        payload = _read_json(sp) or {}
        exp_name = str(payload.get("experiment_name") or sp.name.replace("_status.json", ""))
        if not exp_name:
            continue
        exps.append(exp_name)
        st = str(payload.get("status") or "").lower()
        status_map[exp_name] = st
        if st == "running":
            running += 1
            prog = payload.get("progress") if isinstance(payload.get("progress"), dict) else {}
            resume = payload.get("resume") if isinstance(payload.get("resume"), dict) else {}
            initial = payload.get("initial") if isinstance(payload.get("initial"), dict) else {}
            running_experiments.append(
                {
                    "run_id": str(run_dir.name),
                    "exp": str(exp_name),
                    "updated_at": payload.get("updated_at"),
                    "resume_start_round": resume.get("start_round"),
                    "round": prog.get("round"),
                    "epoch": prog.get("epoch"),
                    "labeled_size": prog.get("labeled_size"),
                    "loss": prog.get("loss"),
                    "miou_live": prog.get("mIoU"),
                    "best_miou_round": prog.get("best_mIoU_round"),
                    "initial_labeled": initial.get("labeled"),
                    "initial_unlabeled": initial.get("unlabeled"),
                    "pools_dir": payload.get("pools_dir"),
                    "checkpoint_path": payload.get("checkpoint_path"),
                }
            )
        if st in ("completed", "finished"):
            done += 1
        prog = payload.get("progress") if isinstance(payload.get("progress"), dict) else {}
        rr = prog.get("round")
        if isinstance(rr, int):
            progress_round[exp_name] = rr

    exps = sorted(set(exps))

    scored = []
    for exp in exps:
        v = _objective_from_status(run_dir, exp, screen_objective)
        scored.append((exp, float(v) if isinstance(v, (int, float)) else float("-inf")))
    scored.sort(key=lambda x: x[1], reverse=True)
    topk = [exp for exp, v in scored[: max(1, min(int(screen_topk), len(scored)))] if v != float("-inf")]
    topk_round = {exp: progress_round.get(exp) for exp in topk}
    topk_status = {exp: status_map.get(exp, "unknown") for exp in topk}

    screen_best_exp = topk[0] if topk else None
    screen_best = None if not topk else (scored[0][1] if scored[0][1] != float("-inf") else None)

    confirm_best_exp = None
    confirm_best = None
    for exp in exps:
        v = _objective_from_status(run_dir, exp, confirm_objective)
        if v is None:
            continue
        if confirm_best is None or float(v) > float(confirm_best):
            confirm_best = float(v)
            confirm_best_exp = exp

    f1_done = bool(exps) and all(progress_round.get(e, -1) >= int(screen_end_round) for e in exps)
    f2_done = bool(topk) and all(progress_round.get(e, -1) >= int(confirm_end_round) for e in topk)

    seed_done: Dict[int, bool] = {}
    seed_round: Dict[int, Optional[int]] = {}
    seed_score: Dict[int, Optional[float]] = {}
    f3_mean: Optional[float] = None
    f3_std: Optional[float] = None
    f3_min: Optional[float] = None
    f3_done = False
    if expected_seeds:
        seed_runs = _find_seed_runs(run_dir.parent, run_dir.name)
        if confirm_best_exp:
            for seed in expected_seeds:
                sdir = run_dir if int(seed) == int(expected_seeds[0]) else seed_runs.get(int(seed))
                if sdir is None:
                    seed_done[int(seed)] = False
                    seed_round[int(seed)] = None
                    seed_score[int(seed)] = None
                    continue

                sp = _read_json(sdir / f"{confirm_best_exp}_status.json") or {}
                prog = sp.get("progress") if isinstance(sp.get("progress"), dict) else {}
                rr = prog.get("round")
                rr_i = int(rr) if isinstance(rr, int) else None
                seed_round[int(seed)] = rr_i
                seed_done[int(seed)] = bool(rr_i is not None and rr_i >= int(confirm_end_round))
                if seed_done[int(seed)]:
                    seed_score[int(seed)] = _objective_from_status_or_md(
                        sdir, confirm_best_exp, confirm_objective
                    )
                else:
                    seed_score[int(seed)] = None
            f3_done = all(seed_done.values()) if seed_done else False
            if f3_done:
                xs = [float(v) for v in seed_score.values() if isinstance(v, (int, float))]
                if len(xs) == len(seed_score) and xs:
                    f3_mean = _mean(xs)
                    f3_std = _std(xs)
                    f3_min = min(xs)

    return AutoTuningIterSummary(
        iteration=int(iteration),
        run_id=str(run_dir.name),
        updated_at=updated_at,
        n_exps=len(exps),
        n_running=int(running),
        n_done=int(done),
        f1_done=bool(f1_done),
        f2_done=bool(f2_done),
        f3_done=bool(f3_done),
        screen_best_exp=screen_best_exp,
        screen_best=screen_best,
        confirm_best_exp=confirm_best_exp,
        confirm_best=confirm_best,
        topk=list(topk),
        topk_round=topk_round,
        topk_status=topk_status,
        seed_done=seed_done,
        seed_round=seed_round,
        seed_score=seed_score,
        f3_mean=f3_mean,
        f3_std=f3_std,
        f3_min=f3_min,
        running_experiments=running_experiments,
    )


def _autotune_stage(summary: AutoTuningIterSummary) -> str:
    if not summary.f1_done:
        return "F1_screening"
    if not summary.f2_done:
        return "F2_confirming"
    if summary.seed_done and not summary.f3_done:
        return "F3_multi_seed"
    if summary.f3_done:
        return "iter_complete"
    return "F2_confirming"


def _format_bool(x: bool) -> str:
    return "yes" if bool(x) else "no"


def _write_autotune_md(
    *,
    out_path: Path,
    iters: List[AutoTuningIterSummary],
    running_experiments: List[Dict[str, Any]],
    screen_objective: str,
    confirm_objective: str,
    screen_end_round: int,
    confirm_end_round: int,
    screen_topk: int,
    expected_seeds: List[int],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cur = iters[-1] if iters else None
    stage = _autotune_stage(cur) if cur else "no_runs"
    active_node = {"F1_screening": "B", "F2_confirming": "D", "F3_multi_seed": "E", "iter_complete": "F"}.get(stage, "B")

    lines: List[str] = []
    lines.append("# Auto Tuning 运行监控（分阶段）\n\n")
    lines.append(f"- screen_objective: `{screen_objective}`（screen_end_round={screen_end_round}）\n")
    lines.append(f"- confirm_objective: `{confirm_objective}`（confirm_end_round={confirm_end_round}）\n")
    lines.append(f"- screen_topk: {screen_topk}\n")
    lines.append(f"- expected_seeds: {','.join(str(s) for s in expected_seeds)}\n\n")

    lines.append("```mermaid\n")
    lines.append("flowchart LR\n")
    lines.append("  A[Branch from incumbent] --> B[F1: Screen]\n")
    lines.append("  B --> C{Pick top-k}\n")
    lines.append("  C --> D[F2: Confirm]\n")
    lines.append("  D --> E[F3: Multi-seed]\n")
    lines.append("  E --> F[Update incumbent]\n")
    lines.append("  classDef active fill:#fffae6,stroke:#d4a106,stroke-width:2px;\n")
    lines.append(f"  class {active_node} active;\n")
    lines.append("```\n\n")

    lines.append("## 迭代总览\n")
    lines.append("| iter | run_id | updated_at | exps | running | F1 | F2 | F3 | top-k | best@screen | best@confirm |\n")
    lines.append("|---:|---|---|---:|---:|---|---|---|---|---|---|\n")
    for s in iters:
        best_screen = "-" if s.screen_best is None else f"{s.screen_best_exp} ({s.screen_best:.4f})"
        best_confirm = "-" if s.confirm_best is None else f"{s.confirm_best_exp} ({s.confirm_best:.4f})"
        lines.append(
            f"| {s.iteration} | {s.run_id} | {s.updated_at} | {s.n_exps} | {s.n_running} | {_format_bool(s.f1_done)} | {_format_bool(s.f2_done)} | {_format_bool(s.f3_done)} | {', '.join(s.topk) if s.topk else '-'} | {best_screen} | {best_confirm} |\n"
        )

    lines.append("\n## 当前进展（最新一轮）\n")
    if cur:
        lines.append(f"- 当前 run_id: `{cur.run_id}`\n")
        lines.append(f"- 当前阶段: `{stage}`\n")
        lines.append(f"- F1 完成: {_format_bool(cur.f1_done)}（{cur.n_done}/{cur.n_exps} 已完成/标记）\n")
        lines.append(f"- F2 完成: {_format_bool(cur.f2_done)}（top-k={', '.join(cur.topk) if cur.topk else '-'}）\n")
        if cur.seed_done:
            lines.append(f"- F3 seeds: `{cur.seed_done}`\n")
        lines.append(f"- best@screen: {cur.screen_best_exp} {cur.screen_best}\n")
        lines.append(f"- best@confirm: {cur.confirm_best_exp} {cur.confirm_best}\n")
        if cur.topk:
            lines.append("\n### F2 候选进度（top-k）\n")
            lines.append("| exp | status | round | target_round |\n")
            lines.append("|---|---|---:|---:|\n")
            for exp in cur.topk:
                st = cur.topk_status.get(exp, "unknown")
                rr = cur.topk_round.get(exp)
                rr_s = "-" if rr is None else str(int(rr))
                lines.append(f"| {exp} | {st} | {rr_s} | {int(confirm_end_round)} |\n")
        if cur.seed_done:
            lines.append("\n### F3 复核进度（best@confirm across seeds）\n")
            lines.append("| seed | done | round | target_round |\n")
            lines.append("|---:|---|---:|---:|\n")
            for seed in expected_seeds:
                done_s = _format_bool(bool(cur.seed_done.get(int(seed), False)))
                rr = cur.seed_round.get(int(seed))
                rr_s = "-" if rr is None else str(int(rr))
                lines.append(f"| {int(seed)} | {done_s} | {rr_s} | {int(confirm_end_round)} |\n")

        lines.append("\n## 正在运行的实验（全局）\n")
        if running_experiments:
            lines.append("| run_id | exp | rd | ep | ep_mIoU | best_rd | best_mIoU | updated_at |\n")
            lines.append("|---|---|---:|---:|---:|---:|---:|---|\n")
            for r in sorted(running_experiments, key=lambda x: (str(x.get("run_id")), str(x.get("exp")))):
                lines.append(
                    "| {run_id} | {exp} | {rd} | {ep} | {ep_miou} | {best_rd} | {best_miou} | {updated_at} |\n".format(
                        run_id=str(r.get("run_id") or "-"),
                        exp=str(r.get("exp") or "-"),
                        rd=str(r.get("rd") if r.get("rd") is not None else "-"),
                        ep=str(r.get("ep") if r.get("ep") is not None else "-"),
                        ep_miou=(
                            "{:.4f}".format(float(r.get("ep_miou")))
                            if isinstance(r.get("ep_miou"), (int, float))
                            else "-"
                        ),
                        best_rd=str(r.get("best_rd") if r.get("best_rd") is not None else "-"),
                        best_miou=(
                            "{:.4f}".format(float(r.get("best_miou")))
                            if isinstance(r.get("best_miou"), (int, float))
                            else "-"
                        ),
                        updated_at=str(r.get("updated_at") or "-"),
                    )
                )

            lines.append("\n### 运行中实验详情\n")
            for r in sorted(running_experiments, key=lambda x: (str(x.get("run_id")), str(x.get("exp")))):
                lines.append(f"- run_id: `{r.get('run_id')}` exp: `{r.get('exp')}`\n")
                lines.append(f"  - resume_start_round: `{r.get('resume_start_round') if r.get('resume_start_round') is not None else '-'}`\n")
                lines.append(f"  - labeled_size: `{r.get('labeled_size') if r.get('labeled_size') is not None else '-'}`\n")
                lines.append(f"  - pools_dir: `{r.get('pools_dir') or '-'}`\n")
                lines.append(f"  - checkpoint_path: `{r.get('checkpoint_path') or '-'}`\n")
        else:
            lines.append("- 无 running 状态的实验\n")
    else:
        lines.append("- 未发现 autotune_opt_iter* 目录\n")

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text("".join(lines), encoding="utf-8")
    os.replace(tmp, out_path)


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
    print("=" * 115)
    print(f"{'Run':<40} | {'Done':<4} | {'Run':<4} | {'Fail':<4} | {'Best':<6} | {'BestExp(Rd)'}")
    print("-" * 115)
    for r in runs[:show_limit]:
        done = int(r.counts.get("completed", 0))
        running = int(r.counts.get("running", 0))
        failed = int(r.counts.get("failed", 0))
        best = "-" if r.best_miou is None else f"{r.best_miou:.4f}"
        if r.best_exp is None:
            be = "-"
        else:
            rd_str = f"(R{r.best_rd})" if r.best_rd is not None else ""
            be = f"{r.best_exp}{rd_str}"
        print(f"{r.run_id:<40} | {done:<4} | {running:<4} | {failed:<4} | {best:<6} | {be}")
    print()


def _print_tuning_curve(runs: List[RunSummary]) -> None:
    ordered = sorted(runs, key=lambda r: _iter_order_key(r.run_id))
    best_so_far_by_type: Dict[str, float] = {}
    print("Tuning Curve (best-per-iter vs best-so-far)")
    print("-" * 115)
    print(f"{'IterRun':<40} | {'Best':<8} | {'BestSoFar':<9} | {'Improved'}")
    for r in ordered:
        if "_seed" in r.run_id:
            seed_type = r.run_id.split("_seed")[-1]
            run_type = f"seed{seed_type}"
        else:
            run_type = "base"

        best_so_far = best_so_far_by_type.get(run_type)
        b = r.best_miou
        if b is None:
            print(f"{r.run_id:<40} | {'-':<8} | {('-' if best_so_far is None else f'{best_so_far:.4f}'):<9} | -")
            continue
        if best_so_far is None or b > best_so_far + 1e-12:
            best_so_far_by_type[run_type] = b
            best_so_far = b
            improved = "Y"
        else:
            improved = "N"
        print(f"{r.run_id:<40} | {b:.4f}   | {best_so_far:.4f}     | {improved}")
    print()


def _print_autotune_f3_curve(
    iters: List[AutoTuningIterSummary],
    *,
    confirm_objective: str,
    confirm_end_round: int,
) -> None:
    ordered = sorted(iters, key=lambda x: (int(x.iteration), str(x.run_id)))
    best_so_far: Optional[float] = None
    print(f"Tuning Curve (F3 @R{int(confirm_end_round)}, objective={confirm_objective})")
    print("-" * 115)
    print(f"{'IterRun':<40} | {'F3Mean':<8} | {'BestSoFar':<9} | {'Improved'} | {'BestExp'}")
    for s in ordered:
        b = s.f3_mean
        if b is None:
            bs = "-" if best_so_far is None else f"{best_so_far:.4f}"
            exp = str(s.confirm_best_exp or "-")
            print(f"{s.run_id:<40} | {'-':<8} | {bs:<9} | -        | {exp}")
            continue
        if best_so_far is None or float(b) > float(best_so_far) + 1e-12:
            best_so_far = float(b)
            improved = "Y"
        else:
            improved = "N"
        exp = str(s.confirm_best_exp or "-")
        print(f"{s.run_id:<40} | {b:.4f}   | {best_so_far:.4f}     | {improved:<8} | {exp}")
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
    parser.add_argument("--autotune-report", action="store_true", default=False)
    parser.add_argument("--autotune-screen-objective", type=str, default="alc")
    parser.add_argument("--autotune-confirm-objective", type=str, default="val")
    parser.add_argument("--autotune-screen-end-round", type=int, default=10)
    parser.add_argument("--autotune-confirm-end-round", type=int, default=16)
    parser.add_argument("--autotune-screen-topk", type=int, default=2)
    parser.add_argument("--autotune-seeds", type=str, default="42,43,44")
    parser.add_argument("--autotune-write-md", type=str, default="results/runs/autotune_progress_report.md")
    parser.add_argument("--once", action="store_true", default=False)
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

        seeds: List[int] = []
        iters: List[AutoTuningIterSummary] = []
        if bool(args.autotune_report):
            seeds = [int(s) for s in str(args.autotune_seeds).split(",") if s.strip()]
            base_runs = _list_autotune_base_runs(runs_dir, max_runs=int(args.max_runs))
            iters = [
                _autotune_iter_summary(
                    d,
                    iteration=it,
                    screen_objective=str(args.autotune_screen_objective),
                    confirm_objective=str(args.autotune_confirm_objective),
                    screen_end_round=int(args.autotune_screen_end_round),
                    confirm_end_round=int(args.autotune_confirm_end_round),
                    screen_topk=int(args.autotune_screen_topk),
                    expected_seeds=seeds,
                )
                for it, d in base_runs
            ]

        os.system("clear")
        _print_overview(summaries, objective=str(args.objective), show_limit=int(args.show))
        if bool(args.autotune_report):
            _print_autotune_f3_curve(
                iters,
                confirm_objective=str(args.autotune_confirm_objective),
                confirm_end_round=int(args.autotune_confirm_end_round),
            )
        else:
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

        if bool(args.autotune_report):
            running_exps: List[Dict[str, Any]] = []
            for d in run_dirs:
                for sp in d.glob("*_status.json"):
                    payload = _read_json(sp) or {}
                    if str(payload.get("status") or "").lower() != "running":
                        continue
                    exp_name = str(payload.get("experiment_name") or sp.name.replace("_status.json", ""))
                    prog = payload.get("progress") if isinstance(payload.get("progress"), dict) else {}
                    resume = payload.get("resume") if isinstance(payload.get("resume"), dict) else {}
                    initial = payload.get("initial") if isinstance(payload.get("initial"), dict) else {}
                    trace_path = d / f"{exp_name}_trace.jsonl"
                    md_path = d / f"{exp_name}.md"
                    last_evt = _latest_epoch_end_from_trace(trace_path)
                    best_rd, best_miou = _best_round_miou_from_md(md_path)
                    rd = None
                    ep = None
                    ep_miou = None
                    if isinstance(last_evt, dict):
                        rr = last_evt.get("round")
                        ee = last_evt.get("epoch")
                        mm = last_evt.get("mIoU")
                        if isinstance(rr, int):
                            rd = int(rr)
                        if isinstance(ee, int):
                            ep = int(ee)
                        if isinstance(mm, (int, float)):
                            ep_miou = float(mm)
                    running_exps.append(
                        {
                            "run_id": str(d.name),
                            "exp": str(exp_name),
                            "updated_at": payload.get("updated_at"),
                            "resume_start_round": resume.get("start_round"),
                            "labeled_size": prog.get("labeled_size"),
                            "rd": rd,
                            "ep": ep,
                            "ep_miou": ep_miou,
                            "best_rd": best_rd,
                            "best_miou": best_miou,
                            "initial_labeled": initial.get("labeled"),
                            "initial_unlabeled": initial.get("unlabeled"),
                            "pools_dir": payload.get("pools_dir"),
                            "checkpoint_path": payload.get("checkpoint_path"),
                            "status_path": str(sp),
                        }
                    )
            out_md = Path(args.autotune_write_md)
            if not out_md.is_absolute():
                out_md = (repo_root / out_md).resolve()
            _write_autotune_md(
                out_path=out_md,
                iters=iters,
                running_experiments=running_exps,
                screen_objective=str(args.autotune_screen_objective),
                confirm_objective=str(args.autotune_confirm_objective),
                screen_end_round=int(args.autotune_screen_end_round),
                confirm_end_round=int(args.autotune_confirm_end_round),
                screen_topk=int(args.autotune_screen_topk),
                expected_seeds=seeds,
            )
            if iters:
                cur = iters[-1]
                print("autotune_progress:", cur.run_id, "stage=", _autotune_stage(cur), "md=", str(out_md))

        if bool(args.once):
            break
        time.sleep(int(args.interval))


if __name__ == "__main__":
    main()
