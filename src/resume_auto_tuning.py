from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timedelta
import re
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__)))

from tuning_opt.evaluator import parse_objective_from_status  # noqa: E402
from tuning_opt.pool_resume import PoolResumeManager  # noqa: E402
from utils import atomic_write_json, read_json_dict  # noqa: E402


def _parse_seed_run_id(run_id: str) -> Optional[Tuple[str, int]]:
    s = str(run_id or "").strip()
    if "_seed" not in s:
        return None
    base, tail = s.rsplit("_seed", 1)
    if not base:
        return None
    if not tail.isdigit():
        return None
    return base, int(tail)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    return read_json_dict(path)


def _safe_int(x: Any) -> Optional[int]:
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float) and int(x) == x:
        return int(x)
    return None


def _progress_round_from_status(path: Path) -> Optional[int]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return None
    prog = payload.get("progress")
    if not isinstance(prog, dict):
        return None
    return _safe_int(prog.get("round"))


def _status_str_from_status(path: Path) -> str:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return "unknown"
    return str(payload.get("status") or "unknown").strip().lower()


_STALE_RUNNING_MINUTES = 10


def _ps_command_rows() -> List[Tuple[int, str]]:
    try:
        r = subprocess.run(
            ["ps", "-axo", "pid,command"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return []
    if r.returncode != 0:
        return []
    lines = [ln.rstrip("\n") for ln in str(r.stdout).splitlines() if ln.strip()]
    rows: List[Tuple[int, str]] = []
    for ln in lines[1:]:
        parts = ln.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid_s, cmd = parts
        if not pid_s.isdigit():
            continue
        rows.append((int(pid_s), str(cmd)))
    return rows


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _has_live_owner_process(*, run_id: str) -> bool:
    rid = str(run_id or "").strip()
    if not rid:
        return False
    for _, cmd in _ps_command_rows():
        s = str(cmd)
        if "run_parallel_strict.py" in s and rid in s:
            return True
        if "run_multi_seed.py" in s and rid in s:
            return True
    return False


def _is_stale_running(
    status_path: Path, *, stale_minutes: int = _STALE_RUNNING_MINUTES
) -> bool:
    """Detect experiments whose status says 'running' but have no live process.

    Heuristic: if updated_at is older than *stale_minutes* and no matching
    ``run_parallel_strict`` / ``spawn_main`` process owns the experiment,
    the status is stale (process crashed without cleanup).
    """
    payload = _read_json(status_path)
    if not isinstance(payload, dict):
        return False
    if str(payload.get("status") or "").strip().lower() != "running":
        return False

    run_id = str(payload.get("run_id") or status_path.parent.name or "").strip()
    if _has_live_owner_process(run_id=run_id):
        return False

    pid = payload.get("pid")
    if isinstance(pid, int) and pid > 0 and _pid_alive(int(pid)):
        return False

    updated = payload.get("updated_at")
    if isinstance(updated, str) and updated.strip():
        try:
            ts = datetime.fromisoformat(updated.strip())
            if datetime.now() - ts > timedelta(minutes=max(1, stale_minutes)):
                return True
        except Exception:
            pass

    return False


def _mark_status_stale(status_path: Path, *, reason: str) -> None:
    payload = _read_json(status_path) or {}
    if not isinstance(payload, dict):
        payload = {}
    if str(payload.get("status") or "").strip().lower() != "running":
        return
    next_payload = dict(payload)
    next_payload["status"] = "stalled"
    next_payload["current_error"] = {
        "type": "stale_running",
        "message": str(reason or "stale_running"),
        "ts": datetime.now().isoformat(),
    }
    atomic_write_json(status_path, next_payload, indent=2)


def _list_exps_in_run(run_dir: Path) -> List[str]:
    exps = []
    for sp in sorted(run_dir.glob("*_status.json")):
        payload = _read_json(sp)
        if not isinstance(payload, dict):
            continue
        exp = str(payload.get("experiment_name") or "").strip()
        if exp:
            exps.append(exp)
    return sorted(set(exps))


def _topk_by_objective(
    *, run_dir: Path, exps: Sequence[str], objective: str, k: int
) -> List[str]:
    scored: List[Tuple[str, float]] = []
    for exp in exps:
        v = parse_objective_from_status(run_dir / f"{exp}_status.json", objective)
        scored.append((exp, float(v) if isinstance(v, (int, float)) else float("-inf")))
    scored.sort(key=lambda x: x[1], reverse=True)
    out = [
        e for e, v in scored[: max(1, min(int(k), len(scored)))] if v != float("-inf")
    ]
    return out


def _best_by_objective(
    *, run_dir: Path, exps: Sequence[str], objective: str
) -> Optional[str]:
    best_exp = None
    best_val = None
    for exp in exps:
        v = parse_objective_from_status(run_dir / f"{exp}_status.json", objective)
        if not isinstance(v, (int, float)):
            continue
        if best_val is None or float(v) > float(best_val):
            best_val = float(v)
            best_exp = exp
    return best_exp


def _ensure_sidecar_has_exps_from_traces(
    *, repo_root: Path, run_dir: Path, exps: Sequence[str]
) -> None:
    sidecar = repo_root / "src" / "experiments" / "auto_tune_configs.json"
    try:
        existing = (
            json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
        )
        if not isinstance(existing, dict):
            existing = {}
    except Exception:
        existing = {}

    changed = False
    for exp in exps:
        trace_path = run_dir / f"{exp}_trace.jsonl"
        if not trace_path.exists():
            continue
        try:
            with trace_path.open("r", encoding="utf-8", errors="ignore") as f:
                first = f.readline()
            evt = json.loads(first)
        except Exception:
            continue
        ab = evt.get("ablation") if isinstance(evt, dict) else None
        if not isinstance(ab, dict):
            continue
        existing_cfg = (
            existing.get(exp) if isinstance(existing.get(exp), dict) else None
        )
        if existing_cfg is not None:
            if "description" not in existing_cfg:
                existing_cfg = dict(existing_cfg)
                existing_cfg["description"] = f"AutoTuneOpt: {exp}"
                existing[exp] = existing_cfg
                changed = True
            continue
        if "description" not in ab:
            ab = dict(ab)
            ab["description"] = f"AutoTuneOpt: {exp}"
        existing[exp] = ab
        changed = True

    if not changed:
        return
    tmp = sidecar.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, sidecar)


def _run_parallel_strict(
    *,
    repo_root: Path,
    run_id: str,
    seed: int,
    include: Sequence[str],
    n_rounds: int,
    agent_workers: int,
    dynamic_agent_workers: bool,
    agent_workers_min: int,
    reserve_free_gb: float,
    mem_per_agent_worker_gb: float,
    dynamic_poll_seconds: float,
) -> None:
    cmd = _build_parallel_strict_cmd(
        run_id=run_id,
        seed=seed,
        include=include,
        n_rounds=n_rounds,
        agent_workers=agent_workers,
        dynamic_agent_workers=dynamic_agent_workers,
        agent_workers_min=agent_workers_min,
        reserve_free_gb=reserve_free_gb,
        mem_per_agent_worker_gb=mem_per_agent_worker_gb,
        dynamic_poll_seconds=dynamic_poll_seconds,
    )
    r = subprocess.run(cmd, cwd=str(repo_root))
    if r.returncode != 0:
        import sys

        print(
            f"[resume] run_parallel_strict returned non-zero exit code {r.returncode} "
            f"for run_id={run_id} include={list(include)}. "
            f"Partial results may exist in results/runs/{run_id}.",
            file=sys.stderr,
        )


def _build_parallel_strict_cmd(
    *,
    run_id: str,
    seed: int,
    include: Sequence[str],
    n_rounds: int,
    agent_workers: int,
    dynamic_agent_workers: bool,
    agent_workers_min: int,
    reserve_free_gb: float,
    mem_per_agent_worker_gb: float,
    dynamic_poll_seconds: float,
) -> List[str]:
    cmd = [
        "python",
        "src/run_parallel_strict.py",
        "--resume",
        str(run_id),
        "--seed",
        str(int(seed)),
        "--include",
        ",".join(include),
        "--n-rounds",
        str(int(n_rounds)),
        "--agent-workers",
        str(int(agent_workers)),
        "--non-agent-workers",
        "0",
        "--owner-pid",
        str(int(os.getpid())),
    ]
    if dynamic_agent_workers:
        cmd.extend(
            [
                "--dynamic-agent-workers",
                "--agent-workers-min",
                str(int(agent_workers_min)),
                "--reserve-free-gb",
                str(float(reserve_free_gb)),
                "--mem-per-agent-worker-gb",
                str(float(mem_per_agent_worker_gb)),
                "--dynamic-poll-seconds",
                str(float(dynamic_poll_seconds)),
            ]
        )
    return cmd


def _vm_stat_pages() -> Dict[str, int]:
    try:
        out = subprocess.check_output(["vm_stat"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return {}
    pages: Dict[str, int] = {}
    m = re.search(
        r"page size of\s+(\d+)\s+bytes", out.splitlines()[0] if out else "", re.I
    )
    if m and m.group(1).isdigit():
        pages["_page_size_bytes"] = int(m.group(1))
    for line in out.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip().lower()
        v = v.strip().strip(".").replace(",", "")
        if not v.isdigit():
            continue
        pages[k] = int(v)
    return pages


def _available_mem_gb() -> float:
    pages = _vm_stat_pages()
    if not pages:
        return 0.0
    page_size = int(pages.get("_page_size_bytes") or 4096)
    free_pages = (
        pages.get("pages free", 0)
        + pages.get("pages speculative", 0)
        + pages.get("pages inactive", 0)
    )
    return (free_pages * page_size) / (1024**3)


def _dynamic_cap(
    *,
    hard_max: int,
    reserve_free_gb: float,
    mem_per_run_gb: float,
) -> int:
    hard_max = max(1, int(hard_max))
    avail = _available_mem_gb()
    headroom = float(avail) - float(reserve_free_gb)
    if headroom <= 0:
        return 1
    if mem_per_run_gb <= 0:
        return hard_max
    cap = int(headroom // float(mem_per_run_gb))
    return max(1, min(hard_max, cap))


def _run_jobs(
    *,
    repo_root: Path,
    jobs: Sequence[Tuple[str, List[str]]],
    max_parallel_runs: int,
    dynamic_parallel_runs: bool,
    reserve_free_gb: float,
    mem_per_run_gb: float,
    poll_seconds: float,
    stall_minutes: int = 60,
) -> None:
    pending: Deque[Tuple[str, List[str]]] = deque(jobs)
    running: List[Tuple[str, subprocess.Popen]] = []
    failures: List[str] = []
    last_progress_at: Dict[str, float] = {}
    stalled_since: Dict[str, float] = {}

    def _parse_label(label: str) -> Tuple[str, str]:
        s = str(label or "")
        if ":" in s:
            a, b = s.split(":", 1)
            return str(a).strip(), str(b).strip()
        return s.strip(), ""

    def _updated_at_ts(path: Path) -> Optional[float]:
        payload = _read_json(path)
        if not isinstance(payload, dict):
            return None
        updated = payload.get("updated_at")
        if isinstance(updated, str) and updated.strip():
            try:
                dt = datetime.fromisoformat(updated.strip())
                return dt.timestamp()
            except Exception:
                return None
        return None

    def _latest_activity_ts(run_id: str, exp: str) -> Optional[float]:
        rid = str(run_id or "").strip()
        if not rid:
            return None
        run_dir = repo_root / "results" / "runs" / rid
        if not run_dir.exists():
            return None
        best = None
        if exp:
            sp = run_dir / f"{exp}_status.json"
            tp = run_dir / f"{exp}_trace.jsonl"
            ts = _updated_at_ts(sp) if sp.exists() else None
            if ts is None and tp.exists():
                try:
                    ts = tp.stat().st_mtime
                except Exception:
                    ts = None
            best = ts
        else:
            for sp in run_dir.glob("*_status.json"):
                ts = _updated_at_ts(sp)
                if ts is None:
                    continue
                best = ts if best is None else max(best, ts)
        return best

    while pending or running:
        cap = (
            _dynamic_cap(
                hard_max=int(max_parallel_runs),
                reserve_free_gb=float(reserve_free_gb),
                mem_per_run_gb=float(mem_per_run_gb),
            )
            if bool(dynamic_parallel_runs)
            else max(1, int(max_parallel_runs))
        )

        while pending and len(running) < cap:
            label, cmd = pending.popleft()
            p = subprocess.Popen(cmd, cwd=str(repo_root), start_new_session=True)
            running.append((label, p))
            last_progress_at[str(label)] = time.time()

        still_running: List[Tuple[str, subprocess.Popen]] = []
        for label, p in running:
            rc = p.poll()
            if rc is None:
                now = time.time()
                if int(stall_minutes) > 0:
                    rid, exp = _parse_label(str(label))
                    latest = _latest_activity_ts(rid, exp)
                    if latest is not None:
                        prev = last_progress_at.get(str(label))
                        if prev is None or float(latest) > float(prev):
                            last_progress_at[str(label)] = float(latest)
                            stalled_since.pop(str(label), None)
                        else:
                            if str(label) not in stalled_since:
                                stalled_since[str(label)] = now
                    since = stalled_since.get(str(label))
                    if since is not None and (now - float(since)) >= float(
                        max(1, int(stall_minutes)) * 60
                    ):
                        try:
                            os.killpg(int(p.pid), signal.SIGTERM)
                        except Exception:
                            pass
                        deadline = time.time() + 2.0
                        while time.time() < deadline:
                            if p.poll() is not None:
                                break
                            time.sleep(0.05)
                        if p.poll() is None:
                            try:
                                os.killpg(int(p.pid), signal.SIGKILL)
                            except Exception:
                                pass
                        failures.append(f"{label} (stalled)")
                        continue
                still_running.append((label, p))
                continue
            if int(rc) != 0:
                failures.append(f"{label} (exit={int(rc)})")
        running = still_running

        if pending or running:
            time.sleep(max(0.2, float(poll_seconds)))

    if failures:
        raise SystemExit("Some resume jobs failed: " + ", ".join(failures))


def _start_monitor_detached(
    *,
    repo_root: Path,
    interval: int,
    screen_objective: str,
    confirm_objective: str,
    screen_end_round: int,
    confirm_end_round: int,
    screen_topk: int,
    seeds: Sequence[int],
    out_md: str,
) -> None:
    try:
        out_md_str = str(out_md)
        out_md_abs = (
            str((repo_root / out_md_str).resolve())
            if not os.path.isabs(out_md_str)
            else out_md_str
        )
        ps = subprocess.check_output(["ps", "-ax", "-o", "command="], text=True)
        for line in ps.splitlines():
            if "monitor_auto_tuning.py" not in line:
                continue
            if "--autotune-write-md" not in line:
                continue
            if out_md_abs in line or out_md_str in line:
                return
    except Exception:
        pass
    cmd = [
        sys.executable,
        "src/monitor_auto_tuning.py",
        "--interval",
        str(int(interval)),
        "--autotune-report",
        "--autotune-screen-objective",
        str(screen_objective),
        "--autotune-confirm-objective",
        str(confirm_objective),
        "--autotune-screen-end-round",
        str(int(screen_end_round)),
        "--autotune-confirm-end-round",
        str(int(confirm_end_round)),
        "--autotune-screen-topk",
        str(int(screen_topk)),
        "--autotune-seeds",
        ",".join(str(int(s)) for s in seeds),
        "--autotune-write-md",
        str(out_md),
    ]
    subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--baseline-run-id", required=True)
    ap.add_argument("--baseline-exp", required=True)
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--branch-round", type=int, default=7)
    ap.add_argument("--screen-objective", type=str, default="alc")
    ap.add_argument("--confirm-objective", type=str, default="val")
    ap.add_argument("--screen-end-round", type=int, default=10)
    ap.add_argument("--confirm-end-round", type=int, default=16)
    ap.add_argument("--screen-topk", type=int, default=2)
    ap.add_argument("--agent-workers", type=int, default=3)
    ap.add_argument("--dynamic-agent-workers", action="store_true", default=False)
    ap.add_argument("--agent-workers-min", type=int, default=1)
    ap.add_argument("--reserve-free-gb", type=float, default=4.0)
    ap.add_argument("--mem-per-agent-worker-gb", type=float, default=2.0)
    ap.add_argument("--dynamic-poll-seconds", type=float, default=2.0)
    ap.add_argument("--max-parallel-runs", type=int, default=3)
    ap.add_argument("--dynamic-parallel-runs", action="store_true", default=False)
    ap.add_argument("--mem-per-run-gb", type=float, default=3.0)
    ap.add_argument("--run-poll-seconds", type=float, default=5.0)
    ap.add_argument("--stall-minutes", type=int, default=60)
    ap.add_argument("--continuous", action="store_true", default=False)
    ap.add_argument("--target-miou", type=float, default=0.725) 
    ap.add_argument("--orch-max-iterations", type=int, default=50)
    ap.add_argument("--orch-screen-epochs-per-round", type=int, default=8)
    ap.add_argument("--no-llm", action="store_true", default=False)
    ap.add_argument("--llm-config", type=str, default="")
    ap.add_argument("--program", type=str, default="src/tuning_program.json")
    ap.add_argument("--start-monitor", action="store_true", default=False)
    ap.add_argument("--monitor-interval", type=int, default=30)
    ap.add_argument(
        "--monitor-out-md", type=str, default="results/runs/autotune_progress_report.md"
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    runs_root = repo_root / "results" / "runs"
    requested_run_id = str(args.run_id)
    requested_run_dir = runs_root / requested_run_id
    if not requested_run_dir.exists():
        raise SystemExit(f"Run not found: {requested_run_dir}")

    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    base_seed = int(args.base_seed)
    if base_seed not in seeds:
        seeds = [base_seed] + [s for s in seeds if s != base_seed]

    if bool(args.start_monitor):
        _start_monitor_detached(
            repo_root=repo_root,
            interval=int(args.monitor_interval),
            screen_objective=str(args.screen_objective),
            confirm_objective=str(args.confirm_objective),
            screen_end_round=int(args.screen_end_round),
            confirm_end_round=int(args.confirm_end_round),
            screen_topk=int(args.screen_topk),
            seeds=seeds,
            out_md=str(args.monitor_out_md),
        )

    def _run_orchestrator(initial_run_id: str, initial_exp: str) -> None:
        cmd = [
            "python",
            "src/tuning_opt/orchestrator.py",
            "--initial-run-id",
            str(initial_run_id),
            "--initial-exp",
            str(initial_exp),
            "--target-miou",
            str(float(args.target_miou)),
            "--max-iterations",
            str(int(args.orch_max_iterations)),
            "--seeds",
            ",".join(str(int(s)) for s in seeds),
            "--max-concurrent",
            str(int(args.agent_workers)),
            "--objective",
            str(args.confirm_objective),
            "--screen-objective",
            str(args.screen_objective),
            "--confirm-objective",
            str(args.confirm_objective),
            "--branch-round",
            str(int(args.branch_round)),
            "--screen-end-round",
            str(int(args.screen_end_round)),
            "--confirm-end-round",
            str(int(args.confirm_end_round)),
            "--screen-epochs-per-round",
            str(int(args.orch_screen_epochs_per_round)),
            "--results-dir",
            "results",
        ]
        if bool(args.dynamic_agent_workers):
            cmd.extend(
                [
                    "--dynamic-agent-workers",
                    "--agent-workers-min",
                    str(int(args.agent_workers_min)),
                    "--reserve-free-gb",
                    str(float(args.reserve_free_gb)),
                    "--mem-per-agent-worker-gb",
                    str(float(args.mem_per_agent_worker_gb)),
                    "--dynamic-poll-seconds",
                    str(float(args.dynamic_poll_seconds)),
                ]
            )
        if bool(args.no_llm):
            cmd.append("--no-llm")
        if str(args.llm_config).strip():
            cmd.extend(["--llm-config", str(args.llm_config)])
        if str(args.program).strip():
            cmd.extend(["--program", str(args.program)])
        r = subprocess.run(cmd, cwd=str(repo_root))
        if r.returncode != 0:
            raise SystemExit(f"orchestrator failed with exit={int(r.returncode)}")

    seed_run = _parse_seed_run_id(requested_run_id)
    if seed_run is not None:
        base_run_id, seed_from_run_id = seed_run
        base_run_dir = runs_root / base_run_id
        if not base_run_dir.exists():
            raise SystemExit(f"Base run not found: {base_run_dir}")
        exps = _list_exps_in_run(base_run_dir)
        if not exps:
            raise SystemExit(f"No experiments detected in base run: {base_run_dir}")

        _ensure_sidecar_has_exps_from_traces(
            repo_root=repo_root, run_dir=base_run_dir, exps=exps
        )
        best_exp = _best_by_objective(
            run_dir=base_run_dir, exps=exps, objective=str(args.confirm_objective)
        )
        if not best_exp:
            topk = _topk_by_objective(
                run_dir=base_run_dir,
                exps=exps,
                objective=str(args.screen_objective),
                k=int(args.screen_topk),
            )
            if not topk:
                raise SystemExit("Unable to pick top-k from base run; missing results?")
            best_exp = topk[0]

        mgr = PoolResumeManager(results_dir="results")

        target_seeds = [int(seed_from_run_id)]
        for seed in seeds:
            if int(seed) == int(seed_from_run_id):
                continue
            target_seeds.append(int(seed))

        jobs: List[Tuple[str, List[str]]] = []
        for seed in target_seeds:
            seed_run_id = f"{base_run_id}_seed{int(seed)}"
            seed_run_dir = runs_root / seed_run_id

            # Resume ALL experiments found in the seed run directory
            current_exps = _list_exps_in_run(seed_run_dir)
            for exp in current_exps:
                seed_status_path = seed_run_dir / f"{exp}_status.json"
                if seed_status_path.exists():
                    st = _status_str_from_status(seed_status_path)
                    rr = _progress_round_from_status(seed_status_path)
                    if st == "running":
                        if not _is_stale_running(seed_status_path):
                            continue
                        try:
                            _mark_status_stale(
                                seed_status_path,
                                reason="status=running but no live owner process",
                            )
                        except Exception:
                            pass
                    if rr is not None and int(rr) >= int(args.confirm_end_round):
                        continue

                    # If we already have progress at or beyond branch_round, don't re-branch
                    if rr is not None and int(rr) >= int(args.branch_round):
                        jobs.append(
                            (
                                f"{seed_run_id}:{exp}",
                                _build_parallel_strict_cmd(
                                    run_id=seed_run_id,
                                    seed=int(seed),
                                    include=[exp],
                                    n_rounds=int(args.confirm_end_round),
                                    agent_workers=int(args.agent_workers),
                                    dynamic_agent_workers=bool(
                                        args.dynamic_agent_workers
                                    ),
                                    agent_workers_min=int(args.agent_workers_min),
                                    reserve_free_gb=float(args.reserve_free_gb),
                                    mem_per_agent_worker_gb=float(
                                        args.mem_per_agent_worker_gb
                                    ),
                                    dynamic_poll_seconds=float(
                                        args.dynamic_poll_seconds
                                    ),
                                ),
                            )
                        )
                        continue

            # Also ensure the current best_exp is started if it hasn't been yet
            seed_status_path = seed_run_dir / f"{best_exp}_status.json"
            if not seed_status_path.exists():
                if not seed_run_dir.exists():
                    seed_run_dir.mkdir(parents=True, exist_ok=True)
                mgr.branch_from_round(
                    source_run_id=str(args.baseline_run_id),
                    source_exp=str(args.baseline_exp),
                    target_run_id=seed_run_id,
                    target_exps=[best_exp],
                    branch_round=int(args.branch_round),
                )
                jobs.append(
                    (
                        f"{seed_run_id}:{best_exp}",
                        _build_parallel_strict_cmd(
                            run_id=seed_run_id,
                            seed=int(seed),
                            include=[best_exp],
                            n_rounds=int(args.confirm_end_round),
                            agent_workers=int(args.agent_workers),
                            dynamic_agent_workers=bool(args.dynamic_agent_workers),
                            agent_workers_min=int(args.agent_workers_min),
                            reserve_free_gb=float(args.reserve_free_gb),
                            mem_per_agent_worker_gb=float(args.mem_per_agent_worker_gb),
                            dynamic_poll_seconds=float(args.dynamic_poll_seconds),
                        ),
                    )
                )

        if jobs:
            _run_jobs(
                repo_root=repo_root,
                jobs=jobs,
                max_parallel_runs=int(args.max_parallel_runs),
                dynamic_parallel_runs=bool(args.dynamic_parallel_runs),
                reserve_free_gb=float(args.reserve_free_gb),
                mem_per_run_gb=float(args.mem_per_run_gb),
                poll_seconds=float(args.run_poll_seconds),
                stall_minutes=int(args.stall_minutes),
            )

        print(
            "resume_ok:",
            base_run_id,
            "best_exp:",
            best_exp,
            "monitor_md:",
            str(args.monitor_out_md),
        )
        if bool(args.continuous):
            print(
                f"[resume] --continuous: launching orchestrator "
                f"initial_run_id={base_run_id} initial_exp={best_exp} "
                f"target_miou={args.target_miou} max_iter={args.orch_max_iterations}",
                flush=True,
            )
            _run_orchestrator(initial_run_id=base_run_id, initial_exp=best_exp)
        return

    base_run_id = requested_run_id
    run_dir = requested_run_dir
    exps = _list_exps_in_run(run_dir)
    if not exps:
        raise SystemExit(f"No experiments detected in {run_dir}")

    _ensure_sidecar_has_exps_from_traces(
        repo_root=repo_root, run_dir=run_dir, exps=exps
    )

    topk = _topk_by_objective(
        run_dir=run_dir,
        exps=exps,
        objective=str(args.screen_objective),
        k=int(args.screen_topk),
    )
    if not topk:
        raise SystemExit("Unable to pick top-k from screen objective; missing results?")

    f2_targets = []
    for exp in topk:
        st = _status_str_from_status(run_dir / f"{exp}_status.json")
        rr = _progress_round_from_status(run_dir / f"{exp}_status.json")
        if st == "running" and not _is_stale_running(run_dir / f"{exp}_status.json"):
            continue
        if rr is None or int(rr) < int(args.confirm_end_round):
            f2_targets.append(exp)

    jobs: List[Tuple[str, List[str]]] = []
    if f2_targets:
        jobs.append(
            (
                f"{base_run_id}:F2",
                _build_parallel_strict_cmd(
                    run_id=base_run_id,
                    seed=base_seed,
                    include=topk,
                    n_rounds=int(args.confirm_end_round),
                    agent_workers=int(args.agent_workers),
                    dynamic_agent_workers=bool(args.dynamic_agent_workers),
                    agent_workers_min=int(args.agent_workers_min),
                    reserve_free_gb=float(args.reserve_free_gb),
                    mem_per_agent_worker_gb=float(args.mem_per_agent_worker_gb),
                    dynamic_poll_seconds=float(args.dynamic_poll_seconds),
                ),
            )
        )

    best_exp = _best_by_objective(
        run_dir=run_dir, exps=exps, objective=str(args.confirm_objective)
    )
    if not best_exp:
        best_exp = topk[0]

    _ensure_sidecar_has_exps_from_traces(
        repo_root=repo_root, run_dir=run_dir, exps=[best_exp]
    )

    mgr = PoolResumeManager(results_dir="results")

    for seed in seeds:
        if int(seed) == int(base_seed):
            continue
        seed_run_id = f"{base_run_id}_seed{int(seed)}"
        seed_run_dir = runs_root / seed_run_id

        # Resume ALL experiments found in the seed run directory
        current_exps = _list_exps_in_run(seed_run_dir)
        for exp in current_exps:
            seed_status_path = seed_run_dir / f"{exp}_status.json"
            if seed_status_path.exists():
                st = _status_str_from_status(seed_status_path)
                rr = _progress_round_from_status(seed_status_path)
                if st == "running":
                    if not _is_stale_running(seed_status_path):
                        continue
                    try:
                        _mark_status_stale(
                            seed_status_path,
                            reason="status=running but no live owner process",
                        )
                    except Exception:
                        pass
                if rr is not None and int(rr) >= int(args.confirm_end_round):
                    continue

                # If we already have progress at or beyond branch_round, don't re-branch
                if rr is not None and int(rr) >= int(args.branch_round):
                    jobs.append(
                        (
                            f"{seed_run_id}:{exp}",
                            _build_parallel_strict_cmd(
                                run_id=seed_run_id,
                                seed=int(seed),
                                include=[exp],
                                n_rounds=int(args.confirm_end_round),
                                agent_workers=int(args.agent_workers),
                                dynamic_agent_workers=bool(args.dynamic_agent_workers),
                                agent_workers_min=int(args.agent_workers_min),
                                reserve_free_gb=float(args.reserve_free_gb),
                                mem_per_agent_worker_gb=float(
                                    args.mem_per_agent_worker_gb
                                ),
                                dynamic_poll_seconds=float(args.dynamic_poll_seconds),
                            ),
                        )
                    )
                    continue

        # Also ensure the current best_exp is started if it hasn't been yet
        seed_status_path = seed_run_dir / f"{best_exp}_status.json"
        if not seed_status_path.exists():
            if not seed_run_dir.exists():
                seed_run_dir.mkdir(parents=True, exist_ok=True)
            mgr.branch_from_round(
                source_run_id=str(args.baseline_run_id),
                source_exp=str(args.baseline_exp),
                target_run_id=seed_run_id,
                target_exps=[best_exp],
                branch_round=int(args.branch_round),
            )
            jobs.append(
                (
                    f"{seed_run_id}:{best_exp}",
                    _build_parallel_strict_cmd(
                        run_id=seed_run_id,
                        seed=int(seed),
                        include=[best_exp],
                        n_rounds=int(args.confirm_end_round),
                        agent_workers=int(args.agent_workers),
                        dynamic_agent_workers=bool(args.dynamic_agent_workers),
                        agent_workers_min=int(args.agent_workers_min),
                        reserve_free_gb=float(args.reserve_free_gb),
                        mem_per_agent_worker_gb=float(args.mem_per_agent_worker_gb),
                        dynamic_poll_seconds=float(args.dynamic_poll_seconds),
                    ),
                )
            )

    if jobs:
        _run_jobs(
            repo_root=repo_root,
            jobs=jobs,
            max_parallel_runs=int(args.max_parallel_runs),
            dynamic_parallel_runs=bool(args.dynamic_parallel_runs),
            reserve_free_gb=float(args.reserve_free_gb),
            mem_per_run_gb=float(args.mem_per_run_gb),
            poll_seconds=float(args.run_poll_seconds),
            stall_minutes=int(args.stall_minutes),
        )

    print(
        "resume_ok:",
        base_run_id,
        "best_exp:",
        best_exp,
        "monitor_md:",
        str(args.monitor_out_md),
    )
    if bool(args.continuous):
        print(
            f"[resume] --continuous: launching orchestrator "
            f"initial_run_id={base_run_id} initial_exp={best_exp} "
            f"target_miou={args.target_miou} max_iter={args.orch_max_iterations}",
            flush=True,
        )
        _run_orchestrator(initial_run_id=base_run_id, initial_exp=best_exp)


if __name__ == "__main__":
    main()
