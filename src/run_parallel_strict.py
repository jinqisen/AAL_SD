import os
import sys
import argparse
import multiprocessing
import json
import hashlib
import re
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
import time
import subprocess
from collections import deque
import signal

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from main import ActiveLearningPipeline
from experiments.ablation_config import ABLATION_SETTINGS, EXPERIMENT_NAME_ALIASES
from utils.logger import logger
from utils import locked_update_json, read_json_dict

PRESETS = {
    "paper_grad_evidence": {
        "include": "full_model_A_lambda_policy,no_agent,baseline_entropy,baseline_random,fixed_lambda,uncertainty_only,knowledge_only",
    },
    "fast_grad_smoke": {
        "include": "full_model_A_lambda_policy,no_agent,baseline_entropy,baseline_random",
        "execution": "sequential",
        "n_rounds": 2,
        "epochs_per_round": 3,
    },
}


def _load_json(path):
    return read_json_dict(path)


def _status_indicates_finished(status_payload, config):
    if not isinstance(status_payload, dict):
        return False
    progress = status_payload.get("progress")
    if isinstance(progress, dict):
        round_val = progress.get("round")
        if isinstance(round_val, (int, float)) and round_val >= getattr(
            config, "N_ROUNDS", 0
        ):
            return True
        if isinstance(round_val, str) and round_val.strip().lower() == "finished":
            return True
    return False


def _checkpoint_indicates_finished(state_payload, config):
    if not isinstance(state_payload, dict):
        return False
    round_val = state_payload.get("round")
    if isinstance(round_val, (int, float)) and round_val >= getattr(
        config, "N_ROUNDS", 0
    ):
        return True
    return False


def _hash_file(path):
    if not os.path.exists(path):
        return None
    try:
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None


def _collect_config_snapshot(config):
    snapshot = {}
    for name in dir(config):
        if not name.isupper() or name.startswith("_"):
            continue
        if any(key in name for key in ("KEY", "TOKEN", "SECRET")):
            continue
        try:
            value = getattr(config, name)
        except Exception:
            continue
        if callable(value):
            continue
        snapshot[name] = value
    return snapshot


def _write_manifest(run_id, config, ablation_settings):
    results_dir = Path(Config.RESULTS_DIR) / "runs" / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = results_dir / "manifest.json"
    lock_path = results_dir / "manifest.json.lock"
    src_dir = Path(__file__).resolve().parent
    def _build(existing: dict) -> dict:
        created_at = existing.get("created_at") or datetime.now().isoformat()
        existing_experiments = existing.get("experiments")
        if not isinstance(existing_experiments, dict):
            existing_experiments = {}
        merged_experiments = dict(existing_experiments)
        if isinstance(ablation_settings, dict):
            merged_experiments.update(ablation_settings)
        return {
            "run_id": run_id,
            "created_at": created_at,
            "updated_at": datetime.now().isoformat(),
            "config": _collect_config_snapshot(config),
            "experiments": merged_experiments,
            "experiment_runtime": {
                "mode": "legacy_adapter",
                "trace_schema_version": 2,
            },
            "code_fingerprint": {
                "run_parallel_strict": _hash_file(__file__),
                "config": _hash_file(str(src_dir / "config.py")),
                "main": _hash_file(str(src_dir / "main.py")),
                "trainer": _hash_file(str(src_dir / "core" / "trainer.py")),
                "ablation_config": _hash_file(
                    str(src_dir / "experiments" / "ablation_config.py")
                ),
                "prompt_template": _hash_file(
                    str(src_dir / "agent" / "prompt_template.py")
                ),
                "plot_paper_figures": _hash_file(
                    str(src_dir / "analysis" / "plot_paper_figures.py")
                ),
            },
        }

    locked_update_json(manifest_path, lock_path=lock_path, update=_build)


def _manifest_lock_paths(run_id: str) -> tuple[Path, Path]:
    results_dir = Path(Config.RESULTS_DIR) / "runs" / str(run_id)
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = results_dir / "manifest.json"
    lock_path = results_dir / "manifest.json.lock"
    return manifest_path, lock_path


def _update_manifest(run_id: str, patch: dict) -> None:
    if not isinstance(patch, dict):
        return
    manifest_path, lock_path = _manifest_lock_paths(str(run_id))
    def _merge(existing: dict) -> dict:
        merged = dict(existing)
        merged["updated_at"] = datetime.now().isoformat()
        for k, v in patch.items():
            if k == "experiment_runtime" and isinstance(v, dict):
                cur = merged.get("experiment_runtime")
                cur_dict = dict(cur) if isinstance(cur, dict) else {}
                cur_dict.update(v)
                merged["experiment_runtime"] = cur_dict
            elif k == "tuning_progress" and isinstance(v, dict):
                cur = merged.get("tuning_progress")
                cur_dict = dict(cur) if isinstance(cur, dict) else {}
                cur_dict.update(v)
                merged["tuning_progress"] = cur_dict
            else:
                merged[k] = v
        return merged

    locked_update_json(manifest_path, lock_path=lock_path, update=_merge)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _terminate_pid(pid: int, *, sig) -> None:
    try:
        os.killpg(int(pid), sig)
    except Exception:
        try:
            os.kill(int(pid), sig)
        except Exception:
            return


def _terminate_process_tree(pids: list[int], *, timeout_seconds: float = 2.0) -> None:
    uniq = [int(x) for x in pids if isinstance(x, int) and int(x) > 0]
    if not uniq:
        return
    for pid in uniq:
        _terminate_pid(pid, sig=signal.SIGTERM)
    deadline = time.time() + max(0.1, float(timeout_seconds))
    while time.time() < deadline:
        alive = [pid for pid in uniq if _pid_alive(pid)]
        if not alive:
            return
        time.sleep(0.05)
    for pid in uniq:
        _terminate_pid(pid, sig=signal.SIGKILL)


def is_experiment_finished(run_id, exp_name, n_rounds=None):
    status_path = os.path.join(
        Config.RESULTS_DIR, "runs", run_id, f"{exp_name}_status.json"
    )
    status_payload = _load_json(status_path)
    config = Config()
    if n_rounds is not None:
        try:
            config.N_ROUNDS = int(n_rounds)
        except Exception:
            pass
    if _status_indicates_finished(status_payload, config):
        return True

    checkpoint_path = os.path.join(
        Config.RESULTS_DIR, "checkpoints", run_id, f"{exp_name}_state.json"
    )
    checkpoint_payload = _load_json(checkpoint_path)
    if _checkpoint_indicates_finished(checkpoint_payload, config):
        return True
    return False


def run_single_experiment(args):
    experiment_name, run_id, gpu_id, resume_mode, seed, n_rounds, epochs_per_round = (
        args
    )

    # Set GPU/MPS visibility if needed (though MPS usually manages itself,
    # explicit CUDA_VISIBLE_DEVICES helps if on CUDA. On Mac MPS, it's shared)
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    start_mode = "resume" if resume_mode else "fresh"

    if start_mode == "resume" and is_experiment_finished(
        run_id, experiment_name, n_rounds
    ):
        print(
            f"[{datetime.now()}] Experiment {experiment_name} already finished. Skipping."
        )
        return (experiment_name, "skipped (finished)")

    print(
        f"[{datetime.now()}] Starting experiment: {experiment_name} (Run ID: {run_id}, Mode: {start_mode})"
    )

    result = None
    pipeline = None
    config = None
    try:
        try:
            import torch

            strategy = os.getenv("AAL_SD_SHARING_STRATEGY", "file_system")
            torch.multiprocessing.set_sharing_strategy(strategy)
        except Exception:
            pass

        # Instantiate Config to ensure properties work
        config = Config()
        if seed is not None:
            config.RANDOM_SEED = int(seed)
        if n_rounds is not None:
            config.N_ROUNDS = int(n_rounds)
        if epochs_per_round is not None:
            config.EPOCHS_PER_ROUND = int(epochs_per_round)
            config.FIX_EPOCHS_PER_ROUND = True

        # Override config based on mode
        config.START_MODE = start_mode

        # Initialize pipeline
        pipeline = ActiveLearningPipeline(config, experiment_name, run_id=run_id)

        # Run
        log_path = os.path.join(
            Config.RESULTS_DIR, "runs", run_id, f"{experiment_name}.md"
        )
        joblib_ctx = nullcontext()
        try:
            import joblib

            joblib_ctx = joblib.parallel_config(backend="threading")
        except Exception:
            pass
        with joblib_ctx:
            pipeline.run_and_collect(log_path=log_path)

        print(f"[{datetime.now()}] Finished experiment: {experiment_name}")
        result = (experiment_name, "success")

    except Exception as e:
        import traceback

        err_msg = traceback.format_exc()
        print(
            f"[{datetime.now()}] Error in experiment {experiment_name}: {e}\n{err_msg}"
        )
        result = (experiment_name, f"failed: {str(e)}")
    finally:
        try:
            pipeline = None
            config = None
            import gc

            gc.collect()
            try:
                import torch

                try:
                    if hasattr(torch, "mps"):
                        torch.mps.empty_cache()
                except Exception:
                    pass
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            pass
    return result


def _run_single_experiment_and_put(queue, args):
    try:
        os.setsid()
    except Exception:
        pass
    try:
        queue.put(run_single_experiment(args))
    except Exception as e:
        try:
            queue.put((str(args[0]), f"failed: {str(e)}"))
        except Exception:
            pass


def _vm_stat_pages() -> dict:
    try:
        out = subprocess.check_output(["vm_stat"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return {}
    pages = {}
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


def _dynamic_worker_cap(
    *,
    hard_max: int,
    hard_min: int,
    reserve_free_gb: float,
    mem_per_worker_gb: float,
) -> int:
    hard_max = max(1, int(hard_max))
    hard_min = max(1, int(hard_min))
    avail = _available_mem_gb()
    headroom = float(avail) - float(reserve_free_gb)
    if headroom <= 0:
        return hard_min
    if mem_per_worker_gb <= 0:
        return hard_max
    cap = int(headroom // float(mem_per_worker_gb))
    return max(hard_min, min(hard_max, cap))


def _run_dynamic_queue(
    *,
    experiments: list,
    run_id: str,
    resume_mode: bool,
    seed: int,
    n_rounds: int,
    epochs_per_round: int,
    workers_max: int,
    workers_min: int,
    reserve_free_gb: float,
    mem_per_worker_gb: float,
    poll_seconds: float,
    owner_pid: int | None = None,
    owner_grace_seconds: float = 60.0,
) -> dict:
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    pending = deque(experiments)
    running = {}
    results = {}

    def _start_one(exp_name: str):
        p = ctx.Process(
            target=_run_single_experiment_and_put,
            args=(
                q,
                (exp_name, run_id, 0, resume_mode, seed, n_rounds, epochs_per_round),
            ),
            daemon=False,
        )
        p.start()
        running[p.pid] = (p, exp_name)

    last_cap = None
    last_print_ts = 0.0
    last_manifest_hb_ts = 0.0
    owner_dead_since: float | None = None
    while pending or running:
        if owner_pid is not None and int(owner_pid) > 0:
            if _pid_alive(int(owner_pid)):
                owner_dead_since = None
            else:
                if owner_dead_since is None:
                    owner_dead_since = time.time()
                if (time.time() - float(owner_dead_since)) >= float(owner_grace_seconds):
                    try:
                        _update_manifest(
                            run_id,
                            {
                                "experiment_runtime": {
                                    "owner_pid": int(owner_pid),
                                    "owner_dead_at": datetime.now().isoformat(),
                                }
                            },
                        )
                    except Exception:
                        pass
                    to_kill = []
                    for pid, (p, _) in list(running.items()):
                        try:
                            if int(pid) > 0 and _pid_alive(int(pid)):
                                to_kill.append(int(pid))
                        except Exception:
                            continue
                    _terminate_process_tree(to_kill, timeout_seconds=2.0)
                    break

        pages = _vm_stat_pages()
        page_size = int(pages.get("_page_size_bytes") or 4096)
        free_pages = (
            pages.get("pages free", 0)
            + pages.get("pages speculative", 0)
            + pages.get("pages inactive", 0)
        )
        avail_gb = (free_pages * page_size) / (1024**3)
        headroom_gb = float(avail_gb) - float(reserve_free_gb)
        cap = _dynamic_worker_cap(
            hard_max=workers_max,
            hard_min=workers_min,
            reserve_free_gb=reserve_free_gb,
            mem_per_worker_gb=mem_per_worker_gb,
        )
        now = time.time()
        if last_cap is None or cap != last_cap or (now - last_print_ts) >= 30.0:
            print(
                f"[{datetime.now()}] DynamicAgentWorkers: page_size={page_size}B avail≈{avail_gb:.2f}GB "
                f"headroom≈{headroom_gb:.2f}GB reserve={float(reserve_free_gb):.2f}GB "
                f"mem_per_worker={float(mem_per_worker_gb):.2f}GB cap={cap} "
                f"running={len(running)} pending={len(pending)}"
            )
            last_print_ts = now
        if (now - last_manifest_hb_ts) >= 30.0:
            try:
                _update_manifest(
                    run_id,
                    {
                        "experiment_runtime": {
                            "driver_pid": int(os.getpid()),
                            "driver_ppid": int(os.getppid()),
                            "heartbeat_at": datetime.now().isoformat(),
                        }
                    },
                )
            except Exception:
                pass
            last_manifest_hb_ts = now
        last_cap = cap
        while pending and len(running) < cap:
            _start_one(pending.popleft())

        while True:
            try:
                exp_name, status = q.get_nowait()
            except Exception:
                break
            results[str(exp_name)] = str(status)

        finished = []
        for pid, (p, exp_name) in running.items():
            if not p.is_alive():
                p.join(timeout=0.1)
                finished.append(pid)
        for pid in finished:
            running.pop(pid, None)

        if pending or running:
            time.sleep(max(0.1, float(poll_seconds)))

    return results


def _run_fixed_parallel_queue(
    *,
    agent_experiments: list[str],
    non_agent_experiments: list[str],
    run_id: str,
    resume_mode: bool,
    seed: int,
    n_rounds: int,
    epochs_per_round: int,
    agent_workers: int,
    non_agent_workers: int,
    poll_seconds: float,
    owner_pid: int | None = None,
    owner_grace_seconds: float = 60.0,
) -> dict:
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    pending_agent = deque(agent_experiments)
    pending_non_agent = deque(non_agent_experiments)
    running: dict[int, tuple[multiprocessing.Process, str, str]] = {}
    results: dict = {}

    def _start_one(exp_name: str, *, kind: str, gpu_id: int):
        p = ctx.Process(
            target=_run_single_experiment_and_put,
            args=(
                q,
                (exp_name, run_id, gpu_id, resume_mode, seed, n_rounds, epochs_per_round),
            ),
            daemon=False,
        )
        p.start()
        running[int(p.pid)] = (p, str(exp_name), str(kind))

    last_manifest_hb_ts = 0.0
    owner_dead_since: float | None = None
    while pending_agent or pending_non_agent or running:
        if owner_pid is not None and int(owner_pid) > 0:
            if _pid_alive(int(owner_pid)):
                owner_dead_since = None
            else:
                if owner_dead_since is None:
                    owner_dead_since = time.time()
                if (time.time() - float(owner_dead_since)) >= float(owner_grace_seconds):
                    try:
                        _update_manifest(
                            run_id,
                            {
                                "experiment_runtime": {
                                    "owner_pid": int(owner_pid),
                                    "owner_dead_at": datetime.now().isoformat(),
                                }
                            },
                        )
                    except Exception:
                        pass
                    to_kill = []
                    for pid, (p, _, _) in list(running.items()):
                        try:
                            if int(pid) > 0 and _pid_alive(int(pid)):
                                to_kill.append(int(pid))
                        except Exception:
                            continue
                    _terminate_process_tree(to_kill, timeout_seconds=2.0)
                    for exp in list(pending_agent):
                        results[str(exp)] = "aborted: owner_dead"
                    for exp in list(pending_non_agent):
                        results[str(exp)] = "aborted: owner_dead"
                    for _, (_, exp, _) in list(running.items()):
                        if str(exp) not in results:
                            results[str(exp)] = "aborted: owner_dead"
                    break

        now = time.time()
        if (now - last_manifest_hb_ts) >= 30.0:
            try:
                _update_manifest(
                    run_id,
                    {
                        "experiment_runtime": {
                            "driver_pid": int(os.getpid()),
                            "driver_ppid": int(os.getppid()),
                            "heartbeat_at": datetime.now().isoformat(),
                        }
                    },
                )
            except Exception:
                pass
            last_manifest_hb_ts = now

        running_agent = sum(1 for _, (_, __, kind) in running.items() if kind == "agent")
        running_non_agent = sum(
            1 for _, (_, __, kind) in running.items() if kind == "non_agent"
        )
        while pending_agent and running_agent < max(1, int(agent_workers)):
            exp_name = pending_agent.popleft()
            gpu_id = running_agent % max(1, int(agent_workers))
            _start_one(str(exp_name), kind="agent", gpu_id=int(gpu_id))
            running_agent += 1
        while pending_non_agent and running_non_agent < max(1, int(non_agent_workers)):
            exp_name = pending_non_agent.popleft()
            gpu_id = running_non_agent % max(1, int(non_agent_workers))
            _start_one(str(exp_name), kind="non_agent", gpu_id=int(gpu_id))
            running_non_agent += 1

        while True:
            try:
                exp_name, status = q.get_nowait()
            except Exception:
                break
            results[str(exp_name)] = str(status)

        finished = []
        for pid, (p, _, _) in list(running.items()):
            if not p.is_alive():
                p.join(timeout=0.1)
                finished.append(pid)
        for pid in finished:
            running.pop(pid, None)

        if pending_agent or pending_non_agent or running:
            time.sleep(max(0.1, float(poll_seconds)))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run strict ablation experiments (parallel or sequential)"
    )
    parser.add_argument("--resume", type=str, help="Resume from an existing Run ID")
    parser.add_argument(
        "--run-id", type=str, help="Use a specific Run ID when starting fresh"
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=tuple(sorted(PRESETS.keys())),
        default=None,
        help="Apply a preset plan (explicit CLI overrides preset values)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Override RANDOM_SEED for this batch run"
    )
    parser.add_argument(
        "--n-rounds",
        type=int,
        default=None,
        help="Override N_ROUNDS for this batch run",
    )
    parser.add_argument(
        "--epochs-per-round",
        type=int,
        default=None,
        help="Override EPOCHS_PER_ROUND for this batch run",
    )
    parser.add_argument(
        "--include",
        type=str,
        default="",
        help="Comma-separated experiment names to run (others will be skipped)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help="Comma-separated experiment names to skip",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Only print plan and write manifest"
    )
    parser.add_argument(
        "--execution",
        type=str,
        choices=("parallel", "sequential"),
        default="parallel",
        help="Execution mode for experiments",
    )
    parser.add_argument(
        "--agent-workers",
        type=int,
        default=1,
        help="Max concurrent workers for agent experiments (parallel mode only)",
    )
    parser.add_argument(
        "--non-agent-workers",
        type=int,
        default=2,
        help="Max concurrent workers for non-agent experiments (parallel mode only)",
    )
    parser.add_argument("--dynamic-agent-workers", action="store_true", default=False)
    parser.add_argument("--agent-workers-min", type=int, default=1)
    parser.add_argument("--reserve-free-gb", type=float, default=4.0)
    parser.add_argument("--mem-per-agent-worker-gb", type=float, default=2.0)
    parser.add_argument("--dynamic-poll-seconds", type=float, default=2.0)
    parser.add_argument("--owner-pid", type=int, default=0)
    parser.add_argument("--owner-grace-seconds", type=float, default=60.0)
    args = parser.parse_args()

    if args.resume:
        run_id = args.resume
        resume_mode = True
        print(f"Resuming Batch Run: {run_id}")
    else:
        run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S_strict")
        resume_mode = False
        print(f"Starting Batch Run: {run_id}")

    if args.preset:
        preset = PRESETS.get(args.preset, {})
        preset_include = preset.get("include")
        preset_exclude = preset.get("exclude")
        preset_seed = preset.get("seed")
        preset_n_rounds = preset.get("n_rounds")
        preset_epochs = preset.get("epochs_per_round")
        preset_execution = preset.get("execution")
        preset_agent_workers = preset.get("agent_workers")
        preset_non_agent_workers = preset.get("non_agent_workers")

        if not str(args.include).strip() and preset_include:
            args.include = preset_include
        if not str(args.exclude).strip() and preset_exclude:
            args.exclude = preset_exclude
        if args.seed is None and preset_seed is not None:
            args.seed = int(preset_seed)
        if args.n_rounds is None and preset_n_rounds is not None:
            args.n_rounds = int(preset_n_rounds)
        if args.epochs_per_round is None and preset_epochs is not None:
            args.epochs_per_round = int(preset_epochs)
        if preset_execution and args.execution == "parallel":
            args.execution = preset_execution
        if preset_agent_workers is not None and args.agent_workers == 1:
            args.agent_workers = int(preset_agent_workers)
        if preset_non_agent_workers is not None and args.non_agent_workers == 1:
            args.non_agent_workers = int(preset_non_agent_workers)

    agent_experiments = []
    non_agent_experiments = []

    for exp_name, config in ABLATION_SETTINGS.items():
        if config.get("use_agent", False):
            agent_experiments.append(exp_name)
        else:
            non_agent_experiments.append(exp_name)

    agent_experiments.sort()
    non_agent_experiments.sort()
    if "full_model_A_lambda_policy" in agent_experiments:
        agent_experiments = ["full_model_A_lambda_policy"] + [
            name for name in agent_experiments if name != "full_model_A_lambda_policy"
        ]

    include_set = {x.strip() for x in str(args.include).split(",") if x.strip()}
    exclude_set = {x.strip() for x in str(args.exclude).split(",") if x.strip()}
    if include_set:
        include_set = {str(EXPERIMENT_NAME_ALIASES.get(x, x)) for x in include_set}
    if exclude_set:
        exclude_set = {str(EXPERIMENT_NAME_ALIASES.get(x, x)) for x in exclude_set}
    if include_set:
        agent_experiments = [x for x in agent_experiments if x in include_set]
        non_agent_experiments = [x for x in non_agent_experiments if x in include_set]
    if exclude_set:
        agent_experiments = [x for x in agent_experiments if x not in exclude_set]
        non_agent_experiments = [
            x for x in non_agent_experiments if x not in exclude_set
        ]

    print(f"Total Experiments: {len(ABLATION_SETTINGS)}")
    print(f"Agent Experiments ({len(agent_experiments)}): {agent_experiments}")
    print(
        f"Non-Agent Experiments ({len(non_agent_experiments)}): {non_agent_experiments}"
    )

    results_dir = Path(Config.RESULTS_DIR) / "runs" / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    base_config = Config()
    if args.seed is not None:
        base_config.RANDOM_SEED = int(args.seed)
    if args.n_rounds is not None:
        base_config.N_ROUNDS = int(args.n_rounds)
    if args.epochs_per_round is not None:
        base_config.EPOCHS_PER_ROUND = int(args.epochs_per_round)
        base_config.FIX_EPOCHS_PER_ROUND = True
    planned_names = []
    planned_names.extend(agent_experiments)
    planned_names.extend([x for x in non_agent_experiments if x not in planned_names])
    planned_settings = {
        name: ABLATION_SETTINGS[name]
        for name in planned_names
        if name in ABLATION_SETTINGS
    }
    _write_manifest(run_id, base_config, planned_settings)
    try:
        _update_manifest(
            run_id,
            {
                "experiment_runtime": {
                    "driver_pid": int(os.getpid()),
                    "driver_ppid": int(os.getppid()),
                    "owner_pid": int(args.owner_pid) if int(args.owner_pid) > 0 else None,
                    "owner_grace_seconds": float(args.owner_grace_seconds),
                    "started_at": datetime.now().isoformat(),
                }
            },
        )
    except Exception:
        pass
    if args.dry_run:
        print("Dry-run enabled: no experiments will be executed.")
        return

    results = {}

    if args.execution == "sequential":
        for exp_name in agent_experiments:
            exp_name, status = run_single_experiment(
                (
                    exp_name,
                    run_id,
                    0,
                    resume_mode,
                    args.seed,
                    args.n_rounds,
                    args.epochs_per_round,
                )
            )
            results[exp_name] = status
        for exp_name in non_agent_experiments:
            exp_name, status = run_single_experiment(
                (
                    exp_name,
                    run_id,
                    0,
                    resume_mode,
                    args.seed,
                    args.n_rounds,
                    args.epochs_per_round,
                )
            )
            results[exp_name] = status
    else:
        max_agent_workers = max(1, int(args.agent_workers))
        max_non_agent_workers = max(1, int(args.non_agent_workers))
        print(
            f"Concurrency Config: Agent={max_agent_workers}, Non-Agent={max_non_agent_workers}, Total={max_agent_workers + max_non_agent_workers}"
        )

        if bool(args.dynamic_agent_workers) and agent_experiments:
            dyn = _run_dynamic_queue(
                experiments=list(agent_experiments),
                run_id=run_id,
                resume_mode=resume_mode,
                seed=args.seed,
                n_rounds=args.n_rounds,
                epochs_per_round=args.epochs_per_round,
                workers_max=max_agent_workers,
                workers_min=int(args.agent_workers_min),
                reserve_free_gb=float(args.reserve_free_gb),
                mem_per_worker_gb=float(args.mem_per_agent_worker_gb),
                poll_seconds=float(args.dynamic_poll_seconds),
                owner_pid=int(args.owner_pid) if int(args.owner_pid) > 0 else None,
                owner_grace_seconds=float(args.owner_grace_seconds),
            )
            results.update(dyn)

            if non_agent_experiments:
                fixed = _run_fixed_parallel_queue(
                    agent_experiments=[],
                    non_agent_experiments=list(non_agent_experiments),
                    run_id=run_id,
                    resume_mode=resume_mode,
                    seed=args.seed,
                    n_rounds=args.n_rounds,
                    epochs_per_round=args.epochs_per_round,
                    agent_workers=1,
                    non_agent_workers=max_non_agent_workers,
                    poll_seconds=max(0.1, float(args.dynamic_poll_seconds)),
                    owner_pid=int(args.owner_pid) if int(args.owner_pid) > 0 else None,
                    owner_grace_seconds=float(args.owner_grace_seconds),
                )
                results.update(fixed)
        else:
            fixed = _run_fixed_parallel_queue(
                agent_experiments=list(agent_experiments),
                non_agent_experiments=list(non_agent_experiments),
                run_id=run_id,
                resume_mode=resume_mode,
                seed=args.seed,
                n_rounds=args.n_rounds,
                epochs_per_round=args.epochs_per_round,
                agent_workers=max_agent_workers,
                non_agent_workers=max_non_agent_workers,
                poll_seconds=max(0.1, float(args.dynamic_poll_seconds)),
                owner_pid=int(args.owner_pid) if int(args.owner_pid) > 0 else None,
                owner_grace_seconds=float(args.owner_grace_seconds),
            )
            results.update(fixed)

    # Summary
    print("\nBatch Run Complete.")
    print("Summary:")
    for exp, status in results.items():
        print(f"  - {exp}: {status}")


if __name__ == "__main__":
    # Mac OS multiprocessing fix
    multiprocessing.set_start_method("spawn", force=True)
    main()
