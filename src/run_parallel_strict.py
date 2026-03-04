
import os
import sys
import argparse
import multiprocessing
import json
import hashlib
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import time

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from main import ActiveLearningPipeline
from experiments.ablation_config import ABLATION_SETTINGS, EXPERIMENT_NAME_ALIASES
from utils.logger import logger

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
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _status_indicates_finished(status_payload, config):
    if not isinstance(status_payload, dict):
        return False
    status = status_payload.get("status")
    if status in ("completed", "finished"):
        return True
    progress = status_payload.get("progress")
    if isinstance(progress, dict):
        round_val = progress.get("round")
        if isinstance(round_val, (int, float)) and round_val >= getattr(config, "N_ROUNDS", 0):
            return True
        if isinstance(round_val, str) and round_val.strip().lower() == "finished":
            return True
    return False


def _checkpoint_indicates_finished(state_payload, config):
    if not isinstance(state_payload, dict):
        return False
    round_val = state_payload.get("round")
    if isinstance(round_val, (int, float)) and round_val >= getattr(config, "N_ROUNDS", 0):
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
    existing = _load_json(str(manifest_path)) or {}
    created_at = existing.get("created_at") or datetime.now().isoformat()
    src_dir = Path(__file__).resolve().parent
    payload = {
        "run_id": run_id,
        "created_at": created_at,
        "updated_at": datetime.now().isoformat(),
        "config": _collect_config_snapshot(config),
        "experiments": ablation_settings,
        "experiment_runtime": {
            "mode": "legacy_adapter",
            "trace_schema_version": 2,
        },
        "code_fingerprint": {
            "run_parallel_strict": _hash_file(__file__),
            "config": _hash_file(str(src_dir / "config.py")),
            "main": _hash_file(str(src_dir / "main.py")),
            "trainer": _hash_file(str(src_dir / "core" / "trainer.py")),
            "ablation_config": _hash_file(str(src_dir / "experiments" / "ablation_config.py")),
            "prompt_template": _hash_file(str(src_dir / "agent" / "prompt_template.py")),
            "plot_paper_figures": _hash_file(str(src_dir / "analysis" / "plot_paper_figures.py")),
        }
    }
    tmp_path = str(manifest_path) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, manifest_path)


def is_experiment_finished(run_id, exp_name):
    status_path = os.path.join(Config.RESULTS_DIR, "runs", run_id, f"{exp_name}_status.json")
    status_payload = _load_json(status_path)
    config = Config()
    if _status_indicates_finished(status_payload, config):
        return True

    checkpoint_path = os.path.join(Config.RESULTS_DIR, "checkpoints", run_id, f"{exp_name}_state.json")
    checkpoint_payload = _load_json(checkpoint_path)
    if _checkpoint_indicates_finished(checkpoint_payload, config):
        return True

    log_path = os.path.join(Config.RESULTS_DIR, "runs", run_id, f"{exp_name}.md")
    if not os.path.exists(log_path):
        return False

    try:
        with open(log_path, "rb") as f:
            try:
                f.seek(-1024, os.SEEK_END)
            except OSError:
                f.seek(0)
            tail = f.read().decode("utf-8", errors="ignore")
        return "## 实验汇总" in tail
    except Exception:
        return False

def run_single_experiment(args):
    experiment_name, run_id, gpu_id, resume_mode, seed, n_rounds, epochs_per_round = args
    
    # Set GPU/MPS visibility if needed (though MPS usually manages itself, 
    # explicit CUDA_VISIBLE_DEVICES helps if on CUDA. On Mac MPS, it's shared)
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id) 
    
    start_mode = "resume" if resume_mode else "fresh"
    
    if start_mode == "resume" and is_experiment_finished(run_id, experiment_name):
        print(f"[{datetime.now()}] Experiment {experiment_name} already finished. Skipping.")
        return (experiment_name, "skipped (finished)")
    
    print(f"[{datetime.now()}] Starting experiment: {experiment_name} (Run ID: {run_id}, Mode: {start_mode})")

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
        log_path = os.path.join(Config.RESULTS_DIR, "runs", run_id, f"{experiment_name}.md")
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
        print(f"[{datetime.now()}] Error in experiment {experiment_name}: {e}\n{err_msg}")
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

def main():
    parser = argparse.ArgumentParser(description="Run strict ablation experiments (parallel or sequential)")
    parser.add_argument("--resume", type=str, help="Resume from an existing Run ID")
    parser.add_argument("--run-id", type=str, help="Use a specific Run ID when starting fresh")
    parser.add_argument(
        "--preset",
        type=str,
        choices=tuple(sorted(PRESETS.keys())),
        default=None,
        help="Apply a preset plan (explicit CLI overrides preset values)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override RANDOM_SEED for this batch run")
    parser.add_argument("--n-rounds", type=int, default=None, help="Override N_ROUNDS for this batch run")
    parser.add_argument("--epochs-per-round", type=int, default=None, help="Override EPOCHS_PER_ROUND for this batch run")
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
    parser.add_argument("--dry-run", action="store_true", help="Only print plan and write manifest")
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
    args = parser.parse_args()

    if args.resume:
        run_id = args.resume
        resume_mode = True
        print(f"Resuming Batch Run: {run_id}")
    else:
        run_id = args.run_id or datetime.now().strftime('%Y%m%d_%H%M%S_strict')
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
        agent_experiments = ["full_model_A_lambda_policy"] + [name for name in agent_experiments if name != "full_model_A_lambda_policy"]

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
        non_agent_experiments = [x for x in non_agent_experiments if x not in exclude_set]
    
    print(f"Total Experiments: {len(ABLATION_SETTINGS)}")
    print(f"Agent Experiments ({len(agent_experiments)}): {agent_experiments}")
    print(f"Non-Agent Experiments ({len(non_agent_experiments)}): {non_agent_experiments}")
    
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
    planned_settings = {name: ABLATION_SETTINGS[name] for name in planned_names if name in ABLATION_SETTINGS}
    _write_manifest(run_id, base_config, planned_settings)
    if args.dry_run:
        print("Dry-run enabled: no experiments will be executed.")
        return
    
    results = {}

    if args.execution == "sequential":
        for exp_name in agent_experiments:
            exp_name, status = run_single_experiment((exp_name, run_id, 0, resume_mode, args.seed, args.n_rounds, args.epochs_per_round))
            results[exp_name] = status
        for exp_name in non_agent_experiments:
            exp_name, status = run_single_experiment((exp_name, run_id, 0, resume_mode, args.seed, args.n_rounds, args.epochs_per_round))
            results[exp_name] = status
    else:
        max_agent_workers = max(1, int(args.agent_workers))
        max_non_agent_workers = max(1, int(args.non_agent_workers))
        print(
            f"Concurrency Config: Agent={max_agent_workers}, Non-Agent={max_non_agent_workers}, Total={max_agent_workers + max_non_agent_workers}"
        )

        futures = []
        with ProcessPoolExecutor(max_workers=max_agent_workers, max_tasks_per_child=1) as agent_executor, \
             ProcessPoolExecutor(max_workers=max_non_agent_workers, max_tasks_per_child=1) as non_agent_executor:
            for i, exp_name in enumerate(agent_experiments):
                gpu_id = i % max_agent_workers
                future = agent_executor.submit(run_single_experiment, (exp_name, run_id, gpu_id, resume_mode, args.seed, args.n_rounds, args.epochs_per_round))
                futures.append(future)

            for i, exp_name in enumerate(non_agent_experiments):
                gpu_id = i % max_non_agent_workers
                future = non_agent_executor.submit(run_single_experiment, (exp_name, run_id, gpu_id, resume_mode, args.seed, args.n_rounds, args.epochs_per_round))
                futures.append(future)

            for future in as_completed(futures):
                exp_name, status = future.result()
                results[exp_name] = status

    # Summary
    print("\nBatch Run Complete.")
    print("Summary:")
    for exp, status in results.items():
        print(f"  - {exp}: {status}")

if __name__ == "__main__":
    # Mac OS multiprocessing fix
    multiprocessing.set_start_method('spawn', force=True)
    main()
