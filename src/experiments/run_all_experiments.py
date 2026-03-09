import os
import sys
import json
import argparse
import hashlib
import platform
import concurrent.futures
import multiprocessing
import traceback
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from main import ActiveLearningPipeline
from experiments.ablation_config import ABLATION_SETTINGS, EXPERIMENT_NAME_ALIASES
from utils.logger import logger


def _run_single_experiment_worker(payload: dict) -> dict:
    results_dir = str(payload["results_dir"])
    run_id = str(payload["run_id"])
    start_mode = str(payload["start_mode"])
    experiment_name = str(payload["experiment_name"])
    config_overrides = payload.get("config_overrides") or {}

    cfg = Config()
    for k, v in config_overrides.items():
        try:
            setattr(cfg, str(k), v)
        except Exception:
            pass

    cfg.START_MODE = start_mode
    cfg.SUPPRESS_MANIFEST_UPDATE = True

    runner = ExperimentRunner(cfg, results_dir, run_id=run_id, start_mode=start_mode)
    try:
        result = runner.run_single_experiment(experiment_name)
        return {"experiment_name": experiment_name, "status": "success", "result": result}
    except Exception as e:
        return {
            "experiment_name": experiment_name,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


class ExperimentRunner:
    def __init__(self, config, results_dir, run_id=None, start_mode="resume"):
        self.config = config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.results_dir / "logs_md"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.all_results = {}
        self.start_time = datetime.now()
        self.start_mode = start_mode
        self.run_id_explicit = run_id is not None
        self.run_id = run_id or self.start_time.strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.results_dir / "runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.config.RESULTS_DIR = str(self.results_dir)
        self.config.CHECKPOINT_DIR = os.path.join(self.config.RESULTS_DIR, "checkpoints")
        os.makedirs(self.config.RESULTS_DIR, exist_ok=True)
        os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)
        self.config.START_MODE = self.start_mode
        try:
            import torch

            strategy = getattr(self.config, "SHARING_STRATEGY", "file_descriptor") or "file_descriptor"
            torch.multiprocessing.set_sharing_strategy(str(strategy))
        except Exception:
            pass

        if not getattr(self.config, "LLM_API_KEY", None):
            self.config.STOP_ON_LLM_FAILURE = False
            self.config.STOP_ALL_EXPERIMENTS_ON_LLM_FAILURE = False
            logger.info("LLM_API_KEY not set. Override: STOP_ON_LLM_FAILURE=False, STOP_ALL_EXPERIMENTS_ON_LLM_FAILURE=False")
        
    def _get_experiment_log_path(self, experiment_name):
        resume_path = self._find_resumable_log_path(experiment_name)
        if resume_path is not None:
            inferred = self._infer_run_id_from_log_path(experiment_name, resume_path)
            if inferred == self.run_id:
                return resume_path
        filename = f"{experiment_name}_{self.run_id}.md"
        return str(self.log_dir / filename)

    def _infer_run_id_from_log_path(self, experiment_name, log_path):
        ts = self._parse_log_timestamp(experiment_name, Path(log_path))
        if ts is not None:
            return ts.strftime('%Y%m%d_%H%M%S')
        try:
            mtime = Path(log_path).stat().st_mtime
        except Exception:
            return self.start_time.strftime('%Y%m%d_%H%M%S')
        return datetime.fromtimestamp(mtime).strftime('%Y%m%d_%H%M%S')

    def _find_resumable_log_path(self, experiment_name):
        if not getattr(self.config, "RESUME_FROM_LOGS", True):
            return None
        incomplete = []
        for path in self.log_dir.glob(f"{experiment_name}_*.md"):
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            if "## 实验汇总" in text:
                continue
            ts = self._parse_log_timestamp(experiment_name, path)
            try:
                mtime = path.stat().st_mtime
            except Exception:
                mtime = 0.0
            incomplete.append((ts, mtime, path))

        if not incomplete:
            return None

        incomplete.sort(key=lambda item: (
            item[0] or datetime.min,
            item[1],
        ))
        return str(incomplete[-1][2])

    def _adopt_run_id_from_latest_incomplete_log(self):
        if not getattr(self.config, "RESUME_FROM_LOGS", True):
            return
        incomplete = []
        for path in self.log_dir.glob("*_*.md"):
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            if "## 实验汇总" in text:
                continue
            try:
                mtime = path.stat().st_mtime
            except Exception:
                mtime = 0.0
            incomplete.append((mtime, path))
        if not incomplete:
            return
        incomplete.sort(key=lambda item: item[0])
        latest = incomplete[-1][1]
        stem = latest.stem
        import re
        m = re.search(r"(\d{8}_\d{6})$", stem)
        if not m:
            return
        inferred = m.group(1)
        self.run_id = inferred
        self.run_dir = self.results_dir / "runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _dataset_fingerprint(self):
        data_dir = getattr(self.config, "DATA_DIR", None)
        if not data_dir:
            return None
        candidates = []
        for rel in ("TrainData/img", "images"):
            img_dir = os.path.join(data_dir, rel)
            if os.path.exists(img_dir):
                try:
                    candidates = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".h5", ".png", ".tif", ".tiff", ".jpg", ".jpeg"))])
                except Exception:
                    candidates = []
                break
        h = hashlib.sha256()
        for name in candidates:
            h.update(name.encode("utf-8", errors="ignore"))
            h.update(b"\n")
        return {"count": len(candidates), "sha256": h.hexdigest()}

    def _write_run_manifest(self):
        manifest_path = self.run_dir / "manifest.json"
        env = {
            "python": sys.version,
            "platform": platform.platform(),
        }
        try:
            import torch

            env["torch"] = getattr(torch, "__version__", None)
        except Exception:
            env["torch"] = None
        try:
            import numpy

            env["numpy"] = getattr(numpy, "__version__", None)
        except Exception:
            env["numpy"] = None
        try:
            import sklearn

            env["sklearn"] = getattr(sklearn, "__version__", None)
        except Exception:
            env["sklearn"] = None
        try:
            import segmentation_models_pytorch as smp

            env["segmentation_models_pytorch"] = getattr(smp, "__version__", None)
        except Exception:
            env["segmentation_models_pytorch"] = None

        manifest = {
            "run_id": self.run_id,
            "created_at": self.start_time.isoformat(),
            "data_dir": getattr(self.config, "DATA_DIR", None),
            "pools_dir": getattr(self.config, "POOLS_DIR", None),
            "checkpoint_dir": getattr(self.config, "CHECKPOINT_DIR", None),
            "dataset_fingerprint": self._dataset_fingerprint(),
            "environment": env,
            "experiments": [],
            "config": {
                "INITIAL_LABELED_SIZE": getattr(self.config, "INITIAL_LABELED_SIZE", None),
                "TRAIN_SPLIT": "train",
                "VAL_SPLIT": "val",
                "TEST_SPLIT": "test",
                "MODEL_SELECTION": getattr(self.config, "MODEL_SELECTION", None),
                "FAIL_ON_NONFINITE": getattr(self.config, "FAIL_ON_NONFINITE", None),
                "N_ROUNDS": getattr(self.config, "N_ROUNDS", None),
                "QUERY_SIZE": getattr(self.config, "QUERY_SIZE", None),
                "TOTAL_BUDGET": getattr(self.config, "TOTAL_BUDGET", None),
                "BUDGET_RATIO": getattr(self.config, "BUDGET_RATIO", None),
                "ESTIMATED_TOTAL_SAMPLES": getattr(self.config, "ESTIMATED_TOTAL_SAMPLES", None),
                "RANDOM_SEED": getattr(self.config, "RANDOM_SEED", None),
                "DETERMINISTIC": getattr(self.config, "DETERMINISTIC", None),
                "EPOCHS_PER_ROUND": getattr(self.config, "EPOCHS_PER_ROUND", None),
                "BATCH_SIZE": getattr(self.config, "BATCH_SIZE", None),
                "LR": getattr(self.config, "LR", None),
                "ALPHA": getattr(self.config, "ALPHA", None),
            },
        }
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    existing.update({k: v for k, v in manifest.items() if v is not None})
                    manifest = existing
            except Exception:
                pass
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    def _parse_log_timestamp(self, experiment_name, path):
        try:
            stem = Path(path).stem
        except Exception:
            return None
        prefix = f"{experiment_name}_"
        if not stem.startswith(prefix):
            return None
        raw = stem[len(prefix):]
        try:
            return datetime.strptime(raw, "%Y%m%d_%H%M%S")
        except Exception:
            return None

    def _extract_epochs_schedule_from_trace(self, trace_path: Path, *, n_rounds: int, default_epochs: int):
        if not trace_path.exists() or n_rounds <= 0:
            return None
        max_ep = {}
        try:
            with trace_path.open("r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        e = json.loads(ln)
                    except Exception:
                        continue
                    if not isinstance(e, dict) or e.get("type") != "epoch_end":
                        continue
                    r = e.get("round")
                    ep = e.get("epoch")
                    try:
                        r = int(r)
                        ep = int(ep)
                    except Exception:
                        continue
                    max_ep[r] = max(max_ep.get(r, 0), ep)
        except Exception:
            return None
        if not max_ep:
            return None
        sched = [int(default_epochs) for _ in range(int(n_rounds))]
        for i in range(1, int(n_rounds) + 1):
            if i in max_ep and int(max_ep[i]) > 0:
                sched[i - 1] = int(max_ep[i])
        return sched
        
    def run_single_experiment(self, experiment_name):
        logger.info(f"\n{'='*80}")
        logger.info(
            f"RUN START | run_id={self.run_id} | experiment={experiment_name} | start_mode={self.start_mode} | "
            f"results_dir={self.results_dir} | pools_dir={getattr(self.config, 'POOLS_DIR', None)} | checkpoint_dir={getattr(self.config, 'CHECKPOINT_DIR', None)}"
        )
        logger.info(f"{'='*80}")
        
        log_path = None
        try:
            log_path = self._get_experiment_log_path(experiment_name)
            requested = str(experiment_name)
            canonical = str(EXPERIMENT_NAME_ALIASES.get(requested, requested))
            exp_cfg = ABLATION_SETTINGS.get(canonical, {})
            schedule_from = exp_cfg.get("epoch_schedule_from") if isinstance(exp_cfg, dict) else None
            original_schedule = getattr(self.config, "EPOCHS_PER_ROUND_SCHEDULE", None)
            original_epochs_per_round = getattr(self.config, "EPOCHS_PER_ROUND", None)
            schedule_applied = None
            fixed_epochs = bool(getattr(self.config, "FIX_EPOCHS_PER_ROUND", False))
            if schedule_from and (not fixed_epochs):
                schedule_applied = self._extract_epochs_schedule_from_trace(
                    self.run_dir / f"{schedule_from}_trace.jsonl",
                    n_rounds=int(getattr(self.config, "N_ROUNDS", 0) or 0),
                    default_epochs=int(getattr(self.config, "EPOCHS_PER_ROUND", 0) or 0),
                )
                if schedule_applied is not None:
                    setattr(self.config, "EPOCHS_PER_ROUND_SCHEDULE", schedule_applied)

            try:
                epochs_override = (
                    exp_cfg.get("epochs_per_round_override")
                    if isinstance(exp_cfg, dict)
                    else None
                )
                if epochs_override is not None:
                    setattr(self.config, "EPOCHS_PER_ROUND", int(epochs_override))
                pipeline = ActiveLearningPipeline(self.config, experiment_name, run_id=self.run_id)
                run_result = pipeline.run_and_collect(log_path=log_path)
            finally:
                setattr(self.config, "EPOCHS_PER_ROUND_SCHEDULE", original_schedule)
                if original_epochs_per_round is not None:
                    setattr(self.config, "EPOCHS_PER_ROUND", original_epochs_per_round)
            result = {
                'experiment_name': experiment_name,
                'description': (exp_cfg.get('description') if isinstance(exp_cfg, dict) else None),
                'performance_history': run_result['performance_history'],
                'budget_history': run_result['budget_history'],
                'fallback_history': run_result.get('fallback_history', []),
                'alc': float(run_result['alc']),
                'final_miou': float(run_result['final_miou']),
                'final_f1': float(run_result['final_f1']),
                'epochs_schedule_source': schedule_from,
                'epochs_schedule': schedule_applied,
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'log_file': log_path
            }
            
            self.all_results[experiment_name] = result
            
            # Concurrency Fix: Disable manifest update in child process to avoid race condition
            # The manifest will be rebuilt by the main process (run_parallel.py) or by run_all_experiments loop.
            # Only update manifest if we are running sequentially (not checked here but assumed safe if sequential)
            # But wait, run_single_experiment is called by run_all_experiments which IS sequential.
            # HOWEVER, if run_parallel calls this via a subprocess, we have a race on manifest.json if multiple subprocesses write to it.
            # To be safe, we disable manifest writing here if we suspect we are in a parallel worker?
            # Or we can just use a file lock. But simpler: assume run_parallel handles manifest reconstruction.
            # So we SKIP manifest writing here? 
            # If we skip it, run_all_experiments (sequential) will also skip it.
            # Better strategy: run_all_experiments calls _write_run_manifest() at the start.
            # And maybe we should update it only at the end?
            
            # Let's use a simple heuristic: if we are in a parallel worker (indicated by not having all experiments in list maybe?)
            # Actually, the safest way is to use file locking or just don't write it here.
            # Let's keep it but wrap in a try-except block with a lock if possible, 
            # or just rely on the fact that run_parallel REBUILDS it at the end.
            # If we have race condition, the file might get corrupted.
            # So we should DISABLE it here and let the orchestrator handle it.
            
            # BUT run_all_experiments needs it.
            # Let's add a flag to Config or Runner to suppress manifest updates.
            if not getattr(self.config, "SUPPRESS_MANIFEST_UPDATE", False):
                try:
                    manifest_path = self.run_dir / "manifest.json"
                    # Simple atomic-ish update: read, update, write. Still racy.
                    # Given run_parallel handles reconstruction, we can skip this if we set the flag.
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
                    if isinstance(manifest, dict):
                        experiments = manifest.get("experiments")
                        if not isinstance(experiments, list):
                            experiments = []
                        if experiment_name not in experiments:
                            experiments.append(experiment_name)
                        manifest["experiments"] = experiments
                        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass
            
            logger.info(f"Experiment {experiment_name} completed. ALC: {result['alc']:.4f}, Final mIoU: {result['final_miou']:.4f}")
            logger.info(f"RUN END   | run_id={self.run_id} | experiment={experiment_name} | status=success")
            
            return result
            
        except Exception as e:
            logger.error(f"Experiment {experiment_name} failed: {str(e)}")
            logger.info(f"RUN END   | run_id={self.run_id} | experiment={experiment_name} | status=failed")
            
            performance_history = []
            budget_history = []
            last_round = None
            
            if log_path and os.path.exists(log_path):
                try:
                    with open(log_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    import re
                    rounds = re.findall(r"## Round (\d+)", content)
                    if rounds:
                        last_round = int(rounds[-1])
                    
                    labeled_sizes = re.findall(r"Labeled Pool Size: (\d+)", content)
                    if labeled_sizes:
                        last_labeled_size = int(labeled_sizes[-1])
                    
                    miou_matches = re.findall(r"Round=(\d+), Labeled=(\d+), mIoU=([\d.]+), F1=([\d.]+)", content)
                    for match in miou_matches:
                        round_num, labeled_size, miou, f1 = match
                        performance_history.append({
                            "round": int(round_num),
                            "mIoU": float(miou),
                            "f1_score": float(f1),
                            "labeled_size": int(labeled_size)
                        })
                    
                    budget_history = [p["labeled_size"] for p in performance_history]
                except Exception as log_error:
                    logger.warning(f"Failed to parse log for resume state: {str(log_error)}")
            
            result = {
                'experiment_name': experiment_name,
                'description': ABLATION_SETTINGS[experiment_name]['description'],
                'status': 'failed',
                'error': str(e),
                'performance_history': performance_history,
                'budget_history': budget_history,
                'last_round': last_round,
                'timestamp': datetime.now().isoformat(),
                'log_file': log_path
            }
            self.all_results[experiment_name] = result
            return result
    
    def _finalize_run_manifest(self, experiment_list: list[str]):
        try:
            manifest_path = self.run_dir / "manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
            if not isinstance(payload, dict):
                payload = {}
            payload["experiments"] = list(experiment_list)
            manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def _export_config_overrides(self) -> dict:
        keys = [
            "RANDOM_SEED",
            "DETERMINISTIC",
            "RESEARCH_MODE",
            "FIX_EPOCHS_PER_ROUND",
            "EPOCHS_PER_ROUND",
            "EPOCHS_PER_ROUND_SCHEDULE",
            "N_ROUNDS",
            "RESULTS_DIR",
            "CHECKPOINT_DIR",
            "POOLS_DIR",
            "DATA_DIR",
            "DEVICE",
            "BATCH_SIZE",
            "LR",
            "ALPHA",
            "SHARING_STRATEGY",
            "STOP_ON_LLM_FAILURE",
            "STOP_ALL_EXPERIMENTS_ON_LLM_FAILURE",
            "RESUME_FROM_LOGS",
            "STRICT_RESUME",
            "FEATURE_NUM_WORKERS",
            "FEATURE_PERSISTENT_WORKERS",
            "FEATURE_PREFETCH_FACTOR",
            "FEATURE_PIN_MEMORY",
            "STRICT_INNOVATION",
            "LLM_API_KEY",
            "LLM_BASE_URL",
            "LLM_MODEL",
            "LLM_TEMPERATURE",
        ]
        out = {}
        for k in keys:
            if hasattr(self.config, k):
                try:
                    out[k] = getattr(self.config, k)
                except Exception:
                    pass
        return out

    def run_all_experiments(self, experiment_list=None, *, parallel_workers: int = 1):
        if experiment_list is None:
            experiment_list = list(ABLATION_SETTINGS.keys())

        if self.start_mode == "resume" and (not self.run_id_explicit):
            self._adopt_run_id_from_latest_incomplete_log()
        self._write_run_manifest()
        self._finalize_run_manifest([str(x) for x in experiment_list])
        
        logger.info(f"Starting {len(experiment_list)} experiments...")

        if int(parallel_workers) <= 1:
            for exp_name in experiment_list:
                result = self.run_single_experiment(exp_name)
                if (
                    isinstance(result, dict)
                    and result.get("status") == "failed"
                    and getattr(self.config, "STOP_ALL_EXPERIMENTS_ON_LLM_FAILURE", True)
                    and "timed out" in str(result.get("error", "")).lower()
                ):
                    raise RuntimeError(result.get("error"))
        else:
            self.config.SUPPRESS_MANIFEST_UPDATE = True
            ctx = multiprocessing.get_context("spawn")
            config_overrides = self._export_config_overrides()
            should_stop_all_on_llm_failure = bool(getattr(self.config, "STOP_ALL_EXPERIMENTS_ON_LLM_FAILURE", True))

            with concurrent.futures.ProcessPoolExecutor(
                max_workers=int(parallel_workers),
                mp_context=ctx,
                max_tasks_per_child=1,
            ) as ex:
                futures = []
                for exp_name in experiment_list:
                    futures.append(
                        ex.submit(
                            _run_single_experiment_worker,
                            {
                                "results_dir": str(self.results_dir),
                                "run_id": str(self.run_id),
                                "start_mode": str(self.start_mode),
                                "experiment_name": str(exp_name),
                                "config_overrides": config_overrides,
                            },
                        )
                    )

                for fut in concurrent.futures.as_completed(futures):
                    payload = fut.result()
                    exp_name = payload.get("experiment_name")
                    if payload.get("status") == "success":
                        result = payload.get("result")
                        if isinstance(exp_name, str) and isinstance(result, dict):
                            self.all_results[exp_name] = result
                    else:
                        error = payload.get("error")
                        if isinstance(exp_name, str):
                            self.all_results[exp_name] = {
                                "experiment_name": exp_name,
                                "description": ABLATION_SETTINGS.get(exp_name, {}).get("description", ""),
                                "status": "failed",
                                "error": str(error),
                                "timestamp": datetime.now().isoformat(),
                            }
                        if should_stop_all_on_llm_failure and "timed out" in str(error or "").lower():
                            for f in futures:
                                try:
                                    f.cancel()
                                except Exception:
                                    pass
                            raise RuntimeError(str(error))
        
        self.save_results()
        self.generate_report()
    
    def save_results(self):
        # Save individual result files to support safe parallel execution
        # Each experiment writes its own file, avoiding race conditions on the main results file
        for exp_name, result in self.all_results.items():
            individual_file = self.run_dir / f"result_{exp_name}.json"
            try:
                with open(individual_file, 'w', encoding='utf-8') as f:
                    json.dump({exp_name: result}, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Failed to save individual result for {exp_name}: {e}")

        # Best-effort update of the main results file
        # In parallel mode, this file might be incomplete or corrupted due to race conditions,
        # but the orchestrator (run_parallel.py) will rebuild it from individual files.
        results_file = self.run_dir / "experiment_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {results_file}")
    
    def generate_report(self):
        from experiments.report_generator import ReportGenerator
        
        generator = ReportGenerator(self.all_results, self.run_dir)
        generator.generate_all_reports()


def main():
    parser = argparse.ArgumentParser(description='Run all baseline and ablation experiments')
    parser.add_argument('--experiments', type=str, nargs='+', 
                        help='Specific experiments to run (default: all)')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--start', type=str, default='resume', choices=['resume', 'fresh'],
                        help='Start mode: resume from latest incomplete run, or start fresh')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Explicit run_id to use (recommended for research)')
    parser.add_argument('--n_rounds', type=int, default=None,
                        help='Number of active learning rounds to run (default: use config value)')
    parser.add_argument('--parallel_workers', type=int, default=1,
                        help='Run multiple experiments in parallel within a single run_id (default: 1)')
    
    args = parser.parse_args()
    
    config = Config()
    if args.config:
        import importlib.util
        spec = importlib.util.spec_from_file_location("config_module", args.config)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config = config_module.Config()
    
    runner = ExperimentRunner(config, args.results_dir, run_id=args.run_id, start_mode=args.start)

    if args.n_rounds is not None and args.n_rounds > 0:
        config.N_ROUNDS = int(args.n_rounds)
        logger.info(f"Config override: N_ROUNDS={config.N_ROUNDS}")
    
    if args.experiments:
        runner.run_all_experiments(args.experiments, parallel_workers=int(args.parallel_workers))
    else:
        runner.run_all_experiments(parallel_workers=int(args.parallel_workers))


if __name__ == '__main__':
    main()
