import argparse
import os
from datetime import datetime
import hashlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import gc

from config import Config
from core.data_preprocessing import DataPreprocessor
from core.dataset import Landslide4SenseDataset
from core.model import LandslideDeepLabV3
from core.trainer import Trainer
from core.sampler import ADKUCSSampler
from agent.agent_manager import AgentManager
from agent.toolbox import Toolbox
from agent.prompt_template import PromptBuilder
from agent.agent_manager import SiliconFlowClient
from agent.config import AgentThresholds
from baselines.bald_sampler import BALDSampler
from experiments.ablation_config import (
    ABLATION_SETTINGS,
    EXPERIMENT_NAME_ALIASES,
    build_spec_from_legacy_dict,
)
from experiments.components import (
    build_sampler,
    build_selection_postprocessor,
)
from utils.logger import logger
from utils.evaluation import calculate_alc
from utils.reproducibility import set_global_seed, worker_init_fn
from core.checkpoint import CheckpointManager


class ActiveLearningPipeline:
    def __init__(self, config, experiment_name, run_id=None):
        self.config = config
        self.experiment_name = experiment_name
        self.run_id = run_id
        if getattr(self.config, "RESEARCH_MODE", False) and not self.run_id:
            raise ValueError("Research mode requires explicit run_id")
        if not self.run_id:
            self.run_id = "default"
        requested = str(experiment_name)
        canonical = str(EXPERIMENT_NAME_ALIASES.get(requested, requested))
        self.exp_config = ABLATION_SETTINGS.get(canonical)
        if not self.exp_config:
            raise ValueError(f"Unknown experiment: {experiment_name}")
        self._apply_agent_threshold_overrides(
            self.exp_config.get("agent_threshold_overrides")
            if isinstance(self.exp_config, dict)
            else None
        )
        self.experiment_spec = build_spec_from_legacy_dict(requested, self.exp_config)
        self.experiment_runtime = self.experiment_spec.build(config)

        # Centralized Randomness Control
        self.seed = int(getattr(self.config, "RANDOM_SEED", 42) or 42)
        self.deterministic = getattr(self.config, "DETERMINISTIC", True)
        set_global_seed(self.seed, deterministic=self.deterministic)

        logger.info(f"Initializing Experiment: {experiment_name}")
        logger.info(f"Description: {self.exp_config.get('description') or ''}")
        logger.info(
            f"Isolation: run_id={self.run_id} experiment={self.experiment_name} pools_dir={getattr(self.config, 'POOLS_DIR', None)} "
            f"results_dir={getattr(self.config, 'RESULTS_DIR', None)} checkpoint_dir={getattr(self.config, 'CHECKPOINT_DIR', None)}"
        )

        self.pools_dir = self._resolve_pools_dir()
        self._last_control_events = {}

        checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, self.run_id)
        self.checkpoint_manager = CheckpointManager(checkpoint_dir, experiment_name)

        self.run_reports_dir = os.path.join(config.RESULTS_DIR, "runs", self.run_id)
        os.makedirs(self.run_reports_dir, exist_ok=True)
        self.status_path = os.path.join(
            self.run_reports_dir, f"{self.experiment_name}_status.json"
        )
        self.trace_path = os.path.join(
            self.run_reports_dir, f"{self.experiment_name}_trace.jsonl"
        )
        self.round_model_dir = os.path.join(
            self.run_reports_dir, f"{self.experiment_name}_round_models"
        )

        # 1. Data Preprocessing
        start_mode = getattr(self.config, "START_MODE", "resume")
        if start_mode == "fresh":
            if os.path.exists(self.checkpoint_manager.checkpoint_path):
                try:
                    os.remove(self.checkpoint_manager.checkpoint_path)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to remove checkpoint for fresh start: {self.checkpoint_manager.checkpoint_path} ({e})"
                    ) from e
            if os.path.exists(self.status_path):
                try:
                    os.remove(self.status_path)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to remove status for fresh start: {self.status_path} ({e})"
                    ) from e
            if os.path.exists(self.trace_path):
                try:
                    os.remove(self.trace_path)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to remove trace for fresh start: {self.trace_path} ({e})"
                    ) from e
            if os.path.isdir(self.round_model_dir):
                try:
                    import shutil

                    shutil.rmtree(self.round_model_dir)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to remove round model dir for fresh start: {self.round_model_dir} ({e})"
                    ) from e

        os.makedirs(self.round_model_dir, exist_ok=True)

        import shutil

        base_subdir = os.path.join(self.run_id, "_base")
        base_preprocessor = DataPreprocessor(config, experiment_name=base_subdir)
        base_preprocessor.create_data_pools(force=(start_mode == "fresh"))

        base_dir = os.path.join(config.POOLS_DIR, base_subdir)
        os.makedirs(self.pools_dir, exist_ok=True)
        for name in (
            "labeled_pool.csv",
            "unlabeled_pool.csv",
            "pools_manifest.json",
        ):
            src = os.path.join(base_dir, name)
            dst = os.path.join(self.pools_dir, name)
            if not os.path.exists(src):
                raise FileNotFoundError(f"Missing base pool file: {src}")
            if start_mode == "fresh" or (not os.path.exists(dst)):
                tmp = dst + ".tmp"
                shutil.copy2(src, tmp)
                os.replace(tmp, dst)

        self._validate_pools_manifest()

        # 2. Load Data Pools
        import pandas as pd

        self.labeled_df = pd.read_csv(os.path.join(self.pools_dir, "labeled_pool.csv"))
        self.unlabeled_df = pd.read_csv(
            os.path.join(self.pools_dir, "unlabeled_pool.csv")
        )

        # Map sample_id to indices in the full dataset is tricky if we don't load full dataset.
        # But Dataset class loads by directory.
        # Strategy: Load Full Train/Val Dataset, then map file names to indices.

        # NOTE: Landslide4SenseDataset loads all files in a dir.
        # We need to construct indices based on file names.

        train_split = (
            str(getattr(config, "TRAIN_SPLIT", "train") or "train").strip().lower()
        )
        self.full_dataset = Landslide4SenseDataset(
            config.DATA_DIR, 
            split=train_split,
            bg_undersample_ratio=getattr(config, "BG_UNDERSAMPLE_RATIO", 1.0)
        )
        self.query_dataset = Landslide4SenseDataset(
            config.DATA_DIR, split=train_split, with_mask=False
        )
        self.dataset = self.full_dataset
        # Assuming 'train' split in dataset class points to the folder containing all training candidates

        self.labeled_indices = self._map_filenames_to_indices(
            self.full_dataset, self.labeled_df["sample_id"].tolist()
        )
        self._initial_labeled_indices = list(self.labeled_indices)
        self.unlabeled_indices = self._map_filenames_to_indices(
            self.full_dataset, self.unlabeled_df["sample_id"].tolist()
        )

        logger.info(
            f"Initial Labeled: {len(self.labeled_indices)}, Unlabeled: {len(self.unlabeled_indices)}"
        )

        self._assert_pool_integrity()
        if getattr(self.config, "TOTAL_BUDGET", None) in (None, 0):
            self.config.TOTAL_BUDGET = max(int(len(self.full_dataset) * 0.1), 1)
        logger.info(
            f"Config budget: ESTIMATED_TOTAL_SAMPLES={getattr(self.config, 'ESTIMATED_TOTAL_SAMPLES', None)} "
            f"BUDGET_RATIO={getattr(self.config, 'BUDGET_RATIO', None)} TOTAL_BUDGET={getattr(self.config, 'TOTAL_BUDGET', None)}"
        )

        # 3. Initialize Sampler
        sampler_result = build_sampler(config, self.exp_config)
        self.sampler = sampler_result.sampler
        self.sampler_type = sampler_result.sampler_type
        self.score_normalization = sampler_result.score_normalization
        self.rollback_config = sampler_result.rollback_config

        # 4. Agent Setup
        self.use_agent = self.exp_config["use_agent"]
        self.agent_manager = None
        if (
            self.use_agent
            and getattr(self.config, "REQUIRE_LLM_FOR_AGENT", True)
            and (not getattr(self.config, "LLM_API_KEY", None))
        ):
            self._append_trace(
                {
                    "type": "llm_degraded",
                    "round": 0,
                    "degraded": {
                        "source": "pipeline_init",
                        "reason_type": "missing_api_key",
                        "policy": "fail_fast",
                        "message": "LLM_API_KEY is required for agent experiments (REQUIRE_LLM_FOR_AGENT=True).",
                    },
                }
            )
            self._write_status(
                {
                    "status": "failed",
                    "error": {
                        "round": 0,
                        "type": "missing_llm_api_key",
                        "message": "LLM_API_KEY is required for agent experiments (REQUIRE_LLM_FOR_AGENT=True).",
                        "exception": "MissingLLMApiKey",
                    },
                }
            )
            raise RuntimeError(
                "LLM_API_KEY is required for agent experiments (REQUIRE_LLM_FOR_AGENT=True)."
            )
        if self.use_agent:
            # We need a model placeholder to init toolbox, but model changes every round
            # We will set toolbox.model later
            dummy_model = None
            # We need a wrapper to act as 'controller' for toolbox (providing dataset/indices)
            self.toolbox = Toolbox(self, self.sampler, dummy_model)

            # Set control permissions for ablation experiments
            control_permissions = self.exp_config.get("control_permissions")
            if control_permissions is None:
                if getattr(self.config, "RESEARCH_MODE", False):
                    raise ValueError(
                        f"Research mode requires explicit control_permissions for agent experiment: {self.experiment_name}"
                    )
                fixed_epochs = bool(getattr(self.config, "FIX_EPOCHS_PER_ROUND", False))
                control_permissions = {
                    "set_lambda": True,
                    "set_query_size": True,
                    "set_epochs_per_round": (not fixed_epochs),
                    "set_alpha": True,
                }

            self.toolbox.set_control_permissions(control_permissions)
            logger.info(f"Control permissions set: {control_permissions}")

            client = SiliconFlowClient(
                api_key=config.LLM_API_KEY,
                base_url=config.LLM_BASE_URL,
                model=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE,
                timeout=config.LLM_TIMEOUT,
                max_retries=int(getattr(config, "LLM_MAX_RETRIES", 0) or 0),
                retry_delay=float(
                    getattr(config, "LLM_RETRY_BASE_SECONDS", 0.0) or 5.0
                ),
                retry_backoff=float(getattr(config, "LLM_RETRY_BACKOFF", 1.0) or 2.0),
            )
            self.agent_manager = AgentManager(self.toolbox, client=client, verbose=True)
            self.agent_manager.llm_max_retries = int(
                getattr(config, "LLM_MAX_RETRIES", 0) or 0
            )
            self.agent_manager.llm_retry_base_seconds = float(
                getattr(config, "LLM_RETRY_BASE_SECONDS", 0.0) or 0.0
            )
            self.agent_manager.llm_retry_backoff = float(
                getattr(config, "LLM_RETRY_BACKOFF", 1.0) or 1.0
            )
            self.agent_manager.llm_retry_max_seconds = float(
                getattr(config, "LLM_RETRY_MAX_SECONDS", 0.0) or 0.0
            )
            self.agent_manager.reset()
        self.selection_postprocessor = build_selection_postprocessor(self)
        self.current_round = None
        self._last_query_selected_count = None
        self._selection_context = None
        self._last_score_precalc_error = None
        self._last_ranking_degraded = None
        self._last_ranking_metadata = None
        self._last_ranked_items = None
        self._last_training_state = None
        self._last_lambda_controller = None
        self._last_selection_summary = None
        self.training_state: dict = {
            "train_u_median_history": [],
            "train_k_median_history": [],
            "max_history_length": 5,
        }
        self.default_query_size = int(getattr(self.config, "QUERY_SIZE", 0) or 0)
        self.default_epochs_per_round = int(
            getattr(self.config, "EPOCHS_PER_ROUND", 0) or 0
        )
        self._pending_round_controls = {}
        self.train_split = (
            str(getattr(self.config, "TRAIN_SPLIT", "train") or "train").strip().lower()
        )
        self.val_split = (
            str(getattr(self.config, "VAL_SPLIT", "val") or "val").strip().lower()
        )
        self.test_split = (
            str(getattr(self.config, "TEST_SPLIT", "test") or "test").strip().lower()
        )

        self._write_status(
            {
                "status": "initialized",
                "pools_dir": self.pools_dir,
                "checkpoint_path": self.checkpoint_manager.checkpoint_path,
                "initial": {
                    "labeled": int(len(self.labeled_indices)),
                    "unlabeled": int(len(self.unlabeled_indices)),
                },
            }
        )

        self._append_trace(
            {
                "type": "initialized",
                "pools_dir": self.pools_dir,
                "checkpoint_path": self.checkpoint_manager.checkpoint_path,
                "trace_schema_version": int(
                    getattr(
                        getattr(self, "experiment_runtime", None), "trace_options", None
                    ).schema_version
                )
                if getattr(
                    getattr(self, "experiment_runtime", None), "trace_options", None
                )
                is not None
                else 1,
                "experiment_runtime": {
                    "mode": "legacy_adapter",
                    "spec": getattr(
                        getattr(self, "experiment_spec", None), "name", None
                    ),
                },
                "ablation": {
                    "use_agent": bool(self.use_agent),
                    "sampler_type": getattr(self, "sampler_type", None),
                    "score_normalization": getattr(self, "score_normalization", None),
                    "lambda_override": self.exp_config.get("lambda_override")
                    if isinstance(self.exp_config, dict)
                    else None,
                    "lambda_controller": self.exp_config.get("lambda_controller")
                    if isinstance(self.exp_config, dict)
                    else None,
                    "lambda_policy": self.exp_config.get("lambda_policy")
                    if isinstance(self.exp_config, dict)
                    else None,
                    "require_explicit_lambda": self.exp_config.get(
                        "require_explicit_lambda"
                    )
                    if isinstance(self.exp_config, dict)
                    else None,
                    "rollback_config": getattr(self, "rollback_config", None),
                    "control_permissions": self.exp_config.get("control_permissions")
                    if isinstance(self.exp_config, dict)
                    else None,
                    "agent_threshold_overrides": self.exp_config.get(
                        "agent_threshold_overrides"
                    )
                    if isinstance(self.exp_config, dict)
                    else None,
                    "enable_l3_selection_logging": self.exp_config.get(
                        "enable_l3_selection_logging"
                    )
                    if isinstance(self.exp_config, dict)
                    else None,
                    "enable_agent_prompt_logging": self.exp_config.get(
                        "enable_agent_prompt_logging"
                    )
                    if isinstance(self.exp_config, dict)
                    else None,
                },
                "runtime": {
                    "fix_epochs_per_round": bool(
                        getattr(self.config, "FIX_EPOCHS_PER_ROUND", False)
                    ),
                    "epochs_per_round": int(
                        getattr(self.config, "EPOCHS_PER_ROUND", 0) or 0
                    ),
                    "query_size": int(getattr(self.config, "QUERY_SIZE", 0) or 0),
                    "total_budget": int(getattr(self.config, "TOTAL_BUDGET", 0) or 0),
                    "n_rounds": int(getattr(self.config, "N_ROUNDS", 0) or 0),
                    "random_seed": int(getattr(self.config, "RANDOM_SEED", 0) or 0),
                    "deterministic": bool(getattr(self.config, "DETERMINISTIC", True)),
                    "train_split": self.train_split,
                    "val_split": self.val_split,
                    "test_split": self.test_split,
                },
                "counts": {
                    "labeled": int(len(self.labeled_indices)),
                    "unlabeled": int(len(self.unlabeled_indices)),
                },
            }
        )

        try:
            import torch

            logger.info(
                f"Runtime: device={getattr(self.config, 'DEVICE', None)} num_workers={getattr(self.config, 'NUM_WORKERS', None)} "
                f"torch_threads={torch.get_num_threads()} torch_interop_threads={torch.get_num_interop_threads()}"
            )
        except Exception:
            logger.info(
                f"Runtime: device={getattr(self.config, 'DEVICE', None)} num_workers={getattr(self.config, 'NUM_WORKERS', None)}"
            )

    def _apply_agent_threshold_overrides(self, overrides):
        if not isinstance(overrides, dict):
            return
        for key, value in overrides.items():
            if not isinstance(key, str):
                continue
            name = key.strip()
            if not name or not hasattr(AgentThresholds, name):
                continue
            applied = value
            if isinstance(value, (int, float)):
                applied = float(value)
            setattr(AgentThresholds, name, applied)

    def _write_status(self, patch):
        import json
        from utils import atomic_write_json

        payload = {}
        try:
            if os.path.exists(self.status_path):
                with open(self.status_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if isinstance(existing, dict):
                    payload.update(existing)
        except Exception:
            payload = {}

        base = {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "updated_at": datetime.now().isoformat(),
            "pid": int(os.getpid()),
            "ppid": int(os.getppid()),
        }
        payload.update(base)
        existing_errors = payload.get("errors")
        errors: list = list(existing_errors) if isinstance(existing_errors, list) else []
        if isinstance(patch, dict):
            patch_dict = dict(patch)
            incoming_error = None
            if "current_error" in patch_dict:
                incoming_error = patch_dict.pop("current_error")
            if "error" in patch_dict:
                incoming_error = patch_dict.pop("error")

            status_next = patch_dict.get("status")
            should_clear_current_error = (
                incoming_error is None
                and isinstance(status_next, str)
                and status_next.strip().lower() in ("running", "completed", "finished")
            )

            payload.update(patch_dict)
            payload["errors"] = errors

            if incoming_error is not None:
                cur = payload.get("current_error")
                if incoming_error != cur:
                    errors.append(
                        {
                            "ts": datetime.now().isoformat(),
                            "status": str(payload.get("status") or ""),
                            "error": incoming_error,
                        }
                    )
                payload["current_error"] = incoming_error
                payload["error"] = incoming_error
            elif should_clear_current_error:
                payload.pop("current_error", None)
                payload.pop("error", None)

        atomic_write_json(self.status_path, payload, indent=2)

    def _append_trace(self, event):
        from utils import append_jsonl

        payload = {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "ts": datetime.now().isoformat(),
        }
        if isinstance(event, dict):
            payload.update(event)
        try:
            et = payload.get("type")
            if isinstance(et, str) and et in (
                "lambda_policy_apply",
                "lambda_override",
                "lambda_guard",
                "selection_guardrail",
            ):
                self._last_control_events[str(et)] = dict(payload)
        except Exception:
            pass
        append_jsonl(self.trace_path, payload)

    def _resolve_pools_dir(self):
        return os.path.join(self.config.POOLS_DIR, self.run_id, self.experiment_name)

    def _map_filenames_to_indices(self, dataset, filenames):
        # Create a map name -> idx
        # dataset.images is list of filenames like 'img_1.h5'
        name_to_idx = {os.path.splitext(f)[0]: i for i, f in enumerate(dataset.images)}
        indices = []
        for name in filenames:
            if name in name_to_idx:
                indices.append(name_to_idx[name])
        return indices

    def _pool_integrity(self):
        labeled_ids = (
            set(self.labeled_df["sample_id"].tolist())
            if hasattr(self, "labeled_df")
            else set()
        )
        unlabeled_ids = (
            set(self.unlabeled_df["sample_id"].tolist())
            if hasattr(self, "unlabeled_df")
            else set()
        )
        overlap_l_u = len(labeled_ids.intersection(unlabeled_ids))
        union = labeled_ids.union(unlabeled_ids)
        return {
            "counts": {
                "labeled": int(len(labeled_ids)),
                "unlabeled": int(len(unlabeled_ids)),
                "union": int(len(union)),
                "dataset": int(len(getattr(self.full_dataset, "images", []) or [])),
            },
            "overlaps": {
                "labeled_unlabeled": int(overlap_l_u),
            },
        }

    def _assert_pool_integrity(self):
        integrity = self._pool_integrity()
        if int(integrity["counts"].get("labeled") or 0) <= 0:
            if getattr(self.config, "STRICT_RESUME", False):
                raise RuntimeError(f"Labeled pool is empty: {integrity}")
        ok = (
            integrity["overlaps"]["labeled_unlabeled"] == 0
            and integrity["counts"]["union"] == integrity["counts"]["dataset"]
        )
        self._write_status({"pool_integrity": integrity, "pool_integrity_ok": bool(ok)})
        if not ok and getattr(self.config, "STRICT_RESUME", False):
            raise RuntimeError(f"Pool integrity check failed: {integrity}")
        return ok

    def _dataset_fingerprint(self):
        import hashlib

        names = list(getattr(self.full_dataset, "images", []) or [])
        names = sorted([os.path.basename(n) for n in names])
        h = hashlib.sha256()
        for name in names:
            h.update(name.encode("utf-8", errors="ignore"))
            h.update(b"\n")
        return {"count": int(len(names)), "sha256": h.hexdigest()}

    def _validate_pools_manifest(self):
        import json
        import hashlib

        manifest_path = os.path.join(self.pools_dir, "pools_manifest.json")
        if not os.path.exists(manifest_path):
            raise RuntimeError(f"Pool manifest missing: {manifest_path}")
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read pool manifest: {manifest_path} ({e})"
            ) from e

        if (
            not isinstance(manifest, dict)
            or int(manifest.get("schema_version") or 0) != 1
        ):
            raise RuntimeError(f"Unsupported pool manifest schema: {manifest_path}")

        def _fingerprint_dir(img_dir: str, mask_dir: str | None) -> dict:
            h = hashlib.sha256()
            img_names = sorted(
                [f for f in os.listdir(img_dir) if str(f).lower().endswith(".h5")]
            )
            for name in img_names:
                h.update(str(name).encode("utf-8", errors="ignore"))
                h.update(b"\n")
            img_fp = {"count": int(len(img_names)), "sha256": str(h.hexdigest())}
            if mask_dir is None:
                return {"images": img_fp, "masks": None}
            mh = hashlib.sha256()
            mask_names = sorted(
                [f for f in os.listdir(mask_dir) if str(f).lower().endswith(".h5")]
            )
            for name in mask_names:
                mh.update(str(name).encode("utf-8", errors="ignore"))
                mh.update(b"\n")
            return {
                "images": img_fp,
                "masks": {"count": int(len(mask_names)), "sha256": str(mh.hexdigest())},
            }

        def _split_paths(split: str) -> tuple[str, str | None]:
            s = str(split).strip().lower()
            if s == "train":
                return (
                    os.path.join(self.config.DATA_DIR, "TrainData", "img"),
                    os.path.join(self.config.DATA_DIR, "TrainData", "mask"),
                )
            if s == "val":
                return (
                    os.path.join(self.config.DATA_DIR, "ValidData", "img"),
                    os.path.join(self.config.DATA_DIR, "ValidData", "mask"),
                )
            if s == "test":
                return (
                    os.path.join(self.config.DATA_DIR, "TestData", "img"),
                    os.path.join(self.config.DATA_DIR, "TestData", "mask"),
                )
            raise RuntimeError(f"Unknown split for manifest validation: {split}")

        splits = (
            manifest.get("splits") if isinstance(manifest.get("splits"), dict) else {}
        )
        for s in ("train", "val"):
            img_dir, mask_dir = _split_paths(s)
            if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir or ""):
                raise RuntimeError(
                    f"Missing required split directories for {s}: {img_dir} {mask_dir}"
                )
            actual = _fingerprint_dir(img_dir, mask_dir)
            expected = splits.get(s)
            if expected != actual:
                raise RuntimeError(
                    f"Pool manifest mismatch for split={s}: expected={expected} actual={actual} manifest={manifest_path}"
                )

        expected_test = splits.get("test")
        if expected_test is not None:
            img_dir, mask_dir = _split_paths("test")
            if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir or ""):
                raise RuntimeError(
                    f"Pool manifest expects test split but directories missing: {img_dir} {mask_dir}"
                )
            actual_test = _fingerprint_dir(img_dir, mask_dir)
            if expected_test != actual_test:
                raise RuntimeError(
                    f"Pool manifest mismatch for split=test: expected={expected_test} actual={actual_test} manifest={manifest_path}"
                )

    def _classify_error(self, message):
        text = (message or "").lower()
        if "dataloader worker" in text and "exited unexpectedly" in text:
            return "dataloader_worker_exit"
        if "out of memory" in text:
            return "out_of_memory"
        if "timed out" in text:
            return "timeout"
        if "selection failed" in text:
            return "selection_failed"
        return "unknown"

    def _sampler_audit(self):
        rollback_cfg = getattr(self, "rollback_config", None)
        if not isinstance(rollback_cfg, dict):
            rollback_cfg = {}
        ranking_meta = getattr(self, "_last_ranking_metadata", None)
        ranking_lambda_eff = (
            ranking_meta.get("lambda_effective")
            if isinstance(ranking_meta, dict)
            else None
        )
        ranking_lambda_source = (
            ranking_meta.get("lambda_source")
            if isinstance(ranking_meta, dict)
            else None
        )
        if ranking_lambda_eff is None:
            round_idx = getattr(self, "current_round", None)
            try:
                round_idx = int(round_idx) if round_idx is not None else None
            except Exception:
                round_idx = None

            def _match_round(ev) -> bool:
                if round_idx is None:
                    return True
                if not isinstance(ev, dict):
                    return False
                try:
                    return int(ev.get("round") or -1) == int(round_idx)
                except Exception:
                    return False

            inferred = None
            inferred_source = None
            try:
                for key, src in (
                    ("lambda_guard", "lambda_guard"),
                    ("selection_guardrail", "selection_guardrail"),
                    ("lambda_override", "lambda_override"),
                    ("lambda_policy_apply", "lambda_policy"),
                ):
                    ev = self._last_control_events.get(key)
                    if not (isinstance(ev, dict) and _match_round(ev)):
                        continue
                    cand = None
                    if key in ("lambda_guard",):
                        cand = ev.get("lambda_after")
                    elif key in ("selection_guardrail",):
                        cand = ev.get("lambda_after")
                    elif key in ("lambda_override", "lambda_policy_apply"):
                        cand = ev.get("applied")
                    if cand is None:
                        continue
                    inferred = float(cand)
                    inferred_source = str(src)
                    break
            except Exception:
                inferred = None
                inferred_source = None

            if inferred is None and hasattr(self, "toolbox"):
                try:
                    state = getattr(self.toolbox, "control_state", None)
                    if (
                        isinstance(state, dict)
                        and state.get("lambda_override_round") is not None
                    ):
                        inferred = float(state.get("lambda_override_round"))
                        inferred_source = "agent_control_state"
                except Exception:
                    inferred = None
                    inferred_source = None

            if inferred is not None:
                ranking_lambda_eff = inferred
                ranking_lambda_source = inferred_source
        lambda_ctrl = (
            self.exp_config.get("lambda_controller")
            if isinstance(getattr(self, "exp_config", None), dict)
            else None
        )
        lambda_mode = lambda_ctrl.get("mode") if isinstance(lambda_ctrl, dict) else None
        lambda_rule = lambda_ctrl.get("rule") if isinstance(lambda_ctrl, dict) else None
        lambda_policy = (
            self.exp_config.get("lambda_policy")
            if isinstance(getattr(self, "exp_config", None), dict)
            else None
        )
        lambda_policy_mode = (
            str(lambda_policy.get("mode"))
            if isinstance(lambda_policy, dict) and lambda_policy.get("mode") is not None
            else None
        )
        return {
            "sampler_type": getattr(self, "sampler_type", None),
            "sampler_class": self.sampler.__class__.__name__
            if hasattr(self, "sampler") and self.sampler is not None
            else None,
            "score_normalization": getattr(self, "score_normalization", None),
            "lambda_effective": ranking_lambda_eff,
            "lambda_source": ranking_lambda_source,
            "lambda_policy_mode": lambda_policy_mode,
            "lambda_controller_mode": lambda_mode,
            "lambda_controller_rule": lambda_rule,
            "rollback_mode": rollback_cfg.get("mode"),
            "rollback_threshold": rollback_cfg.get("threshold"),
            "rollback_std_factor": rollback_cfg.get("std_factor"),
        }

    def _clamp_lambda(self, value: float, min_v: float, max_v: float) -> float:
        low = float(min_v)
        high = float(max_v)
        if low > high:
            low, high = high, low
        return float(min(high, max(low, float(value))))

    def _apply_lambda_controller(self, round_idx: int | None) -> float | None:
        cfg = (
            self.exp_config.get("lambda_controller")
            if isinstance(self.exp_config, dict)
            else None
        )
        if not isinstance(cfg, dict):
            return None
        mode = str(cfg.get("mode") or "").strip().lower()
        if not mode:
            return None
        lmin = float(cfg.get("lambda_min", 0.0))
        lmax = float(cfg.get("lambda_max", 1.0))
        value = None

        def _mode_random():
            seed = int(self.seed) + int(round_idx or 0) * 9973
            rng = np.random.default_rng(seed)
            return float(rng.uniform(lmin, lmax))

        def _mode_rule_based():
            rule = str(cfg.get("rule") or "r1").strip().lower()
            base = cfg.get("lambda_init")
            prev = getattr(self, "_last_rule_lambda", None)
            if prev is None:
                prev = (
                    float(base)
                    if isinstance(base, (int, float))
                    else float((lmin + lmax) * 0.5)
                )

            state = self._last_training_state or {}
            rollback_flag = bool(state.get("rollback_flag"))
            miou_delta = state.get("miou_delta")
            step_up = float(cfg.get("step_up", 0.05))
            step_down = float(cfg.get("step_down", 0.1))
            threshold = float(cfg.get("miou_delta_threshold", 0.0))
            schedule = (
                cfg.get("schedule") if isinstance(cfg.get("schedule"), list) else None
            )

            def _rule_r1():
                return self._adjust_lambda_by_performance(
                    prev, miou_delta, rollback_flag, step_up, step_down, threshold
                )

            def _rule_r2():
                return self._schedule_lambda(schedule, round_idx, prev)

            def _rule_r3():
                base_val = self._schedule_lambda(schedule, round_idx, prev)
                return self._adjust_lambda_by_performance(
                    base_val, miou_delta, rollback_flag, step_up, step_down, threshold
                )

            rules = {"r1": _rule_r1, "r2": _rule_r2, "r3": _rule_r3}
            try:
                out = rules[rule]()
            except KeyError:
                out = _rule_r1()

            self._last_rule_lambda = float(out)
            return float(out)

        modes = {"random": _mode_random, "rule_based": _mode_rule_based}
        try:
            value = float(modes[mode]())
        except KeyError:
            return None
        value = None if value is None else self._clamp_lambda(value, lmin, lmax)
        self._last_lambda_controller = {
            "mode": mode,
            "lambda": value,
            "round": int(round_idx or 0),
        }
        if value is not None:
            self._append_trace(
                {
                    "type": "lambda_controller_apply",
                    "round": int(round_idx or 0),
                    "mode": mode,
                    "lambda": float(value),
                }
            )
        return value

    def _resolve_lambda_override(
        self, round_idx: int | None
    ) -> tuple[float | None, str | None]:
        exp_config = getattr(self, "exp_config", None)
        if not isinstance(exp_config, dict):
            return None, None

        exp_override = exp_config.get("lambda_override")
        if exp_override is not None:
            return float(exp_override), "exp_override"

        if bool(getattr(self, "use_agent", False)) and hasattr(self, "toolbox"):
            override = getattr(self.toolbox, "control_state", {}).get(
                "lambda_override_round"
            )
            if override is not None:
                return float(override), "agent_override"

            if isinstance(exp_config.get("lambda_policy"), dict):
                applied = self.toolbox.apply_round_lambda_policy()
                if applied is None:
                    raise RuntimeError(
                        "lambda_policy is configured but apply_round_lambda_policy returned None"
                    )
                return float(applied), "lambda_policy"

        if not bool(getattr(self, "use_agent", False)):
            if isinstance(exp_config.get("lambda_controller"), dict):
                value = self._apply_lambda_controller(round_idx)
                if value is not None:
                    return float(value), "lambda_controller"

        return None, None

    def _schedule_lambda(
        self, schedule, round_idx: int | None, fallback: float
    ) -> float:
        if not isinstance(schedule, list) or not schedule:
            return float(fallback)
        current = float(fallback)
        r = int(round_idx or 0)
        for item in schedule:
            if not isinstance(item, dict):
                continue
            rr = item.get("round")
            lv = item.get("lambda")
            if rr is None or lv is None:
                continue
            if int(rr) <= r:
                current = float(lv)
        return float(current)

    def _adjust_lambda_by_performance(
        self,
        base_value: float,
        miou_delta: float | None,
        rollback_flag: bool,
        step_up: float,
        step_down: float,
        threshold: float,
    ) -> float:
        value = float(base_value)
        if rollback_flag:
            return float(value - step_down)
        if miou_delta is None:
            return float(value)
        if float(miou_delta) < float(threshold):
            return float(value - step_down)
        return float(value + step_up)

    def _append_l3_selection(self, selected_ids, source: str | None = None):
        opts = getattr(getattr(self, "experiment_runtime", None), "trace_options", None)
        if opts is not None:
            if not bool(getattr(opts, "enable_l3_selection_logging", False)):
                return
            topk = int(getattr(opts, "l3_topk", 256) or 256)
            max_selected_opt = getattr(opts, "l3_max_selected", None)
            max_selected = (
                int(max_selected_opt) if max_selected_opt is not None else int(topk)
            )
        else:
            if not isinstance(self.exp_config, dict):
                return
            if not self.exp_config.get("enable_l3_selection_logging"):
                return
            topk = int(self.exp_config.get("l3_topk", 256) or 256)
            max_selected = int(self.exp_config.get("l3_max_selected", topk) or topk)
        ranked_items = list(getattr(self, "_last_ranked_items", []) or [])
        if not ranked_items:
            return
        top_items = ranked_items[: max(0, topk)]
        index = {str(item.get("sample_id")): item for item in top_items}
        selected_rows = []
        for sid in list(selected_ids or [])[: max(0, max_selected)]:
            key = str(sid)
            ref = index.get(key)
            if isinstance(ref, dict):
                selected_rows.append(
                    {
                        "sample_id": ref.get("sample_id"),
                        "final_score": ref.get("final_score"),
                        "uncertainty": ref.get("uncertainty"),
                        "knowledge_gain": ref.get("knowledge_gain"),
                        "lambda_t": ref.get("lambda_t"),
                    }
                )
            else:
                selected_rows.append({"sample_id": key})
        top_rows = []
        for item in top_items:
            top_rows.append(
                {
                    "sample_id": item.get("sample_id"),
                    "final_score": item.get("final_score"),
                    "uncertainty": item.get("uncertainty"),
                    "knowledge_gain": item.get("knowledge_gain"),
                    "lambda_t": item.get("lambda_t"),
                }
            )
        selected_u = []
        selected_k = []
        for row in selected_rows:
            if not isinstance(row, dict):
                continue
            if row.get("uncertainty") is not None:
                try:
                    selected_u.append(float(row.get("uncertainty")))
                except Exception:
                    pass
            if row.get("knowledge_gain") is not None:
                try:
                    selected_k.append(float(row.get("knowledge_gain")))
                except Exception:
                    pass
        top_u = []
        top_k = []
        for row in top_rows:
            if not isinstance(row, dict):
                continue
            if row.get("uncertainty") is not None:
                try:
                    top_u.append(float(row.get("uncertainty")))
                except Exception:
                    pass
            if row.get("knowledge_gain") is not None:
                try:
                    top_k.append(float(row.get("knowledge_gain")))
                except Exception:
                    pass
        u_median_selected = (
            float(np.median(np.asarray(selected_u, dtype=float)))
            if selected_u
            else None
        )
        k_median_selected = (
            float(np.median(np.asarray(selected_k, dtype=float)))
            if selected_k
            else None
        )
        u_median_top = (
            float(np.median(np.asarray(top_u, dtype=float))) if top_u else None
        )
        k_median_top = (
            float(np.median(np.asarray(top_k, dtype=float))) if top_k else None
        )
        selected_ids_list = [
            str(sid) for sid in list(selected_ids or [])[: max(0, max_selected)]
        ]
        top_ids_set = set(
            str(item.get("sample_id"))
            for item in top_rows
            if isinstance(item, dict) and item.get("sample_id") is not None
        )
        hit_count = int(sum(1 for sid in selected_ids_list if sid in top_ids_set))
        coverage_selected_in_top = (
            float(hit_count) / float(len(selected_ids_list))
            if int(len(selected_ids_list)) > 0
            else None
        )
        topk_stats_warning = None
        if (
            coverage_selected_in_top is not None
            and float(coverage_selected_in_top) < 0.5
        ):
            topk_stats_warning = "topk_stats_low_coverage_diagnostics_only"
        self._append_trace(
            {
                "type": "l3_selection",
                "round": int(self.current_round)
                if self.current_round is not None
                else None,
                "source": source,
                "topk": int(topk),
                "selected_limit": int(max_selected),
                "top_items": top_rows,
                "selected_items": selected_rows,
            }
        )
        self._append_trace(
            {
                "type": "l3_selection_stats",
                "round": int(self.current_round)
                if self.current_round is not None
                else None,
                "source": source,
                "topk": int(topk),
                "selected_limit": int(max_selected),
                "u_median_selected": u_median_selected,
                "k_median_selected": k_median_selected,
                "u_median_top": u_median_top,
                "k_median_top": k_median_top,
                "coverage_selected_in_top": coverage_selected_in_top,
                "topk_stats_for_closed_loop": False,
                "topk_stats_warning": topk_stats_warning,
            }
        )
        # P0: 保存到实例变量，供 training_state 使用
        self._last_u_median_selected = u_median_selected
        self._last_k_median_selected = k_median_selected
        self._last_u_median_top = u_median_top
        self._last_k_median_top = k_median_top

    def _append_score_snapshot(self, selected_ids, source: str | None = None):
        opts = getattr(getattr(self, "experiment_runtime", None), "trace_options", None)
        if opts is not None:
            if not bool(getattr(opts, "enable_score_snapshot_logging", False)):
                return
            boundary_window = int(
                getattr(opts, "score_snapshot_boundary_window", 64) or 64
            )
            max_pool_items_opt = getattr(opts, "score_snapshot_max_pool_items", None)
            max_pool_items = (
                None if max_pool_items_opt is None else int(max_pool_items_opt)
            )
        else:
            if not isinstance(self.exp_config, dict):
                return
            if not bool(
                self.exp_config.get(
                    "enable_score_snapshot_logging",
                    self.exp_config.get("enable_l3_selection_logging"),
                )
            ):
                return
            boundary_window = int(
                self.exp_config.get("score_snapshot_boundary_window", 64) or 64
            )
            max_pool_items_cfg = self.exp_config.get("score_snapshot_max_pool_items")
            max_pool_items = (
                None if max_pool_items_cfg is None else int(max_pool_items_cfg)
            )
        ranked_items = list(getattr(self, "_last_ranked_items", []) or [])
        if not ranked_items:
            return
        query_size = int(getattr(self.config, "QUERY_SIZE", 0) or 0)
        selected_id_set = {str(sid) for sid in list(selected_ids or [])}

        def _row(item, rank: int):
            sample_id = item.get("sample_id") if isinstance(item, dict) else None
            return {
                "rank": int(rank),
                "sample_id": sample_id,
                "final_score": item.get("final_score") if isinstance(item, dict) else None,
                "uncertainty": item.get("uncertainty") if isinstance(item, dict) else None,
                "knowledge_gain": item.get("knowledge_gain")
                if isinstance(item, dict)
                else None,
                "lambda_t": item.get("lambda_t") if isinstance(item, dict) else None,
                "selected": bool(str(sample_id) in selected_id_set),
            }

        pool_limit = len(ranked_items)
        if max_pool_items is not None and int(max_pool_items) > 0:
            pool_limit = min(len(ranked_items), int(max_pool_items))
        pool_rows = [_row(item, idx + 1) for idx, item in enumerate(ranked_items[:pool_limit])]

        boundary_center = min(max(query_size, 1), len(ranked_items))
        boundary_start = max(0, boundary_center - max(0, boundary_window))
        boundary_end = min(len(ranked_items), boundary_center + max(0, boundary_window))
        boundary_rows = [
            _row(item, idx + 1)
            for idx, item in enumerate(
                ranked_items[boundary_start:boundary_end], start=boundary_start
            )
        ]

        self._append_trace(
            {
                "type": "score_snapshot",
                "round": int(self.current_round)
                if self.current_round is not None
                else None,
                "source": source,
                "pool_n": int(len(ranked_items)),
                "query_size": int(query_size),
                "pool_rows_limit": int(pool_limit),
                "pool_rows_truncated": bool(pool_limit < len(ranked_items)),
                "boundary_window": int(boundary_window),
                "boundary_start_rank": int(boundary_start + 1)
                if boundary_rows
                else None,
                "boundary_end_rank": int(boundary_end) if boundary_rows else None,
                "rows": pool_rows,
                "boundary_rows": boundary_rows,
            }
        )

    def _append_round_summary(
        self, round_idx: int, best_miou: float, best_f1: float, labeled_size: int
    ):
        selection = (
            dict(self._last_selection_summary or {})
            if isinstance(self._last_selection_summary, dict)
            else None
        )
        ranking = (
            dict(self._last_ranking_metadata or {})
            if isinstance(self._last_ranking_metadata, dict)
            else None
        )
        lambda_ctrl = (
            dict(self._last_lambda_controller or {})
            if isinstance(self._last_lambda_controller, dict)
            else None
        )
        training_state = (
            dict(self._last_training_state or {})
            if isinstance(self._last_training_state, dict)
            else None
        )
        controls = {
            "lambda_policy_apply": None,
            "lambda_override": None,
            "lambda_guard": None,
            "selection_guardrail": None,
        }
        try:
            for k in list(controls.keys()):
                ev = self._last_control_events.get(k)
                if isinstance(ev, dict) and int(ev.get("round") or -1) == int(
                    round_idx
                ):
                    controls[k] = dict(ev)
        except Exception:
            pass
        self._append_trace(
            {
                "type": "round_summary",
                "round": int(round_idx),
                "labeled_size": int(labeled_size),
                "mIoU": float(best_miou),
                "f1": float(best_f1),
                "selection": selection,
                "ranking": ranking,
                "lambda_controller": lambda_ctrl,
                "training_state": training_state,
                "controls": controls,
                "sampler": self._sampler_audit(),
                "score_precalc_error": dict(self._last_score_precalc_error or {})
                if isinstance(getattr(self, "_last_score_precalc_error", None), dict)
                else None,
                "ranking_degraded": dict(self._last_ranking_degraded or {})
                if isinstance(getattr(self, "_last_ranking_degraded", None), dict)
                else None,
            }
        )

    def _compute_ranking_metadata(self, ranked, top_k: int):
        if not ranked:
            return None
        k = max(1, int(top_k))
        query_size = k

        def _extract(values, key: str, limit=None, allow_none: bool = False):
            out = []
            take = len(values) if limit is None else min(int(limit), len(values))
            for item in values[:take]:
                if not isinstance(item, dict):
                    continue
                v = item.get(key)
                if v is None:
                    if allow_none:
                        out.append(None)
                    continue
                try:
                    out.append(float(v))
                except Exception:
                    continue
            return out

        def _stats(vals, qts=(0.1, 0.25, 0.5, 0.75, 0.9, 0.99)):
            xs = [float(v) for v in (vals or []) if v is not None]
            if not xs:
                return None
            arr = np.asarray(xs, dtype=float)
            q = np.quantile(arr, np.asarray(list(qts), dtype=float)).tolist()
            q_map = {
                f"p{int(round(float(p) * 100)):02d}": float(v) for p, v in zip(qts, q)
            }
            return {
                "n": int(arr.size),
                "mean": float(arr.mean()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                **q_map,
            }

        top_u = _extract(ranked, "uncertainty", limit=k)
        top_kg = _extract(ranked, "knowledge_gain", limit=k)
        top_fs = _extract(ranked, "final_score", limit=k)
        pool_u = _extract(ranked, "uncertainty")
        pool_kg = _extract(ranked, "knowledge_gain")
        pool_fs = _extract(ranked, "final_score")
        pool_lambda = _extract(ranked, "lambda_t", limit=1)

        meta = {
            "pool_n": int(len(ranked)),
            "topk_n": int(k),
        }
        if pool_lambda:
            meta["lambda_effective"] = float(pool_lambda[0])
        if top_u:
            meta["avg_uncertainty"] = float(np.mean(np.asarray(top_u, dtype=float)))
        if top_kg:
            meta["avg_knowledge_gain"] = float(np.mean(np.asarray(top_kg, dtype=float)))
        if top_fs:
            meta["avg_final_score"] = float(np.mean(np.asarray(top_fs, dtype=float)))

        stats = {}
        u_top = _stats(top_u)
        u_pool = _stats(pool_u)
        if u_top or u_pool:
            stats["uncertainty"] = {"topk": u_top, "pool": u_pool}
        k_top = _stats(top_kg)
        k_pool = _stats(pool_kg)
        if k_top or k_pool:
            stats["knowledge_gain"] = {"topk": k_top, "pool": k_pool}
        s_top = _stats(top_fs)
        s_pool = _stats(pool_fs)
        if s_top or s_pool:
            stats["final_score"] = {"topk": s_top, "pool": s_pool}
        if stats:
            meta["score_stats"] = stats

        if pool_u and pool_kg and len(pool_u) == len(pool_kg):
            u_arr = np.asarray(pool_u, dtype=float)
            k_arr = np.asarray(pool_kg, dtype=float)
            pool_n = int(len(u_arr))
            boundary_ratio = 0.2
            sensitivity_delta = 0.1
            if isinstance(getattr(self, "exp_config", None), dict):
                try:
                    boundary_ratio = float(
                        self.exp_config.get("geometry_boundary_delta_ratio", 0.2)
                        or 0.2
                    )
                except Exception:
                    boundary_ratio = 0.2
                try:
                    sensitivity_delta = float(
                        self.exp_config.get(
                            "geometry_sensitivity_delta_lambda", 0.1
                        )
                        or 0.1
                    )
                except Exception:
                    sensitivity_delta = 0.1
            boundary_half_width = max(1, int(np.ceil(float(query_size) * float(boundary_ratio))))
            boundary_start = max(0, int(query_size) - boundary_half_width)
            boundary_end = min(pool_n, int(query_size) + boundary_half_width)
            boundary_u = u_arr[boundary_start:boundary_end]
            boundary_k = k_arr[boundary_start:boundary_end]
            geometry = {
                "boundary_delta_ratio": float(boundary_ratio),
                "boundary_half_width": int(boundary_half_width),
                "boundary_start_rank": int(boundary_start + 1)
                if boundary_end > boundary_start
                else None,
                "boundary_end_rank": int(boundary_end)
                if boundary_end > boundary_start
                else None,
                "boundary_n": int(max(0, boundary_end - boundary_start)),
                "boundary_u_std": float(np.std(boundary_u))
                if boundary_u.size > 0
                else None,
                "boundary_k_std": float(np.std(boundary_k))
                if boundary_k.size > 0
                else None,
            }
            try:
                from scipy.stats import spearmanr

                rho_uk, pval = spearmanr(u_arr, k_arr)
                geometry["spearman_rho_uk"] = (
                    float(rho_uk) if rho_uk is not None and not np.isnan(rho_uk) else None
                )
                geometry["spearman_pvalue_uk"] = (
                    float(pval) if pval is not None and not np.isnan(pval) else None
                )
            except Exception:
                geometry["spearman_rho_uk"] = None
                geometry["spearman_pvalue_uk"] = None

            lambda_current = None
            if pool_lambda:
                try:
                    lambda_current = float(pool_lambda[0])
                except Exception:
                    lambda_current = None
            if lambda_current is not None and pool_n > 0:
                current_scores = (
                    (1.0 - float(lambda_current)) * u_arr
                    + float(lambda_current) * k_arr
                )
                current_order = np.argsort(-current_scores)
                current_top = set(
                    int(idx) for idx in current_order[: min(int(query_size), pool_n)]
                )
                current_rank = np.empty(pool_n, dtype=int)
                current_rank[current_order] = np.arange(pool_n, dtype=int)

                def _jaccard_metrics(lambda_shifted: float):
                    shifted_scores = (
                        (1.0 - float(lambda_shifted)) * u_arr
                        + float(lambda_shifted) * k_arr
                    )
                    shifted_order = np.argsort(-shifted_scores)
                    shifted_top = set(
                        int(idx)
                        for idx in shifted_order[: min(int(query_size), pool_n)]
                    )
                    union_n = len(current_top | shifted_top)
                    overlap_n = len(current_top & shifted_top)
                    shifted_rank = np.empty(pool_n, dtype=int)
                    shifted_rank[shifted_order] = np.arange(pool_n, dtype=int)
                    boundary_index = current_order[boundary_start:boundary_end]
                    boundary_pairs_flipped = 0
                    boundary_pair_total = 0
                    if boundary_index.size >= 2:
                        for i_pos in range(int(boundary_index.size)):
                            ii = int(boundary_index[i_pos])
                            for j_pos in range(i_pos + 1, int(boundary_index.size)):
                                jj = int(boundary_index[j_pos])
                                current_lt = int(current_rank[ii]) < int(current_rank[jj])
                                shifted_lt = int(shifted_rank[ii]) < int(shifted_rank[jj])
                                boundary_pair_total += 1
                                if current_lt != shifted_lt:
                                    boundary_pairs_flipped += 1
                    return {
                        "lambda": float(lambda_shifted),
                        "overlap": float(overlap_n) / float(union_n)
                        if union_n > 0
                        else None,
                        "distance": 1.0 - (float(overlap_n) / float(union_n))
                        if union_n > 0
                        else None,
                        "replacements": int(min(int(query_size), pool_n) - overlap_n),
                        "boundary_pairs_flipped": int(boundary_pairs_flipped),
                        "boundary_pair_total": int(boundary_pair_total),
                        "crossing_density": float(boundary_pairs_flipped)
                        / float(boundary_pair_total)
                        if boundary_pair_total > 0
                        else None,
                    }

                lambda_up = min(1.0, lambda_current + float(sensitivity_delta))
                lambda_down = max(0.0, lambda_current - float(sensitivity_delta))
                geometry["sensitivity_delta_lambda"] = float(sensitivity_delta)

                up_metrics = (
                    _jaccard_metrics(lambda_up)
                    if abs(lambda_up - lambda_current) > 1e-12
                    else None
                )
                down_metrics = (
                    _jaccard_metrics(lambda_down)
                    if abs(lambda_down - lambda_current) > 1e-12
                    else None
                )
                if up_metrics is not None:
                    geometry["lambda_probe_up"] = float(up_metrics["lambda"])
                    geometry["sens_up_overlap"] = up_metrics["overlap"]
                    geometry["sens_up"] = up_metrics["distance"]
                    geometry["topk_replacements_up"] = up_metrics["replacements"]
                if down_metrics is not None:
                    geometry["lambda_probe_down"] = float(down_metrics["lambda"])
                    geometry["sens_down_overlap"] = down_metrics["overlap"]
                    geometry["sens_down"] = down_metrics["distance"]
                    geometry["topk_replacements_down"] = down_metrics["replacements"]
                if up_metrics is not None and down_metrics is not None:
                    sens_down = down_metrics["distance"]
                    sens_up = up_metrics["distance"]
                    geometry["asymmetry_ratio"] = (
                        float(sens_up) / float(sens_down)
                        if sens_down is not None and abs(float(sens_down)) > 1e-12
                        else None
                    )
                crossing_source = up_metrics if up_metrics is not None else down_metrics
                if crossing_source is not None:
                    geometry["boundary_pairs_flipped"] = crossing_source[
                        "boundary_pairs_flipped"
                    ]
                    geometry["boundary_pair_total"] = crossing_source[
                        "boundary_pair_total"
                    ]
                    geometry["crossing_density"] = crossing_source["crossing_density"]
                if up_metrics is not None:
                    geometry["local_jaccard_overlap"] = up_metrics["overlap"]
                    geometry["local_jaccard_distance"] = up_metrics["distance"]
            meta["selection_geometry"] = geometry
        return meta or None

    def _deterministic_hash_order(self, indices, salt: str):
        def _key(x):
            try:
                v = int(x)
            except Exception:
                v = str(x)
            return hashlib.sha256(f"{salt}:{v}".encode("utf-8")).hexdigest()

        return sorted(list(indices or []), key=_key)

    def _append_md(self, log_path, content):
        if not log_path:
            return
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(content)

    def _round_checkpoint_path(self, round_idx: int):
        return os.path.join(
            self.round_model_dir, f"round_{int(round_idx):02d}_best_val.pt"
        )

    def _save_round_best_val_model(
        self, round_idx: int, state_dict: dict, metadata: dict
    ):
        os.makedirs(self.round_model_dir, exist_ok=True)
        path = self._round_checkpoint_path(round_idx)
        tmp_path = f"{path}.tmp"
        payload = {
            "state_dict": state_dict,
            "metadata": dict(metadata or {}),
        }
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
        return path

    def _round_model_retention(self) -> tuple[str, int]:
        mode = None
        keep_last_n = None
        if isinstance(getattr(self, "exp_config", None), dict):
            v = self.exp_config.get("round_model_retention")
            if v is not None:
                mode = str(v)
            v = self.exp_config.get("round_model_keep_last_n")
            if v is not None:
                try:
                    keep_last_n = int(v)
                except Exception:
                    keep_last_n = None
        if mode is None:
            try:
                mode = str(getattr(self.config, "ROUND_MODEL_RETENTION", None) or "")
            except Exception:
                mode = ""
        mode = str(mode or "").strip().lower()
        if not mode:
            mode = "all"
        if keep_last_n is None:
            try:
                keep_last_n = int(
                    getattr(self.config, "ROUND_MODEL_KEEP_LAST_N", 0) or 0
                )
            except Exception:
                keep_last_n = 0

        if mode in ("final_only", "latest_only", "last_only"):
            return "last_n", 1
        if mode in ("last_n", "keep_last_n"):
            k = int(keep_last_n or 0)
            return "last_n", max(1, k)
        return "all", 0

    def _prune_round_best_val_models(self, current_round: int) -> None:
        mode, keep_n = self._round_model_retention()
        if mode != "last_n":
            return
        keep_n = int(keep_n or 0)
        if keep_n <= 0:
            return
        if not self.round_model_dir or not os.path.isdir(self.round_model_dir):
            return
        lo = int(max(1, int(current_round) - int(keep_n) + 1))
        hi = int(current_round)
        try:
            for name in os.listdir(self.round_model_dir):
                if not (name.startswith("round_") and name.endswith("_best_val.pt")):
                    if name.startswith("round_") and name.endswith("_best_val.pt.tmp"):
                        try:
                            os.remove(os.path.join(self.round_model_dir, name))
                        except Exception:
                            pass
                    continue
                try:
                    round_part = name.split("_", 2)[1]
                    r = int(round_part)
                except Exception:
                    continue
                if r < lo or r > hi:
                    try:
                        os.remove(os.path.join(self.round_model_dir, name))
                    except Exception:
                        pass
        except Exception:
            return

    def _parse_log_resume_state(self, log_path):
        if not log_path or not os.path.exists(log_path):
            return None, None, None
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            return None, None, None

        if "## 实验汇总" in content:
            return None, None, None

        import re

        rounds = re.findall(r"## Round (\d+)", content)
        if not rounds:
            return None, None, None

        last_round = int(rounds[-1])

        labeled_sizes = re.findall(r"Labeled Pool Size: (\d+)", content)
        if not labeled_sizes:
            return None, None, None

        last_labeled_size = int(labeled_sizes[-1])

        performance_history = []
        miou_matches = re.findall(
            r"Round=(\d+), Labeled=(\d+), mIoU=([\d.]+), F1=([\d.]+)", content
        )
        for match in miou_matches:
            round_num, labeled_size, miou, f1 = match
            performance_history.append(
                {
                    "round": int(round_num),
                    "mIoU": float(miou),
                    "f1_score": float(f1),
                    "labeled_size": int(labeled_size),
                }
            )

        return last_round, last_labeled_size, performance_history

    def run_and_collect(self, log_path=None):
        performance_history = []
        budget_history = []
        best_miou_so_far = 0.0
        final_report = None
        test_split = None

        start_mode = getattr(self.config, "START_MODE", "resume")

        dataset_fingerprint = self._dataset_fingerprint()
        self._write_status({"dataset_fingerprint": dataset_fingerprint})
        self._append_trace(
            {"type": "dataset_fingerprint", "fingerprint": dataset_fingerprint}
        )

        manifest_path = os.path.join(self.run_reports_dir, "manifest.json")
        if os.path.exists(manifest_path):
            try:
                import json

                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                expected_fp = (
                    manifest.get("dataset_fingerprint")
                    if isinstance(manifest, dict)
                    else None
                )
                if (
                    isinstance(expected_fp, dict)
                    and expected_fp.get("sha256")
                    and expected_fp != dataset_fingerprint
                ):
                    if getattr(self.config, "STRICT_RESUME", False):
                        raise RuntimeError(
                            f"Dataset fingerprint mismatch: expected={expected_fp} actual={dataset_fingerprint}"
                        )
                    self._write_status(
                        {
                            "dataset_fingerprint_mismatch": {
                                "expected": expected_fp,
                                "actual": dataset_fingerprint,
                            }
                        }
                    )

                if start_mode != "fresh" and isinstance(manifest, dict):
                    cfg_snapshot = manifest.get("config")
                    if isinstance(cfg_snapshot, dict):
                        manifest_exempt_keys = set()
                        try:
                            if (
                                isinstance(getattr(self, "exp_config", None), dict)
                                and self.exp_config.get("epochs_per_round_override")
                                is not None
                            ):
                                manifest_exempt_keys.add("EPOCHS_PER_ROUND")
                        except Exception:
                            manifest_exempt_keys = set()
                        tracked_keys = (
                            "RANDOM_SEED",
                            "DETERMINISTIC",
                            "ESTIMATED_TOTAL_SAMPLES",
                            "BUDGET_RATIO",
                            "TOTAL_BUDGET",
                            "INITIAL_LABELED_SIZE",
                            "TRAIN_SPLIT",
                            "VAL_SPLIT",
                            "TEST_SPLIT",
                            "MODEL_SELECTION",
                            "N_ROUNDS",
                            "QUERY_SIZE",
                            "EPOCHS_PER_ROUND",
                            "BATCH_SIZE",
                            "LR",
                            "ALPHA",
                        )
                        mismatches = {}
                        for key in tracked_keys:
                            if key in manifest_exempt_keys:
                                continue
                            if key not in cfg_snapshot:
                                continue
                            value = cfg_snapshot.get(key)
                            if value is None:
                                continue
                            current = getattr(self.config, key, None)
                            if current is not None and current != value:
                                mismatches[key] = {
                                    "current": current,
                                    "manifest": value,
                                }
                            setattr(self.config, key, value)
                        if mismatches:
                            self._write_status({"manifest_config_mismatch": mismatches})
                            self._append_trace(
                                {
                                    "type": "manifest_config_mismatch",
                                    "mismatches": mismatches,
                                }
                            )
                            if getattr(self.config, "STRICT_RESUME", False):
                                raise RuntimeError(
                                    f"Manifest config mismatch: {mismatches}"
                                )
                        applied = {
                            k: cfg_snapshot.get(k)
                            for k in tracked_keys
                            if k in cfg_snapshot
                        }
                        self._append_trace(
                            {"type": "manifest_config_applied", "config": applied}
                        )
            except Exception as e:
                if getattr(self.config, "STRICT_RESUME", False):
                    raise
                self._write_status({"manifest_read_error": str(e)})

        if start_mode == "fresh":
            if log_path:
                try:
                    with open(log_path, "w", encoding="utf-8") as f:
                        f.write(
                            f"# 实验日志\n\n实验名称: {self.experiment_name}\n描述: {self.exp_config['description']}\n开始时间: {datetime.now().isoformat()}\n\n"
                        )
                except Exception:
                    pass
            start_round = 0
        else:
            state = self.checkpoint_manager.load()
            if state:
                performance_history = state.get("performance_history", [])
                budget_history = state.get("budget_history", [])
                start_round = state.get("round", 0)

                logger.info(f"Resumed from checkpoint: Round {start_round}")

                # 1. Rollback Pools if necessary (Atomic Guarantee)
                target_size = state.get(
                    "labeled_size",
                    len(self.labeled_indices)
                    if hasattr(self, "labeled_indices")
                    else 0,
                )
                # Note: self.labeled_indices might not be inited yet if we don't load pools first.
                # Actually we call _load_pool_states later.
                # But _rollback_pools reads CSV directly.

                # Check if we need to rollback before loading pools?
                # Actually, _rollback_pools writes CSVs.
                if not self._rollback_pools(target_size):
                    if getattr(self.config, "STRICT_RESUME", False):
                        logger.error("Pool rollback failed.")
                        # If rollback failed (e.g. pool files missing), we rely on _load_pool_states to check.

                # 2. Truncate Trace (Idempotency Guarantee)
                self._truncate_trace(start_round)

                # 3. Restore RNG States (Reproducibility Guarantee)
                self._set_rng_states(state.get("rng_states", {}))

                if self.use_agent and self.agent_manager:
                    # Restore pending controls
                    self._pending_round_controls = state.get(
                        "pending_round_controls", {}
                    )

                fixed_epochs = bool(getattr(self.config, "FIX_EPOCHS_PER_ROUND", False))
                if fixed_epochs and isinstance(self._pending_round_controls, dict):
                    self._pending_round_controls.pop("epochs_round", None)

                loaded = self._load_pool_states()
                if not loaded:
                    if getattr(self.config, "STRICT_RESUME", False):
                        raise RuntimeError("Checkpoint found but pool files missing")
                    logger.warning(
                        "Checkpoint found but pool files missing! State might be inconsistent."
                    )
                else:
                    self._assert_pool_integrity()

                if log_path:
                    self._append_md(
                        log_path,
                        f"\n--- [Checkpoint] 续跑开始时间: {datetime.now().isoformat()} ---\n\n",
                    )
            else:
                if log_path:
                    # Legacy log parsing
                    last_round, last_labeled_size, log_performance_history = (
                        self._parse_log_resume_state(log_path)
                    )
                    if last_round is not None:
                        performance_history = log_performance_history
                        budget_history = [
                            p["labeled_size"] for p in performance_history
                        ]
                        logger.info(
                            f"Resuming from Round {last_round}, Labeled Pool Size: {last_labeled_size}"
                        )
                        loaded = self._load_pool_states()
                        if not loaded:
                            if getattr(self.config, "STRICT_RESUME", False):
                                raise RuntimeError(
                                    "Legacy log resume requested but pool files missing"
                                )
                            logger.warning(
                                "Failed to load pool states, using current state"
                            )
                        else:
                            self._assert_pool_integrity()
                        self._append_md(
                            log_path,
                            f"\n--- [Legacy Log] 续跑开始时间: {datetime.now().isoformat()} ---\n\n",
                        )
                    else:
                        self._append_md(
                            log_path,
                            f"# 实验日志\n\n实验名称: {self.experiment_name}\n描述: {self.exp_config['description']}\n开始时间: {datetime.now().isoformat()}\n\n",
                        )
                start_round = 0
                if performance_history:
                    start_round = performance_history[-1]["round"]

        self._write_status(
            {"status": "running", "resume": {"start_round": int(start_round)}}
        )

        for round_idx in range(start_round, self.config.N_ROUNDS):
            self.current_round = round_idx + 1
            self._selection_context = None
            self._last_score_precalc_error = None
            self._last_ranking_degraded = None
            self._last_ranking_metadata = None
            self._last_ranked_items = None
            self._last_selection_summary = None
            self._last_lambda_controller = None

            fixed_epochs = bool(getattr(self.config, "FIX_EPOCHS_PER_ROUND", False))
            epochs_schedule = getattr(self.config, "EPOCHS_PER_ROUND_SCHEDULE", None)
            scheduled_epochs = None
            if (
                (not fixed_epochs)
                and isinstance(epochs_schedule, (list, tuple))
                and round_idx < len(epochs_schedule)
            ):
                scheduled_epochs = epochs_schedule[round_idx]

            self.config.QUERY_SIZE = int(self.default_query_size)
            self.config.EPOCHS_PER_ROUND = int(self.default_epochs_per_round)
            if (not fixed_epochs) and scheduled_epochs is not None:
                try:
                    self.config.EPOCHS_PER_ROUND = int(scheduled_epochs)
                except Exception:
                    self.config.EPOCHS_PER_ROUND = int(self.default_epochs_per_round)

            if self.use_agent and self.agent_manager:
                self.toolbox.reset_round_controls()
                pending_epochs = None
                if (not fixed_epochs) and isinstance(
                    self._pending_round_controls, dict
                ):
                    pending_epochs = self._pending_round_controls.pop(
                        "epochs_round", None
                    )
                allow_epoch_control = bool(
                    getattr(self.toolbox, "control_permissions", {}).get(
                        "set_epochs_per_round", False
                    )
                )
                if (
                    (not fixed_epochs)
                    and allow_epoch_control
                    and pending_epochs is not None
                ):
                    self.config.EPOCHS_PER_ROUND = int(pending_epochs)
                    self.toolbox.control_state["epochs_round"] = int(pending_epochs)
            logger.info(f"=== Round {round_idx + 1}/{self.config.N_ROUNDS} ===")
            logger.info(f"Labeled Pool Size: {len(self.labeled_indices)}")
            self._append_md(
                log_path,
                f"## Round {round_idx + 1}\n\nLabeled Pool Size: {len(self.labeled_indices)}\n\n",
            )

            try:
                early_stop = False
                # Unified Seeding: Ensure each round has a deterministic but distinct seed
                current_seed = self.seed + int(round_idx + 1)
                set_global_seed(current_seed, deterministic=self.deterministic)

                # Generator for DataLoader shuffling
                g = torch.Generator()
                g.manual_seed(current_seed)

                model = LandslideDeepLabV3(
                    in_channels=self.config.IN_CHANNELS, classes=self.config.NUM_CLASSES
                )
                trainer = Trainer(
                    model, self.config, self.config.DEVICE, training_state=self.training_state
                )

                grad_probe_source = "official_val"
                grad_probe_holdout_frac = 0.1
                grad_probe_holdout_min = int(getattr(self.config, "BATCH_SIZE", 1) or 1)
                if isinstance(getattr(self, "exp_config", None), dict):
                    v = self.exp_config.get("grad_probe_source")
                    if v is not None:
                        grad_probe_source = str(v)
                    v = self.exp_config.get("grad_probe_holdout_frac")
                    if v is not None:
                        try:
                            grad_probe_holdout_frac = float(v)
                        except Exception:
                            pass
                    v = self.exp_config.get("grad_probe_holdout_min")
                    if v is not None:
                        try:
                            grad_probe_holdout_min = int(v)
                        except Exception:
                            pass
                grad_probe_source = (
                    str(grad_probe_source or "official_val").strip().lower()
                )

                labeled_ids_train = list(self.labeled_indices)
                labeled_ids_probe = []
                if (
                    bool(getattr(self.config, "GRAD_LOG_VAL_ALIGNMENT", False))
                    and grad_probe_source == "train_holdout"
                ):
                    train_ids, probe_ids = self._split_labeled_indices_for_grad_probe(
                        list(self.labeled_indices),
                        frac=float(grad_probe_holdout_frac),
                        min_count=int(grad_probe_holdout_min),
                        seed=int(self.seed),
                    )
                    if train_ids and probe_ids:
                        labeled_ids_train = list(train_ids)
                        labeled_ids_probe = list(probe_ids)

                labeled_subset = Subset(self.full_dataset, labeled_ids_train)

                train_num_workers = int(getattr(self.config, "NUM_WORKERS", 0) or 0)
                train_loader_kwargs = self._build_loader_kwargs(
                    batch_size=self.config.BATCH_SIZE,
                    shuffle=True,
                    num_workers=train_num_workers,
                    generator=g,
                    worker_init_fn=worker_init_fn,
                    drop_last=True,
                    pin_memory=bool(getattr(self.config, "PIN_MEMORY", False)),
                    persistent_workers=bool(
                        getattr(self.config, "PERSISTENT_WORKERS", False)
                    ),
                    prefetch_factor=int(
                        getattr(self.config, "PREFETCH_FACTOR", 2) or 2
                    ),
                )
                labeled_loader = DataLoader(labeled_subset, **train_loader_kwargs)

                probe_loader = None
                if labeled_ids_probe:
                    probe_subset = Subset(self.full_dataset, labeled_ids_probe)
                    probe_loader_kwargs = self._build_loader_kwargs(
                        batch_size=self.config.BATCH_SIZE,
                        shuffle=False,
                        num_workers=train_num_workers,
                        generator=g,
                        worker_init_fn=worker_init_fn,
                        drop_last=False,
                        pin_memory=bool(getattr(self.config, "PIN_MEMORY", False)),
                        persistent_workers=bool(
                            getattr(self.config, "PERSISTENT_WORKERS", False)
                        ),
                        prefetch_factor=int(
                            getattr(self.config, "PREFETCH_FACTOR", 2) or 2
                        ),
                    )
                    probe_loader = DataLoader(probe_subset, **probe_loader_kwargs)

                val_dataset = Landslide4SenseDataset(
                    self.config.DATA_DIR, split=self.val_split
                )
                val_loader_kwargs = self._build_loader_kwargs(
                    batch_size=self.config.BATCH_SIZE,
                    shuffle=False,
                    num_workers=train_num_workers,
                    generator=g,
                    worker_init_fn=worker_init_fn,
                    drop_last=False,
                    pin_memory=bool(getattr(self.config, "PIN_MEMORY", False)),
                    persistent_workers=bool(
                        getattr(self.config, "PERSISTENT_WORKERS", False)
                    ),
                    prefetch_factor=int(
                        getattr(self.config, "PREFETCH_FACTOR", 2) or 2
                    ),
                )
                val_loader = DataLoader(val_dataset, **val_loader_kwargs)
                val_eval_source = (
                    "official_val" if str(self.val_split) == "val" else "val"
                )

                model_selection = (
                    str(
                        getattr(self.config, "MODEL_SELECTION", "last_epoch")
                        or "last_epoch"
                    )
                    .strip()
                    .lower()
                )
                if model_selection not in ("last_epoch", "best_val"):
                    raise ValueError(
                        f"Unsupported MODEL_SELECTION={model_selection}. Expected 'last_epoch' or 'best_val'."
                    )
                is_test_only_round = bool(round_idx == self.config.N_ROUNDS - 1)
                selected_from_round = int(round_idx + 1)

                best_miou = 0.0
                best_f1 = 0.0
                best_epoch = None
                best_state_dict = None
                last_miou_epoch = 0.0
                last_f1_epoch = 0.0
                epoch_mious = []
                grad_tvc_values = []
                grad_probe_loader = None
                if bool(getattr(self.config, "GRAD_LOG_VAL_ALIGNMENT", False)):
                    if grad_probe_source == "official_val":
                        grad_probe_loader = val_loader
                    elif grad_probe_source == "train_holdout":
                        grad_probe_loader = probe_loader
                if not is_test_only_round:
                    for epoch in range(self.config.EPOCHS_PER_ROUND):
                        grad = None
                        out = trainer.train_one_epoch(
                            labeled_loader,
                            grad_probe_loader=grad_probe_loader,
                        )

                        if isinstance(out, tuple) and len(out) == 2:
                            loss, grad = out
                        else:
                            loss = out

                        if (
                            isinstance(grad, dict)
                            and grad.get("train_val_cos") is not None
                        ):
                            grad_tvc_values.append(float(grad.get("train_val_cos")))
                        metrics = trainer.evaluate(val_loader)
                        last_miou_epoch = float(metrics.get("mIoU", 0.0))
                        last_f1_epoch = float(metrics.get("f1_score", 0.0))
                        epoch_mious.append(float(last_miou_epoch))
                        logger.info(
                            f"Epoch {epoch + 1}: Loss={loss:.4f}, mIoU={metrics['mIoU']:.4f}, F1={metrics['f1_score']:.4f}"
                        )
                        self._append_md(
                            log_path,
                            f"- Epoch {epoch + 1}: Loss={loss:.4f}, mIoU={metrics['mIoU']:.4f}, F1={metrics['f1_score']:.4f}\n",
                        )
                        if metrics["mIoU"] > best_miou:
                            best_miou = metrics["mIoU"]
                            best_f1 = metrics["f1_score"]
                            best_epoch = int(epoch + 1)
                            best_state_dict = {
                                k: v.detach().cpu().clone()
                                for k, v in trainer.model.state_dict().items()
                            }

                        warnings = []
                        import math

                        if not math.isfinite(float(loss)):
                            warnings.append("loss_not_finite")
                        if not math.isfinite(float(metrics.get("mIoU", 0.0))):
                            warnings.append("miou_not_finite")
                        if not math.isfinite(float(metrics.get("f1_score", 0.0))):
                            warnings.append("f1_not_finite")
                        if warnings and bool(
                            getattr(self.config, "FAIL_ON_NONFINITE", True)
                        ):
                            raise RuntimeError(
                                f"Non-finite metrics detected at round={int(round_idx + 1)} epoch={int(epoch + 1)} warnings={warnings}"
                            )

                        self._write_status(
                            {
                                "status": "running",
                                "progress": {
                                    "round": int(round_idx + 1),
                                    "epoch": int(epoch + 1),
                                    "labeled_size": int(len(self.labeled_indices)),
                                    "loss": float(loss),
                                    "mIoU": float(metrics["mIoU"]),
                                    "f1": float(metrics["f1_score"]),
                                    "best_mIoU_round": float(best_miou),
                                    "warnings": warnings,
                                },
                            }
                        )
                        self._append_trace(
                            {
                                "type": "epoch_end",
                                "round": int(round_idx + 1),
                                "epoch": int(epoch + 1),
                                "labeled_size": int(len(self.labeled_indices)),
                                "loss": float(loss),
                                "mIoU": float(metrics["mIoU"]),
                                "f1": float(metrics["f1_score"]),
                                "eval_source": str(val_eval_source),
                                "eval_split": str(self.val_split),
                                "eval_img_dir": getattr(val_dataset, "img_dir", None),
                                "eval_mask_dir": getattr(val_dataset, "mask_dir", None),
                                "warnings": warnings,
                                "grad": grad,
                            }
                        )

                    selected_epoch = int(self.config.EPOCHS_PER_ROUND)
                    selected_miou = float(last_miou_epoch)
                    selected_f1 = float(last_f1_epoch)
                    if best_state_dict is None or best_epoch is None:
                        raise RuntimeError(
                            "No best validation checkpoint was captured in this round"
                        )
                    if model_selection == "best_val":
                        trainer.model.load_state_dict(best_state_dict, strict=True)
                        selected_epoch = int(best_epoch)
                        selected_miou = float(best_miou)
                        selected_f1 = float(best_f1)

                    round_best_ckpt_path = self._save_round_best_val_model(
                        int(round_idx + 1),
                        best_state_dict,
                        {
                            "run_id": str(self.run_id),
                            "experiment_name": str(self.experiment_name),
                            "round": int(round_idx + 1),
                            "labeled_size": int(len(self.labeled_indices)),
                            "best_epoch": int(best_epoch),
                            "best_miou": float(best_miou),
                            "best_f1": float(best_f1),
                            "selected_epoch": int(selected_epoch),
                            "selected_miou": float(selected_miou),
                            "selected_f1": float(selected_f1),
                            "model_selection": str(model_selection),
                            "eval_source": str(val_eval_source),
                            "eval_split": str(self.val_split),
                        },
                    )
                    self._prune_round_best_val_models(int(round_idx + 1))
                    self._append_trace(
                        {
                            "type": "round_best_val_checkpoint",
                            "round": int(round_idx + 1),
                            "path": str(round_best_ckpt_path),
                            "best_epoch": int(best_epoch),
                            "best_miou": float(best_miou),
                            "best_f1": float(best_f1),
                        }
                    )
                else:
                    previous_round = int(round_idx)
                    if previous_round <= 0:
                        raise RuntimeError(
                            "Final test-only round requires previous training round"
                        )
                    previous_round_path = self._round_checkpoint_path(previous_round)
                    if not os.path.exists(previous_round_path):
                        raise RuntimeError(
                            f"Missing previous round best checkpoint for final test-only round: {previous_round_path}"
                        )
                    payload = torch.load(previous_round_path, map_location="cpu")
                    if not isinstance(payload, dict) or "state_dict" not in payload:
                        raise RuntimeError(
                            f"Invalid previous round checkpoint payload: {previous_round_path}"
                        )
                    previous_state = payload.get("state_dict")
                    if not isinstance(previous_state, dict):
                        raise RuntimeError(
                            f"Invalid state_dict in previous round checkpoint: {previous_round_path}"
                        )
                    trainer.model.load_state_dict(previous_state, strict=True)
                    meta = payload.get("metadata")
                    if not isinstance(meta, dict):
                        meta = {}
                    selected_from_round = int(previous_round)
                    model_selection = "prev_round_best_val"
                    selected_epoch = int(
                        meta.get("best_epoch")
                        or meta.get("selected_epoch")
                        or int(getattr(self.config, "EPOCHS_PER_ROUND", 0) or 0)
                    )
                    selected_miou = float(
                        meta.get("best_miou")
                        if meta.get("best_miou") is not None
                        else meta.get("selected_miou", 0.0)
                    )
                    selected_f1 = float(
                        meta.get("best_f1")
                        if meta.get("best_f1") is not None
                        else meta.get("selected_f1", 0.0)
                    )
                    best_epoch = int(selected_epoch)
                    best_miou = float(selected_miou)
                    best_f1 = float(selected_f1)
                    best_state_dict = previous_state
                    round_best_ckpt_path = str(previous_round_path)
                    logger.info(
                        f"Round {round_idx + 1} uses test-only mode: load Round {selected_from_round} best_val checkpoint "
                        f"(epoch={selected_epoch}, val_mIoU={selected_miou:.4f}, val_F1={selected_f1:.4f})"
                    )
                    self._append_md(
                        log_path,
                        f"- Test-only mode: Round {round_idx + 1} loads Round {selected_from_round} best_val checkpoint "
                        f"(epoch={selected_epoch}, val_mIoU={selected_miou:.4f}, val_F1={selected_f1:.4f})\n",
                    )
                    self._append_trace(
                        {
                            "type": "final_round_test_only_load",
                            "round": int(round_idx + 1),
                            "selected_from_round": int(selected_from_round),
                            "path": str(previous_round_path),
                            "selected_epoch": int(selected_epoch),
                            "selected_miou": float(selected_miou),
                            "selected_f1": float(selected_f1),
                        }
                    )

                performance_history.append(
                    {
                        "round": round_idx + 1,
                        "mIoU": float(selected_miou),
                        "f1_score": float(selected_f1),
                        "labeled_size": len(self.labeled_indices),
                        "model_selection": str(model_selection),
                        "selected_epoch": int(selected_epoch),
                        "selected_from_round": int(selected_from_round),
                        "best_val_epoch": int(best_epoch),
                        "best_val_mIoU": float(best_miou),
                        "best_val_f1": float(best_f1),
                        "best_val_checkpoint": str(round_best_ckpt_path),
                    }
                )
                if float(selected_miou) > best_miou_so_far:
                    best_miou_so_far = float(selected_miou)
                budget_history.append(len(self.labeled_indices))
                selection_audit = (
                    f"{model_selection} (source_round={selected_from_round}, epoch={selected_epoch})"
                    if int(selected_from_round) != int(round_idx + 1)
                    else f"{model_selection} (epoch={selected_epoch})"
                )
                self._append_md(
                    log_path,
                    f"\n本轮结果: Round={round_idx + 1}, Labeled={len(self.labeled_indices)}, Selection={selection_audit}, mIoU={selected_miou:.4f}, F1={selected_f1:.4f}, peak_mIoU={best_miou:.4f}\n\nRound={round_idx + 1}, Labeled={len(self.labeled_indices)}, mIoU={selected_miou:.4f}, F1={selected_f1:.4f}\n\n",
                )

                # --- 基础设施增强: 无论是否使用 Agent，都计算并记录过拟合信号 (TVC) ---
                tvc_mean = None
                tvc_min = None
                tvc_max = None
                tvc_last = None
                tvc_p10 = None
                tvc_neg_rate = None
                tvc_sign_flip_rate = None
                overfit_risk = None
                if grad_tvc_values:
                    try:
                        arr = np.asarray(grad_tvc_values, dtype=float)
                        arr = arr[np.isfinite(arr)]
                        if int(arr.size) > 0:
                            tvc_mean = float(np.mean(arr))
                            tvc_min = float(np.min(arr))
                            tvc_max = float(np.max(arr))
                            tvc_last = float(arr[-1])
                            tvc_p10 = float(np.quantile(arr, 0.1))
                            tvc_neg_rate = float(np.mean(arr < 0))
                            if int(arr.size) >= 2:
                                signs = np.sign(arr)
                                signs = signs[np.isfinite(signs)]
                                sign_den = int(max(int(signs.size) - 1, 0))
                                if sign_den > 0:
                                    sign_flips = int(
                                        np.sum((signs[1:] * signs[:-1]) < 0)
                                    )
                                    tvc_sign_flip_rate = float(sign_flips) / float(
                                        sign_den
                                    )
                                else:
                                    tvc_sign_flip_rate = 0.0
                            else:
                                tvc_sign_flip_rate = 0.0
                            overfit_risk = float(
                                (tvc_neg_rate or 0.0)
                                + max(0.0, -tvc_min) * 0.5
                                + max(0.0, -tvc_last) * 0.5
                            )
                    except Exception:
                        tvc_mean = None
                        tvc_min = None
                        tvc_max = None
                        tvc_last = None
                        tvc_p10 = None
                        tvc_neg_rate = None
                        tvc_sign_flip_rate = None
                        overfit_risk = None
                epoch_miou_volatility = None
                try:
                    if isinstance(epoch_mious, list) and len(epoch_mious) >= 2:
                        epoch_miou_volatility = float(
                            np.std(np.asarray(epoch_mious, dtype=float), ddof=1)
                        )
                    elif isinstance(epoch_mious, list) and len(epoch_mious) == 1:
                        epoch_miou_volatility = 0.0
                except Exception:
                    epoch_miou_volatility = None

                self._append_trace(
                    {
                        "type": "overfit_signal",
                        "round": int(round_idx + 1),
                        "grad_train_val_cos_mean": tvc_mean,
                        "grad_train_val_cos_min": tvc_min,
                        "grad_train_val_cos_max": tvc_max,
                        "grad_train_val_cos_last": tvc_last,
                        "grad_train_val_cos_p10": tvc_p10,
                        "grad_train_val_cos_neg_rate": tvc_neg_rate,
                        "tvc_sign_flip_rate": tvc_sign_flip_rate,
                        "epoch_miou_volatility": epoch_miou_volatility,
                        "overfit_risk": overfit_risk,
                        # P0: 标记梯度探针数据来源（学术审计）
                        "grad_probe_source": grad_probe_source,
                    }
                )
                # ------------------------------------------------------------------

                miou_policy = (
                    self.exp_config.get("training_state_policy")
                    if isinstance(self.exp_config, dict)
                    else None
                )
                if not isinstance(miou_policy, dict):
                    miou_policy = None
                miou_last_epoch = (
                    float(epoch_mious[-1]) if epoch_mious else float(best_miou)
                )
                last_k = int(miou_policy.get("last_k", 3)) if miou_policy else 0
                last_k = max(int(last_k), 0)
                miou_last_k_mean = float(
                    np.mean(epoch_mious[-last_k:])
                    if (epoch_mious and last_k > 0)
                    else miou_last_epoch
                )
                ema_alpha = (
                    float(miou_policy.get("ema_alpha", 0.6)) if miou_policy else 0.0
                )
                ema_alpha = float(min(max(ema_alpha, 0.0), 1.0))
                miou_ema = None
                if epoch_mious:
                    for val in epoch_mious:
                        miou_ema = (
                            float(val)
                            if miou_ema is None
                            else ema_alpha * float(val)
                            + (1.0 - ema_alpha) * float(miou_ema)
                        )

                raw_signal_key = (
                    str(
                        miou_policy.get(
                            "miou_signal",
                            "last_epoch" if model_selection == "last_epoch" else "peak",
                        )
                    )
                    .strip()
                    .lower()
                    if miou_policy
                    else ("last" if model_selection == "last_epoch" else "peak")
                )
                signal_alias = {
                    "miou_ema": "ema",
                    "lastk": "last_k_mean",
                    "last_k": "last_k_mean",
                    "last_epoch": "last",
                    "epoch_last": "last",
                }
                miou_signal_type = (
                    signal_alias.get(raw_signal_key, raw_signal_key) or "peak"
                )

                signal_getters = {
                    "peak": lambda: float(best_miou),
                    "ema": lambda: float(miou_ema)
                    if miou_ema is not None
                    else float(best_miou),
                    "last_k_mean": lambda: float(miou_last_k_mean),
                    "last": lambda: float(miou_last_epoch),
                }
                miou_signal = float(
                    signal_getters.get(miou_signal_type, signal_getters["peak"])()
                )

                prev_miou = None
                delta_source = (
                    str(miou_policy.get("delta_source", "miou_signal")).strip().lower()
                    if miou_policy
                    else "history"
                )

                def _prev_from_state():
                    state = (
                        self._last_training_state
                        if isinstance(self._last_training_state, dict)
                        else {}
                    )
                    return state.get("miou_signal")

                def _prev_from_history():
                    return (
                        performance_history[-2]["mIoU"]
                        if len(performance_history) > 1
                        else None
                    )

                prev_getters = {
                    "miou_signal": _prev_from_state,
                    "history": _prev_from_history,
                }
                prev_miou = prev_getters.get(delta_source, _prev_from_history)()
                prev_miou = prev_miou if prev_miou is not None else _prev_from_history()
                miou_delta = (
                    None if prev_miou is None else float(miou_signal) - float(prev_miou)
                )

                if miou_delta is not None:
                    low_gain_thresh = float(
                        getattr(AgentThresholds, "MIOU_LOW_GAIN_THRESH", 0.001)
                    )
                    if float(miou_delta) < low_gain_thresh:
                        self._main_miou_low_gain_streak = (
                            getattr(self, "_main_miou_low_gain_streak", 0) + 1
                        )
                    else:
                        self._main_miou_low_gain_streak = 0
                else:
                    self._main_miou_low_gain_streak = 0

                rollback_cfg = getattr(self, "rollback_config", None)
                if not isinstance(rollback_cfg, dict):
                    rollback_cfg = {}
                rollback_mode = "adaptive_threshold"

                rollback_threshold_used = None
                if miou_delta is not None:
                    std_factor = abs(float(rollback_cfg.get("std_factor", 1.5)))
                    tau_min = abs(float(rollback_cfg.get("tau_min", 0.005)))
                    epoch_std = 0.0
                    try:
                        if isinstance(epoch_mious, list) and len(epoch_mious) >= 2:
                            epoch_std = float(
                                np.std(np.array(epoch_mious, dtype=float), ddof=1)
                            )
                    except Exception:
                        epoch_std = 0.0
                    if not np.isfinite(epoch_std) or float(epoch_std) < 0.0:
                        epoch_std = 0.0
                    tau = max(float(tau_min), float(std_factor) * float(epoch_std))
                    if (not np.isfinite(tau)) or float(tau) <= 0.0:
                        raise RuntimeError(
                            f"adaptive rollback threshold requires positive finite tau, got {tau}"
                        )
                    rollback_threshold_used = -tau
                    rollback_flag = bool(
                        float(miou_delta) < float(rollback_threshold_used)
                    )
                else:
                    rollback_flag = False

                remaining_budget = None
                try:
                    remaining_budget = int(self.config.TOTAL_BUDGET) - int(
                        len(self.labeled_indices)
                    )
                except Exception:
                    remaining_budget = None
                _u_med = getattr(self, "_last_u_median_selected", None)
                _k_med = getattr(self, "_last_k_median_selected", None)
                training_state_update = {
                    "round_idx": int(round_idx + 1),
                    "last_miou": float(miou_signal)
                    if miou_policy
                    else float(best_miou),
                    "prev_miou": prev_miou if prev_miou is not None else 0.0,
                    "best_miou_so_far": best_miou_so_far,
                    "miou_delta": miou_delta,
                    "miou_signal": float(miou_signal),
                    "miou_signal_type": str(miou_signal_type),
                    "miou_peak": float(best_miou),
                    "miou_last_epoch": float(miou_last_epoch),
                    "miou_last_k_mean": float(miou_last_k_mean),
                    "miou_ema": float(miou_ema) if miou_ema is not None else None,
                    "model_selection": str(model_selection),
                    "selected_epoch": int(selected_epoch),
                    "selected_miou": float(selected_miou),
                    "selected_f1": float(selected_f1),
                    "rollback_flag": bool(rollback_flag),
                    "rollback_mode": rollback_mode,
                    "rollback_threshold": float(rollback_threshold_used)
                    if rollback_threshold_used is not None
                    else None,
                    "current_labeled_count": int(len(self.labeled_indices)),
                    "total_budget": int(self.config.TOTAL_BUDGET),
                    "remaining_budget": remaining_budget,
                    "last_round_selected_count": self._last_query_selected_count,
                    "grad_train_val_cos_mean": tvc_mean,
                    "grad_train_val_cos_min": tvc_min,
                    "grad_train_val_cos_max": tvc_max,
                    "grad_train_val_cos_last": tvc_last,
                    "grad_train_val_cos_p10": tvc_p10,
                    "grad_train_val_cos_neg_rate": tvc_neg_rate,
                    "tvc_sign_flip_rate": tvc_sign_flip_rate,
                    "epoch_miou_volatility": epoch_miou_volatility,
                    "overfit_risk": overfit_risk,
                    "u_median_selected": _u_med,
                    "k_median_selected": _k_med,
                    "u_median_top": getattr(self, "_last_u_median_top", None),
                    "k_median_top": getattr(self, "_last_k_median_top", None),
                    "grad_probe_source": grad_probe_source,
                    "train_u_median_selected": _u_med,
                    "train_k_median_selected": _k_med,
                    "selection_geometry": (
                        dict(
                            (
                                getattr(self, "_last_ranking_metadata", {}) or {}
                            ).get("selection_geometry")
                            or {}
                        )
                        if isinstance(
                            (
                                getattr(self, "_last_ranking_metadata", {}) or {}
                            ).get("selection_geometry"),
                            dict,
                        )
                        else None
                    ),
                }
                if not isinstance(getattr(self, "training_state", None), dict):
                    self.training_state = {
                        "train_u_median_history": [],
                        "train_k_median_history": [],
                        "max_history_length": 5,
                    }
                self.training_state.update(training_state_update)
                try:
                    trainer._update_uncertainty_history(int(round_idx + 1))
                except Exception:
                    pass
                training_state = dict(self.training_state)
                self._last_training_state = dict(training_state)
                if self.use_agent and self.agent_manager:
                    self.toolbox.set_training_state(training_state)

                report_metrics = None
                report_miou = None
                report_f1 = None
                is_final_round = round_idx == self.config.N_ROUNDS - 1
                if is_final_round:
                    report_dataset = Landslide4SenseDataset(
                        self.config.DATA_DIR, split=self.test_split
                    )
                    mask_dir = getattr(report_dataset, "mask_dir", None)
                    mask_map = getattr(report_dataset, "_mask_by_id", None)
                    if (
                        mask_dir is None
                        or not isinstance(mask_map, dict)
                        or int(len(mask_map)) <= 0
                    ):
                        raise RuntimeError(
                            "TestData/mask is missing; cannot run final report evaluation"
                        )
                    missing_mask_ids = []
                    for image_name in getattr(report_dataset, "images", []) or []:
                        sid = os.path.splitext(str(image_name))[0]
                        if sid not in mask_map:
                            missing_mask_ids.append(sid)
                            if len(missing_mask_ids) >= 3:
                                break
                    if missing_mask_ids:
                        raise RuntimeError(
                            f"TestData/mask is incomplete; missing mask for sample_ids={missing_mask_ids}"
                        )
                    report_loader = DataLoader(report_dataset, **val_loader_kwargs)
                    report_metrics = trainer.evaluate(report_loader)
                    if isinstance(report_metrics, dict):
                        try:
                            report_miou = float(report_metrics.get("mIoU", 0.0))
                        except Exception:
                            report_miou = None
                        try:
                            report_f1 = float(report_metrics.get("f1_score", 0.0))
                        except Exception:
                            report_f1 = None
                    final_report = (
                        dict(report_metrics)
                        if isinstance(report_metrics, dict)
                        else report_metrics
                    )
                    test_split = str(self.test_split)
                    if (
                        isinstance(self._last_training_state, dict)
                        and report_miou is not None
                        and report_f1 is not None
                    ):
                        training_state_with_report = dict(self._last_training_state)
                        training_state_with_report["report_split"] = "test"
                        training_state_with_report["report_eval_source"] = (
                            "official_test"
                            if str(self.test_split) == "test"
                            else "test"
                        )
                        training_state_with_report["report_miou"] = float(report_miou)
                        training_state_with_report["report_f1"] = float(report_f1)
                        self._last_training_state = training_state_with_report
                    self._append_trace(
                        {
                            "type": "report_eval",
                            "round": int(round_idx + 1),
                            "report_split": "test",
                            "eval_source": "official_test"
                            if str(self.test_split) == "test"
                            else "test",
                            "eval_split": str(self.test_split),
                            "eval_img_dir": getattr(report_dataset, "img_dir", None),
                            "eval_mask_dir": getattr(report_dataset, "mask_dir", None),
                            "model_selection": str(model_selection),
                            "selected_epoch": int(selected_epoch),
                            "selected_from_round": int(selected_from_round),
                            "metrics": dict(report_metrics)
                            if isinstance(report_metrics, dict)
                            else report_metrics,
                        }
                    )

                if round_idx < self.config.N_ROUNDS - 1:
                    new_indices = self._query_samples(trainer.model)
                    if new_indices is not None:
                        update_result = self.update(new_indices)
                        if (
                            isinstance(update_result, dict)
                            and update_result.get("status") != "success"
                        ):
                            raise ValueError(f"Selection failed: {update_result}")
                        expected = int(
                            update_result.get("expected_count")
                            or self.config.QUERY_SIZE
                        )
                        selected = int(update_result.get("selected_count") or 0)
                        if expected > 0 and selected < expected:
                            early_stop = True
                            early_stop_reason = "selection_short"
                        if expected > 0 and selected < expected:
                            self._write_status(
                                {
                                    "progress": {
                                        "round": int(round_idx + 1),
                                        "selection": {
                                            "expected": int(expected),
                                            "selected": int(selected),
                                            "early_stop": True,
                                        },
                                    }
                                }
                            )
                    # Early stopping logic
                    # 1. Selection Short
                    if self.use_agent and self._last_query_selected_count is not None:
                        expected = int(getattr(self.config, "QUERY_SIZE", 0) or 0)
                        if (
                            expected > 0
                            and int(self._last_query_selected_count) < expected
                        ):
                            early_stop = True
                            early_stop_reason = "selection_short"

                    # 2. mIoU Low Gain (Disabled, used for lambda adjustment instead)
                    low_gain_streak = getattr(self, "_main_miou_low_gain_streak", 0)
                    low_gain_thresh_streak = getattr(
                        AgentThresholds, "MIOU_LOW_GAIN_STREAK", 3
                    )
                    if (
                        low_gain_streak >= low_gain_thresh_streak
                        and round_idx >= self.config.N_ROUNDS // 2
                    ):
                        logger.warning(
                            f"Low gain warning: mIoU low gain for {low_gain_streak} consecutive rounds (threshold={low_gain_thresh_streak}). NOT stopping early, letting agent/policy adjust."
                        )
                        # early_stop = True # Disabled to allow policy adjustment
                        # if not early_stop_reason: early_stop_reason = "miou_low_gain"

                    self._append_round_summary(
                        int(round_idx + 1),
                        float(selected_miou),
                        float(selected_f1),
                        int(len(self.labeled_indices)),
                    )
                else:
                    self._append_round_summary(
                        int(round_idx + 1),
                        float(report_miou)
                        if report_miou is not None
                        else float(selected_miou),
                        float(report_f1)
                        if report_f1 is not None
                        else float(selected_f1),
                        int(len(self.labeled_indices)),
                    )

                self._save_pool_states()

                # Checkpoint - Atomic Guarantee
                state_dict = {
                    "round": int(round_idx + 1),
                    "performance_history": performance_history,
                    "budget_history": budget_history,
                    "labeled_indices": self.labeled_indices,
                    "labeled_size": int(
                        len(self.labeled_indices)
                    ),  # Crucial for rollback
                    "unlabeled_size": int(len(self.unlabeled_indices)),
                    "rng_states": self._get_rng_states(),  # Reproducibility
                    "model_selection": str(model_selection),
                    "selected_epoch": int(selected_epoch),
                    "selected_from_round": int(selected_from_round),
                    "best_val_epoch": int(best_epoch),
                    "best_val_miou": float(best_miou),
                    "best_val_checkpoint": str(round_best_ckpt_path),
                }
                if self.use_agent and self.agent_manager:
                    state_dict["pending_round_controls"] = dict(
                        self._pending_round_controls
                    )

                self.checkpoint_manager.save(state_dict)
                self._write_status(
                    {
                        "status": "running",
                        "progress": {
                            "round": int(round_idx + 1),
                            "epoch": "finished",
                            "labeled_size": int(len(self.labeled_indices)),
                        },
                    }
                )

                if early_stop:
                    self._append_trace(
                        {
                            "type": "early_stop",
                            "round": int(round_idx + 1),
                            "reason": early_stop_reason
                            if "early_stop_reason" in locals()
                            else "unknown",
                            "details": {
                                "low_gain_streak": getattr(
                                    self, "_main_miou_low_gain_streak", 0
                                ),
                                "selection_short": (
                                    int(self._last_query_selected_count) < expected
                                )
                                if (
                                    self.use_agent
                                    and self._last_query_selected_count is not None
                                    and expected > 0
                                )
                                else False,
                            },
                        }
                    )
                    break
            except Exception as e:
                logger.error(f"Round {round_idx + 1} failed: {str(e)}")
                error_type = None
                try:
                    error_type = self._classify_error(str(e))
                except Exception:
                    error_type = "unknown"
                self._write_status(
                    {
                        "status": "failed",
                        "error": {
                            "round": int(round_idx + 1),
                            "message": str(e),
                            "type": error_type,
                            "exception": e.__class__.__name__,
                        },
                    }
                )
                self._append_trace(
                    {
                        "type": "failed",
                        "round": int(round_idx + 1),
                        "message": str(e),
                        "error_type": error_type,
                        "exception": e.__class__.__name__,
                    }
                )
                self._append_md(
                    log_path, f"\n**[ERROR] Round {round_idx + 1} 失败: {str(e)}**\n\n"
                )
                raise
            finally:
                self._cleanup_resources(
                    labeled_loader=labeled_loader
                    if "labeled_loader" in locals()
                    else None,
                    val_loader=val_loader if "val_loader" in locals() else None,
                    probe_loader=probe_loader if "probe_loader" in locals() else None,
                    report_loader=report_loader
                    if "report_loader" in locals()
                    else None,
                    trainer=trainer if "trainer" in locals() else None,
                    model=model if "model" in locals() else None,
                )

        alc = calculate_alc(
            [p["mIoU"] for p in performance_history],
            budget_history,
            total_budget=int(getattr(self.config, "TOTAL_BUDGET", 0) or 0),
            pad_to_total_budget=True,
        )
        final_report_miou = None
        final_report_f1 = None
        if isinstance(final_report, dict):
            try:
                final_report_miou = float(final_report.get("mIoU", 0.0))
            except Exception:
                final_report_miou = None
            try:
                final_report_f1 = float(final_report.get("f1_score", 0.0))
            except Exception:
                final_report_f1 = None
        selected_final_miou = None
        selected_final_f1 = None
        if performance_history:
            try:
                selected_final_miou = float(performance_history[-1].get("mIoU", 0.0))
            except Exception:
                selected_final_miou = None
            try:
                selected_final_f1 = float(performance_history[-1].get("f1_score", 0.0))
            except Exception:
                selected_final_f1 = None
        final_miou_source = "final_report"
        if final_report_miou is None:
            final_miou_source = "selected_val_fallback"
        final_f1_source = "final_report"
        if final_report_f1 is None:
            final_f1_source = "selected_val_fallback"
        result = {
            "performance_history": performance_history,
            "budget_history": budget_history,
            "alc": float(alc),
            "final_miou": float(final_report_miou)
            if final_report_miou is not None
            else float(selected_final_miou or 0.0),
            "final_f1": float(final_report_f1)
            if final_report_f1 is not None
            else float(selected_final_f1 or 0.0),
            "final_miou_source": str(final_miou_source),
            "final_f1_source": str(final_f1_source),
            "final_selected_miou": float(selected_final_miou)
            if selected_final_miou is not None
            else None,
            "final_selected_f1": float(selected_final_f1)
            if selected_final_f1 is not None
            else None,
            "final_report_miou": float(final_report_miou)
            if final_report_miou is not None
            else None,
            "final_report_f1": float(final_report_f1)
            if final_report_f1 is not None
            else None,
            "test_split": test_split,
            "final_report": final_report,
        }
        self._append_md(
            log_path,
            f"\n## 实验汇总\n\n预算历史: {budget_history}\nALC(基于每轮选模val mIoU): {result['alc']:.4f}\n最后一轮选模 mIoU(val): {result['final_selected_miou']}\n最后一轮选模 F1(val): {result['final_selected_f1']}\n最终报告 mIoU({result['test_split']}): {result['final_report_miou']}\n最终报告 F1({result['test_split']}): {result['final_report_f1']}\n最终输出 mIoU: {result['final_miou']:.4f} (source={result['final_miou_source']})\n最终输出 F1: {result['final_f1']:.4f} (source={result['final_f1_source']})\n最终 Test Split: {result['test_split']}\n最终 Report: {result['final_report']}\n",
        )
        self._write_status(
            {
                "status": "completed",
                "result": {
                    "alc": float(result["alc"]),
                    "final_mIoU": float(result["final_miou"]),
                    "final_f1": float(result["final_f1"]),
                    "final_selected_mIoU": float(result["final_selected_miou"])
                    if result.get("final_selected_miou") is not None
                    else None,
                    "final_selected_f1": float(result["final_selected_f1"])
                    if result.get("final_selected_f1") is not None
                    else None,
                    "final_report_mIoU": float(result["final_report_miou"])
                    if result.get("final_report_miou") is not None
                    else None,
                    "final_report_f1": float(result["final_report_f1"])
                    if result.get("final_report_f1") is not None
                    else None,
                    "final_miou_source": str(result.get("final_miou_source") or ""),
                    "final_f1_source": str(result.get("final_f1_source") or ""),
                    "test_split": str(result.get("test_split") or ""),
                    "budget_history": list(result["budget_history"]),
                },
            }
        )
        return result

    def _build_loader_kwargs(
        self,
        *,
        batch_size,
        shuffle,
        num_workers,
        generator,
        worker_init_fn,
        drop_last=False,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
    ):
        kwargs = {
            "batch_size": batch_size,
            "shuffle": bool(shuffle),
            "num_workers": int(num_workers),
            "persistent_workers": bool(
                int(num_workers) > 0 and bool(persistent_workers)
            ),
            "drop_last": bool(drop_last),
            "generator": generator,
            "worker_init_fn": worker_init_fn,
            "pin_memory": bool(pin_memory),
        }
        if int(num_workers) > 0:
            kwargs["prefetch_factor"] = int(prefetch_factor or 2)
        return kwargs

    @staticmethod
    def _split_labeled_indices_for_grad_probe(
        labeled_indices: list[int],
        *,
        frac: float,
        min_count: int,
        seed: int,
    ) -> tuple[list[int], list[int]]:
        import hashlib

        ids = [int(x) for x in (labeled_indices or [])]
        n = int(len(ids))
        if n <= 0:
            return [], []
        frac = float(min(max(frac, 0.0), 0.8))
        k = int(round(float(n) * float(frac)))
        k = max(int(min_count), k)
        k = min(k, max(0, n - 1))
        if k <= 0:
            return ids, []

        scored = []
        salt = str(int(seed))
        for sid in ids:
            h = hashlib.sha256(
                f"grad_probe|{salt}|{int(sid)}".encode("utf-8", errors="ignore")
            ).digest()
            u = int.from_bytes(h[:8], "big") / float(2**64 - 1)
            scored.append((u, int(sid)))
        scored.sort(key=lambda x: (float(x[0]), int(x[1])))
        probe = [sid for _, sid in scored[:k]]
        probe_set = set(probe)
        train = [sid for sid in ids if sid not in probe_set]
        return train, probe

    def _cleanup_loader(self, loader):
        """显式清理 DataLoader 资源"""
        if loader is None:
            return
        try:
            if hasattr(loader, "_iterator") and loader._iterator is not None:
                try:
                    loader._iterator._shutdown_workers()
                except Exception:
                    pass
                del loader._iterator
        except Exception:
            pass
        finally:
            del loader

    def _cleanup_resources(
        self,
        labeled_loader=None,
        val_loader=None,
        probe_loader=None,
        report_loader=None,
        trainer=None,
        model=None,
    ):
        """统一资源清理入口"""
        # 1. Cleanup Loaders
        self._cleanup_loader(labeled_loader)
        self._cleanup_loader(val_loader)
        self._cleanup_loader(probe_loader)
        self._cleanup_loader(report_loader)

        # 2. Cleanup Trainer
        if trainer is not None:
            try:
                trainer.cleanup()
            except Exception:
                pass
            del trainer

        # 3. Cleanup Model
        if model is not None:
            try:
                model.to("cpu")
            except Exception:
                pass
            del model

        # 4. Force GC and Empty Cache
        import gc

        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps"):
                torch.mps.empty_cache()
        except Exception:
            pass

    def run(self):
        result = self.run_and_collect()
        logger.info("=== Experiment Finished ===")
        logger.info(f"Experiment: {self.experiment_name}")
        logger.info(f"Budget History: {result['budget_history']}")
        logger.info(
            f"Performance (mIoU) History: {[p['mIoU'] for p in result['performance_history']]}"
        )
        logger.info(f"ALC: {result['alc']:.4f}")

    def _query_samples(self, model):
        logger.info("Querying samples...")

        # Common logic: rank samples
        # Need unlabeled info
        # To avoid re-extracting features for baseline samplers if they don't need model,
        # we check sampler type. But AD-KUCS needs model.

        # For simplicity, we assume samplers might need model features or predictions.
        # We need a helper to get unlabeled data info efficiently.
        # But our sampler interface varies.

        # 统一接口: rank_samples(model, dataset, unlabeled_indices, labeled_indices, ...)
        # Or specialized handling.

        n_samples = int(self.config.QUERY_SIZE)

        if self.use_agent and self.agent_manager:
            self.toolbox.model = model
            self._last_query_selected_count = None
            try:
                result = self.agent_manager.run_cycle()
            except Exception as e:
                logger.error(f"Agent run_cycle failed: {str(e)}")
                raise RuntimeError(
                    f"LLM Agent failed at Round {self.current_round}: {str(e)}"
                )

            n_samples = int(self.config.QUERY_SIZE)
            if not isinstance(result, dict):
                logger.error(
                    f"Agent returned non-dict result at Round {self.current_round}: {type(result).__name__}"
                )
                raise RuntimeError(
                    f"LLM Agent failed at Round {self.current_round}: invalid_result_type {type(result).__name__}"
                )

            if result.get("status") != "success":
                error_type = result.get("error_type") or "AgentError"
                message = result.get("message") or json.dumps(
                    result, ensure_ascii=False
                )
                logger.error(
                    f"Agent failed at Round {self.current_round}: {error_type}: {message}"
                )
                raise RuntimeError(
                    f"LLM Agent failed at Round {self.current_round}: {error_type}: {message}"
                )

            control_applied = result.get("control_applied")
            if (
                isinstance(control_applied, dict)
                and control_applied.get("epochs") is not None
            ):
                fixed_epochs = bool(getattr(self.config, "FIX_EPOCHS_PER_ROUND", False))
                allow_epoch_control = bool(
                    getattr(
                        getattr(self, "toolbox", None), "control_permissions", {}
                    ).get("set_epochs_per_round", False)
                )
                if (not fixed_epochs) and allow_epoch_control:
                    self._pending_round_controls["epochs_round"] = int(
                        control_applied.get("epochs")
                    )

            selected_ids = (
                result.get("valid_candidates") or result.get("selected_ids") or []
            )
            selected_count = int(result.get("selected_count") or 0)
            self._last_query_selected_count = selected_count

            if selected_count <= 0:
                logger.error(
                    f"Agent returned success but selected_count={selected_count} at Round {self.current_round}"
                )
                raise RuntimeError(
                    f"Agent selection incomplete at Round {self.current_round}: selected_count={selected_count}, required={n_samples}"
                )

            if not isinstance(selected_ids, list) or len(selected_ids) < n_samples:
                error_msg = f"Agent selected {len(selected_ids) if isinstance(selected_ids, list) else 0} samples, but {n_samples} required"
                logger.error(
                    f"Agent selection incomplete at Round {self.current_round}: {error_msg}"
                )
                raise RuntimeError(
                    f"Agent selection incomplete at Round {self.current_round}: {error_msg}"
                )

            return None

        ranked_ids = self._rank_samples_by_sampler(model)
        selected = ranked_ids[:n_samples]
        policy = "pipeline_rank_topk"
        if isinstance(getattr(self, "_last_ranking_metadata", None), dict):
            post_meta = self._last_ranking_metadata.get("postprocess")
            if isinstance(post_meta, dict) and post_meta.get("applied"):
                policy = "pipeline_rank_postprocess"
        self._selection_context = {
            "source": "pipeline",
            "policy": policy,
            "fallback_used": False,
            **self._sampler_audit(),
            **(getattr(self, "_last_ranking_metadata", {}) or {}),
            "degraded": dict(self._last_ranking_degraded)
            if isinstance(getattr(self, "_last_ranking_degraded", None), dict)
            else None,
        }
        return selected

    def _rank_samples_by_sampler(self, model):
        if hasattr(self.sampler, "set_round"):
            self.sampler.set_round(self.current_round)
        if isinstance(self.sampler, BALDSampler):
            protocol = None
            if isinstance(getattr(self, "exp_config", None), dict):
                protocol = self.exp_config.get("acquisition_protocol")
            protocol = protocol if isinstance(protocol, dict) else {}
            u_agg = (
                str(protocol.get("uncertainty_aggregation", "mean") or "mean")
                .strip()
                .lower()
            )
            needs_post = bool(
                hasattr(self, "selection_postprocessor")
                and self.selection_postprocessor is not None
            )
            use_pipeline_bald = needs_post or (
                u_agg not in ("mean", "full_mean", "none", "")
            )

            if use_pipeline_bald:
                unlabeled_info, _ = self._prepare_unlabeled_info(model, mc_dropout=True)
                ranked = [
                    {
                        "sample_id": sid,
                        "final_score": float(
                            (info or {}).get("uncertainty_score") or 0.0
                        ),
                    }
                    for sid, info in unlabeled_info.items()
                ]
                ranked.sort(
                    key=lambda x: float(x.get("final_score") or 0.0), reverse=True
                )
                self._last_ranked_items = list(ranked or [])
                self._last_ranking_metadata = (
                    self._compute_ranking_metadata(ranked, int(self.config.QUERY_SIZE))
                    or None
                )
                if needs_post:
                    ranked_ids, post_meta = self.selection_postprocessor.apply(
                        ranked, unlabeled_info, int(self.config.QUERY_SIZE)
                    )
                    if not isinstance(self._last_ranking_metadata, dict):
                        self._last_ranking_metadata = {}
                    if isinstance(post_meta, dict):
                        self._last_ranking_metadata["postprocess"] = post_meta
                    return ranked_ids
                return [item["sample_id"] for item in ranked]

            subset_u = Subset(self.query_dataset, self.unlabeled_indices)
            feature_workers = int(getattr(self.config, "FEATURE_NUM_WORKERS", 0) or 0)
            seed_base = int(self.seed) + int(self.current_round or 0)
            g = torch.Generator()
            g.manual_seed(seed_base)
            loader_kwargs = self._build_loader_kwargs(
                batch_size=self.config.BATCH_SIZE,
                shuffle=False,
                num_workers=feature_workers,
                generator=g,
                worker_init_fn=worker_init_fn,
                drop_last=False,
                pin_memory=bool(getattr(self.config, "FEATURE_PIN_MEMORY", False)),
                persistent_workers=bool(
                    getattr(self.config, "FEATURE_PERSISTENT_WORKERS", False)
                ),
                prefetch_factor=int(
                    getattr(self.config, "FEATURE_PREFETCH_FACTOR", 2) or 2
                ),
            )
            loader_u = DataLoader(subset_u, **loader_kwargs)
            sample_indices = list(self.unlabeled_indices)
            try:
                ranked = self.sampler.rank_samples(
                    {},
                    model=model,
                    data_loader=loader_u,
                    sample_indices=sample_indices,
                )
            finally:
                self._cleanup_loader(loader_u)

            self._last_ranked_items = list(ranked or [])
            self._last_ranking_metadata = (
                self._compute_ranking_metadata(ranked, int(self.config.QUERY_SIZE))
                or None
            )
            return [item["sample_id"] for item in ranked]

        if (
            isinstance(self.sampler, ADKUCSSampler)
            or self.sampler.__class__.__name__ == "ADKUCSSampler"
        ):
            unlabeled_info, labeled_features = self._prepare_unlabeled_info(model)
            lambda_override, lambda_source = self._resolve_lambda_override(
                self.current_round
            )

            # Explicitly log lambda usage for debugging
            logger.info(
                f"ADKUCSSampler: Preparing rank. lambda_override={lambda_override} source={lambda_source}"
            )
            current_iteration = len(self.labeled_indices)
            total_iterations = self.config.TOTAL_BUDGET
            lambda_used = float(
                self.sampler._get_adaptive_weight(
                    current_iteration, total_iterations, override=lambda_override
                )
            )
            lambda_effective = float(lambda_used)
            lambda_effective_source = (
                "sampler_adaptive" if lambda_override is None else str(lambda_source)
            )

            if not unlabeled_info:
                raise RuntimeError("ADKUCSSampler: unlabeled_info is empty")

            ranked = self.sampler.rank_samples(
                unlabeled_info,
                labeled_features,
                current_iteration=current_iteration,
                total_iterations=total_iterations,
                lambda_override=lambda_override,
            )
            self._last_ranked_items = list(ranked or [])
            if ranked:
                top_item = ranked[0]
                top_k = int(self.config.QUERY_SIZE)
                avg_uncertainty = float(
                    np.mean([r["uncertainty"] for r in ranked[:top_k]])
                )
                avg_knowledge_gain = float(
                    np.mean([r["knowledge_gain"] for r in ranked[:top_k]])
                )
                protocol = None
                if isinstance(getattr(self, "exp_config", None), dict):
                    protocol = self.exp_config.get("acquisition_protocol")
                protocol = protocol if isinstance(protocol, dict) else {}
                u_agg = (
                    str(protocol.get("uncertainty_aggregation", "mean") or "mean")
                    .strip()
                    .lower()
                )
                if u_agg in ("mean", "full_mean", "none", ""):
                    u_type = "pixel_mean_entropy_log2"
                    u_def = (
                        "U(I)=mean_i H(p_i), H(p_i)=-sum_c p_{i,c} log2(p_{i,c}+1e-10)"
                    )
                else:
                    tau = float(protocol.get("entropy_threshold", 0.5) or 0.5)
                    u_type = "pixel_high_entropy_mean_log2"
                    u_def = (
                        "U(I)=mean_{i: H(p_i)>tau} H(p_i) else mean_i H(p_i), "
                        "H(p_i)=-sum_c p_{i,c} log2(p_{i,c}+1e-10), "
                        f"tau={tau}"
                    )
                self._last_ranking_metadata = {
                    "lambda_effective": float(lambda_effective),
                    "lambda_source": str(lambda_effective_source),
                    "avg_uncertainty": avg_uncertainty,
                    "avg_knowledge_gain": avg_knowledge_gain,
                    "uncertainty_type": str(u_type),
                    "uncertainty_definition": str(u_def),
                }
                base_meta = (
                    self._compute_ranking_metadata(ranked, int(self.config.QUERY_SIZE))
                    or {}
                )
                if isinstance(base_meta, dict):
                    base_meta.update(self._last_ranking_metadata)
                    self._last_ranking_metadata = base_meta
                logger.info(
                    f"AD-KUCS Strategy: lambda_effective={lambda_effective:.4f} source={lambda_effective_source} (Top Sample: {top_item.get('sample_id')}) "
                    f"avg_uncertainty(top{top_k})={avg_uncertainty:.6f} avg_knowledge_gain(top{top_k})={avg_knowledge_gain:.6f} "
                    f"U_def={u_type}"
                )
            if (
                hasattr(self, "selection_postprocessor")
                and self.selection_postprocessor is not None
            ):
                ranked_ids, post_meta = self.selection_postprocessor.apply(
                    ranked, unlabeled_info, int(self.config.QUERY_SIZE)
                )
                if not isinstance(self._last_ranking_metadata, dict):
                    self._last_ranking_metadata = {}
                if isinstance(post_meta, dict):
                    self._last_ranking_metadata["postprocess"] = post_meta
                return ranked_ids
        else:
            unlabeled_info, labeled_features = self._prepare_unlabeled_info(model)
            ranked = self.sampler.rank_samples(
                unlabeled_info, labeled_features=labeled_features
            )
            self._last_ranked_items = list(ranked or [])
            self._last_ranking_metadata = (
                self._compute_ranking_metadata(ranked, int(self.config.QUERY_SIZE))
                or None
            )
            if (
                hasattr(self, "selection_postprocessor")
                and self.selection_postprocessor is not None
            ):
                ranked_ids, post_meta = self.selection_postprocessor.apply(
                    ranked, unlabeled_info, int(self.config.QUERY_SIZE)
                )
                if not isinstance(self._last_ranking_metadata, dict):
                    self._last_ranking_metadata = {}
                if isinstance(post_meta, dict):
                    self._last_ranking_metadata["postprocess"] = post_meta
                return ranked_ids
        return [item["sample_id"] for item in ranked]

    def _select_diverse_items(self, items, unlabeled_info, k, post_cfg):
        if not items or int(k) <= 0:
            return []
        mode = str(post_cfg.get("mode", "none")).strip().lower() or "none"

        def _select_topk():
            return list(items[: int(k)])

        def _select_fps_feature():
            features = []
            valid_items = []
            if isinstance(unlabeled_info, dict):
                for item in items:
                    sid = item.get("sample_id")
                    info = unlabeled_info.get(sid, {})
                    feat = info.get("feature")
                    if feat is None:
                        continue
                    try:
                        features.append(np.asarray(feat, dtype=float))
                        valid_items.append(item)
                    except Exception:
                        continue
            if len(valid_items) <= int(k):
                return valid_items
            features_array = np.vstack(features)
            seed = post_cfg.get("seed")
            seed_base = int(seed) if seed is not None else 0
            if self.current_round is not None:
                seed_base += int(self.current_round)
            rng = np.random.default_rng(seed_base)
            start_idx = int(rng.integers(0, len(valid_items)))
            selected_idx = [start_idx]
            dist = np.linalg.norm(features_array - features_array[start_idx], axis=1)
            for _ in range(int(k) - 1):
                next_idx = int(np.argmax(dist))
                selected_idx.append(next_idx)
                new_dist = np.linalg.norm(
                    features_array - features_array[next_idx], axis=1
                )
                dist = np.minimum(dist, new_dist)
            return [valid_items[i] for i in selected_idx]

        selectors = {
            "none": _select_topk,
            "fps_feature": _select_fps_feature,
        }
        return selectors.get(mode, _select_topk)()

    def update(self, new_indices):
        expected = int(getattr(self.config, "QUERY_SIZE", 0) or 0)
        if expected <= 0:
            raise ValueError(f"Invalid QUERY_SIZE: {expected}")
        if not getattr(self, "unlabeled_indices", None):
            raise RuntimeError("unlabeled_indices is empty")
        if expected > int(len(self.unlabeled_indices)):
            raise RuntimeError(
                f"QUERY_SIZE {expected} exceeds remaining unlabeled {len(self.unlabeled_indices)}"
            )
        if not new_indices:
            raise ValueError("new_indices is empty")

        normalized = []
        seen = set()
        for item in new_indices:
            idx = int(item)
            if idx in seen:
                raise ValueError(f"Duplicate selected id: {idx}")
            seen.add(idx)
            normalized.append(idx)

        valid = [idx for idx in normalized if idx in self.unlabeled_indices]
        if not valid:
            raise RuntimeError("No valid selected ids in unlabeled_indices")
        if len(valid) < expected:
            raise RuntimeError(
                f"Only {len(valid)} valid selections, but {expected} required"
            )

        limit = expected if expected > 0 else len(valid)
        selected = valid[: min(limit, len(valid))]
        for idx in selected:
            self.unlabeled_indices.remove(idx)
            self.labeled_indices.append(idx)

        self._save_pool_states()
        logger.info(
            f"Updated pools. Added {len(selected)} samples. New Labeled: {len(self.labeled_indices)}"
        )

        ctx = None
        if isinstance(getattr(self, "_selection_context", None), dict):
            ctx = dict(self._selection_context)
        if isinstance(ctx, dict) and isinstance(
            getattr(self, "_last_ranked_items", None), list
        ):
            ranked_items = list(getattr(self, "_last_ranked_items", []) or [])
            if ranked_items:
                qts = (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)

                def _stats(vals):
                    xs = [float(v) for v in (vals or []) if v is not None]
                    if not xs:
                        return None
                    arr = np.asarray(xs, dtype=float)
                    q = np.quantile(arr, np.asarray(list(qts), dtype=float)).tolist()
                    q_map = {
                        f"p{int(round(float(p) * 100)):02d}": float(v)
                        for p, v in zip(qts, q)
                    }
                    return {
                        "n": int(arr.size),
                        "mean": float(arr.mean()),
                        "min": float(arr.min()),
                        "max": float(arr.max()),
                        **q_map,
                    }

                want = {str(i) for i in selected}
                selected_rows = []
                for item in ranked_items:
                    if not isinstance(item, dict):
                        continue
                    sid = item.get("sample_id")
                    if sid is None or str(sid) not in want:
                        continue
                    selected_rows.append(item)
                    if len(selected_rows) >= int(len(selected)):
                        break

                def _extract(items, key: str):
                    out = []
                    for it in items:
                        if not isinstance(it, dict):
                            continue
                        v = it.get(key)
                        if v is None:
                            continue
                        try:
                            out.append(float(v))
                        except Exception:
                            continue
                    return out

                sel_stats = {}
                for key in ("final_score", "uncertainty", "knowledge_gain", "lambda_t"):
                    st = _stats(_extract(selected_rows, key))
                    if st is not None:
                        sel_stats[key] = st
                if sel_stats:
                    ctx["selected_score_stats"] = sel_stats
        self._selection_context = None
        self._write_status(
            {
                "progress": {
                    "round": int(self.current_round)
                    if self.current_round is not None
                    else None,
                    "selection": {
                        "expected": int(expected),
                        "selected": int(len(selected)),
                        "selected_ids": list(selected),
                        "context": ctx,
                    },
                }
            }
        )
        self._append_trace(
            {
                "type": "selection",
                "round": int(self.current_round)
                if self.current_round is not None
                else None,
                **self._sampler_audit(),
                "expected": int(expected),
                "selected": int(len(selected)),
                "selected_ids": list(selected),
                "context": ctx,
            }
        )
        self._last_selection_summary = {
            "expected": int(expected),
            "selected": int(len(selected)),
            "selected_ids": list(selected),
            "context": ctx,
        }
        self._append_l3_selection(
            selected, source=ctx.get("source") if isinstance(ctx, dict) else None
        )
        self._append_score_snapshot(
            selected, source=ctx.get("source") if isinstance(ctx, dict) else None
        )
        return {
            "status": "success",
            "expected_count": expected,
            "selected_count": len(selected),
            "selected_ids": selected,
            "exhausted": len(selected) < expected or not self.unlabeled_indices,
        }

    def _load_pool_states(self):
        import pandas as pd

        pools_dir = self.pools_dir

        labeled_path = os.path.join(pools_dir, "labeled_pool.csv")
        unlabeled_path = os.path.join(pools_dir, "unlabeled_pool.csv")

        if (not os.path.exists(labeled_path)) or (not os.path.exists(unlabeled_path)):
            raise FileNotFoundError(
                f"Pool state files not found in {pools_dir} (required: labeled/unlabeled)"
            )

        self.labeled_df = pd.read_csv(labeled_path)
        self.unlabeled_df = pd.read_csv(unlabeled_path)

        self.labeled_indices = self._map_filenames_to_indices(
            self.full_dataset, self.labeled_df["sample_id"].tolist()
        )
        self.unlabeled_indices = self._map_filenames_to_indices(
            self.full_dataset, self.unlabeled_df["sample_id"].tolist()
        )

        logger.info(
            f"Loaded pool states: Labeled={len(self.labeled_indices)}, Unlabeled={len(self.unlabeled_indices)}"
        )
        self._assert_pool_integrity()
        return True

    def _get_rng_states(self):
        """Capture RNG states for reproducibility."""
        # numpy.random.get_state() returns (str, ndarray, int, int, float)
        # ndarray is not JSON serializable, so we convert it to list
        np_state = np.random.get_state()
        np_state_list = list(np_state)
        np_state_list[1] = np_state[1].tolist()

        states = {
            "python": random.getstate(),
            "numpy": np_state_list,
            "torch": torch.get_rng_state().tolist(),  # Convert to list for JSON serialization
        }
        if torch.cuda.is_available():
            # torch.cuda.get_rng_state_all() returns a list of ByteTensors
            states["torch_cuda"] = [t.tolist() for t in torch.cuda.get_rng_state_all()]
        return states

    def _set_rng_states(self, states):
        """Restore RNG states."""
        if not states:
            return

        try:
            if "python" in states:
                # python random state is a tuple, but JSON loads as list
                # random.setstate expects tuple
                py_state = states["python"]
                if isinstance(py_state, list):
                    py_state = tuple(py_state)
                    # The second element is the state, which is also a tuple
                    if len(py_state) > 1 and isinstance(py_state[1], list):
                        py_state_list = list(py_state)
                        py_state_list[1] = tuple(py_state[1])
                        py_state = tuple(py_state_list)
                random.setstate(py_state)

            if "numpy" in states:
                # numpy state is tuple(str, ndarray, int, int, float)
                # JSON loads ndarray as list
                np_state = states["numpy"]
                if isinstance(np_state, list):
                    # Need to convert back to tuple and ensure ndarray
                    state_str = np_state[0]
                    state_arr = np.array(np_state[1], dtype=np.uint32)
                    pos = np_state[2]
                    has_gauss = np_state[3]
                    cached_gauss = np_state[4]
                    np.random.set_state(
                        (state_str, state_arr, pos, has_gauss, cached_gauss)
                    )

            if "torch" in states:
                torch.set_rng_state(torch.tensor(states["torch"], dtype=torch.uint8))

            if "torch_cuda" in states and torch.cuda.is_available():
                cuda_states = states["torch_cuda"]
                # Convert list of lists back to list of ByteTensors
                if isinstance(cuda_states, list):
                    tensor_states = [
                        torch.tensor(s, dtype=torch.uint8) for s in cuda_states
                    ]
                    torch.cuda.set_rng_state_all(tensor_states)

        except Exception as e:
            logger.warning(f"Failed to restore RNG states: {e}")

    def _rollback_pools(self, target_labeled_size):
        """
        Rollback pools to match the checkpoint state.
        This handles the case where Pool CSVs were updated but Checkpoint JSON was not.
        """
        if not os.path.exists(os.path.join(self.pools_dir, "labeled_pool.csv")):
            return False

        import pandas as pd

        labeled_df = pd.read_csv(os.path.join(self.pools_dir, "labeled_pool.csv"))
        current_size = len(labeled_df)

        if current_size == target_labeled_size:
            return True

        if current_size < target_labeled_size:
            logger.error(
                f"Pool corruption: Labeled pool size ({current_size}) < Checkpoint size ({target_labeled_size})"
            )
            return False

        logger.warning(
            f"Rolling back pools: Truncating labeled pool from {current_size} to {target_labeled_size}"
        )

        # 1. Truncate Labeled Pool
        # Keep first N rows
        valid_labeled_df = labeled_df.iloc[:target_labeled_size]
        labeled_path = os.path.join(self.pools_dir, "labeled_pool.csv")
        labeled_tmp = labeled_path + ".tmp"
        valid_labeled_df.to_csv(labeled_tmp, index=False)
        os.replace(labeled_tmp, labeled_path)
        self.labeled_df = valid_labeled_df

        # 2. Reconstruct Unlabeled Pool
        # Unlabeled = Full Dataset - Labeled

        # Get all sample IDs from full dataset
        all_samples = [
            os.path.splitext(os.path.basename(x))[0]
            for x in getattr(self.full_dataset, "images", [])
        ]
        labeled_ids = set(valid_labeled_df["sample_id"].tolist())

        unlabeled_ids = [s for s in all_samples if s not in labeled_ids]

        # Create unlabeled DataFrame
        # Assuming original unlabeled pool has 'sample_id' column.
        # If it had other columns (like scores), they are lost for the rolled-back samples,
        # but that's fine as they need to be re-scored anyway.
        unlabeled_df = pd.DataFrame({"sample_id": unlabeled_ids})
        unlabeled_path = os.path.join(self.pools_dir, "unlabeled_pool.csv")
        unlabeled_tmp = unlabeled_path + ".tmp"
        unlabeled_df.to_csv(unlabeled_tmp, index=False)
        os.replace(unlabeled_tmp, unlabeled_path)
        self.unlabeled_df = unlabeled_df

        # Update indices in memory
        self.labeled_indices = self._map_filenames_to_indices(
            self.full_dataset, self.labeled_df["sample_id"].tolist()
        )
        self.unlabeled_indices = self._map_filenames_to_indices(
            self.full_dataset, self.unlabeled_df["sample_id"].tolist()
        )

        logger.info(
            f"Pool rollback complete. Labeled: {len(self.labeled_indices)}, Unlabeled: {len(self.unlabeled_indices)}"
        )
        return True

    def _truncate_trace(self, target_round):
        """
        Remove trace entries that belong to rounds > target_round.
        Also remove duplicate 'epoch_end' for target_round if any (though usually we keep completed round).

        If we are resuming from Round N (completed), we expect next to be Round N+1.
        So we keep everything up to Round N.
        """
        if not os.path.exists(self.trace_path):
            return

        import json

        valid_lines = []
        with open(self.trace_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            try:
                entry = json.loads(line)
                r = entry.get("round")
                if r is None:
                    valid_lines.append(line)  # Keep initialization/metadata lines
                    continue

                if r <= target_round:
                    valid_lines.append(line)
            except Exception:
                continue

        # Add resume marker
        resume_marker = json.dumps(
            {"type": "resume", "round": target_round, "ts": datetime.now().isoformat()}
        )
        valid_lines.append(resume_marker + "\n")

        tmp_path = str(self.trace_path) + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.writelines(valid_lines)
        os.replace(tmp_path, self.trace_path)

    def _save_pool_states(self):
        import pandas as pd

        labeled_rows = []
        unlabeled_rows = []

        for idx in self.labeled_indices:
            img_file = self.full_dataset.images[idx]
            sample_id = os.path.splitext(img_file)[0]
            img_path = os.path.join(self.full_dataset.img_dir, img_file)
            mask_path = os.path.join(self.full_dataset.mask_dir, img_file)
            labeled_rows.append(
                {"sample_id": sample_id, "image_path": img_path, "mask_path": mask_path}
            )

        for idx in self.unlabeled_indices:
            img_file = self.full_dataset.images[idx]
            sample_id = os.path.splitext(img_file)[0]
            img_path = os.path.join(self.full_dataset.img_dir, img_file)
            mask_path = os.path.join(self.full_dataset.mask_dir, img_file)
            unlabeled_rows.append(
                {"sample_id": sample_id, "image_path": img_path, "mask_path": mask_path}
            )

        labeled_df = pd.DataFrame(labeled_rows)
        unlabeled_df = pd.DataFrame(unlabeled_rows)
        pools_dir = self.pools_dir
        os.makedirs(pools_dir, exist_ok=True)
        labeled_path = os.path.join(pools_dir, "labeled_pool.csv")
        unlabeled_path = os.path.join(pools_dir, "unlabeled_pool.csv")
        labeled_tmp = labeled_path + ".tmp"
        unlabeled_tmp = unlabeled_path + ".tmp"
        labeled_df.to_csv(labeled_tmp, index=False)
        unlabeled_df.to_csv(unlabeled_tmp, index=False)
        os.replace(labeled_tmp, labeled_path)
        os.replace(unlabeled_tmp, unlabeled_path)

    def _unpack_images(self, batch):
        if isinstance(batch, dict):
            return batch.get("image")
        if isinstance(batch, (list, tuple)):
            return batch[0] if len(batch) >= 1 else None
        return batch

    def _resolve_feature_layer(self, model):
        layer = None
        if hasattr(model, "backbone"):
            layer = getattr(model.backbone, "layer4", None)
        if (
            layer is None
            and hasattr(model, "model")
            and hasattr(model.model, "encoder")
        ):
            layer = getattr(model.model, "encoder", None)
            if layer is not None:
                layer = getattr(layer, "layer4", None)
        if layer is None:
            candidates = ["features", "avgpool", "layer4", "layer3", "mixed_7c"]
            for name in candidates:
                if hasattr(model, name):
                    layer = getattr(model, name)
                    break
                if hasattr(model, "module") and hasattr(model.module, name):
                    layer = getattr(model.module, name)
                    break
        return layer

    def _extract_features_only(self, model, data_loader):
        model.eval()
        features_list = []

        def hook_fn(module, input, output):
            gap = nn.functional.adaptive_avg_pool2d(output, (1, 1)).flatten(1)
            features_list.append(gap.detach().cpu())

        layer = self._resolve_feature_layer(model)
        handle = layer.register_forward_hook(hook_fn) if layer is not None else None
        try:
            with torch.no_grad():
                for batch in data_loader:
                    images = self._unpack_images(batch)
                    if images is None:
                        continue
                    images = images.to(self.config.DEVICE)
                    _ = model(images)
        finally:
            if handle is not None:
                handle.remove()

        return torch.cat(features_list) if features_list else None

    def _extract_uncertainty_and_features(
        self,
        model,
        data_loader,
        method: str = "entropy",
        n_mc_samples: int = 10,
        pos_class: int | None = None,
        pos_threshold: float = 0.5,
    ):
        method = str(method or "entropy").strip().lower()
        eps = 1e-10
        protocol = None
        if isinstance(getattr(self, "exp_config", None), dict):
            protocol = self.exp_config.get("acquisition_protocol")
        protocol = protocol if isinstance(protocol, dict) else {}
        u_agg = (
            str(protocol.get("uncertainty_aggregation", "mean") or "mean")
            .strip()
            .lower()
        )
        tau = float(protocol.get("entropy_threshold", 0.5) or 0.5)
        min_frac = float(protocol.get("entropy_min_frac", 0.01) or 0.01)

        def _aggregate_uncertainty(ent_map: torch.Tensor) -> torch.Tensor:
            if ent_map.numel() == 0:
                return torch.zeros((int(ent_map.shape[0]),), device=ent_map.device)
            if u_agg in ("mean", "full_mean", "none", ""):
                return ent_map.mean(dim=(1, 2))
            mask = ent_map > float(tau)
            min_keep = max(
                1, int(float(min_frac) * float(ent_map.shape[1] * ent_map.shape[2]))
            )
            keep = mask.view(mask.shape[0], -1).sum(dim=1)
            fallback = ent_map.mean(dim=(1, 2))
            masked_sum = (ent_map * mask).view(ent_map.shape[0], -1).sum(dim=1)
            denom = keep.clamp_min(1).to(ent_map.dtype)
            masked_mean = masked_sum / denom
            return torch.where(keep >= int(min_keep), masked_mean, fallback)

        if method == "bald":
            uncertainties = []
            pos_areas = [] if pos_class is not None else None
            model.eval()
            for module in model.modules():
                if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d)):
                    module.train()
            try:
                with torch.no_grad():
                    for batch in data_loader:
                        images = self._unpack_images(batch)
                        if images is None:
                            continue
                        images = images.to(self.config.DEVICE)
                        batch_size = int(images.shape[0])
                        sum_probs = None
                        sum_ent = torch.zeros((batch_size,), device=self.config.DEVICE)
                        for _ in range(int(n_mc_samples or 10)):
                            logits = model(images)
                            probs = torch.softmax(logits, dim=1)
                            sum_probs = (
                                probs if sum_probs is None else (sum_probs + probs)
                            )
                            ent = -torch.sum(probs * torch.log2(probs + eps), dim=1)
                            sum_ent = sum_ent + _aggregate_uncertainty(ent)
                        mean_probs = sum_probs / float(int(n_mc_samples or 10))
                        pred_ent = -torch.sum(
                            mean_probs * torch.log2(mean_probs + eps), dim=1
                        )
                        mi = _aggregate_uncertainty(pred_ent) - (
                            sum_ent / float(int(n_mc_samples or 10))
                        )
                        uncertainties.extend(mi.detach().cpu().tolist())
                        if pos_areas is not None:
                            channel = mean_probs[:, int(pos_class), :, :]
                            area = (
                                (channel > float(pos_threshold))
                                .float()
                                .mean(dim=(1, 2))
                            )
                            pos_areas.extend(area.detach().cpu().tolist())
            finally:
                model.eval()

            features_tensor = self._extract_features_only(model, data_loader)
            unc_arr = np.asarray(uncertainties, dtype=np.float32)
            pos_arr = (
                np.asarray(pos_areas, dtype=np.float32)
                if pos_areas is not None
                else None
            )
            return unc_arr, features_tensor, pos_arr

        model.eval()
        uncertainties = []
        pos_areas = [] if pos_class is not None else None
        features_list = []

        def hook_fn(module, input, output):
            gap = nn.functional.adaptive_avg_pool2d(output, (1, 1)).flatten(1)
            features_list.append(gap.detach().cpu())

        layer = self._resolve_feature_layer(model)
        handle = layer.register_forward_hook(hook_fn) if layer is not None else None
        try:
            with torch.no_grad():
                for batch in data_loader:
                    images = self._unpack_images(batch)
                    if images is None:
                        continue
                    images = images.to(self.config.DEVICE)
                    logits = model(images)
                    probs = torch.softmax(logits, dim=1)
                    ent = -torch.sum(probs * torch.log2(probs + eps), dim=1)
                    ent_mean = _aggregate_uncertainty(ent)
                    uncertainties.extend(ent_mean.detach().cpu().tolist())
                    if pos_areas is not None:
                        channel = probs[:, int(pos_class), :, :]
                        area = (channel > float(pos_threshold)).float().mean(dim=(1, 2))
                        pos_areas.extend(area.detach().cpu().tolist())
        finally:
            if handle is not None:
                handle.remove()

        features_tensor = torch.cat(features_list) if features_list else None
        unc_arr = np.asarray(uncertainties, dtype=np.float32)
        pos_arr = (
            np.asarray(pos_areas, dtype=np.float32) if pos_areas is not None else None
        )
        return unc_arr, features_tensor, pos_arr

    def _prepare_unlabeled_info(self, model, mc_dropout=False, n_mc_samples=10):
        subset_u = Subset(self.query_dataset, self.unlabeled_indices)
        feature_workers = int(getattr(self.config, "FEATURE_NUM_WORKERS", 0) or 0)
        seed_base = int(self.seed) + int(self.current_round or 0)
        g = torch.Generator()
        g.manual_seed(seed_base)
        feature_loader_kwargs = self._build_loader_kwargs(
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=feature_workers,
            generator=g,
            worker_init_fn=worker_init_fn,
            drop_last=False,
            pin_memory=bool(getattr(self.config, "FEATURE_PIN_MEMORY", False)),
            persistent_workers=bool(
                getattr(self.config, "FEATURE_PERSISTENT_WORKERS", False)
            ),
            prefetch_factor=int(
                getattr(self.config, "FEATURE_PREFETCH_FACTOR", 2) or 2
            ),
        )
        loader_u = DataLoader(subset_u, **feature_loader_kwargs)
        constraints = (
            self.exp_config.get("candidate_constraints")
            if isinstance(getattr(self, "exp_config", None), dict)
            else None
        )
        use_pos_area = bool(
            isinstance(constraints, dict)
            and bool(constraints.get("use_pred_pos_area", False))
        )
        pos_class = int(constraints.get("pos_class", 1)) if use_pos_area else None
        pos_threshold = (
            float(constraints.get("pos_threshold", 0.5)) if use_pos_area else 0.5
        )

        unlabeled_info = {}
        uncertainty_method = "bald" if bool(mc_dropout) else "entropy"
        if not mc_dropout:
            method_from_sampler = getattr(self.sampler, "uncertainty_method", None)
            if isinstance(method_from_sampler, str) and method_from_sampler.strip():
                uncertainty_method = method_from_sampler.strip().lower()
        effective_mc = int(
            getattr(self.sampler, "n_mc_samples", n_mc_samples) or n_mc_samples
        )

        unc_arr, features_tensor, pos_arr = self._extract_uncertainty_and_features(
            model,
            loader_u,
            method=uncertainty_method,
            n_mc_samples=effective_mc,
            pos_class=pos_class,
            pos_threshold=pos_threshold,
        )

        if features_tensor is None:
            raise RuntimeError("_prepare_unlabeled_info failed: features=None")
        if unc_arr is None or int(len(unc_arr)) != int(len(self.unlabeled_indices)):
            raise RuntimeError(
                "_prepare_unlabeled_info failed: uncertainty size mismatch"
            )

        for i, idx in enumerate(self.unlabeled_indices):
            info = {
                "feature": features_tensor[i].numpy().copy(),
                "uncertainty_score": float(unc_arr[i]),
            }
            if pos_arr is not None and i < int(len(pos_arr)):
                info["pos_area"] = float(pos_arr[i])
            unlabeled_info[idx] = info

        labeled_features = None
        if self.labeled_indices:
            subset_l = Subset(self.query_dataset, self.labeled_indices)
            loader_l = DataLoader(subset_l, **feature_loader_kwargs)
            l_feats = self._extract_features_only(model, loader_l)
            if l_feats is None:
                raise RuntimeError(
                    "_prepare_unlabeled_info failed: labeled feature extraction returned None"
                )
            labeled_features = l_feats.numpy()
            self._cleanup_loader(loader_l)
        self._cleanup_loader(loader_u)
        return unlabeled_info, labeled_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="full_model_A_lambda_policy",
        help="Name of the experiment to run",
    )
    parser.add_argument(
        "--run_id", type=str, required=True, help="Run id for research isolation"
    )
    parser.add_argument(
        "--start",
        type=str,
        default="resume",
        choices=["resume", "fresh"],
        help="Start mode: resume or fresh",
    )
    parser.add_argument(
        "--n_rounds",
        type=int,
        default=None,
        help="Number of active learning rounds to run (default: use config value)",
    )
    args = parser.parse_args()

    config = Config()
    try:
        strategy = getattr(config, "SHARING_STRATEGY", "file_descriptor")
        torch.multiprocessing.set_sharing_strategy(strategy)
    except Exception:
        pass
    try:
        num_threads = int(getattr(config, "TORCH_NUM_THREADS", 0) or 0)
        if num_threads > 0:
            torch.set_num_threads(num_threads)
    except Exception:
        pass
    try:
        num_interop = int(getattr(config, "TORCH_NUM_INTEROP_THREADS", 0) or 0)
        if num_interop > 0:
            torch.set_num_interop_threads(num_interop)
    except Exception:
        pass
    try:
        if (
            str(getattr(config, "DEVICE", "")).lower() == "cuda"
            and torch.cuda.is_available()
        ):
            if bool(getattr(config, "TF32", False)):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                try:
                    torch.set_float32_matmul_precision("high")
                except Exception:
                    pass
            torch.backends.cudnn.benchmark = bool(
                getattr(config, "CUDNN_BENCHMARK", False)
            )
    except Exception:
        pass
    config.START_MODE = args.start
    if args.n_rounds is not None and args.n_rounds > 0:
        config.N_ROUNDS = int(args.n_rounds)
    pipeline = ActiveLearningPipeline(config, args.experiment_name, run_id=args.run_id)
    pipeline.run()


if __name__ == "__main__":
    main()
