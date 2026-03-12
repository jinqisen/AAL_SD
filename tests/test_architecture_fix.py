import sys
from unittest.mock import MagicMock

# Mock heavy dependencies before importing src modules
mock_torch = MagicMock()
sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['torch.optim'] = MagicMock()
sys.modules['torch.utils'] = MagicMock()
sys.modules['torch.utils.data'] = MagicMock()
sys.modules['torchvision'] = MagicMock()
sys.modules['torchvision.transforms'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.model_selection'] = MagicMock()
sys.modules['sklearn.metrics'] = MagicMock()
sys.modules['sklearn.cluster'] = MagicMock()
sys.modules['scipy'] = MagicMock()
sys.modules['scipy.spatial'] = MagicMock()
sys.modules['scipy.spatial.distance'] = MagicMock()
sys.modules['scipy.stats'] = MagicMock()
sys.modules['segmentation_models_pytorch'] = MagicMock()
sys.modules['tqdm'] = MagicMock()

import unittest
import os
import shutil
import tempfile
import copy
import pandas as pd
import json
import hashlib
from unittest.mock import patch

# Add src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Now we can safely import modules
from agent.toolbox import Toolbox
from agent.agent_manager import AgentManager
from main import ActiveLearningPipeline
from config import Config
from experiments.ablation_config import ABLATION_SETTINGS

class TestArchitectureFix(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for pools
        self.test_dir = tempfile.mkdtemp()
        self.config = Config()
        self.config.REQUIRE_LLM_FOR_AGENT = False
        self.config.POOLS_DIR = os.path.join(self.test_dir, 'pools')
        self.config.DATA_DIR = os.path.join(self.test_dir, 'data')
        self.config.RESULTS_DIR = os.path.join(self.test_dir, 'results')
        self.config.CHECKPOINT_DIR = os.path.join(self.config.RESULTS_DIR, 'checkpoints')
        self.config.DEVICE = 'cpu'
        os.makedirs(self.config.POOLS_DIR, exist_ok=True)
        os.makedirs(self.config.DATA_DIR, exist_ok=True)
        os.makedirs(self.config.RESULTS_DIR, exist_ok=True)
        os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)
        for rel in (
            ("TrainData", "img"),
            ("TrainData", "mask"),
            ("ValidData", "img"),
            ("ValidData", "mask"),
        ):
            os.makedirs(os.path.join(self.config.DATA_DIR, *rel), exist_ok=True)
        base_dir = os.path.join(self.config.POOLS_DIR, "unit_test", "_base")
        os.makedirs(base_dir, exist_ok=True)
        pd.DataFrame({"sample_id": []}).to_csv(os.path.join(base_dir, "labeled_pool.csv"), index=False)
        pd.DataFrame({"sample_id": []}).to_csv(os.path.join(base_dir, "unlabeled_pool.csv"), index=False)
        def _sha(names):
            h = hashlib.sha256()
            for n in sorted(names):
                h.update(str(n).encode("utf-8", errors="ignore"))
                h.update(b"\n")
            return h.hexdigest()

        empty = {"count": 0, "sha256": _sha([])}
        manifest = {
            "schema_version": 1,
            "created_at": "1970-01-01T00:00:00",
            "data_root": str(self.config.DATA_DIR),
            "splits": {
                "train": {"images": empty, "masks": empty},
                "val": {"images": empty, "masks": empty},
                "test": None,
            },
            "split_policy": {
                "name": "unit_test",
                "initial_labeled_size": 0.0,
                "random_seed": 0,
                "stratify_key": "has_positive",
                "train_counts": {"pos": 0, "neg": 0, "total": 0},
            },
            "pools": {"labeled": 0, "unlabeled": 0, "files": {"labeled_pool": "labeled_pool.csv", "unlabeled_pool": "unlabeled_pool.csv"}},
        }
        with open(os.path.join(base_dir, "pools_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_toolbox_reset(self):
        """Test if Toolbox.reset() clears cache and scores."""
        # Mock dependencies
        controller = MagicMock()
        strategy = MagicMock()
        model = MagicMock()
        
        toolbox = Toolbox(controller, strategy, model)
        
        # Modify state
        toolbox.candidates_cache = [{'id': '1', 'score': 0.9}]
        toolbox.current_scores = {'1': {'U': 0.5, 'K': 0.4}}
        toolbox.training_state = {'epoch': 5}
        
        # Act
        toolbox.reset()
        
        # Assert
        self.assertEqual(toolbox.candidates_cache, [])
        self.assertEqual(toolbox.current_scores, {})
        self.assertEqual(toolbox.training_state, {})
        print("\n✅ Toolbox.reset() works correctly.")

    def test_agent_manager_reset(self):
        """Test if AgentManager.reset() clears history and resets toolbox."""
        # Mock dependencies
        toolbox = MagicMock()
        client = MagicMock()
        
        agent = AgentManager(toolbox, client, verbose=False)
        
        # Modify state
        agent.history = [{'role': 'user', 'content': 'hi'}]
        
        # Act
        agent.reset()
        
        # Assert
        self.assertEqual(agent.history, [])
        toolbox.reset.assert_called_once()
        print("✅ AgentManager.reset() works correctly.")

    @patch('main.Landslide4SenseDataset')
    @patch('main.DataPreprocessor')
    @patch('main.SiliconFlowClient')
    def test_pipeline_initialization_resets_agent(self, MockClient, MockPreprocessor, MockDataset):
        """Test if ActiveLearningPipeline resets agent on initialization."""
        self.config.STRICT_RESUME = False
        # Setup mocks
        mock_dataset_instance = MockDataset.return_value
        mock_dataset_instance.images = [] # Empty dataset
        MockPreprocessor.return_value.create_data_pools.return_value = None

        with patch('main.ABLATION_SETTINGS', {'test_exp': {'description': 'test', 'sampler_type': 'ad_kucs', 'use_agent': True, 'control_permissions': {}}}):
            pipeline = ActiveLearningPipeline(self.config, 'test_exp', run_id='unit_test')
            self.assertEqual(pipeline.agent_manager.history, [])
            self.assertEqual(pipeline.toolbox.candidates_cache, [])
            pipeline.agent_manager.history.append('dirty')
            pipeline.agent_manager.reset()
            self.assertEqual(pipeline.agent_manager.history, [])
            print("✅ Pipeline initializes with clean Agent state.")

    @patch('main.Landslide4SenseDataset')
    @patch('main.DataPreprocessor')
    def test_load_pool_states(self, MockPreprocessor, MockDataset):
        """Test _load_pool_states functionality."""
        exp_name = 'resume_exp'
        exp_dir = os.path.join(self.config.POOLS_DIR, 'unit_test', exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Create dummy pool files
        pd.DataFrame({'sample_id': ['img1', 'img2']}).to_csv(os.path.join(exp_dir, 'labeled_pool.csv'), index=False)
        pd.DataFrame({'sample_id': ['img3']}).to_csv(os.path.join(exp_dir, 'unlabeled_pool.csv'), index=False)
        
        # Mock dataset to have these images
        mock_dataset_instance = MockDataset.return_value
        # Mock images list for _map_filenames_to_indices
        mock_dataset_instance.images = ['img1.h5', 'img2.h5', 'img3.h5']
        
        # Mock ABLATION_SETTINGS
        with patch('main.ABLATION_SETTINGS', {exp_name: {'description': 'test', 'sampler_type': 'random', 'use_agent': False}}):
             # Initialize pipeline
             # __init__ will load pools from disk if they exist.
             pipeline = ActiveLearningPipeline(self.config, exp_name, run_id='unit_test')
             
             # Verify initial load worked (proving __init__ uses correct paths)
             self.assertEqual(set(pipeline.labeled_indices), {0, 1}) # img1, img2
             self.assertEqual(set(pipeline.unlabeled_indices), {2})  # img3
             
             # Now clear state manually to test _load_pool_states explicitly
             pipeline.labeled_indices = []
             pipeline.unlabeled_indices = []
             
             # Call _load_pool_states
             result = pipeline._load_pool_states()
             
             self.assertTrue(result)
             # img1(0), img2(1) should be labeled
             self.assertEqual(set(pipeline.labeled_indices), {0, 1})
             # img3(2) should be unlabeled
             self.assertEqual(set(pipeline.unlabeled_indices), {2})
             
             print("✅ _load_pool_states loads state correctly.")

    def test_ablation_settings_have_distinct_effective_configs(self):
        def freeze(v):
            if isinstance(v, dict):
                return tuple(sorted((str(k), freeze(val)) for k, val in v.items()))
            if isinstance(v, (list, tuple)):
                return tuple(freeze(x) for x in v)
            return v

        signatures = {}
        duplicates = []
        for name, cfg in (ABLATION_SETTINGS or {}).items():
            effective = {
                "use_agent": cfg.get("use_agent", False),
                "sampler_type": cfg.get("sampler_type"),
                "k_definition": cfg.get("k_definition", "coreset_to_labeled"),
                "score_normalization": cfg.get("score_normalization", True),
                "uncertainty_method": cfg.get("uncertainty_method"),
                "n_mc_samples": cfg.get("n_mc_samples"),
                "acquisition_protocol": cfg.get("acquisition_protocol"),
                "lambda_override": cfg.get("lambda_override"),
                "lambda_controller": cfg.get("lambda_controller"),
                "lambda_policy": cfg.get("lambda_policy"),
                "grad_probe_source": cfg.get("grad_probe_source"),
                "grad_probe_holdout_frac": cfg.get("grad_probe_holdout_frac"),
                "grad_probe_holdout_min": cfg.get("grad_probe_holdout_min"),
                "rollback_config": cfg.get("rollback_config"),
                "control_permissions": cfg.get("control_permissions"),
                "agent_threshold_overrides": cfg.get("agent_threshold_overrides"),
                "epochs_per_round_override": cfg.get("epochs_per_round_override"),
            }
            sig = freeze(effective)
            other = signatures.get(sig)
            if other is not None:
                duplicates.append((other, name))
            else:
                signatures[sig] = name

        self.assertEqual(duplicates, [], f"Duplicate effective ablation configs found: {duplicates}")

    def test_toolbox_lambda_policy_threshold_overrides_change_lambda(self):
        class DummyController:
            def __init__(self, exp_config):
                self.exp_config = exp_config
                self.current_round = 5
                self.config = MagicMock()
                self.config.RANDOM_SEED = 123
                self.events = []

            def _append_trace(self, event):
                self.events.append(event)

        def make_toolbox(exp_config):
            controller = DummyController(exp_config)
            toolbox = Toolbox(controller, MagicMock(), MagicMock())
            toolbox._last_lambda_applied = 0.45
            toolbox.current_scores = {
                "1": {"U": 0.1, "K": 0.9},
                "2": {"U": 0.1, "K": 0.9},
            }
            toolbox.set_training_state(
                {
                    "round_idx": 5,
                    "last_miou": 0.5,
                    "prev_miou": 0.49,
                    "miou_delta": 0.0,
                    "rollback_flag": False,
                    "rollback_mode": "adaptive_threshold",
                    "rollback_threshold": 0.0,
                    "k_definition": "coreset_to_labeled",
                    "current_labeled_count": 10,
                    "total_budget": 100,
                    "overfit_risk": 0.1,
                }
            )
            return toolbox, controller

        cfg_full = ABLATION_SETTINGS["full_model_A_lambda_policy"]
        cfg_alt = copy.deepcopy(cfg_full)
        cfg_alt.setdefault("agent_threshold_overrides", {})
        cfg_alt["agent_threshold_overrides"]["LAMBDA_DELTA_UP"] = 0.10
        tb_full, c_full = make_toolbox(cfg_full)
        tb_alt, c_alt = make_toolbox(cfg_alt)

        lam_full = tb_full.apply_round_lambda_policy()
        lam_alt = tb_alt.apply_round_lambda_policy()

        self.assertNotEqual(lam_full, lam_alt)
        self.assertTrue(any(e.get("type") == "lambda_policy_apply" for e in c_full.events))
        self.assertTrue(any(e.get("type") == "lambda_policy_apply" for e in c_alt.events))

    def test_lambda_policy_r1_lambda_zero_applies_in_r1_r2(self):
        class DummyController:
            def __init__(self, exp_config, current_round):
                self.exp_config = exp_config
                self.current_round = current_round
                self.config = MagicMock()
                self.config.RANDOM_SEED = 123
                self.events = []

            def _append_trace(self, event):
                self.events.append(event)

        for r in (1, 2):
            controller = DummyController(ABLATION_SETTINGS["full_model_A_lambda_policy"], current_round=r)
            toolbox = Toolbox(controller, MagicMock(), MagicMock())
            toolbox._last_lambda_applied = 0.55
            toolbox.set_training_state(
                {
                    "round_idx": r,
                    "last_miou": 0.5,
                    "prev_miou": 0.49,
                    "miou_delta": 0.01,
                    "rollback_flag": False,
                    "rollback_mode": "adaptive_threshold",
                    "rollback_threshold": 0.0,
                    "k_definition": "coreset_to_labeled",
                    "current_labeled_count": 10,
                    "total_budget": 100,
                    "overfit_risk": 0.1,
                }
            )
            lam = toolbox.apply_round_lambda_policy()
            self.assertEqual(lam, 0.0)
            self.assertEqual(float(toolbox.control_state.get("lambda_override_round")), 0.0)
            self.assertTrue(any(e.get("type") == "lambda_policy_apply" and int(e.get("round")) == r for e in controller.events))

    def test_set_lambda_allows_r1_lambda_below_clamp_min_in_uncertainty_phase(self):
        class DummyController:
            def __init__(self, exp_config, current_round):
                self.exp_config = exp_config
                self.current_round = current_round
                self.config = MagicMock()
                self.config.RANDOM_SEED = 123
                self.events = []

            def _append_trace(self, event):
                self.events.append(event)

        def make_toolbox(round_idx):
            exp_cfg = ABLATION_SETTINGS["full_model_B_lambda_agent"]
            controller = DummyController(exp_cfg, current_round=round_idx)
            toolbox = Toolbox(controller, MagicMock(), MagicMock())
            toolbox.set_control_permissions(exp_cfg.get("control_permissions") or {})
            toolbox.set_training_state(
                {
                    "round_idx": round_idx,
                    "last_miou": 0.5,
                    "prev_miou": 0.49,
                    "miou_delta": 0.01,
                    "rollback_flag": False,
                    "rollback_mode": "adaptive_threshold",
                    "rollback_threshold": 0.0,
                    "k_definition": "coreset_to_labeled",
                    "current_labeled_count": 10,
                    "total_budget": 100,
                    "overfit_risk": 0.1,
                }
            )
            return toolbox

        tb_r1 = make_toolbox(1)
        out_r1 = json.loads(tb_r1.set_lambda(0.0, scope="round"))
        self.assertEqual(out_r1.get("status"), "success")
        self.assertEqual(float(out_r1.get("applied")), 0.0)
        self.assertEqual(float(tb_r1.control_state.get("lambda_override_round")), 0.0)

        tb_r3 = make_toolbox(3)
        out_r3 = json.loads(tb_r3.set_lambda(0.0, scope="round"))
        self.assertEqual(out_r3.get("status"), "success")
        self.assertGreaterEqual(float(out_r3.get("applied")), 0.05)
        self.assertEqual(float(tb_r3.control_state.get("lambda_override_round")), float(out_r3.get("applied")))

    def test_get_top_k_samples_uses_lambda_policy_bounds_in_r1(self):
        class DummyController:
            def __init__(self, exp_config, current_round):
                self.exp_config = exp_config
                self.current_round = current_round
                self.config = MagicMock()
                self.config.RANDOM_SEED = 123

        controller = DummyController(ABLATION_SETTINGS["full_model_A_lambda_policy"], current_round=1)
        toolbox = Toolbox(controller, MagicMock(), MagicMock())
        toolbox.current_scores = {
            "1": {"U": 1.0, "K": 0.0},
            "2": {"U": 0.0, "K": 1.0},
        }
        toolbox.set_training_state(
            {
                "round_idx": 1,
                "last_miou": 0.5,
                "prev_miou": 0.49,
                "miou_delta": 0.01,
                "rollback_flag": False,
                "rollback_mode": "adaptive_threshold",
                "rollback_threshold": 0.0,
                "k_definition": "coreset_to_labeled",
                "current_labeled_count": 10,
                "total_budget": 100,
                "overfit_risk": 0.1,
            }
        )
        out = json.loads(toolbox.get_top_k_samples(k=1))
        self.assertEqual(out.get("status"), "success")
        meta = out.get("meta") or {}
        self.assertEqual(float(meta.get("lambda_param")), 0.0)
        items = out.get("result") or []
        self.assertEqual(items[0].get("id"), "1")

    @patch("main.Landslide4SenseDataset")
    @patch("main.DataPreprocessor")
    @patch("main.SiliconFlowClient")
    def test_agent_finalize_selection_writes_l3_selection(self, MockClient, MockPreprocessor, MockDataset):
        self.config.STRICT_RESUME = False
        self.config.STRICT_INNOVATION = False
        self.config.QUERY_SIZE = 1

        mock_dataset_instance = MockDataset.return_value
        mock_dataset_instance.images = ["s0.h5", "s1.h5", "s2.h5"]
        MockPreprocessor.return_value.create_data_pools.return_value = None

        def fake_read_csv(path, *args, **kwargs):
            filename = os.path.basename(str(path))
            if filename == "labeled_pool.csv":
                return pd.DataFrame({"sample_id": ["s0"]})
            if filename == "unlabeled_pool.csv":
                return pd.DataFrame({"sample_id": ["s1", "s2"]})
            return pd.DataFrame({"sample_id": []})

        with patch("pandas.read_csv", side_effect=fake_read_csv):
            with patch("main.ABLATION_SETTINGS", {"full_model_A_lambda_policy": ABLATION_SETTINGS["full_model_A_lambda_policy"]}):
                pipeline = ActiveLearningPipeline(self.config, "full_model_A_lambda_policy", run_id="unit_test")

        events = []
        pipeline._append_trace = lambda e: events.append(e)
        pipeline._last_ranked_items = []
        pipeline.toolbox.candidates_cache = [
            {"id": "1", "U_score": 0.9, "K_score": 0.1, "final_score": 0.5},
            {"id": "2", "U_score": 0.1, "K_score": 0.9, "final_score": 0.5},
        ]
        pipeline.toolbox.control_state["lambda_override_round"] = 0.55
        pipeline.toolbox.set_training_state(
            {
                "round_idx": 1,
                "last_miou": 0.5,
                "prev_miou": 0.49,
                "miou_delta": 0.01,
                "rollback_flag": False,
                "rollback_mode": "adaptive_threshold",
                "rollback_threshold": 0.0,
                "k_definition": "coreset_to_labeled",
                "current_labeled_count": 1,
                "total_budget": 100,
                "overfit_risk": 0.1,
            }
        )
        pipeline.current_round = 1
        pipeline.toolbox.finalize_selection(["1"], reason="unit_test", thought=None)

        self.assertTrue(any(e.get("type") == "l3_selection" for e in events))

    def test_lambda_policy_overrides_change_sampler_ranking(self):
        class DummySampler:
            def __init__(self, *args, **kwargs):
                pass

            def _get_adaptive_weight(self, current_iteration, total_iterations, override=None):
                return override if override is not None else 0.5

            def rank_samples(
                self,
                unlabeled_info,
                labeled_features,
                current_iteration,
                total_iterations,
                lambda_override=None,
            ):
                lam = 0.5 if lambda_override is None else float(lambda_override)
                ranked = []
                for sid, info in unlabeled_info.items():
                    u = float(info["uncertainty"])
                    k = float(info["knowledge_gain"])
                    score = (1.0 - lam) * u + lam * k
                    ranked.append(
                        {
                            "sample_id": sid,
                            "uncertainty": u,
                            "knowledge_gain": k,
                            "final_score": score,
                            "lambda_t": lam,
                        }
                    )
                ranked.sort(key=lambda x: x["final_score"], reverse=True)
                return ranked

        class DummyController:
            def __init__(self, exp_config):
                self.exp_config = exp_config
                self.current_round = 5
                self.config = MagicMock()
                self.config.RANDOM_SEED = 123

            def _append_trace(self, event):
                return None

        def make_lambda(exp_config):
            controller = DummyController(exp_config)
            toolbox = Toolbox(controller, MagicMock(), MagicMock())
            toolbox._last_lambda_applied = 0.45
            toolbox.current_scores = {
                "1": {"U": 0.1, "K": 0.9},
                "2": {"U": 0.1, "K": 0.9},
            }
            toolbox.set_training_state(
                {
                    "round_idx": 5,
                    "last_miou": 0.5,
                    "prev_miou": 0.49,
                    "miou_delta": 0.0,
                    "rollback_flag": False,
                    "rollback_mode": "adaptive_threshold",
                    "rollback_threshold": 0.0,
                    "k_definition": "coreset_to_labeled",
                    "current_labeled_count": 10,
                    "total_budget": 100,
                    "overfit_risk": 0.1,
                }
            )
            return toolbox.apply_round_lambda_policy()

        unlabeled_info = {
            1: {"uncertainty": 0.9, "knowledge_gain": 0.1},
            2: {"uncertainty": 0.1, "knowledge_gain": 0.9},
        }
        sampler = DummySampler()
        cfg_full = ABLATION_SETTINGS["full_model_A_lambda_policy"]
        cfg_alt = copy.deepcopy(cfg_full)
        cfg_alt.setdefault("agent_threshold_overrides", {})
        cfg_alt["agent_threshold_overrides"]["LAMBDA_DELTA_UP"] = 0.10
        lam_full = make_lambda(cfg_full)
        lam_alt = make_lambda(cfg_alt)
        ranked_full = sampler.rank_samples(
            unlabeled_info, None, current_iteration=10, total_iterations=100, lambda_override=lam_full
        )
        ranked_alt = sampler.rank_samples(
            unlabeled_info, None, current_iteration=10, total_iterations=100, lambda_override=lam_alt
        )
        self.assertNotAlmostEqual(float(lam_full), float(lam_alt), places=6)
        self.assertNotAlmostEqual(float(ranked_full[0]["lambda_t"]), float(ranked_alt[0]["lambda_t"]), places=6)
        self.assertNotAlmostEqual(float(ranked_full[0]["final_score"]), float(ranked_alt[0]["final_score"]), places=6)

    def test_resolve_lambda_override_precedence_and_strictness(self):
        pipeline = ActiveLearningPipeline.__new__(ActiveLearningPipeline)
        pipeline.current_round = 3

        class DummyToolbox:
            def __init__(self, control_state, applied):
                self.control_state = dict(control_state or {})
                self._applied = applied

            def apply_round_lambda_policy(self):
                return self._applied

        pipeline.use_agent = True
        pipeline.exp_config = {"lambda_override": 0.11, "lambda_policy": {"mode": "warmup_risk_closed_loop"}}
        pipeline.toolbox = DummyToolbox({"lambda_override_round": 0.22}, applied=0.33)
        value, source = pipeline._resolve_lambda_override(pipeline.current_round)
        self.assertAlmostEqual(value, 0.11, places=6)
        self.assertEqual(source, "exp_override")

        pipeline.exp_config = {"lambda_policy": {"mode": "warmup_risk_closed_loop"}}
        pipeline.toolbox = DummyToolbox({"lambda_override_round": 0.22}, applied=0.33)
        value, source = pipeline._resolve_lambda_override(pipeline.current_round)
        self.assertAlmostEqual(value, 0.22, places=6)
        self.assertEqual(source, "agent_override")

        pipeline.exp_config = {"lambda_policy": {"mode": "warmup_risk_closed_loop"}}
        pipeline.toolbox = DummyToolbox({}, applied=0.33)
        value, source = pipeline._resolve_lambda_override(pipeline.current_round)
        self.assertAlmostEqual(value, 0.33, places=6)
        self.assertEqual(source, "lambda_policy")

        pipeline.exp_config = {"lambda_policy": {"mode": "warmup_risk_closed_loop"}}
        pipeline.toolbox = DummyToolbox({}, applied=None)
        with self.assertRaises(RuntimeError):
            pipeline._resolve_lambda_override(pipeline.current_round)

        pipeline.use_agent = False
        pipeline.exp_config = {"lambda_controller": {"mode": "rule_based", "rule": "r1", "lambda_min": 0.0, "lambda_max": 1.0}}
        pipeline._apply_lambda_controller = lambda round_idx: 0.44
        value, source = pipeline._resolve_lambda_override(pipeline.current_round)
        self.assertAlmostEqual(value, 0.44, places=6)
        self.assertEqual(source, "lambda_controller")

    def test_sampler_audit_uses_lambda_effective(self):
        pipeline = ActiveLearningPipeline.__new__(ActiveLearningPipeline)
        pipeline.rollback_config = {"mode": "adaptive_threshold", "threshold": 0.1, "std_factor": 1.5}
        pipeline.exp_config = {}
        pipeline.sampler_type = "ad_kucs"
        pipeline.k_definition = "coreset_to_labeled"
        pipeline.score_normalization = True
        pipeline.sampler = MagicMock()
        pipeline._last_ranking_metadata = {"lambda_effective": 0.55, "lambda_source": "lambda_policy"}

        audit = pipeline._sampler_audit()
        self.assertAlmostEqual(audit.get("lambda_effective"), 0.55, places=6)
        self.assertEqual(audit.get("lambda_source"), "lambda_policy")

if __name__ == '__main__':
    unittest.main()
