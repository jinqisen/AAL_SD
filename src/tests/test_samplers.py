import unittest
import numpy as np
import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from baselines.random_sampler import RandomSampler
from baselines.entropy_sampler import EntropySampler
from baselines.coreset_sampler import CoresetSampler
from baselines.bald_sampler import BALDSampler
from baselines.llm_us_sampler import LLMUncertaintySampler
from baselines.dial_sampler import DIALStyleSampler
from baselines.wang_sampler import WangStyleSampler
from core.sampler import ADKUCSSampler
from utils.evaluation import calculate_alc


class TestRandomSampler(unittest.TestCase):
    def setUp(self):
        self.sampler = RandomSampler()
        self.unlabeled_info = {
            "sample_1": {"prob_map": np.random.rand(2, 128, 128)},
            "sample_2": {"prob_map": np.random.rand(2, 128, 128)},
            "sample_3": {"prob_map": np.random.rand(2, 128, 128)}
        }

    def test_output_structure(self):
        result = self.sampler.rank_samples(self.unlabeled_info)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        
        for item in result:
            self.assertIn("sample_id", item)
            self.assertIn("final_score", item)
            self.assertIsInstance(item["sample_id"], str)
            self.assertIsInstance(item["final_score"], float)
            self.assertGreaterEqual(item["final_score"], 0.0)
            self.assertLessEqual(item["final_score"], 1.0)

    def test_empty_input(self):
        result = self.sampler.rank_samples({})
        self.assertEqual(result, [])


class TestEntropySampler(unittest.TestCase):
    def setUp(self):
        self.sampler = EntropySampler()
        self.unlabeled_info = {
            "sample_1": {"prob_map": np.array([[[0.9, 0.1], [0.1, 0.9]], [[0.1, 0.9], [0.9, 0.1]]])},
            "sample_2": {"prob_map": np.array([[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]])},
            "sample_3": {"prob_map": np.array([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]])}
        }

    def test_output_structure(self):
        result = self.sampler.rank_samples(self.unlabeled_info)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        
        for item in result:
            self.assertIn("sample_id", item)
            self.assertIn("final_score", item)
            self.assertIsInstance(item["sample_id"], str)
            self.assertIsInstance(item["final_score"], float)

    def test_entropy_calculation(self):
        result = self.sampler.rank_samples(self.unlabeled_info)
        
        sample_2 = next(item for item in result if item["sample_id"] == "sample_2")
        sample_3 = next(item for item in result if item["sample_id"] == "sample_3")
        
        self.assertGreater(sample_2["final_score"], sample_3["final_score"])

    def test_missing_prob_map(self):
        unlabeled_info = {
            "sample_1": {"feature": np.random.rand(512)},
            "sample_2": {}
        }
        result = self.sampler.rank_samples(unlabeled_info)
        
        self.assertEqual(len(result), 2)
        for item in result:
            self.assertEqual(item["final_score"], 0.0)


class TestCoresetSampler(unittest.TestCase):
    def setUp(self):
        self.sampler = CoresetSampler()
        # Greedy selection test case:
        # Labeled: (0,0)
        # u1: (10, 0) -> far from L
        # u2: (10, 0.1) -> far from L, but close to u1
        # u3: (5, 0) -> medium from L
        self.labeled_features = np.array([[0.0, 0.0]])
        self.unlabeled_info = {
            "u1": {"feature": np.array([10.0, 0.0])},
            "u2": {"feature": np.array([10.0, 0.1])},
            "u3": {"feature": np.array([5.0, 0.0])}
        }

    def test_greedy_selection(self):
        # Without greedy: u2 (~10), u1 (10), u3 (5) -> [u2, u1, u3]
        # With greedy:
        # 1. Pick u2 (max dist to L ~10.0005)
        # 2. Update dists:
        #    u1: min(10, dist(u1, u2)=0.1) = 0.1
        #    u3: min(5, dist(u3, u2)~5) = 5
        #    Pick u3 (5 > 0.1)
        # 3. Pick u1
        # Expected: [u2, u3, u1]
        
        result = self.sampler.rank_samples(self.unlabeled_info, labeled_features=self.labeled_features)
        ids = [item["sample_id"] for item in result]
        
        self.assertEqual(ids[0], "u2")
        self.assertEqual(ids[1], "u3")
        self.assertEqual(ids[2], "u1")

    def test_output_structure(self):
        result = self.sampler.rank_samples(self.unlabeled_info, labeled_features=self.labeled_features)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        
        for item in result:
            self.assertIn("sample_id", item)


class TestLLMUncertaintySampler(unittest.TestCase):
    def setUp(self):
        self.sampler = LLMUncertaintySampler()
        self.unlabeled_info = {
            "s1": {"uncertainty_score": 0.9},
            "s2": {"uncertainty_score": 0.1},
            "s3": {"U": 0.5} # Fallback key
        }

    def test_ranking(self):
        result = self.sampler.rank_samples(self.unlabeled_info)
        ids = [item["sample_id"] for item in result]
        # 0.9 > 0.5 > 0.1
        self.assertEqual(ids[0], "s1")
        self.assertEqual(ids[1], "s3")
        self.assertEqual(ids[2], "s2")

    def test_missing_scores_raises_error(self):
        bad_info = {
            "s1": {}, # Missing
            "s2": {"uncertainty_score": None} # None
        }
        with self.assertRaises(ValueError):
            self.sampler.rank_samples(bad_info)

    def test_partial_missing_scores(self):
        info = {
            "s1": {"uncertainty_score": 0.8},
            "s2": {}
        }
        result = self.sampler.rank_samples(info)
        self.assertEqual(len(result), 2)
        s2 = next(item for item in result if item["sample_id"] == "s2")
        self.assertEqual(s2["final_score"], 0.0)


class TestBALDSampler(unittest.TestCase):
    def setUp(self):
        self.sampler = BALDSampler()
        # Mock MC predictions: (n_samples=10, C=2, H=1, W=1)
        self.n_samples = 10
        self.mc_preds_high_uncertainty = np.zeros((self.n_samples, 2, 1, 1))
        # Half predict class 0, Half predict class 1 -> High disagreement -> High BALD
        self.mc_preds_high_uncertainty[:5, 0, :, :] = 1.0
        self.mc_preds_high_uncertainty[5:, 1, :, :] = 1.0
        
        self.mc_preds_low_uncertainty = np.zeros((self.n_samples, 2, 1, 1))
        # All predict class 0 -> No disagreement -> Low BALD (0)
        self.mc_preds_low_uncertainty[:, 0, :, :] = 1.0
        
        self.unlabeled_info = {
            "sample_high": {},
            "sample_low": {}
        }

    def test_rank_with_mc_predictions(self):
        mc_predictions = np.stack([self.mc_preds_high_uncertainty, self.mc_preds_low_uncertainty])
        
        result = self.sampler.rank_samples(
            self.unlabeled_info, 
            mc_predictions=mc_predictions,
            sample_indices=["sample_high", "sample_low"]
        )
        
        self.assertEqual(len(result), 2)
        sample_high = next(item for item in result if item["sample_id"] == "sample_high")
        sample_low = next(item for item in result if item["sample_id"] == "sample_low")
        
        self.assertGreater(sample_high["final_score"], sample_low["final_score"])
        self.assertAlmostEqual(sample_low["final_score"], 0.0)

    def test_missing_mc_predictions_raises_error(self):
        with self.assertRaises(ValueError):
            self.sampler.rank_samples(self.unlabeled_info)


class TestDialAndWangSamplers(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.unlabeled_info = {}
        for i in range(6):
            raw = rng.random(size=(2, 4, 4)).astype(np.float32)
            prob_map = raw / np.sum(raw, axis=0, keepdims=True)
            self.unlabeled_info[f"s{i}"] = {
                "prob_map": prob_map,
                "feature": rng.normal(size=(8,)).astype(np.float32),
            }

    def test_dial_style_outputs(self):
        sampler = DIALStyleSampler()
        ranked = sampler.rank_samples(self.unlabeled_info)
        self.assertEqual(len(ranked), len(self.unlabeled_info))
        for item in ranked:
            self.assertIn("sample_id", item)
            self.assertIn("final_score", item)

    def test_wang_style_outputs(self):
        sampler = WangStyleSampler()
        ranked = sampler.rank_samples(self.unlabeled_info)
        self.assertEqual(len(ranked), len(self.unlabeled_info))
        for item in ranked:
            self.assertIn("sample_id", item)
            self.assertIn("final_score", item)


class TestALCCalculation(unittest.TestCase):
    def test_alc_total_budget_padding(self):
        perf = [0.2, 0.4]
        budget = [100, 200]
        alc = calculate_alc(perf, budget, total_budget=400, pad_to_total_budget=True)
        self.assertAlmostEqual(alc, 0.275, places=6)

    def test_alc_legacy_span_normalization(self):
        perf = [0.2, 0.4]
        budget = [100, 200]
        alc = calculate_alc(perf, budget)
        self.assertAlmostEqual(alc, 0.3, places=6)

    def test_alc_deduplicate_budgets(self):
        perf = [0.1, 0.2, 0.3]
        budget = [100, 100, 200]
        alc = calculate_alc(perf, budget, total_budget=200, pad_to_total_budget=True)
        self.assertAlmostEqual(alc, 0.125, places=6)


class TestADKUCSSCoresetToLabeled(unittest.TestCase):
    def test_coreset_to_labeled_scores_equivalent(self):
        rng = np.random.default_rng(123)
        labeled = rng.normal(size=(12, 16)).astype(np.float64)
        unlabeled = rng.normal(size=(9, 16)).astype(np.float64)

        def ref_max_pairwise_distance(x: np.ndarray) -> float:
            max_d = 0.0
            for i in range(len(x)):
                for j in range(i + 1, len(x)):
                    d = float(np.linalg.norm(x[i] - x[j]))
                    if d > max_d:
                        max_d = d
            return max_d

        max_d = ref_max_pairwise_distance(labeled)
        if max_d < 1e-10:
            expected = np.zeros((len(unlabeled),), dtype=np.float64)
        else:
            expected = np.array(
                [
                    float(np.min(np.linalg.norm(labeled - u, axis=1)) / max_d)
                    for u in unlabeled
                ],
                dtype=np.float64,
            )

        sampler = ADKUCSSampler()
        got = sampler._coreset_to_labeled_scores(unlabeled, labeled)

        self.assertEqual(got.shape, expected.shape)
        self.assertTrue(np.all(np.isfinite(got)))
        self.assertTrue(np.allclose(got, expected, rtol=1e-6, atol=1e-6))

    def test_rank_samples_runs_with_coreset_k(self):
        rng = np.random.default_rng(7)
        labeled = rng.normal(size=(6, 8)).astype(np.float32)
        unlabeled_info = {}
        for i in range(10):
            raw = rng.random(size=(2, 8, 8)).astype(np.float32)
            prob_map = raw / np.sum(raw, axis=0, keepdims=True)
            unlabeled_info[f"u{i}"] = {
                "prob_map": prob_map,
                "feature": rng.normal(size=(8,)).astype(np.float32),
            }

        sampler = ADKUCSSampler()
        ranked = sampler.rank_samples(
            unlabeled_info,
            labeled_features=labeled,
            current_iteration=1,
            total_iterations=10,
            lambda_override=0.5,
        )

        self.assertEqual(len(ranked), 10)
        for item in ranked:
            self.assertIn("sample_id", item)
            self.assertIn("final_score", item)
            self.assertIn("knowledge_gain", item)
            self.assertIn("uncertainty", item)
            self.assertIn("lambda_t", item)
            self.assertIsInstance(item["final_score"], float)
            self.assertIsInstance(item["knowledge_gain"], float)
            self.assertIsInstance(item["uncertainty"], float)
            self.assertIsInstance(item["lambda_t"], float)

    def test_rank_samples_raises_when_labeled_empty(self):
        rng = np.random.default_rng(7)
        raw = rng.random(size=(2, 8, 8)).astype(np.float32)
        prob_map = raw / np.sum(raw, axis=0, keepdims=True)
        unlabeled_info = {
            "u0": {
                "prob_map": prob_map,
                "feature": rng.normal(size=(8,)).astype(np.float32),
            }
        }
        sampler = ADKUCSSampler()
        ranked = sampler.rank_samples(
            unlabeled_info,
            labeled_features=np.zeros((0, 8), dtype=np.float32),
            current_iteration=1,
            total_iterations=10,
            lambda_override=0.5,
        )
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["sample_id"], "u0")

    def test_score_normalization_toggle(self):
        rng = np.random.default_rng(1234)
        unlabeled_info = {}
        for i in range(2):
            raw = np.array(
                [
                    [[1.0 / 3.0, 1.0 / 3.0], [1.0 / 3.0, 1.0 / 3.0]],
                    [[1.0 / 3.0, 1.0 / 3.0], [1.0 / 3.0, 1.0 / 3.0]],
                    [[1.0 / 3.0, 1.0 / 3.0], [1.0 / 3.0, 1.0 / 3.0]],
                ],
                dtype=np.float32,
            )
            if i == 1:
                raw = np.array(
                    [
                        [[1.0, 0.0], [1.0, 0.0]],
                        [[0.0, 1.0], [0.0, 1.0]],
                        [[0.0, 0.0], [0.0, 0.0]],
                    ],
                    dtype=np.float32,
                )
            raw = raw / np.sum(raw, axis=0, keepdims=True)
            unlabeled_info[f"u{i}"] = {
                "prob_map": raw,
                "feature": rng.normal(size=(4,)).astype(np.float32),
            }
        labeled_features = rng.normal(size=(3, 4)).astype(np.float32)

        sampler_norm = ADKUCSSampler(score_normalization=True)
        sampler_raw = ADKUCSSampler(score_normalization=False)

        ranked_norm = sampler_norm.rank_samples(
            unlabeled_info,
            labeled_features=labeled_features,
            current_iteration=1,
            total_iterations=5,
            lambda_override=0.0,
        )
        ranked_raw = sampler_raw.rank_samples(
            unlabeled_info,
            labeled_features=labeled_features,
            current_iteration=1,
            total_iterations=5,
            lambda_override=0.0,
        )

        norm_scores = [item["final_score"] for item in ranked_norm]
        raw_scores = [item["final_score"] for item in ranked_raw]

        self.assertGreaterEqual(max(norm_scores), 0.0)
        self.assertLessEqual(max(norm_scores), 1.0)
        self.assertGreater(max(raw_scores), 1.0)


class TestADKUCSSBaldScores(unittest.TestCase):
    def test_calculate_bald_scores_matches_uncertainty_pipeline(self):
        from torch.utils.data import Dataset, DataLoader, Subset

        class TinySegDataset(Dataset):
            def __init__(self, n: int):
                self.n = int(n)

            def __len__(self):
                return self.n

            def __getitem__(self, idx: int):
                base = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 4, 4)
                return {"image": base * float(idx + 1)}

        class TinyDropoutModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.drop = torch.nn.Dropout2d(p=0.5)

            def forward(self, x):
                logits = x[:, :2, :, :]
                return self.drop(logits)

        dataset = TinySegDataset(16)
        model = TinyDropoutModel()
        sampler = ADKUCSSampler(device="cpu")
        sampler.uncertainty_method = "bald"
        sampler.n_mc_samples = 5

        unlabeled_indices = list(range(len(dataset)))

        torch.manual_seed(0)
        scores, ids = sampler.calculate_bald_scores(
            model, dataset, unlabeled_indices, n_mc_samples=5
        )
        self.assertEqual(ids, unlabeled_indices)

        subset = Subset(dataset, unlabeled_indices)
        loader = DataLoader(subset, batch_size=8, shuffle=False, num_workers=0)
        torch.manual_seed(0)
        unc_arr, _, _ = sampler.get_uncertainty_and_features(model, loader)
        self.assertTrue(
            np.allclose(
                np.asarray(scores, dtype=np.float32),
                np.asarray(unc_arr, dtype=np.float32),
                rtol=1e-6,
                atol=1e-6,
            )
        )


class TestHookExceptionSafety(unittest.TestCase):
    def test_sampler_removes_forward_hook_on_exception(self):
        from contextlib import ExitStack
        from unittest.mock import patch

        class FakeHandle:
            def __init__(self, layer, key):
                self._layer = layer
                self._key = key

            def remove(self):
                self._layer._forward_hooks.pop(self._key, None)

        class FakeLayer:
            def __init__(self):
                self._forward_hooks = {}

            def register_forward_hook(self, fn):
                key = object()
                self._forward_hooks[key] = fn
                return FakeHandle(self, key)

        class FakeModel:
            def __init__(self):
                self.layer4 = FakeLayer()

            def eval(self):
                return self

            def __call__(self, images):
                return None

        class CrashIterable:
            def __iter__(self):
                raise RuntimeError("intentional crash")

        sampler = ADKUCSSampler(device="cpu")
        model = FakeModel()

        with ExitStack() as stack:
            for target in ("core.sampler.tqdm", "src.core.sampler.tqdm"):
                try:
                    stack.enter_context(patch(target, new=lambda it, **kwargs: it))
                except Exception:
                    pass
            with self.assertRaises(RuntimeError):
                sampler.get_features_only(model, CrashIterable())
        self.assertEqual(len(model.layer4._forward_hooks), 0)

        sampler.uncertainty_method = "entropy"
        with ExitStack() as stack:
            for target in ("core.sampler.tqdm", "src.core.sampler.tqdm"):
                try:
                    stack.enter_context(patch(target, new=lambda it, **kwargs: it))
                except Exception:
                    pass
            with self.assertRaises(RuntimeError):
                sampler.get_uncertainty_and_features(model, CrashIterable())
        self.assertEqual(len(model.layer4._forward_hooks), 0)

        with ExitStack() as stack:
            for target in ("core.sampler.tqdm", "src.core.sampler.tqdm"):
                try:
                    stack.enter_context(patch(target, new=lambda it, **kwargs: it))
                except Exception:
                    pass
            with self.assertRaises(RuntimeError):
                sampler.get_predictions_and_features(model, CrashIterable(), mc_dropout=False)
        self.assertEqual(len(model.layer4._forward_hooks), 0)

    def test_trainer_removes_forward_hook_on_exception(self):
        from core.trainer import Trainer
        from contextlib import ExitStack
        from unittest.mock import patch

        class FakeHandle:
            def __init__(self, layer, key):
                self._layer = layer
                self._key = key

            def remove(self):
                self._layer._forward_hooks.pop(self._key, None)

        class FakeLayer:
            def __init__(self):
                self._forward_hooks = {}

            def register_forward_hook(self, fn):
                key = object()
                self._forward_hooks[key] = fn
                return FakeHandle(self, key)

        class FakeEncoder:
            def __init__(self):
                self.layer4 = FakeLayer()

        class FakeInner:
            def __init__(self):
                self.encoder = FakeEncoder()

        class FakeModel:
            def __init__(self):
                self.model = FakeInner()

            def eval(self):
                return self

            def __call__(self, images):
                return None

        class CrashIterable:
            def __iter__(self):
                raise RuntimeError("intentional crash")

        model = FakeModel()
        trainer = Trainer.__new__(Trainer)
        trainer.model = model
        trainer.device = "cpu"
        trainer._amp_enabled = False
        trainer._amp_dtype = "float16"

        with ExitStack() as stack:
            for target in ("core.trainer.tqdm", "src.core.trainer.tqdm"):
                try:
                    stack.enter_context(patch(target, new=lambda it, **kwargs: it))
                except Exception:
                    pass
            with self.assertRaises(RuntimeError):
                trainer.predict_probs(CrashIterable())
        self.assertEqual(len(trainer.model.model.encoder.layer4._forward_hooks), 0)


if __name__ == '__main__':
    unittest.main()
