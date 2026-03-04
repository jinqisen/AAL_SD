import unittest
from unittest.mock import MagicMock
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent.agent_manager import AgentManager
from agent.toolbox import Toolbox
from main import ActiveLearningPipeline

class MockLLMClient:
    def chat(self, messages):
        action = {
            "tool_name": "Final Answer",
            "parameters": {"selected_sample_ids": ["1"], "reasoning": "Highest score."}
        }
        return f"Thought: Select best.\nAction: {json.dumps(action)}"

class MockLLMClientMissingActionThenRecover:
    def __init__(self):
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        if self.calls == 1:
            return "Thought: I will decide after seeing candidates."
        action = {
            "tool_name": "finalize_selection",
            "parameters": {"sample_ids": ["1"], "reason": "Recover with strict action."}
        }
        return f"Thought: Done.\nAction：{json.dumps(action)}  \n\n"

class MockLLMClientTimeout:
    def chat(self, messages):
        return "Error calling API: HTTPSConnectionPool(host='api.siliconflow.cn', port=443): Read timed out. (read timeout=60)"

class MockLLMClientRepeatedStatusThenFinalize:
    def __init__(self):
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        last_user = ""
        try:
            for msg in reversed(messages or []):
                if msg.get("role") == "user":
                    last_user = str(msg.get("content") or "")
                    break
        except Exception:
            last_user = ""

        if "InvalidAction" in last_user:
            action = {
                "tool_name": "finalize_selection",
                "parameters": {"sample_ids": ["1"], "reason": "Stop repeating observation; finalize."}
            }
            return f"Thought: Finalize now.\nAction: {json.dumps(action)}"

        if self.calls <= 3:
            action = {"tool_name": "get_system_status", "parameters": {}}
            return f"Thought: Check status again.\nAction: {json.dumps(action)}"

        action = {
            "tool_name": "finalize_selection",
            "parameters": {"sample_ids": ["1"], "reason": "Proceed after status."}
        }
        return f"Thought: Finalize.\nAction: {json.dumps(action)}"


class MockLLMClientNeverFinalizes:
    def chat(self, messages):
        action = {"tool_name": "get_system_status", "parameters": {}}
        return f"Thought: Keep observing.\nAction: {json.dumps(action)}"

class TestLLMAgent(unittest.TestCase):
    def setUp(self):
        self.tools = MagicMock()
        self.tools.training_state = {}
        self.tools.controller = None
        self.tools.get_system_status.return_value = json.dumps({
            "current_labeled_count": 1,
            "total_budget": 10,
            "lambda_t": 0.5
        })
        self.tools.finalize_selection.return_value = json.dumps({
            "status": "success",
            "selected_count": 1,
            "reason_recorded": "Highest score candidate."
        })

    def test_mock_llm_cycle(self):
        mock_client = MockLLMClient()
        agent = AgentManager(tools=self.tools, client=mock_client, verbose=False)
        result = agent.run_cycle()
        
        self.tools.precalculate_scores.assert_called_once()
        self.tools.get_system_status.assert_called()
        self.tools.finalize_selection.assert_called()
        
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['selected_count'], 1)

    def test_missing_action_then_recover(self):
        mock_client = MockLLMClientMissingActionThenRecover()
        agent = AgentManager(tools=self.tools, client=mock_client, verbose=False)
        result = agent.run_cycle()

        self.tools.finalize_selection.assert_called()
        self.assertEqual(result['status'], 'success')

    def test_llm_timeout_fast_fail(self):
        mock_client = MockLLMClientTimeout()
        agent = AgentManager(tools=self.tools, client=mock_client, verbose=False, llm_max_retries=0, llm_retry_base_seconds=0.0)
        result = agent.run_cycle()

        self.tools.precalculate_scores.assert_called_once()
        self.tools.get_system_status.assert_called()
        self.tools.finalize_selection.assert_not_called()
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "LLMAPIError")

    def test_repeated_observation_is_limited_and_recovers(self):
        mock_client = MockLLMClientRepeatedStatusThenFinalize()
        agent = AgentManager(tools=self.tools, client=mock_client, verbose=False)
        result = agent.run_cycle()

        self.tools.finalize_selection.assert_called()
        self.assertEqual(result["status"], "success")

    def test_agent_max_steps_error_when_never_finalizes(self):
        mock_client = MockLLMClientNeverFinalizes()
        agent = AgentManager(tools=self.tools, client=mock_client, verbose=False)
        result = agent.run_cycle()

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "AgentMaxSteps")

if __name__ == '__main__':
    unittest.main()


class DummyController:
    def __init__(self):
        self.dataset = object()
        self.unlabeled_indices = [3, 1, 2]
        self.labeled_indices = []
        self.run_id = "r"
        self.experiment_name = "e"
        self.current_round = 1
        self._last_score_precalc_error = None
        self._last_ranking_degraded = None


class DummyStrategyRaises:
    def calculate_scores(self, model, dataset, unlabeled_indices, labeled_indices=None):
        raise RuntimeError("boom")


class TestToolboxDegrade(unittest.TestCase):
    def test_precalculate_scores_failure_sets_error(self):
        controller = DummyController()
        tools = Toolbox(controller, DummyStrategyRaises(), model=None)
        with self.assertRaises(RuntimeError):
            tools.precalculate_scores()
        self.assertIsInstance(controller._last_score_precalc_error, dict)
        self.assertEqual(controller._last_score_precalc_error.get("phase"), "score_precalc")

    def test_precalculate_scores_raises_under_strict_innovation(self):
        class _StrictCfg:
            STRICT_INNOVATION = True
            FAIL_ON_RANKING_DEGRADED = True

        class _StrictController(DummyController):
            def __init__(self):
                super().__init__()
                self.config = _StrictCfg()

            def _append_trace(self, event):
                return None

        controller = _StrictController()
        tools = Toolbox(controller, DummyStrategyRaises(), model=None)
        with self.assertRaises(RuntimeError):
            tools.precalculate_scores()

    def test_get_top_k_samples_degraded_without_scores(self):
        controller = DummyController()
        tools = Toolbox(controller, DummyStrategyRaises(), model=None)
        with self.assertRaises(RuntimeError):
            tools.get_top_k_samples(k=2, lambda_param=0.5)

    def test_get_top_k_samples_raises_under_strict_innovation(self):
        class _StrictCfg:
            STRICT_INNOVATION = True
            FAIL_ON_RANKING_DEGRADED = True

        class _StrictController(DummyController):
            def __init__(self):
                super().__init__()
                self.config = _StrictCfg()

            def _append_trace(self, event):
                return None

        controller = _StrictController()
        tools = Toolbox(controller, DummyStrategyRaises(), model=None)
        with self.assertRaises(RuntimeError):
            tools.get_top_k_samples(k=2, lambda_param=0.5)


class DummyConfigStopOnLlmFailure:
    QUERY_SIZE = 3
    STOP_ON_LLM_FAILURE = True
    STRICT_INNOVATION = False
    FAIL_ON_AGENT_FALLBACK = True


class DummyAgentManagerIncompleteSelection:
    def run_cycle(self):
        return {"status": "success", "selected_count": 0, "valid_candidates": []}


class DummyToolbox:
    def __init__(self):
        self.model = None


class TestPipelineQueryFallback(unittest.TestCase):
    def test_stop_on_llm_failure_does_not_block_fallback_on_incomplete_selection(self):
        p = type("P", (), {})()
        p.config = DummyConfigStopOnLlmFailure()
        p.use_agent = True
        p.agent_manager = DummyAgentManagerIncompleteSelection()
        p.toolbox = DummyToolbox()
        p.current_round = 6
        p.fallback_history = []
        p._selection_context = None

        def _sampler_audit():
            return {"sampler_type": "ad_kucs", "sampler_class": "X"}

        def _append_trace(event):
            return None

        p._sampler_audit = _sampler_audit
        p._append_trace = _append_trace
        with self.assertRaises(RuntimeError):
            _ = ActiveLearningPipeline._query_samples(p, model=None)


class _PolicyCfg:
    TOTAL_BUDGET = 100


class _PolicyController:
    def __init__(self):
        self.dataset = []
        self.full_dataset = []
        self.unlabeled_indices = []
        self.labeled_indices = []
        self.config = _PolicyCfg()
        self.current_round = 1
        self.exp_config = {
            "lambda_policy": {
                "mode": "warmup_risk_closed_loop",
                "r1_lambda": 0.0,
                "warmup_start_round": 2,
                "warmup_rounds": 1,
                "warmup_lambda_range": [0.2, 0.25],
                "risk_control_start_round": 4,
            }
        }

    def _append_trace(self, event):
        return None


class TestLambdaPolicyWarmupRisk(unittest.TestCase):
    def setUp(self):
        self.controller = _PolicyController()
        self.tools = Toolbox(self.controller, MagicMock(), model=None)

    def test_round1_forces_lambda_zero(self):
        self.controller.current_round = 1
        self.tools.reset_round_controls()
        v = self.tools.apply_round_lambda_policy()
        self.assertAlmostEqual(float(v), 0.0, places=6)

    def test_round2_can_be_forced_uncertainty_only_by_policy(self):
        self.controller.exp_config["lambda_policy"]["uncertainty_only_rounds"] = 2
        self.controller.current_round = 2
        self.tools.reset_round_controls()
        v = self.tools.apply_round_lambda_policy()
        self.assertAlmostEqual(float(v), 0.0, places=6)

    def test_get_top_k_samples_respects_uncertainty_only_bounds(self):
        self.controller.exp_config["lambda_policy"]["uncertainty_only_rounds"] = 2
        self.controller.current_round = 1
        self.tools.reset_round_controls()
        self.tools.current_scores = {
            "a": {"U": 1.0, "K": 0.0},
            "b": {"U": 0.0, "K": 1.0},
        }
        raw = self.tools.get_top_k_samples(k=1)
        payload = json.loads(raw)
        self.assertEqual(payload["status"], "success")
        self.assertAlmostEqual(float(payload["meta"]["lambda_param"]), 0.0, places=12)
        self.assertFalse(bool(payload["meta"]["lambda_clamped"]))
        self.assertEqual(payload["result"][0]["id"], "a")

    def test_round2_warmup_samples_lambda_in_range_deterministically(self):
        self.controller.current_round = 2
        self.tools.reset_round_controls()
        v1 = float(self.tools.apply_round_lambda_policy())
        self.assertGreaterEqual(v1, 0.2)
        self.assertLessEqual(v1, 0.25)

        self.tools.reset_round_controls()
        v2 = float(self.tools.apply_round_lambda_policy())
        self.assertAlmostEqual(v1, v2, places=12)

    def test_round4_severe_risk_decreases_and_clamps_to_policy_min(self):
        self.controller.current_round = 2
        self.tools.reset_round_controls()
        warmup_v = float(self.tools.apply_round_lambda_policy())
        self.assertGreaterEqual(warmup_v, 0.2)
        self.assertLessEqual(warmup_v, 0.25)

        self.controller.current_round = 4
        self.tools.reset_round_controls()
        self.tools.set_training_state({
            "last_miou": 0.7,
            "prev_miou": 0.71,
            "miou_delta": -0.01,
            "rollback_flag": False,
            "current_labeled_count": 10,
            "total_budget": 100,
            "overfit_risk": 0.95,
            "grad_train_val_cos_min": 0.0,
        })
        v = self.tools.apply_round_lambda_policy()
        self.assertAlmostEqual(float(v), 0.2, places=6)

    def test_round4_low_risk_low_gain_allows_small_increase(self):
        self.controller.current_round = 2
        self.tools.reset_round_controls()
        warmup_v = float(self.tools.apply_round_lambda_policy())

        self.controller.current_round = 4
        self.tools.reset_round_controls()
        for _ in range(3):
            self.tools.set_training_state({
                "last_miou": 0.75,
                "prev_miou": 0.75,
                "miou_delta": 0.0,
                "rollback_flag": False,
                "current_labeled_count": 10,
                "total_budget": 100,
                "overfit_risk": 0.1,
                "grad_train_val_cos_min": 0.0,
            })
        v = self.tools.apply_round_lambda_policy()
        self.assertAlmostEqual(float(v), float(warmup_v) + 0.05, places=12)

    def test_round4_low_risk_k_dominant_allows_increase_without_low_risk_streak(self):
        self.controller.current_round = 2
        self.tools.reset_round_controls()
        warmup_v = float(self.tools.apply_round_lambda_policy())

        self.tools.current_scores = {
            "a": {"U": 0.1, "K": 0.9},
            "b": {"U": 0.2, "K": 0.8},
            "c": {"U": 0.15, "K": 0.85},
        }

        self.controller.current_round = 4
        self.tools.reset_round_controls()
        self.tools.set_training_state({
            "last_miou": 0.75,
            "prev_miou": 0.75,
            "miou_delta": 0.0,
            "rollback_flag": False,
            "current_labeled_count": 10,
            "total_budget": 100,
            "overfit_risk": 0.4,
            "grad_train_val_cos_min": 0.0,
        })
        v = float(self.tools.apply_round_lambda_policy())
        self.assertAlmostEqual(v, float(warmup_v) + 0.05, places=12)
