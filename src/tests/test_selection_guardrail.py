import json
import unittest
from unittest.mock import MagicMock

from src.agent.toolbox import Toolbox


class _Cfg:
    QUERY_SIZE = 4
    RANDOM_SEED = 42


class _Controller:
    def __init__(self):
        self.config = _Cfg()
        self.current_round = 10
        self.sampler_type = "ad_kucs"
        self.experiment_name = "full_model_A_lambda_policy"
        self.exp_config = {
            "lambda_policy": {
                "mode": "warmup_risk_closed_loop",
                "r1_lambda": 0.0,
                "uncertainty_only_rounds": 2,
                "warmup_start_round": 3,
                "warmup_rounds": 1,
                "warmup_lambda": 0.2,
                "risk_control_start_round": 4,
                "lambda_smoothing": "none",
                "late_stage_ramp": {"start_round": 9, "end_round": 12, "start_lambda": 0.40, "end_lambda": 0.70},
                "selection_guardrail": {
                    "enabled": True,
                    "u_median_min": 0.45,
                    "u_low_thresh": 0.40,
                    "u_low_frac_max": 0.20,
                    "lambda_step_down": 0.10,
                    "max_steps": 5,
                    "fallback_quota_u_frac": 0.70,
                },
            }
        }
        self._selection_context = None
        self._trace = []
        self.update_called_with = None

    def _append_trace(self, event):
        self._trace.append(event)

    def update(self, sample_ids):
        self.update_called_with = list(sample_ids or [])
        return {
            "status": "success",
            "expected_count": int(self.config.QUERY_SIZE),
            "selected_count": int(self.config.QUERY_SIZE),
            "selected_ids": list(sample_ids or []),
            "exhausted": False,
        }


class TestSelectionGuardrail(unittest.TestCase):
    def setUp(self):
        self.controller = _Controller()
        self.tools = Toolbox(self.controller, MagicMock(), model=None)
        self.tools.current_scores = {
            "0": {"U": 0.10, "K": 1.00},
            "1": {"U": 0.10, "K": 0.95},
            "2": {"U": 0.10, "K": 0.90},
            "3": {"U": 0.10, "K": 0.85},
            "4": {"U": 0.90, "K": 0.00},
            "5": {"U": 0.85, "K": 0.00},
            "6": {"U": 0.80, "K": 0.00},
            "7": {"U": 0.75, "K": 0.00},
        }

    def test_guardrail_applies_and_changes_selection(self):
        self.tools.reset_round_controls()
        v = float(self.tools.apply_round_lambda_policy())
        self.assertGreaterEqual(v, 0.40)

        raw = self.tools.finalize_selection(["0", "1", "2", "3"], reason="test")
        payload = json.loads(raw)
        self.assertEqual(payload.get("status"), "success")
        self.assertIsNotNone(self.controller.update_called_with)
        self.assertEqual(self.controller.update_called_with, ["4", "5", "6", "7"])
        meta = self.tools.control_meta.get("lambda_guardrail")
        self.assertIsInstance(meta, dict)
        self.assertLess(float(meta.get("lambda_after")), float(meta.get("lambda_before")))

    def test_guardrail_noop_when_selection_ok(self):
        self.tools.reset_round_controls()
        _ = self.tools.apply_round_lambda_policy()

        raw = self.tools.finalize_selection(["4", "5", "6", "7"], reason="test")
        payload = json.loads(raw)
        self.assertEqual(payload.get("status"), "success")
        self.assertEqual(self.controller.update_called_with, ["4", "5", "6", "7"])
        self.assertFalse("lambda_guardrail" in (self.tools.control_meta or {}))


if __name__ == "__main__":
    unittest.main()

