
import unittest
from unittest.mock import MagicMock
from src.agent.toolbox import Toolbox
from src.agent.config import AgentThresholds

class TestProposals(unittest.TestCase):
    def setUp(self):
        self.mock_controller = MagicMock()
        self.mock_controller.config.RANDOM_SEED = 42
        self.mock_strategy = MagicMock()
        self.mock_model = MagicMock()
        self.toolbox = Toolbox(self.mock_controller, self.mock_strategy, self.mock_model)

    def test_proposal_a_consistent_lambda(self):
        """Test that lambda generation is consistent across different run_ids (Proposal A)."""
        policy = {
            "mode": "warmup_risk_closed_loop",
            "warmup_rounds": 5,
            "warmup_start_round": 1,
            "warmup_lambda_range": [0.2, 0.8]
        }
        
        # Run 1
        self.mock_controller.run_id = "run_1"
        self.mock_controller.experiment_name = "exp_1"
        lambda_1 = self.toolbox._compute_policy_lambda_for_round(2, policy)["applied"]
        
        # Run 2
        self.mock_controller.run_id = "run_2"
        self.mock_controller.experiment_name = "exp_2"
        lambda_2 = self.toolbox._compute_policy_lambda_for_round(2, policy)["applied"]
        
        self.assertEqual(lambda_1, lambda_2, "Lambda should be consistent regardless of run_id")
        self.assertNotEqual(lambda_1, 0.2, "Lambda should be a random value, not clamped default")

    def test_proposal_b_thresholds(self):
        """Test that severe overfit thresholds are relaxed (Proposal B)."""
        # Check static values
        self.assertEqual(AgentThresholds.OVERFIT_RISK_HI, 0.9, "Risk threshold should be 0.9")
        self.assertEqual(AgentThresholds.OVERFIT_TVC_MIN_HI, 0.6, "TVC Min threshold should be 0.6")

        # Check logic behavior
        # Setup state that would trigger OLD threshold (0.8) but NOT NEW (0.9)
        self.toolbox.training_state = {
            "overfit_risk": 0.85, # > 0.8 (Old) but < 0.9 (New)
            "grad_train_val_cos_min": -0.55, # < -0.5 (Old) but > -0.6 (New, wait, -0.55 > -0.6? No, -0.6 is smaller. -0.55 is "less negative" than -0.6? No.)
            # Wait, threshold logic:
            # if tvc_min <= -tvc_min_hi: severe = True
            # Old: -0.5. -0.55 <= -0.5 is True. Severe.
            # New: -0.6. -0.55 <= -0.6 is False. Not Severe.
            # So -0.55 is a good test case.
            "rollback_flag": False
        }
        self.toolbox._last_lambda_applied = 0.5
        
        policy = {
            "mode": "warmup_risk_closed_loop",
            "warmup_rounds": 0, # Disable warmup to test risk control logic directly
            "risk_control_start_round": 1 # Enable risk control immediately
        }
        
        result = self.toolbox._compute_policy_lambda_for_round(2, policy)
        self.assertNotEqual(result["rule"], "severe_overfit_lambda_down", 
                           "Should NOT trigger severe overfit with risk=0.85/min_cos=-0.55 under new thresholds")
        
        # Setup state that triggers NEW threshold
        self.toolbox.training_state["overfit_risk"] = 0.95
        result = self.toolbox._compute_policy_lambda_for_round(2, policy)
        self.assertEqual(result["rule"], "severe_overfit_lambda_down", 
                        "Should trigger severe overfit with risk=0.95")

if __name__ == '__main__':
    unittest.main()
