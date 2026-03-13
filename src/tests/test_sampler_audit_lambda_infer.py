import unittest


from src.main import ActiveLearningPipeline


class TestSamplerAuditLambdaInference(unittest.TestCase):
    def _make_pipeline(self):
        p = ActiveLearningPipeline.__new__(ActiveLearningPipeline)
        p.rollback_config = {}
        p.exp_config = {"lambda_policy": {"mode": "warmup_risk_closed_loop"}}
        p._last_ranking_metadata = None
        p._last_control_events = {}
        p.toolbox = None
        p.sampler = None
        p.sampler_type = "ad_kucs"
        p.k_definition = "coreset_to_labeled"
        p.score_normalization = True
        return p

    def test_infer_lambda_from_lambda_policy_apply(self):
        p = self._make_pipeline()
        p.current_round = 3
        p._last_control_events["lambda_policy_apply"] = {
            "type": "lambda_policy_apply",
            "round": 3,
            "applied": 0.35,
        }
        audit = p._sampler_audit()
        self.assertAlmostEqual(float(audit.get("lambda_effective")), 0.35)
        self.assertEqual(audit.get("lambda_source"), "lambda_policy")

    def test_infer_lambda_from_selection_guardrail(self):
        p = self._make_pipeline()
        p.current_round = 10
        p._last_control_events["selection_guardrail"] = {
            "type": "selection_guardrail",
            "round": 10,
            "lambda_before": 0.6,
            "lambda_after": 0.3,
        }
        audit = p._sampler_audit()
        self.assertAlmostEqual(float(audit.get("lambda_effective")), 0.3)
        self.assertEqual(audit.get("lambda_source"), "selection_guardrail")


if __name__ == "__main__":
    unittest.main()

