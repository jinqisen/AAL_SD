import unittest
from types import SimpleNamespace

from src.experiments.specs.types import ExperimentRuntime, TraceOptions
from src.main import ActiveLearningPipeline


class TestSelectionGeometryLogging(unittest.TestCase):
    def _make_pipeline(self):
        p = ActiveLearningPipeline.__new__(ActiveLearningPipeline)
        p.exp_config = {
            "geometry_boundary_delta_ratio": 0.5,
            "geometry_sensitivity_delta_lambda": 0.1,
        }
        p.config = SimpleNamespace(QUERY_SIZE=2)
        p.current_round = 1
        p._traces = []
        p._append_trace = lambda payload: p._traces.append(payload)
        p.experiment_runtime = ExperimentRuntime(
            experiment_name="test",
            description="test",
            legacy_config={},
            trace_options=TraceOptions(
                enable_score_snapshot_logging=True,
                score_snapshot_boundary_window=1,
            ),
        )
        return p

    def test_compute_ranking_metadata_contains_selection_geometry(self):
        p = self._make_pipeline()
        ranked = [
            {
                "sample_id": 1,
                "final_score": 0.95,
                "uncertainty": 0.90,
                "knowledge_gain": 0.30,
                "lambda_t": 0.2,
            },
            {
                "sample_id": 2,
                "final_score": 0.80,
                "uncertainty": 0.70,
                "knowledge_gain": 0.60,
                "lambda_t": 0.2,
            },
            {
                "sample_id": 3,
                "final_score": 0.78,
                "uncertainty": 0.40,
                "knowledge_gain": 0.95,
                "lambda_t": 0.2,
            },
            {
                "sample_id": 4,
                "final_score": 0.50,
                "uncertainty": 0.10,
                "knowledge_gain": 0.20,
                "lambda_t": 0.2,
            },
        ]

        meta = p._compute_ranking_metadata(ranked, top_k=2)

        self.assertAlmostEqual(meta["lambda_effective"], 0.2)
        self.assertIn("selection_geometry", meta)
        geometry = meta["selection_geometry"]
        self.assertEqual(geometry["boundary_n"], 2)
        self.assertIsNotNone(geometry["boundary_u_std"])
        self.assertIsNotNone(geometry["boundary_k_std"])
        self.assertIsNotNone(geometry["spearman_rho_uk"])
        self.assertIsNotNone(geometry["sens_up"])
        self.assertIsNotNone(geometry["sens_down"])
        self.assertIsNotNone(geometry["crossing_density"])
        self.assertIsNotNone(geometry["local_jaccard_distance"])

    def test_append_score_snapshot_logs_ranked_rows_and_boundary_rows(self):
        p = self._make_pipeline()
        p._last_ranked_items = [
            {
                "sample_id": 1,
                "final_score": 0.95,
                "uncertainty": 0.90,
                "knowledge_gain": 0.30,
                "lambda_t": 0.2,
            },
            {
                "sample_id": 2,
                "final_score": 0.80,
                "uncertainty": 0.70,
                "knowledge_gain": 0.60,
                "lambda_t": 0.2,
            },
            {
                "sample_id": 3,
                "final_score": 0.78,
                "uncertainty": 0.40,
                "knowledge_gain": 0.95,
                "lambda_t": 0.2,
            },
            {
                "sample_id": 4,
                "final_score": 0.50,
                "uncertainty": 0.10,
                "knowledge_gain": 0.20,
                "lambda_t": 0.2,
            },
        ]

        p._append_score_snapshot([1, 3], source="test")

        self.assertEqual(len(p._traces), 1)
        payload = p._traces[0]
        self.assertEqual(payload["type"], "score_snapshot")
        self.assertEqual(payload["pool_n"], 4)
        self.assertEqual(payload["query_size"], 2)
        self.assertEqual(len(payload["rows"]), 4)
        self.assertEqual(payload["boundary_start_rank"], 2)
        self.assertEqual(payload["boundary_end_rank"], 3)
        selected_rows = [row for row in payload["rows"] if row["selected"]]
        self.assertEqual({row["sample_id"] for row in selected_rows}, {1, 3})


if __name__ == "__main__":
    unittest.main()
