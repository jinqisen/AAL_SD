import os
import tempfile
import unittest

from src.main import ActiveLearningPipeline


class _Cfg:
    ROUND_MODEL_RETENTION = "all"
    ROUND_MODEL_KEEP_LAST_N = 0


class TestRoundModelRetention(unittest.TestCase):
    def test_final_only_keeps_latest_round_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            p = ActiveLearningPipeline.__new__(ActiveLearningPipeline)
            p.round_model_dir = td
            p.exp_config = {"round_model_retention": "final_only"}
            p.config = _Cfg()

            for r in range(1, 6):
                with open(os.path.join(td, f"round_{r:02d}_best_val.pt"), "wb") as f:
                    f.write(b"x")
            with open(os.path.join(td, "round_03_best_val.pt.tmp"), "wb") as f:
                f.write(b"tmp")

            p._prune_round_best_val_models(5)

            remaining = sorted(os.listdir(td))
            self.assertEqual(remaining, ["round_05_best_val.pt"])

    def test_keep_last_n_keeps_window(self):
        with tempfile.TemporaryDirectory() as td:
            p = ActiveLearningPipeline.__new__(ActiveLearningPipeline)
            p.round_model_dir = td
            p.exp_config = {"round_model_retention": "last_n", "round_model_keep_last_n": 2}
            p.config = _Cfg()

            for r in range(1, 6):
                with open(os.path.join(td, f"round_{r:02d}_best_val.pt"), "wb") as f:
                    f.write(b"x")

            p._prune_round_best_val_models(5)

            remaining = sorted(os.listdir(td))
            self.assertEqual(remaining, ["round_04_best_val.pt", "round_05_best_val.pt"])


if __name__ == "__main__":
    unittest.main()

