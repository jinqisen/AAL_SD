import unittest

from src.main import ActiveLearningPipeline


class TestGradProbeSplit(unittest.TestCase):
    def test_split_is_deterministic_and_disjoint(self):
        labeled = list(range(1, 21))
        train1, probe1 = ActiveLearningPipeline._split_labeled_indices_for_grad_probe(
            labeled, frac=0.1, min_count=3, seed=42
        )
        train2, probe2 = ActiveLearningPipeline._split_labeled_indices_for_grad_probe(
            labeled, frac=0.1, min_count=3, seed=42
        )
        self.assertEqual(train1, train2)
        self.assertEqual(probe1, probe2)
        self.assertEqual(len(probe1), 3)
        self.assertTrue(set(train1).isdisjoint(set(probe1)))
        self.assertEqual(sorted(train1 + probe1), sorted(labeled))


if __name__ == "__main__":
    unittest.main()

