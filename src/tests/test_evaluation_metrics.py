import math
import unittest

import numpy as np

from utils.evaluation import calculate_metrics


class TestCalculateMetrics(unittest.TestCase):
    def test_absent_class_is_nan_in_per_class_metrics(self):
        y_true = np.zeros((2, 2), dtype=np.int64)
        y_pred = np.zeros((2, 2), dtype=np.int64)

        out = calculate_metrics(y_true, y_pred, num_classes=2)
        self.assertAlmostEqual(float(out["mIoU"]), 1.0, places=12)
        self.assertAlmostEqual(float(out["f1_score"]), 1.0, places=12)
        self.assertEqual(int(out["valid_iou_classes"]), 1)
        self.assertEqual(int(out["valid_f1_classes"]), 1)
        self.assertTrue(math.isnan(float(out["per_class_iou"][1])))
        self.assertTrue(math.isnan(float(out["per_class_f1"][1])))

    def test_macro_metrics_include_only_valid_classes(self):
        y_true = np.array([[0, 0], [0, 0]], dtype=np.int64)
        y_pred = np.array([[0, 1], [0, 1]], dtype=np.int64)

        out = calculate_metrics(y_true, y_pred, num_classes=3)
        self.assertEqual(int(out["valid_iou_classes"]), 2)
        self.assertTrue(math.isnan(float(out["per_class_iou"][2])))
        self.assertGreaterEqual(float(out["mIoU"]), 0.0)
        self.assertLessEqual(float(out["mIoU"]), 1.0)


if __name__ == "__main__":
    unittest.main()

