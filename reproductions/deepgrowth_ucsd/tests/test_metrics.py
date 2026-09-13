import unittest

import numpy as np

from ucsd_repro.metrics import case_metrics, dice, hd95


class MetricTests(unittest.TestCase):
    def test_identical_masks(self):
        mask = np.zeros((16, 16, 16), dtype=np.uint8)
        mask[4:12, 4:12, 4:12] = 1
        self.assertEqual(dice(mask, mask), 1.0)
        self.assertEqual(hd95(mask, mask, 0.58), 0.0)
        metrics = case_metrics(mask, mask, (0.58, 0.58, 0.58))
        self.assertEqual(metrics["absolute_rvd"], 0.0)

    def test_one_voxel_translation_has_physical_distance(self):
        first = np.zeros((9, 9, 9), dtype=np.uint8)
        second = np.zeros_like(first)
        first[3:6, 3:6, 3:6] = 1
        second[4:7, 3:6, 3:6] = 1
        self.assertGreater(hd95(first, second, 0.5), 0)
        self.assertLessEqual(hd95(first, second, 0.5), 0.5)


if __name__ == "__main__":
    unittest.main()

