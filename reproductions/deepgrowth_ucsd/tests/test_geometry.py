import unittest

import numpy as np

from ucsd_repro.geometry import (
    crop_or_pad,
    mask_centroid,
    normalize_intensity,
    signed_distance,
    touches_border,
)


class GeometryTests(unittest.TestCase):
    def test_signed_distance_uses_released_code_sign(self):
        mask = np.zeros((9, 9, 9), dtype=np.uint8)
        mask[3:6, 3:6, 3:6] = 1
        sdf = signed_distance(mask, 1.0)
        self.assertLess(sdf[4, 4, 4], 0)
        self.assertGreater(sdf[0, 0, 0], 0)

    def test_crop_pads_at_boundary_without_wrapping(self):
        source = np.zeros((5, 5, 5), dtype=np.uint8)
        source[0, 0, 0] = 1
        cropped = crop_or_pad(source, (0, 0, 0), (5, 5, 5))
        self.assertEqual(int(cropped.sum()), 1)
        self.assertEqual(int(cropped[2, 2, 2]), 1)
        self.assertFalse(cropped[-1, -1, -1])

    def test_centroid_and_border(self):
        mask = np.zeros((8, 8, 8), dtype=np.uint8)
        mask[2:4, 2:4, 2:4] = 1
        self.assertEqual(mask_centroid(mask), (2.5, 2.5, 2.5))
        self.assertFalse(touches_border(mask))
        mask[0, 3, 3] = 1
        self.assertTrue(touches_border(mask))

    def test_normalization_replaces_nonfinite_background(self):
        image = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
        image[0, 0, 0] = np.nan
        normalized = normalize_intensity(image)
        self.assertTrue(np.isfinite(normalized).all())
        self.assertEqual(float(normalized[0, 0, 0]), -1.0)


if __name__ == "__main__":
    unittest.main()

