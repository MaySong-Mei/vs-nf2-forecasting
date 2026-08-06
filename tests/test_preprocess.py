import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.preprocess_ucsd import failure_category, geometry_matches, process_triplet
from scripts.render_qc import render_montage
from ucsd_repro.manifest import iter_triplets


@unittest.skipUnless(importlib.util.find_spec("SimpleITK"), "SimpleITK is not installed")
class PreprocessTests(unittest.TestCase):
    def test_geometry_comparison_tolerates_header_roundoff(self):
        class Header:
            def __init__(self, direction):
                self.direction = direction

            def GetSize(self):
                return (16, 16, 16)

            def GetSpacing(self):
                return (1.0, 1.0, 1.0)

            def GetOrigin(self):
                return (0.0, 0.0, 0.0)

            def GetDirection(self):
                return self.direction

        identity = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        rounded = (1.0, 2.4e-5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        self.assertTrue(geometry_matches(Header(identity), Header(rounded)))

    def test_failure_categories_remain_auditable(self):
        self.assertEqual(
            failure_category(ValueError("registered mask at timepoint 1 is empty")),
            "registered_mask_empty",
        )
        self.assertEqual(failure_category(ValueError("source mask is empty")), "source_mask_empty")
        self.assertEqual(
            failure_category(ValueError("cannot compute an SDF for an empty mask")),
            "cropped_mask_empty",
        )
        self.assertEqual(
            failure_category(ValueError("image has no finite nonzero voxels")),
            "cropped_image_no_signal",
        )

    def test_qc_montage_is_deidentified_pixel_data(self):
        images = np.zeros((3, 64, 64, 64), dtype=np.float32)
        masks = np.zeros_like(images, dtype=bool)
        images[:, 24:40, 24:40, 24:40] = 1.0
        masks[:, 28:36, 28:36, 28:36] = True

        montage = render_montage(images, masks)

        self.assertEqual(montage.shape, (192, 192, 3))
        self.assertEqual(montage.dtype, np.uint8)
        self.assertTrue((montage[..., 0] > montage[..., 1]).any())

    def test_nifti_triplet_is_finite_and_emits_qc_metadata(self):
        import SimpleITK as sitk

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            grid = np.indices((16, 16, 16), dtype=np.float32)
            image_array = grid[0] + 2.0 * grid[1] + 3.0 * grid[2] + 1.0
            mask_array = (
                (grid[0] - 8.0) ** 2
                + (grid[1] - 8.0) ** 2
                + (grid[2] - 8.0) ** 2
                <= 9.0
            ).astype(np.uint8)
            for index in range(3):
                image_path = root / f"VS_0001_0{index + 1}_t1post.nii.gz"
                mask_path = root / f"VS_0001_0{index + 1}_seg_t1post.nii.gz"
                image = sitk.GetImageFromArray(image_array + index)
                mask = sitk.GetImageFromArray(mask_array)
                image.SetSpacing((1.0, 1.0, 1.0))
                mask.CopyInformation(image)
                sitk.WriteImage(image, str(image_path))
                sitk.WriteImage(mask, str(mask_path))
                rows.append(
                    {
                        "patient_id": "VS_0001",
                        "timepoint_id": f"VS_0001_0{index + 1}",
                        "study_days": float(index * 100),
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                        "intervened_since_previous": False,
                        "no_residual_vs": False,
                    }
                )

            triplet = next(iter_triplets(pd.DataFrame(rows)))
            arrays, metadata = process_triplet(
                triplet,
                root / "manifest.csv",
                None,
                {
                    "spacing_mm": 1.0,
                    "crop_size": [16, 16, 16],
                    "registration_iterations": 2,
                    "intensity_percentiles": [0.5, 99.5],
                    "sdf_clip_mm": 8.0,
                },
            )

            self.assertEqual(arrays["images"].shape, (3, 16, 16, 16))
            self.assertEqual(arrays["masks"].shape, (3, 16, 16, 16))
            self.assertEqual(arrays["sdfs"].shape, (3, 16, 16, 16))
            self.assertTrue(np.isfinite(arrays["images"]).all())
            self.assertTrue(np.isfinite(arrays["sdfs"]).all())
            self.assertTrue(metadata["preprocessed_finite"])
            self.assertEqual(len(json.loads(metadata["source_geometry"])), 3)
            self.assertEqual(len(json.loads(metadata["registration_overlap_dice_to_t2"])), 3)


if __name__ == "__main__":
    unittest.main()
