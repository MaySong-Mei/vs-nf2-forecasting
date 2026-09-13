import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.preprocess_ucsd import failure_category, geometry_matches, process_triplet
from scripts.render_qc import (
    border_contacts,
    main as render_qc_main,
    render_case_montage,
    render_montage,
    sdf_heatmap,
    select_cases,
)
from ucsd_repro.manifest import iter_triplets


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

    def test_systematic_qc_panels_cover_sdf_sign_and_crop_faces(self):
        images = np.zeros((3, 16, 16, 16), dtype=np.float32)
        masks = np.zeros_like(images, dtype=bool)
        images[:, 2:14, 2:14, 2:14] = np.linspace(0.0, 1.0, 12)[None, :, None, None]
        masks[:, 5:11, 5:11, 5:11] = True
        masks[2, -1, 7:9, 7:9] = True
        sdfs = np.where(masks, -1.0, 1.0).astype(np.float32)

        montage = render_case_montage(images, masks, sdfs, sdf_clip_mm=2.0)
        heatmap = sdf_heatmap(sdfs[0, 8] * 2.0, masks[0, 8], 2.0)

        self.assertEqual(montage.shape, (128, 48, 3))
        self.assertIn("z_max", border_contacts(masks[2]))
        self.assertGreater(int(heatmap[7, 7, 2]), int(heatmap[7, 7, 0]))
        self.assertGreater(int(heatmap[0, 0, 0]), int(heatmap[0, 0, 2]))

    def test_qc_selection_is_complete_and_deterministic(self):
        frame = pd.DataFrame(
            {
                "patient_id": [f"p{index}" for index in range(8)],
                "triplet_id": [f"private-{index}" for index in range(8)],
                "prepared_path": [f"triplets/{index}.npz" for index in range(8)],
                "preprocessing_qc_pass": [False, False, True, True, True, True, True, True],
                "preprocessing_qc_reasons": [
                    '["crop_touches_border"]',
                    '["registration_overlap_below_threshold"]',
                    "[]",
                    "[]",
                    "[]",
                    "[]",
                    "[]",
                    "[]",
                ],
                "registration_overlap_dice_to_t2": [
                    "[0.2, 1, 0.2]",
                    "[0.0, 1, 0.0]",
                    "[0.2, 1, 0.2]",
                    "[0.3, 1, 0.3]",
                    "[0.4, 1, 0.4]",
                    "[0.5, 1, 0.5]",
                    "[0.6, 1, 0.6]",
                    "[0.7, 1, 0.7]",
                ],
            }
        )
        first = select_cases(frame, all_flagged=True, sample_passed=3, max_cases=1, seed=100)
        second = select_cases(frame, all_flagged=True, sample_passed=3, max_cases=1, seed=100)

        self.assertEqual(len(first), 5)
        self.assertEqual(set(first.loc[~first["_qc_pass"], "triplet_id"]), {"private-0", "private-1"})
        self.assertEqual(first["triplet_id"].tolist(), second["triplet_id"].tolist())

    @unittest.skipUnless(importlib.util.find_spec("SimpleITK"), "SimpleITK is not installed")
    def test_qc_cli_keeps_identifiers_out_of_public_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared"
            triplets = prepared / "triplets"
            output = root / "qc"
            triplets.mkdir(parents=True)
            images = np.zeros((3, 16, 16, 16), dtype=np.float32)
            images[:, 2:14, 2:14, 2:14] = 1.0
            masks = np.zeros_like(images, dtype=np.uint8)
            masks[:, 5:11, 5:11, 5:11] = 1
            sdfs = np.where(masks, -1.0, 1.0).astype(np.float32)
            np.savez_compressed(
                triplets / "private-case.npz", images=images, masks=masks, sdfs=sdfs
            )
            metadata = prepared / "metadata.csv"
            pd.DataFrame(
                [
                    {
                        "patient_id": "PRIVATE_PATIENT",
                        "triplet_id": "PRIVATE_TRIPLET",
                        "prepared_path": "triplets/private-case.npz",
                        "preprocessing_qc_pass": False,
                        "preprocessing_qc_reasons": '["crop_touches_border"]',
                        "registration_overlap_dice_to_t2": "[0.2, 1.0, 0.3]",
                        "registration_overlap_review_threshold": 0.1,
                    }
                ]
            ).to_csv(metadata, index=False)
            argv = [
                "render_qc.py",
                "--metadata",
                str(metadata),
                "--prepared-root",
                str(prepared),
                "--output-dir",
                str(output),
                "--all-flagged",
                "--include-sdf",
            ]
            with patch("sys.argv", argv), redirect_stdout(io.StringIO()):
                self.assertEqual(render_qc_main(), 0)

            review_path = next(output.glob("review_index_*.csv"))
            mapping_path = next(output.glob("private_case_mapping_*.csv"))
            review_text = review_path.read_text()
            mapping_text = mapping_path.read_text()
            public_files = [path.name for path in output.glob("*.png")]
            self.assertNotIn("PRIVATE_PATIENT", review_text)
            self.assertNotIn("PRIVATE_TRIPLET", review_text)
            self.assertTrue(all("PRIVATE" not in name for name in public_files))
            self.assertIn("PRIVATE_PATIENT", mapping_text)
            self.assertIn("PRIVATE_TRIPLET", mapping_text)

    @unittest.skipUnless(importlib.util.find_spec("SimpleITK"), "SimpleITK is not installed")
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
