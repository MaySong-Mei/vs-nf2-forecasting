import base64
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.index_ucsd import index_official
from scripts.merge_clinical import official_clinical_rows
from scripts.select_ucsd_files import selected_relative_paths
from scripts.download_ucsd_aspera import (
    completed_paths,
    extract_public_context,
    load_paths,
)
from scripts.validate_manifest import validate_nifti_pairs


class UcsdAdapterTests(unittest.TestCase):
    def test_official_layout_pairs_mask_with_declared_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exam = root / "release" / "VS_0001_01"
            exam.mkdir(parents=True)
            for name in (
                "VS_0001_01_t1post.nii.gz",
                "VS_0001_01_t1postIAC.nii.gz",
                "VS_0001_01_seg_t1postIAC.nii.gz",
            ):
                (exam / name).touch()
            rows, ambiguous = index_official(root)
            self.assertFalse(ambiguous)
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["image_path"].endswith("_t1postIAC.nii.gz"))
            self.assertTrue(rows[0]["mask_path"].endswith("_seg_t1postIAC.nii.gz"))

    def test_official_clinical_timing_and_treatment_transition(self):
        clinical = pd.DataFrame(
            {
                "ID": ["VS_0001_01", "VS_0001_02", "VS_0001_03"],
                "Days from prior scan": [None, 100, 200],
                "Pre- vs Post-Tx": ["Pre", "Post", "Post"],
                "Longitudinal classification (respect to prior scan)": [None, "Unchanged", "Increased"],
                "No residual VS": ["No", "No", "Yes"],
                "Days between scan and treatment #1": [-50, 50, 250],
                "Days between scan and treatment #2": [None, None, None],
            }
        )
        rows = official_clinical_rows(clinical)
        self.assertEqual(rows["study_days_clinical"].tolist(), [0.0, 100.0, 300.0])
        self.assertEqual(rows["intervened_since_previous_clinical"].tolist(), [False, True, False])
        self.assertEqual(rows["no_residual_vs_clinical"].tolist(), [False, False, True])

    def test_minimum_download_paths_follow_reference_sequence(self):
        clinical = pd.DataFrame(
            {
                "ID": ["VS_0001_01", "VS_0002_01"],
                "Has T1post": ["True", "True"],
                "Has T1postIAC": ["True", "False"],
            }
        )
        paths = selected_relative_paths(clinical)
        self.assertEqual(len(paths), 4)
        self.assertIn("UCSD-VS-Longitudinal/VS_0001_01/VS_0001_01_t1postIAC.nii.gz", paths)
        self.assertIn("UCSD-VS-Longitudinal/VS_0002_01/VS_0002_01_seg_t1post.nii.gz", paths)

    def test_aspera_context_and_path_validation_do_not_expose_credentials(self):
        context = {
            "resource": "packages",
            "package_id": "1285",
            "passcode": "not-logged",
        }
        encoded = base64.b64encode(json.dumps(context).encode()).decode()
        parsed_encoded, parsed = extract_public_context(f"href='?context={encoded}'")
        self.assertEqual(parsed_encoded, encoded)
        self.assertEqual(parsed["package_id"], "1285")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path_file = root / "paths.txt"
            relative = "UCSD-VS-Longitudinal/VS_0001_01/VS_0001_01_t1post.nii.gz"
            path_file.write_text(relative + "\n", encoding="utf-8")
            self.assertEqual(load_paths(path_file), [relative])
            downloaded = root / relative
            downloaded.parent.mkdir(parents=True)
            downloaded.write_bytes(b"nifti")
            self.assertEqual(completed_paths(root, [relative]), 1)

    def test_nifti_qc_allows_declared_empty_no_residual_mask(self):
        import nibabel as nib
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.nii.gz"
            mask_path = root / "mask.nii.gz"
            affine = np.diag([0.5, 0.5, 0.5, 1.0])
            nib.save(nib.Nifti1Image(np.ones((4, 5, 6), dtype=np.float32), affine), image_path)
            nib.save(nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.uint8), affine), mask_path)
            frame = pd.DataFrame(
                {
                    "image_path": [image_path.name],
                    "mask_path": [mask_path.name],
                    "no_residual_vs": [True],
                }
            )
            errors, stats = validate_nifti_pairs(frame, root / "manifest.csv", root)
            self.assertFalse(errors)
            self.assertEqual(stats["pairs_checked"], 1)
            self.assertEqual(stats["empty_masks"], 1)


if __name__ == "__main__":
    unittest.main()
