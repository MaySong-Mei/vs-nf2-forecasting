import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ucsd_repro.manifest import iter_triplets, load_manifest, parse_optional_bool


class ManifestTests(unittest.TestCase):
    def make_manifest(self, directory: Path) -> Path:
        path = directory / "manifest.csv"
        pd.DataFrame(
            [
                {"patient_id": "A", "timepoint_id": "t0", "study_days": 0, "image_path": "a0.nii.gz", "mask_path": "a0_mask.nii.gz", "intervened_since_previous": ""},
                {"patient_id": "A", "timepoint_id": "t1", "study_days": 100, "image_path": "a1.nii.gz", "mask_path": "a1_mask.nii.gz", "intervened_since_previous": False},
                {"patient_id": "A", "timepoint_id": "t2", "study_days": 200, "image_path": "a2.nii.gz", "mask_path": "a2_mask.nii.gz", "intervened_since_previous": False},
                {"patient_id": "A", "timepoint_id": "t3", "study_days": 300, "image_path": "a3.nii.gz", "mask_path": "a3_mask.nii.gz", "intervened_since_previous": True},
            ]
        ).to_csv(path, index=False)
        return path

    def test_rolling_triples_and_intervention_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = load_manifest(self.make_manifest(Path(directory)))
            filtered = list(iter_triplets(frame, exclude_intervened=True))
            all_triples = list(iter_triplets(frame, exclude_intervened=False))
            self.assertEqual(len(filtered), 1)
            self.assertEqual(len(all_triples), 2)

    def test_optional_boolean(self):
        self.assertIsNone(parse_optional_bool(""))
        self.assertTrue(parse_optional_bool("yes"))
        self.assertFalse(parse_optional_bool("0"))
        with self.assertRaises(ValueError):
            parse_optional_bool("maybe")


if __name__ == "__main__":
    unittest.main()

