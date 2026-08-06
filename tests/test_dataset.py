import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ucsd_repro.dataset import load_split_metadata


class DatasetTests(unittest.TestCase):
    def test_preprocessing_qc_filter_is_default_and_overridable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = pd.DataFrame(
                {
                    "patient_id": [f"p{i}" for i in range(6)],
                    "triplet_id": [f"t{i}" for i in range(6)],
                    "prepared_path": [f"t{i}.npz" for i in range(6)],
                    "preprocessing_qc_pass": [True, True, True, True, True, False],
                }
            )
            splits = pd.DataFrame(
                {"patient_id": [f"p{i}" for i in range(6)], "fold": [0, 1, 2, 3, 4, 2]}
            )
            metadata_path = root / "metadata.csv"
            splits_path = root / "splits.csv"
            metadata.to_csv(metadata_path, index=False)
            splits.to_csv(splits_path, index=False)

            filtered = load_split_metadata(metadata_path, splits_path, 0)
            included = load_split_metadata(
                metadata_path, splits_path, 0, include_qc_failed=True
            )

            self.assertEqual(sum(map(len, filtered)), 5)
            self.assertEqual(sum(map(len, included)), 6)


if __name__ == "__main__":
    unittest.main()
