import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.freeze_cohort import freeze_cohort


class CohortFreezeTests(unittest.TestCase):
    def test_freeze_filters_qc_and_preserves_patient_folds_with_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared"
            prepared.mkdir()
            rows = []
            for index in range(6):
                path = prepared / f"case-{index}.npz"
                path.write_bytes(b"prepared")
                rows.append(
                    {
                        "patient_id": f"p{index}",
                        "triplet_id": f"t{index}",
                        "prepared_path": path.name,
                        "preprocessing_qc_pass": index < 5,
                    }
                )
            metadata = prepared / "metadata.csv"
            pd.DataFrame(rows).to_csv(metadata, index=False)
            splits = root / "splits.csv"
            pd.DataFrame(
                {"patient_id": [f"p{index}" for index in range(6)], "fold": [0, 1, 2, 3, 4, 0]}
            ).to_csv(splits, index=False)
            config = root / "config.yaml"
            config.write_text("experiment:\n  seed: 100\n", encoding="utf-8")
            review = root / "review.csv"
            pd.DataFrame(
                {
                    "reviewer_decision": ["pass", "fail"],
                    "rendering_config_hash": ["abc", "abc"],
                }
            ).to_csv(review, index=False)

            summary = freeze_cohort(
                metadata,
                splits,
                config,
                root / "frozen",
                review_index=review,
                expected_cases=5,
                expected_patients=5,
            )

            self.assertEqual(summary["modeling_triplets"], 5)
            self.assertEqual(summary["modeling_patients"], 5)
            self.assertEqual(sorted(summary["fold_counts"]), ["0", "1", "2", "3", "4"])
            self.assertEqual(len(summary["hashes"]["modeling_metadata_sha256"]), 64)
            frozen = pd.read_csv(root / "frozen" / "modeling_metadata.csv")
            self.assertTrue(all(not Path(value).is_absolute() for value in frozen["prepared_path"]))
            self.assertEqual(
                json.loads((root / "frozen" / "freeze_summary.json").read_text())["seed"],
                100,
            )


if __name__ == "__main__":
    unittest.main()
