import unittest

import numpy as np
import pandas as pd

from scripts.aggregate_folds import METRICS, summarize_metrics
from scripts.evaluate_baselines import finite_summary
from scripts.train_ucsd import finite_mean


class ReportingTests(unittest.TestCase):
    def test_baseline_summary_counts_nonfinite_metrics(self):
        frame = pd.DataFrame(
            {"patient_id": ["a", "b", "c"], "dice": [0.5, np.inf, np.nan]}
        )
        summary = finite_summary(frame, ["dice"])["dice"]
        self.assertEqual(summary["finite_cases"], 1)
        self.assertEqual(summary["nonfinite_cases"], 2)
        self.assertEqual(summary["mean"], 0.5)

    def test_fold_summary_is_strict_json_safe(self):
        data = {"patient_id": ["a", "b"]}
        for metric in METRICS:
            data[metric] = [1.0, np.inf]
        summary = summarize_metrics(pd.DataFrame(data))
        for metric in METRICS:
            self.assertEqual(summary[metric]["finite_cases"], 1)
            self.assertEqual(summary[metric]["nonfinite_cases"], 1)
            self.assertTrue(np.isfinite(summary[metric]["mean"]))

    def test_training_summary_ignores_infinite_metrics(self):
        self.assertEqual(finite_mean(pd.Series([0.5, np.inf, np.nan])), 0.5)


if __name__ == "__main__":
    unittest.main()
