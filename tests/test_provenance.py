import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ucsd_repro.provenance import (
    collect_runtime_provenance,
    validation_problems,
    write_strict_json,
)


class ProvenanceTests(unittest.TestCase):
    def test_report_is_strict_json_and_excludes_machine_identity(self):
        packages = {
            "numpy": "2.0",
            "scipy": "1.0",
            "pandas": "2.0",
            "pyyaml": "6.0",
            "nibabel": "5.0",
            "simpleitk": "2.0",
            "torch": "2.0+cu130",
        }
        cuda = {
            "available": True,
            "runtime_version": "13.0",
            "device_name": "NVIDIA GeForce RTX 3080 Laptop GPU",
            "total_vram_bytes": 16 * 2**30,
            "compute_capability": "8.6",
            "driver_version": "596.08",
        }
        with (
            patch("ucsd_repro.provenance._package_versions", return_value=packages),
            patch("ucsd_repro.provenance._cuda_details", return_value=cuda),
            patch(
                "ucsd_repro.provenance._git_state",
                return_value={"commit": "a" * 40, "dirty": False},
            ),
        ):
            report = collect_runtime_provenance(Path("C:/Users/private/project"))

        encoded = json.dumps(report, allow_nan=False)
        self.assertIn("RTX 3080", encoded)
        for forbidden in ("hostname", "username", "private/project", "filesystem_root"):
            self.assertNotIn(forbidden, encoded.lower())

    def test_cuda_requirement_is_validated(self):
        report = {
            "packages": {"numpy": "2.0", "torch": "2.0"},
            "cuda": {"available": False},
        }
        self.assertEqual(validation_problems(report), [])
        self.assertEqual(validation_problems(report, require_cuda=True), ["CUDA-enabled PyTorch/GPU"])

    def test_strict_json_writer_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "environment.json"
            write_strict_json(path, {"finite": 1.0, "optional": None})
            self.assertEqual(json.loads(path.read_text()), {"finite": 1.0, "optional": None})


if __name__ == "__main__":
    unittest.main()
