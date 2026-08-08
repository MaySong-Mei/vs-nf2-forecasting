"""Privacy-preserving runtime provenance for reproducible local experiments."""

from __future__ import annotations

import importlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_IMPORTS = {
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "pyyaml": "yaml",
    "nibabel": "nibabel",
    "simpleitk": "SimpleITK",
    "torch": "torch",
}


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for label, import_name in PACKAGE_IMPORTS.items():
        try:
            module = importlib.import_module(import_name)
            versions[label] = str(getattr(module, "__version__", "installed"))
        except (ImportError, OSError):
            versions[label] = None
    return versions


def _nvidia_driver_version() -> str | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        return lines[0] if lines else None
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _cuda_details() -> dict[str, Any]:
    details: dict[str, Any] = {
        "available": False,
        "runtime_version": None,
        "device_name": None,
        "total_vram_bytes": None,
        "compute_capability": None,
        "driver_version": _nvidia_driver_version(),
    }
    try:
        import torch
    except (ImportError, OSError):
        return details

    details["runtime_version"] = torch.version.cuda
    details["available"] = bool(torch.cuda.is_available())
    if details["available"]:
        properties = torch.cuda.get_device_properties(0)
        details.update(
            {
                "device_name": str(torch.cuda.get_device_name(0)),
                "total_vram_bytes": int(properties.total_memory),
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
        )
    return details


def _git_state(repo_root: Path) -> dict[str, str | bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "-c", f"safe.directory={repo_root.as_posix()}", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
        status = subprocess.check_output(
            ["git", "-c", f"safe.directory={repo_root.as_posix()}", "status", "--porcelain"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {"commit": "unknown", "dirty": None}


def collect_runtime_provenance(repo_root: Path) -> dict[str, Any]:
    """Collect reproducibility metadata without hostnames, usernames, or paths."""
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "packages": _package_versions(),
        "cuda": _cuda_details(),
        "git": _git_state(repo_root.resolve()),
    }


def validation_problems(report: dict[str, Any], require_cuda: bool = False) -> list[str]:
    packages = report["packages"]
    problems = [name for name, version in packages.items() if version is None]
    if require_cuda and not report["cuda"]["available"]:
        problems.append("CUDA-enabled PyTorch/GPU")
    return problems


def write_strict_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
