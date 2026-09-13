from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = (
    "patient_id",
    "timepoint_id",
    "study_days",
    "image_path",
    "mask_path",
)


def parse_optional_bool(value: object) -> bool | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"invalid optional boolean: {value!r}")


def load_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"patient_id": str, "timepoint_id": str})
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"manifest is missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("manifest contains no rows")

    frame = frame.copy()
    frame["study_days"] = pd.to_numeric(frame["study_days"], errors="coerce")
    if frame["study_days"].isna().any():
        rows = frame.index[frame["study_days"].isna()].tolist()[:10]
        raise ValueError(f"study_days is missing/non-numeric at rows: {rows}")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("required manifest fields cannot be blank")
    if frame.duplicated(["patient_id", "timepoint_id"]).any():
        raise ValueError("patient_id + timepoint_id must be unique")

    for _, group in frame.groupby("patient_id"):
        days = group["study_days"].sort_values().to_numpy()
        if len(days) > 1 and not (days[1:] > days[:-1]).all():
            patient = group["patient_id"].iloc[0]
            raise ValueError(f"study_days must be strictly increasing for {patient}")

    return frame.sort_values(["patient_id", "study_days", "timepoint_id"]).reset_index(drop=True)


def resolve_path(value: str, manifest_path: str | Path, data_root: str | Path | None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    base = Path(data_root).expanduser() if data_root else Path(manifest_path).resolve().parent
    return (base / path).resolve()


@dataclass(frozen=True)
class Triplet:
    patient_id: str
    triplet_id: str
    rows: tuple[dict, dict, dict]
    intervention_unknown: bool


def triplet_exclusion_reasons(
    triplet: Triplet,
    exclude_intervened: bool = True,
    exclude_no_residual: bool = False,
) -> list[str]:
    reasons = []
    if exclude_intervened and any(
        parse_optional_bool(row.get("intervened_since_previous")) is True
        for row in triplet.rows[1:]
    ):
        reasons.append("intervention_between_scans")
    if exclude_no_residual and any(
        parse_optional_bool(row.get("no_residual_vs")) is True for row in triplet.rows
    ):
        reasons.append("no_residual_tumor_sdf_undefined")
    return reasons


def iter_triplets(
    frame: pd.DataFrame,
    exclude_intervened: bool = True,
    exclude_no_residual: bool = False,
) -> Iterable[Triplet]:
    for patient_id, group in frame.groupby("patient_id", sort=True):
        records = group.sort_values("study_days").to_dict("records")
        for start in range(max(0, len(records) - 2)):
            rows = tuple(records[start : start + 3])
            statuses = [
                parse_optional_bool(rows[1].get("intervened_since_previous")),
                parse_optional_bool(rows[2].get("intervened_since_previous")),
            ]
            timepoints = "__".join(str(row["timepoint_id"]) for row in rows)
            triplet = Triplet(
                patient_id=str(patient_id),
                triplet_id=f"{patient_id}__{timepoints}",
                rows=rows,  # type: ignore[arg-type]
                intervention_unknown=any(value is None for value in statuses),
            )
            if triplet_exclusion_reasons(triplet, exclude_intervened, exclude_no_residual):
                continue
            yield triplet


def validate_files(
    frame: pd.DataFrame, manifest_path: str | Path, data_root: str | Path | None = None
) -> list[str]:
    errors: list[str] = []
    for row_number, row in frame.iterrows():
        for column in ("image_path", "mask_path"):
            path = resolve_path(str(row[column]), manifest_path, data_root)
            if not path.is_file():
                errors.append(f"row {row_number} {column}: not found: {path}")
    return errors

