from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PreparedTripletDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame, points_per_volume: int, seed: int = 100):
        self.metadata = metadata.reset_index(drop=True).copy()
        self.points_per_volume = int(points_per_volume)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.metadata.iloc[index]
        with np.load(Path(row["prepared_path"]), allow_pickle=False) as archive:
            images = archive["images"].astype(np.float32)
            masks = archive["masks"].astype(np.float32)
            sdfs = archive["sdfs"].astype(np.float32)
            days = archive["days"].astype(np.float32)

        total = int(np.prod(sdfs.shape[1:]))
        rng = np.random.default_rng(self.seed + index + int(torch.initial_seed() % 1_000_000))
        replace = self.points_per_volume > total
        flat_indices = rng.choice(total, size=self.points_per_volume, replace=replace)
        z, y, x = np.unravel_index(flat_indices, sdfs.shape[1:])
        depth, height, width = sdfs.shape[1:]
        coords = np.stack(
            [
                2.0 * x / max(width - 1, 1) - 1.0,
                2.0 * y / max(height - 1, 1) - 1.0,
                2.0 * z / max(depth - 1, 1) - 1.0,
            ],
            axis=-1,
        ).astype(np.float32)
        sampled_sdfs = sdfs.reshape(3, -1)[:, flat_indices]
        observed = np.stack([images[:2], masks[:2]], axis=1)
        return {
            "observed": torch.from_numpy(observed),
            "coords": torch.from_numpy(coords),
            "sampled_sdfs": torch.from_numpy(sampled_sdfs),
            "days": torch.from_numpy(days),
            "case_id": str(row["triplet_id"]),
        }


def load_split_metadata(
    metadata_path: str | Path,
    splits_path: str | Path,
    fold: int,
    include_qc_failed: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(metadata_path)
    if "preprocessing_qc_pass" in metadata and not include_qc_failed:
        passed = metadata["preprocessing_qc_pass"].astype(str).str.lower().eq("true")
        metadata = metadata[passed].copy()
    metadata_root = Path(metadata_path).resolve().parent
    metadata["prepared_path"] = metadata["prepared_path"].map(
        lambda value: str((metadata_root / str(value)).resolve())
        if not Path(str(value)).is_absolute()
        else str(Path(str(value)).resolve())
    )
    splits = pd.read_csv(splits_path, dtype={"patient_id": str})
    if splits["patient_id"].duplicated().any():
        raise ValueError("split file contains duplicate patients")
    merged = metadata.merge(splits[["patient_id", "fold"]], on="patient_id", how="left", validate="many_to_one")
    if merged["fold"].isna().any():
        missing = sorted(merged.loc[merged["fold"].isna(), "patient_id"].astype(str).unique())
        raise ValueError(f"patients missing from split file: {missing[:10]}")
    fold_count = int(splits["fold"].max()) + 1
    if fold_count < 3:
        raise ValueError("at least three folds are required for train/validation/test separation")
    validation_fold = (int(fold) + 1) % fold_count
    test = merged[merged["fold"] == fold].copy()
    validation = merged[merged["fold"] == validation_fold].copy()
    train = merged[~merged["fold"].isin([fold, validation_fold])].copy()
    return train, validation, test
