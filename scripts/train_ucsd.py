#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ucsd_repro.dataset import PreparedTripletDataset, load_split_metadata
from ucsd_repro.metrics import case_metrics
from ucsd_repro.model import DeepGrowthLite


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the RTX-2080 DeepGrowth-style profile")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-cases", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    return torch.device(requested)


def evaluate(
    model: DeepGrowthLite,
    frame: pd.DataFrame,
    device: torch.device,
    inference_chunk: int,
    prediction_dir: Path,
    max_cases: int | None,
) -> pd.DataFrame:
    model.eval()
    prediction_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    selected = frame.iloc[:max_cases] if max_cases else frame
    with torch.no_grad():
        for _, item in selected.iterrows():
            with np.load(Path(item["prepared_path"]), allow_pickle=False) as archive:
                images = archive["images"].astype(np.float32)
                masks = archive["masks"].astype(np.float32)
                days = archive["days"].astype(np.float32)
                spacing = archive["spacing_mm"].astype(float)
            observed = torch.from_numpy(np.stack([images[:2], masks[:2]], axis=1)[None]).to(device)
            day_tensor = torch.from_numpy(days[None]).to(device)
            sdf = model.predict_sdf_grid(observed, day_tensor, tuple(masks.shape[1:]), inference_chunk)
            prediction = (sdf[0] < 0).cpu().numpy()
            case_id = str(item["triplet_id"])
            np.savez_compressed(
                prediction_dir / f"{safe_name(case_id)}.npz", prediction=prediction.astype(np.uint8)
            )
            rows.append(
                {
                    "patient_id": str(item["patient_id"]),
                    "triplet_id": case_id,
                    **case_metrics(prediction, masks[2] > 0, spacing),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    args = arguments()
    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    seed = int(config["experiment"]["seed"]) + args.fold
    seed_everything(seed)
    device = choose_device(args.device)
    train_config = config["training"]
    model_config = config["model"]
    train_frame, validation_frame, test_frame = load_split_metadata(
        args.metadata, args.splits, args.fold
    )
    if min(len(train_frame), len(validation_frame), len(test_frame)) == 0:
        raise SystemExit(
            f"empty partition: train={len(train_frame)}, val={len(validation_frame)}, test={len(test_frame)}"
        )
    dataset = PreparedTripletDataset(
        train_frame, int(train_config["points_per_volume"]), seed=seed
    )
    workers = int(train_config["workers"]) if device.type == "cuda" else 0
    loader = DataLoader(
        dataset,
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    model = DeepGrowthLite(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    use_amp = bool(train_config["amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    accumulation = int(train_config["gradient_accumulation"])
    epochs = args.epochs if args.epochs is not None else int(train_config["epochs"])
    run_dir = Path(train_config["checkpoint_dir"]) / config["experiment"]["name"] / f"fold_{args.fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_dice = -1.0
    optimizer.zero_grad(set_to_none=True)
    print(
        f"Device={device}; train/val/test={len(train_frame)}/{len(validation_frame)}/{len(test_frame)}; "
        f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}"
    )
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for batch_number, batch in enumerate(loader, start=1):
            observed = batch["observed"].to(device, non_blocking=True)
            coords = batch["coords"].to(device, non_blocking=True)
            target_sdf = batch["sampled_sdfs"].to(device, non_blocking=True)
            days = batch["days"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                reconstructed, predicted, regularization = model(observed, days, coords)
                reconstruction_loss = F.l1_loss(reconstructed, target_sdf[:, :2])
                prediction_loss = F.l1_loss(predicted, target_sdf[:, 2])
                loss = reconstruction_loss + prediction_loss + float(train_config["latent_weight"]) * regularization
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            is_last = batch_number == len(loader) or (
                args.max_train_batches is not None and batch_number >= args.max_train_batches
            )
            if batch_number % accumulation == 0 or is_last:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            epoch_losses.append(float(loss.detach().cpu()))
            if args.max_train_batches is not None and batch_number >= args.max_train_batches:
                break
        validation = evaluate(
            model,
            validation_frame,
            device,
            int(train_config["inference_chunk"]),
            run_dir / "validation_predictions",
            args.max_val_cases,
        )
        validation_dice = float(validation["dice"].mean())
        record = {
            "epoch": epoch,
            "training_loss": float(np.mean(epoch_losses)),
            "validation_dice": validation_dice,
            "validation_hd95_mm": float(validation["hd95_mm"].mean()),
        }
        history.append(record)
        print(json.dumps(record))
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "fold": args.fold,
        }
        torch.save(checkpoint, run_dir / "latest.pt")
        if validation_dice > best_dice:
            best_dice = validation_dice
            torch.save(checkpoint, run_dir / "best.pt")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    best = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model"])
    test = evaluate(
        model,
        test_frame,
        device,
        int(train_config["inference_chunk"]),
        run_dir / "test_predictions",
        args.max_val_cases,
    )
    test.to_csv(run_dir / "test_per_case.csv", index=False)
    summary = {
        "fold": args.fold,
        "best_epoch": int(best["epoch"]),
        "test_cases": int(len(test)),
        "test_dice_mean": float(test["dice"].mean()),
        "test_hd95_mm_mean": float(test["hd95_mm"].mean()),
        "test_absolute_rvd_mean": float(test["absolute_rvd"].mean()),
    }
    with (run_dir / "test_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
