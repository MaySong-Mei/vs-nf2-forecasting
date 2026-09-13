#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import time
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
from ucsd_repro.provenance import collect_runtime_provenance, write_strict_json


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the memory-bounded DeepGrowth-style profile")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-cases", type=int)
    parser.add_argument("--max-test-cases", type=int)
    parser.add_argument("--resume", type=Path, help="resume model and optimizer state from a checkpoint")
    parser.add_argument(
        "--include-qc-failed",
        action="store_true",
        help="diagnostic only: include prepared cases that failed crop/registration QC",
    )
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


def finite_mean(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    finite = numeric[np.isfinite(numeric)]
    return float(finite.mean()) if len(finite) else None


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
            voxel_volume = float(np.prod(spacing))
            previous_volume = float(masks[1].sum() * voxel_volume)
            target_volume = float(masks[2].sum() * voxel_volume)
            target_relative_change = (
                (target_volume - previous_volume) / previous_volume
                if previous_volume
                else float("nan")
            )
            np.savez_compressed(
                prediction_dir / f"{safe_name(case_id)}.npz", prediction=prediction.astype(np.uint8)
            )
            rows.append(
                {
                    "patient_id": str(item["patient_id"]),
                    "triplet_id": case_id,
                    "target_relative_change": target_relative_change,
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
    input_metadata_cases = len(pd.read_csv(args.metadata))
    train_frame, validation_frame, test_frame = load_split_metadata(
        args.metadata, args.splits, args.fold, include_qc_failed=args.include_qc_failed
    )
    modeling_cases = len(train_frame) + len(validation_frame) + len(test_frame)
    print(
        f"Preprocessing QC: using {modeling_cases}/{input_metadata_cases} prepared cases; "
        f"include_qc_failed={args.include_qc_failed}"
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
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    accumulation = int(train_config["gradient_accumulation"])
    epochs = args.epochs if args.epochs is not None else int(train_config["epochs"])
    run_dir = Path(train_config["checkpoint_dir"]) / config["experiment"]["name"] / f"fold_{args.fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime_provenance = collect_runtime_provenance(Path(__file__).resolve().parents[1])
    commit_sha = str(runtime_provenance["git"]["commit"])
    git_dirty = runtime_provenance["git"]["dirty"]
    with (run_dir / "resolved_config.yaml").open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    run_metadata = {
        "seed": seed,
        "fold": args.fold,
        "commit_sha": commit_sha,
        "git_dirty": git_dirty,
        "device": str(device),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "prepared_input_cases": input_metadata_cases,
        "modeling_cases_after_qc": modeling_cases,
        "included_qc_failed": args.include_qc_failed,
        "runtime_provenance": runtime_provenance,
    }
    write_strict_json(run_dir / "run_metadata.json", run_metadata)
    history = []
    best_dice = -1.0
    start_epoch = 1
    if args.resume:
        resumed = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        start_epoch = int(resumed["epoch"]) + 1
        best_path = run_dir / "best.pt"
        if best_path.exists():
            best_dice = float(torch.load(best_path, map_location="cpu", weights_only=False).get("best_dice", -1.0))
        history_path = run_dir / "history.csv"
        if history_path.exists():
            history = pd.read_csv(history_path).to_dict("records")
        print(f"Resumed {args.resume} at epoch {start_epoch}")
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()
    print(
        f"Device={device}; train/val/test={len(train_frame)}/{len(validation_frame)}/{len(test_frame)}; "
        f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}"
    )
    for epoch in range(start_epoch, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        epoch_losses = []
        for batch_number, batch in enumerate(loader, start=1):
            observed = batch["observed"].to(device, non_blocking=True)
            coords = batch["coords"].to(device, non_blocking=True)
            target_sdf = batch["sampled_sdfs"].to(device, non_blocking=True)
            days = batch["days"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
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
            "validation_hd95_mm": finite_mean(validation["hd95_mm"]),
            "epoch_seconds": float(time.perf_counter() - epoch_started),
        }
        history.append(record)
        print(json.dumps(record))
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "fold": args.fold,
            "seed": seed,
            "commit_sha": commit_sha,
            "best_dice": max(best_dice, validation_dice),
        }
        torch.save(checkpoint, run_dir / "latest.pt")
        if validation_dice > best_dice:
            best_dice = validation_dice
            torch.save(checkpoint, run_dir / "best.pt")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    if not (run_dir / "best.pt").exists():
        raise SystemExit("no best checkpoint exists; epochs must include at least one training epoch")
    best = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test = evaluate(
        model,
        test_frame,
        device,
        int(train_config["inference_chunk"]),
        run_dir / "test_predictions",
        args.max_test_cases,
    )
    test.to_csv(run_dir / "test_per_case.csv", index=False)
    summary = {
        "fold": args.fold,
        "best_epoch": int(best["epoch"]),
        "test_cases": int(len(test)),
        "test_dice_mean": finite_mean(test["dice"]),
        "test_hd95_mm_mean": finite_mean(test["hd95_mm"]),
        "test_absolute_rvd_mean": finite_mean(test["absolute_rvd"]),
        "training_seconds_this_invocation": float(time.perf_counter() - training_started),
        "peak_memory_allocated_gib": (
            float(torch.cuda.max_memory_allocated(device) / 2**30) if device.type == "cuda" else 0.0
        ),
        "peak_memory_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 2**30) if device.type == "cuda" else 0.0
        ),
        **run_metadata,
    }
    with (run_dir / "test_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
