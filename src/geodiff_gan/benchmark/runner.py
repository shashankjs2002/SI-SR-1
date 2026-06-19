from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from ..data import SentinelPatchDataset
from ..losses import charbonnier, gradient_loss, ssim, wavelet_loss
from ..metrics import OptionalMetricSuite, basic_metrics
from .models import MODEL_SPECS, build_benchmark_model


class CroppedDataset(Dataset):
    def __init__(self, dataset: Dataset, lr_crop: int = 64, scale: int = 4) -> None:
        self.dataset = dataset
        self.lr_crop = lr_crop
        self.scale = scale

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        lr = sample["lr"]
        clean_lr = sample["clean_lr"]
        hr = sample["hr"]
        crop = min(self.lr_crop, lr.shape[-2], lr.shape[-1])
        top = int(torch.randint(0, lr.shape[-2] - crop + 1, ()).item())
        left = int(torch.randint(0, lr.shape[-1] - crop + 1, ()).item())
        hr_top, hr_left = top * self.scale, left * self.scale
        sample["lr"] = lr[:, top : top + crop, left : left + crop]
        sample["clean_lr"] = clean_lr[:, top : top + crop, left : left + crop]
        sample["hr"] = hr[
            :,
            hr_top : hr_top + crop * self.scale,
            hr_left : hr_left + crop * self.scale,
        ]
        return sample


@dataclass
class BenchmarkConfig:
    model: str
    source_root: Path
    manifest: Path
    output: Path
    max_updates: int = 50000
    batch_size: int = 1
    accumulation: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    lr_crop: int = 64
    num_workers: int = 4
    validation_limit: int = 64
    test_limit: int | None = 40
    validate_every: int = 2000
    early_stopping_patience: int = 6
    degradation_seed: int = 42
    degradation_severity: str = "mild"
    amp: bool = True
    optional_metrics: bool = False
    seed: int = 42


def _save_rgb(tensor: torch.Tensor, path: Path) -> None:
    array = (
        tensor.detach()
        .float()
        .clamp(0, 1)
        .mul(255)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    Image.fromarray(array).save(path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dataset(config: BenchmarkConfig, split: str, augment: bool) -> SentinelPatchDataset:
    return SentinelPatchDataset(
        config.manifest,
        split=split,
        scale=4,
        caption_file=None,
        augment=augment,
        random_degradation=augment,
        degradation_seed=config.degradation_seed,
        degradation_severity=config.degradation_severity,
    )


def _loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    values = {
        "charbonnier": charbonnier(prediction, target),
        "ssim": 1 - ssim(prediction, target),
        "gradient": gradient_loss(prediction, target),
        "wavelet": wavelet_loss(prediction, target),
    }
    total = (
        values["charbonnier"]
        + 0.2 * values["ssim"]
        + 0.1 * values["gradient"]
        + 0.05 * values["wavelet"]
    )
    return total, {name: float(value.detach()) for name, value in values.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    config: BenchmarkConfig,
    split: str = "val",
    limit: int | None = None,
    save_images: int = 0,
) -> dict[str, float]:
    device = next(model.parameters()).device
    dataset = _dataset(config, split, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    total = min(len(dataset), limit) if limit is not None else len(dataset)
    if total == 0:
        raise RuntimeError(f"No {split} records in {config.manifest}")
    optional = OptionalMetricSuite(device, enabled=config.optional_metrics)
    totals: defaultdict[str, float] = defaultdict(float)
    image_dir = config.output / "images" / split
    if save_images:
        image_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for index, batch in enumerate(
        tqdm(loader, total=total, desc=f"{config.model} {split}", unit="patch")
    ):
        if index >= total:
            break
        lr = batch["lr"].to(device)
        clean_lr = batch["clean_lr"].to(device)
        target = batch["hr"].to(device)
        degradation = batch["degradation"].to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=config.amp and device.type == "cuda",
        ):
            prediction = model(lr).clamp(0, 1)
        values = basic_metrics(
            prediction.float(),
            target.float(),
            clean_lr.float(),
            degradation.float(),
            scale=4,
            severity=config.degradation_severity,
        )
        values.update(optional(prediction.float(), target.float()))
        for name, value in values.items():
            totals[name] += value
        if index < save_images:
            stem = Path(batch["patch"][0]).stem
            _save_rgb(lr[0], image_dir / f"{index:03d}_{stem}_lr.png")
            _save_rgb(prediction[0], image_dir / f"{index:03d}_{stem}_sr.png")
            _save_rgb(target[0], image_dir / f"{index:03d}_{stem}_hr.png")
            error = (prediction[0] - target[0]).abs().mean(0)
            error = error / error.quantile(0.99).clamp_min(1e-6)
            Image.fromarray(error.clamp(0, 1).mul(255).byte().cpu().numpy()).save(
                image_dir / f"{index:03d}_{stem}_error.png"
            )
    return {name: value / total for name, value in totals.items()} | {"count": total}


def train(config: BenchmarkConfig) -> dict[str, Any]:
    _seed_everything(config.seed)
    config.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = build_benchmark_model(config.model, config.source_root).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    train_dataset = CroppedDataset(
        _dataset(config, "train", augment=True),
        lr_crop=config.lr_crop,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.99),
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.max_updates, 1),
        eta_min=config.learning_rate * 0.05,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.amp and device.type == "cuda",
    )
    latest_path = config.output / "latest.pt"
    best_path = config.output / "best.pt"
    history_path = config.output / "history.jsonl"
    update = 0
    best_l1 = math.inf
    stale_validations = 0
    if latest_path.exists():
        state = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        update = int(state["update"])
        best_l1 = float(state.get("best_l1", best_l1))
        stale_validations = int(state.get("stale_validations", 0))
        print(f"[{config.model}] resumed at update {update:,}", flush=True)
    metadata = {
        "model": config.model,
        "spec": MODEL_SPECS[config.model].__dict__,
        "parameters": parameter_count,
        "pixelshuffle": False,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in config.__dict__.items()
        },
    }
    (config.output / "run.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(
        total=config.max_updates,
        initial=update,
        desc=config.model,
        unit="update",
    )
    started = time.monotonic()
    iterator = iter(loader)
    while update < config.max_updates:
        accumulated = 0.0
        components: defaultdict[str, float] = defaultdict(float)
        for _ in range(config.accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            lr = batch["lr"].to(device, non_blocking=True)
            target = batch["hr"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=config.amp and device.type == "cuda",
            ):
                prediction = model(lr)
                loss, values = _loss(prediction, target)
                scaled_loss = loss / config.accumulation
            scaler.scale(scaled_loss).backward()
            accumulated += float(loss.detach()) / config.accumulation
            for name, value in values.items():
                components[name] += value / config.accumulation
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        update += 1
        progress.update(1)
        progress.set_postfix(loss=f"{accumulated:.4f}", best_l1=f"{best_l1:.5f}")
        record: dict[str, Any] = {
            "update": update,
            "loss": accumulated,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **components,
        }
        should_validate = (
            update % config.validate_every == 0 or update == config.max_updates
        )
        if should_validate:
            validation = evaluate(
                model,
                config,
                split="val",
                limit=config.validation_limit,
            )
            record.update({f"val_{key}": value for key, value in validation.items()})
            improved = validation["l1"] < best_l1 - 1e-6
            if improved:
                best_l1 = validation["l1"]
                stale_validations = 0
            else:
                stale_validations += 1
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "update": update,
                "best_l1": best_l1,
                "stale_validations": stale_validations,
            }
            torch.save(state, latest_path)
            if improved:
                torch.save(state, best_path)
            elapsed = time.monotonic() - started
            print(
                f"\n[{config.model}] update={update:,} "
                f"val_psnr={validation['psnr']:.2f} "
                f"val_ssim={validation['ssim']:.4f} "
                f"val_l1={validation['l1']:.5f} "
                f"best_l1={best_l1:.5f} elapsed={elapsed / 3600:.2f}h",
                flush=True,
            )
            if stale_validations >= config.early_stopping_patience:
                print(f"[{config.model}] early stopping", flush=True)
                with history_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                break
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    progress.close()
    checkpoint = best_path if best_path.exists() else latest_path
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    test_metrics = evaluate(
        model,
        config,
        "test",
        limit=config.test_limit,
        save_images=5,
    )
    (config.output / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2),
        encoding="utf-8",
    )
    return metadata | {"test": test_metrics, "checkpoint": str(checkpoint)}
