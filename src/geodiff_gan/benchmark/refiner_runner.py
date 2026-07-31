from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
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
from ..models.blocks import high_pass
from ..models.degradation import sensor_degrade
from .models import MODEL_SPECS, build_benchmark_model
from .refiner import (
    FrozenBackboneRefiner,
    RefinerOutput,
    SensorNullspaceHighFrequencyRefiner,
)
from .runner import CroppedDataset


@dataclass
class RefinerConfig:
    model: str
    source_root: Path
    manifest: Path
    base_checkpoint: Path
    output: Path
    max_updates: int = 10000
    batch_size: int = 1
    accumulation: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    validation_limit: int = 64
    test_limit: int | None = 80
    validate_every: int = 500
    early_stopping_patience: int = 6
    degradation_seed: int = 42
    degradation_severity: str = "mild"
    channels: int = 32
    condition_dim: int = 64
    blocks_per_level: tuple[int, int, int] = (2, 2, 3)
    max_residual: float = 0.12
    nullspace_iterations: int = 1
    nullspace_step: float = 0.75
    amp: bool = True
    optional_metrics: bool = False
    save_images: int = 5
    bootstrap_samples: int = 2000
    seed: int = 42


LOSS_WEIGHTS = {
    "charbonnier": 1.0,
    "ssim": 0.2,
    "gradient": 0.1,
    "wavelet": 0.2,
    "detail": 0.5,
    "consistency": 1.0,
    "gate_calibration": 0.05,
    "smooth_suppression": 0.02,
}


METRIC_DIRECTIONS = {
    "l1": "lower",
    "psnr": "higher",
    "ssim": "higher",
    "edge_f1": "higher",
    "redegradation_l1": "lower",
    "wavelet_l1": "lower",
    "lpips": "lower",
    "dists": "lower",
}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _module_signature(model: nn.Module) -> str:
    layout = "\n".join(
        f"{name}:{tuple(value.shape)}:{value.dtype}"
        for name, value in model.state_dict().items()
    )
    return hashlib.sha256(layout.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset(config: RefinerConfig, split: str, augment: bool) -> Dataset:
    native_size = MODEL_SPECS[config.model].native_lr_size
    dataset = SentinelPatchDataset(
        config.manifest,
        split=split,
        scale=MODEL_SPECS[config.model].scale,
        caption_file=None,
        augment=augment,
        random_degradation=augment,
        degradation_seed=config.degradation_seed,
        degradation_severity=config.degradation_severity,
    )
    return CroppedDataset(
        dataset,
        lr_crop=native_size,
        scale=MODEL_SPECS[config.model].scale,
        random_crop=augment,
    )


def _load_base(config: RefinerConfig, device: torch.device) -> nn.Module:
    base = build_benchmark_model(
        config.model,
        config.source_root,
        architecture_mode="official",
    )
    state = torch.load(
        config.base_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_mode = state.get("architecture_mode", "official")
    if checkpoint_mode != "official":
        raise RuntimeError(
            "SN-HFR requires an unchanged official backbone checkpoint; "
            f"received architecture_mode={checkpoint_mode!r}."
        )
    checkpoint_signature = state.get("model_signature")
    actual_signature = _module_signature(base)
    if checkpoint_signature and checkpoint_signature != actual_signature:
        raise RuntimeError(
            "Base checkpoint signature does not match the selected official model."
        )
    base.load_state_dict(state["model"])
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    return base.to(device)


def _build_pair(
    config: RefinerConfig,
    device: torch.device,
) -> FrozenBackboneRefiner:
    base = _load_base(config, device)
    refiner = SensorNullspaceHighFrequencyRefiner(
        channels=config.channels,
        condition_dim=config.condition_dim,
        blocks_per_level=config.blocks_per_level,
        scale=MODEL_SPECS[config.model].scale,
        max_residual=config.max_residual,
        nullspace_iterations=config.nullspace_iterations,
        nullspace_step=config.nullspace_step,
        degradation_severity=config.degradation_severity,
    )
    return FrozenBackboneRefiner(base, refiner).to(device)


def _refiner_loss(
    output: RefinerOutput,
    base: torch.Tensor,
    target: torch.Tensor,
    clean_lr: torch.Tensor,
    degradation: torch.Tensor,
    config: RefinerConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_detail = high_pass(target - base)
    detail_magnitude = target_detail.abs().mean(dim=1, keepdim=True)
    with torch.no_grad():
        normalizer = torch.quantile(
            detail_magnitude.flatten(1),
            0.9,
            dim=1,
        ).view(-1, 1, 1, 1).clamp_min(1e-4)
        target_gate = (detail_magnitude / normalizer).clamp(0, 1)

    values = {
        "charbonnier": charbonnier(output.refined, target),
        "ssim": 1 - ssim(output.refined, target),
        "gradient": gradient_loss(output.refined, target),
        "wavelet": wavelet_loss(output.refined, target),
        "detail": F.l1_loss(output.residual, target_detail),
        "consistency": charbonnier(
            sensor_degrade(
                output.refined,
                degradation,
                scale=MODEL_SPECS[config.model].scale,
                severity=config.degradation_severity,
            ),
            clean_lr,
        ),
        "gate_calibration": F.smooth_l1_loss(output.confidence, target_gate),
        "smooth_suppression": (
            (1 - target_gate) * output.residual.abs()
        ).mean(),
    }
    total = sum(LOSS_WEIGHTS[name] * value for name, value in values.items())
    return total, {name: float(value.detach()) for name, value in values.items()}


@torch.no_grad()
def validate_pair(
    pair: FrozenBackboneRefiner,
    config: RefinerConfig,
    split: str = "val",
    limit: int | None = None,
) -> dict[str, float]:
    device = next(pair.refiner.parameters()).device
    dataset = _dataset(config, split, augment=False)
    total = min(len(dataset), limit) if limit is not None else len(dataset)
    if total == 0:
        raise RuntimeError(f"No {split} records in {config.manifest}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    totals: defaultdict[str, float] = defaultdict(float)
    pair.eval()
    for index, batch in enumerate(
        tqdm(loader, total=total, desc=f"{config.model} refiner {split}", unit="patch")
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
            output = pair(lr, degradation)
        base_metrics = basic_metrics(
            output.base.float(),
            target.float(),
            clean_lr.float(),
            degradation.float(),
            scale=MODEL_SPECS[config.model].scale,
            severity=config.degradation_severity,
        )
        refined_metrics = basic_metrics(
            output.refined.float(),
            target.float(),
            clean_lr.float(),
            degradation.float(),
            scale=MODEL_SPECS[config.model].scale,
            severity=config.degradation_severity,
        )
        for name, value in base_metrics.items():
            totals[f"base_{name}"] += value
        for name, value in refined_metrics.items():
            totals[f"refined_{name}"] += value
        totals["confidence_mean"] += float(output.confidence.mean())
        totals["residual_abs_mean"] += float(output.residual.abs().mean())
    return {name: value / total for name, value in totals.items()} | {"count": total}


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


def _normalized_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> np.ndarray:
    error = (prediction - target).abs().mean(dim=0)
    error = error / torch.quantile(error, 0.99).clamp_min(1e-6)
    return error.clamp(0, 1).detach().float().cpu().numpy()


def _spectrum(tensor: torch.Tensor) -> np.ndarray:
    gray = tensor.detach().float().mean(dim=0)
    spectrum = torch.fft.fftshift(torch.fft.fft2(gray)).abs()
    spectrum = torch.log1p(spectrum)
    spectrum = spectrum / torch.quantile(spectrum, 0.99).clamp_min(1e-6)
    return spectrum.clamp(0, 1).cpu().numpy()


def _save_overview(
    lr: torch.Tensor,
    base: torch.Tensor,
    output: RefinerOutput,
    target: torch.Tensor,
    path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    lr_up = F.interpolate(
        lr[None],
        size=target.shape[-2:],
        mode="bicubic",
        align_corners=False,
    )[0]
    residual_display = (
        0.5 + output.residual[0].detach().float() / (2 * 0.12)
    ).clamp(0, 1)
    panels: list[tuple[str, np.ndarray, str | None]] = [
        ("Observed LR", lr_up.permute(1, 2, 0).cpu().numpy(), None),
        ("Official base", base.permute(1, 2, 0).cpu().numpy(), None),
        ("Base + SN-HFR", output.refined[0].permute(1, 2, 0).float().cpu().numpy(), None),
        ("Target HR", target.permute(1, 2, 0).cpu().numpy(), None),
        ("Confidence", output.confidence[0, 0].float().cpu().numpy(), "viridis"),
        ("Residual (centered)", residual_display.permute(1, 2, 0).cpu().numpy(), None),
        ("Base absolute error", _normalized_error(base, target), "turbo"),
        (
            "Refined absolute error",
            _normalized_error(output.refined[0], target),
            "turbo",
        ),
        ("Target detail spectrum", _spectrum(high_pass(target[None])[0]), "magma"),
        ("Predicted residual spectrum", _spectrum(output.residual[0]), "magma"),
    ]
    figure, axes = plt.subplots(2, 5, figsize=(20, 8))
    for axis, (panel_title, image, cmap) in zip(axes.flat, panels):
        axis.imshow(np.clip(image, 0, 1), cmap=cmap)
        axis.set_title(panel_title)
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _paired_summary(
    rows: list[dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    metric_names = sorted(
        set.intersection(
            *(set(row["base"]) & set(row["refined"]) for row in rows)
        )
    )
    base = {
        name: float(np.mean([row["base"][name] for row in rows]))
        for name in metric_names
    }
    refined = {
        name: float(np.mean([row["refined"][name] for row in rows]))
        for name in metric_names
    }
    rng = np.random.default_rng(seed)
    paired: dict[str, Any] = {}
    for name in metric_names:
        deltas = np.asarray(
            [row["refined"][name] - row["base"][name] for row in rows],
            dtype=np.float64,
        )
        bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
        for index in range(bootstrap_samples):
            sampled = rng.integers(0, len(deltas), size=len(deltas))
            bootstrap[index] = deltas[sampled].mean()
        lower, upper = np.quantile(bootstrap, (0.025, 0.975))
        direction = METRIC_DIRECTIONS.get(name, "lower")
        useful_delta = deltas if direction == "higher" else -deltas
        paired[name] = {
            "direction": direction,
            "mean_delta_refined_minus_base": float(deltas.mean()),
            "bootstrap_95ci": [float(lower), float(upper)],
            "improved_patch_fraction": float((useful_delta > 0).mean()),
        }
    return {
        "count": len(rows),
        "base": base,
        "refined": refined,
        "paired": paired,
    }


@torch.no_grad()
def evaluate_refiner(
    config: RefinerConfig,
    checkpoint: Path | None = None,
    split: str = "test",
) -> dict[str, Any]:
    _seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pair = _build_pair(config, device)
    checkpoint = checkpoint or select_refiner_checkpoint(config.output)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("base_signature") != _module_signature(pair.backbone):
        raise RuntimeError(
            "Refiner checkpoint does not match the selected base architecture."
        )
    if state.get("base_checkpoint_sha256") != _file_sha256(config.base_checkpoint):
        raise RuntimeError(
            "Refiner checkpoint does not match the selected frozen base weights."
        )
    if state.get("refiner_signature") != _module_signature(pair.refiner):
        raise RuntimeError(
            "Refiner checkpoint does not match the selected SN-HFR architecture."
        )
    pair.refiner.load_state_dict(state["refiner"])
    pair.eval()

    dataset = _dataset(config, split, augment=False)
    total = (
        min(len(dataset), config.test_limit)
        if config.test_limit is not None
        else len(dataset)
    )
    if total == 0:
        raise RuntimeError(f"No {split} records in {config.manifest}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    optional = OptionalMetricSuite(device, enabled=config.optional_metrics)
    image_dir = config.output / "paired_images" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for index, batch in enumerate(
        tqdm(loader, total=total, desc=f"{config.model} paired {split}", unit="patch")
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
            output = pair(lr, degradation)
        base = output.base
        base_metrics = basic_metrics(
            base.float(),
            target.float(),
            clean_lr.float(),
            degradation.float(),
            scale=MODEL_SPECS[config.model].scale,
            severity=config.degradation_severity,
        )
        refined_metrics = basic_metrics(
            output.refined.float(),
            target.float(),
            clean_lr.float(),
            degradation.float(),
            scale=MODEL_SPECS[config.model].scale,
            severity=config.degradation_severity,
        )
        base_metrics["wavelet_l1"] = float(wavelet_loss(base.float(), target.float()))
        refined_metrics["wavelet_l1"] = float(
            wavelet_loss(output.refined.float(), target.float())
        )
        base_metrics.update(optional(base.float(), target.float()))
        refined_metrics.update(optional(output.refined.float(), target.float()))
        row = {
            "index": index,
            "patch": batch["patch"][0],
            "tile_id": batch["tile_id"][0],
            "base": base_metrics,
            "refined": refined_metrics,
            "confidence_mean": float(output.confidence.mean()),
            "residual_abs_mean": float(output.residual.abs().mean()),
        }
        rows.append(row)
        if index < config.save_images:
            stem = Path(batch["patch"][0]).stem
            _save_rgb(lr[0], image_dir / f"{index:03d}_{stem}_lr.png")
            _save_rgb(base[0], image_dir / f"{index:03d}_{stem}_base.png")
            _save_rgb(
                output.refined[0],
                image_dir / f"{index:03d}_{stem}_refined.png",
            )
            _save_rgb(target[0], image_dir / f"{index:03d}_{stem}_hr.png")
            _save_overview(
                lr[0].float().cpu(),
                base[0].float().cpu(),
                RefinerOutput(
                    base=output.base.float().cpu(),
                    refined=output.refined.float().cpu(),
                    residual=output.residual.float().cpu(),
                    raw_residual=output.raw_residual.float().cpu(),
                    confidence=output.confidence.float().cpu(),
                    sensor_error=output.sensor_error.float().cpu(),
                    degraded_base=output.degraded_base.float().cpu(),
                ),
                target[0].float().cpu(),
                image_dir / f"{index:03d}_{stem}_overview.png",
                f"{MODEL_SPECS[config.model].name}: official base vs SN-HFR",
            )

    summary = _paired_summary(
        rows,
        bootstrap_samples=config.bootstrap_samples,
        seed=config.seed,
    )
    summary.update(
        {
            "model": config.model,
            "checkpoint": str(checkpoint),
            "native_lr_size": MODEL_SPECS[config.model].native_lr_size,
            "native_hr_size": (
                MODEL_SPECS[config.model].native_lr_size
                * MODEL_SPECS[config.model].scale
            ),
            "comparison": "official base vs same frozen base plus SN-HFR",
        }
    )
    config.output.mkdir(parents=True, exist_ok=True)
    (config.output / f"{split}_paired_metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    with (config.output / f"{split}_per_patch.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return summary


def select_refiner_checkpoint(output: Path) -> Path:
    best = output / "best.pt"
    latest = output / "latest.pt"
    if best.exists():
        return best
    if latest.exists():
        return latest
    raise FileNotFoundError(f"No SN-HFR checkpoint found in {output}")


def train_refiner(
    config: RefinerConfig,
    evaluate_test: bool = True,
) -> dict[str, Any]:
    _seed_everything(config.seed)
    config.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    pair = _build_pair(config, device)
    refiner_signature = _module_signature(pair.refiner)
    base_signature = _module_signature(pair.backbone)
    base_checkpoint_sha256 = _file_sha256(config.base_checkpoint)
    refiner_parameters = sum(
        parameter.numel() for parameter in pair.refiner.parameters()
    )
    train_dataset = _dataset(config, "train", augment=True)
    loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )
    optimizer = torch.optim.AdamW(
        pair.refiner.parameters(),
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
        if state.get("base_signature") != base_signature:
            raise RuntimeError(
                "Refiner checkpoint belongs to a different base architecture."
            )
        if state.get("base_checkpoint_sha256") != base_checkpoint_sha256:
            raise RuntimeError(
                "Refiner checkpoint belongs to different frozen base weights."
            )
        if state.get("refiner_signature") != refiner_signature:
            raise RuntimeError(
                "Refiner checkpoint architecture does not match this configuration."
            )
        pair.refiner.load_state_dict(state["refiner"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        update = int(state["update"])
        best_l1 = float(state.get("best_l1", best_l1))
        stale_validations = int(state.get("stale_validations", 0))
        print(f"[{config.model} SN-HFR] resumed at update {update:,}", flush=True)

    metadata = {
        "model": config.model,
        "base_checkpoint": str(config.base_checkpoint),
        "base_signature": base_signature,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "refiner_signature": refiner_signature,
        "refiner_parameters": refiner_parameters,
        "base_trainable_parameters": 0,
        "native_lr_size": MODEL_SPECS[config.model].native_lr_size,
        "native_hr_size": (
            MODEL_SPECS[config.model].native_lr_size
            * MODEL_SPECS[config.model].scale
        ),
        "loss_weights": LOSS_WEIGHTS,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
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
        desc=f"{config.model} SN-HFR",
        unit="update",
    )
    iterator = iter(loader)
    started = time.monotonic()
    stopped_early = stale_validations >= config.early_stopping_patience
    if stopped_early:
        print(
            f"[{config.model} SN-HFR] checkpoint already satisfied early stopping; "
            "skipping additional updates.",
            flush=True,
        )
    while update < config.max_updates and not stopped_early:
        pair.train()
        accumulated = 0.0
        components: defaultdict[str, float] = defaultdict(float)
        for _ in range(config.accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            lr = batch["lr"].to(device, non_blocking=True)
            clean_lr = batch["clean_lr"].to(device, non_blocking=True)
            target = batch["hr"].to(device, non_blocking=True)
            degradation = batch["degradation"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=config.amp and device.type == "cuda",
            ):
                output = pair(lr, degradation)
                base = output.base.detach()
                loss, values = _refiner_loss(
                    output,
                    base,
                    target,
                    clean_lr,
                    degradation,
                    config,
                )
                scaled_loss = loss / config.accumulation
            scaler.scale(scaled_loss).backward()
            accumulated += float(loss.detach()) / config.accumulation
            for name, value in values.items():
                components[name] += value / config.accumulation
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(pair.refiner.parameters(), 1.0)
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
            validation = validate_pair(
                pair,
                config,
                split="val",
                limit=config.validation_limit,
            )
            record.update({f"val_{key}": value for key, value in validation.items()})
            refined_l1 = validation["refined_l1"]
            improved = refined_l1 < best_l1 - 1e-6
            if improved:
                best_l1 = refined_l1
                stale_validations = 0
            else:
                stale_validations += 1
            state = {
                "refiner": pair.refiner.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "update": update,
                "best_l1": best_l1,
                "stale_validations": stale_validations,
                "base_signature": base_signature,
                "base_checkpoint_sha256": base_checkpoint_sha256,
                "refiner_signature": refiner_signature,
                "model": config.model,
            }
            torch.save(state, latest_path)
            if improved:
                torch.save(state, best_path)
            elapsed = time.monotonic() - started
            print(
                f"\n[{config.model} SN-HFR] update={update:,} "
                f"base_psnr={validation['base_psnr']:.2f} "
                f"refined_psnr={validation['refined_psnr']:.2f} "
                f"base_l1={validation['base_l1']:.5f} "
                f"refined_l1={refined_l1:.5f} "
                f"best_l1={best_l1:.5f} "
                f"early_stop={stale_validations}/{config.early_stopping_patience} "
                f"elapsed={elapsed / 3600:.2f}h",
                flush=True,
            )
            if stale_validations >= config.early_stopping_patience:
                print(f"[{config.model} SN-HFR] early stopping", flush=True)
                stopped_early = True
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if stopped_early:
            break
    progress.close()
    checkpoint = select_refiner_checkpoint(config.output)
    result = metadata | {"checkpoint": str(checkpoint)}
    if evaluate_test:
        result["test"] = evaluate_refiner(
            config,
            checkpoint=checkpoint,
            split="test",
        )
    return result
