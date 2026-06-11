from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.nn import functional as F

from .models.degradation import sensor_degrade


def tensor_statistics(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach()
    finite = torch.isfinite(value)
    finite_values = value[finite]
    stats: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype).replace("torch.", ""),
        "device": str(value.device),
        "numel": value.numel(),
        "nan_count": int(torch.isnan(value).sum()),
        "inf_count": int(torch.isinf(value).sum()),
        "zero_fraction": float((value == 0).float().mean()) if value.numel() else 0.0,
    }
    if finite_values.numel():
        finite_float = finite_values.float()
        stats.update(
            {
                "min": float(finite_float.min()),
                "max": float(finite_float.max()),
                "mean": float(finite_float.mean()),
                "std": float(finite_float.std(unbiased=False)),
                "abs_mean": float(finite_float.abs().mean()),
                "l2_rms": float(finite_float.square().mean().sqrt()),
            }
        )
    return stats


def _first_sample(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().float().cpu()
    return value[0] if value.ndim == 4 else value


def _normalize(value: torch.Tensor, symmetric: bool = False) -> torch.Tensor:
    if symmetric:
        scale = value.abs().quantile(0.995).clamp_min(1e-8)
        return (value / (2 * scale) + 0.5).clamp(0, 1)
    low = value.quantile(0.01)
    high = value.quantile(0.99)
    return ((value - low) / (high - low).clamp_min(1e-8)).clamp(0, 1)


def _rgb_array(tensor: torch.Tensor, mode: str) -> np.ndarray:
    value = _first_sample(tensor)
    if mode == "rgb":
        if value.shape[0] == 1:
            value = value.repeat(3, 1, 1)
        value = value[:3].clamp(0, 1)
    elif mode == "residual":
        value = _normalize(value[:3], symmetric=True)
    else:
        if value.ndim == 3:
            value = value.abs().mean(dim=0)
        value = _normalize(value)
        red = torch.clamp(1.5 * value - 0.5, 0, 1)
        green = torch.clamp(1.5 - (2 * value - 1).abs() * 1.5, 0, 1)
        blue = torch.clamp(1 - 1.5 * value, 0, 1)
        value = torch.stack((red, green, blue))
    return (value.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def _feature_grid(tensor: torch.Tensor, max_channels: int = 16) -> np.ndarray:
    value = _first_sample(tensor)
    if value.ndim == 2:
        value = value.unsqueeze(0)
    count = min(value.shape[0], max_channels)
    columns = min(4, count)
    rows = math.ceil(count / columns)
    tiles = []
    for index in range(count):
        channel = _normalize(value[index])
        tiles.append((channel.numpy() * 255).round().astype(np.uint8))
    height, width = tiles[0].shape
    grid = np.zeros((rows * height, columns * width), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        grid[row * height : (row + 1) * height, column * width : (column + 1) * width] = tile
    return np.repeat(grid[:, :, None], 3, axis=2)


def _resize(array: np.ndarray, size: int = 320) -> Image.Image:
    image = Image.fromarray(array)
    scale = min(size / image.width, size / image.height)
    target = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    resampling = (
        Image.Resampling.NEAREST
        if image.width < size or image.height < size
        else Image.Resampling.LANCZOS
    )
    image = image.resize(target, resampling)
    canvas = Image.new("RGB", (size, size), "white")
    offset = ((size - image.width) // 2, (size - image.height) // 2)
    canvas.paste(image, offset)
    return canvas


def _contact_sheet(
    panels: list[tuple[str, np.ndarray]],
    destination: Path,
    columns: int = 4,
    panel_size: int = 320,
) -> None:
    if not panels:
        return
    rows = math.ceil(len(panels) / columns)
    title_height = 44
    sheet = Image.new(
        "RGB",
        (columns * panel_size, rows * (panel_size + title_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    for index, (title, array) in enumerate(panels):
        row, column = divmod(index, columns)
        x = column * panel_size
        y = row * (panel_size + title_height)
        draw.rectangle((x, y, x + panel_size, y + title_height), fill=(242, 246, 252))
        draw.text((x + 10, y + 12), title, fill=(20, 37, 63), font=font)
        sheet.paste(_resize(array, panel_size), (x, y + title_height))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


class DiagnosticRecorder:
    def __init__(
        self,
        output_dir: str | Path,
        verbose: bool = True,
        fail_on_nonfinite: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.fail_on_nonfinite = fail_on_nonfinite
        self.tensors: dict[str, dict[str, Any]] = {}
        self.scalars: dict[str, float | int | str | bool] = {}
        self.visuals: dict[str, tuple[torch.Tensor, str]] = {}
        self.diffusion_snapshots: list[tuple[int, torch.Tensor, torch.Tensor]] = []

    def capture(
        self,
        name: str,
        tensor: torch.Tensor,
        visual: str | None = None,
    ) -> None:
        stats = tensor_statistics(tensor)
        self.tensors[name] = stats
        if visual:
            self.visuals[name] = (tensor.detach().float().cpu(), visual)
        if self.verbose:
            print(
                f"[debug] {name}: shape={tuple(stats['shape'])} "
                f"min={stats.get('min', float('nan')):.5f} "
                f"max={stats.get('max', float('nan')):.5f} "
                f"mean={stats.get('mean', float('nan')):.5f} "
                f"std={stats.get('std', float('nan')):.5f}",
                flush=True,
            )
        if self.fail_on_nonfinite and (stats["nan_count"] or stats["inf_count"]):
            raise FloatingPointError(
                f"Non-finite values in {name}: "
                f"nan={stats['nan_count']} inf={stats['inf_count']}"
            )

    def scalar(self, name: str, value: float | int | str | bool | torch.Tensor) -> None:
        if isinstance(value, torch.Tensor):
            value = float(value.detach().float().mean())
        self.scalars[name] = value
        if self.verbose:
            print(f"[debug] {name}={value}", flush=True)

    def diffusion_step(
        self,
        index: int,
        timestep: int,
        latent: torch.Tensor,
        clean: torch.Tensor,
    ) -> None:
        self.diffusion_snapshots.append(
            (timestep, latent.detach().float().cpu(), clean.detach().float().cpu())
        )
        self.capture(f"diffusion.step_{index:03d}.latent_t{timestep}", latent)
        self.capture(f"diffusion.step_{index:03d}.clean_t{timestep}", clean)

    def add_spatial_metrics(
        self,
        lr: torch.Tensor,
        base: torch.Tensor,
        residual: torch.Tensor,
        output: torch.Tensor,
        degradation: torch.Tensor,
        scale: int,
        target: torch.Tensor | None = None,
        consistency_lr: torch.Tensor | None = None,
        degradation_severity: str = "mild",
    ) -> None:
        with torch.no_grad():
            consistency_lr = lr if consistency_lr is None else consistency_lr
            degraded_base = sensor_degrade(
                base,
                degradation,
                scale=scale,
                severity=degradation_severity,
            )
            degraded_output = sensor_degrade(
                output,
                degradation,
                scale=scale,
                severity=degradation_severity,
            )
            self.capture("consistency.degraded_base", degraded_base, visual="rgb")
            self.capture("consistency.degraded_output", degraded_output, visual="rgb")
            self.capture("consistency.clean_lr", consistency_lr, visual="rgb")
            self.capture("consistency.observed_noise", lr - consistency_lr, visual="residual")
            self.capture(
                "consistency.lr_error",
                (degraded_output - consistency_lr).abs(),
                visual="heatmap",
            )
            self.scalar("spatial.lr_base_l1", F.l1_loss(degraded_base, consistency_lr))
            self.scalar("spatial.lr_output_l1", F.l1_loss(degraded_output, consistency_lr))
            self.scalar("degradation.observed_noise_l1", F.l1_loss(lr, consistency_lr))
            self.scalar(
                "degradation.noise_to_signal_ratio",
                (lr - consistency_lr).abs().mean()
                / consistency_lr.abs().mean().clamp_min(1e-8),
            )
            self.scalar("spatial.residual_abs_mean", residual.abs().mean())
            self.scalar(
                "spatial.residual_to_base_ratio",
                residual.abs().mean() / base.abs().mean().clamp_min(1e-8),
            )
            self.scalar(
                "spatial.output_clipped_fraction",
                ((output <= 0) | (output >= 1)).float().mean(),
            )
            if target is not None:
                self.capture("target.hr", target, visual="rgb")
                self.capture("target.absolute_error", (output - target).abs(), visual="heatmap")
                self.scalar("spatial.output_target_l1", F.l1_loss(output, target))
                self.scalar("spatial.base_target_l1", F.l1_loss(base, target))

    def export(self, extra: dict[str, Any] | None = None) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "scalars": self.scalars,
            "tensors": self.tensors,
            "extra": extra or {},
        }
        report_path = self.output_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        overview_order = [
            "input.lr",
            "base.hr",
            "decoder.residual",
            "output.hr",
            "target.hr",
            "consistency.clean_lr",
            "consistency.degraded_output",
            "consistency.lr_error",
            "target.absolute_error",
        ]
        overview = []
        for name in overview_order:
            if name in self.visuals:
                tensor, mode = self.visuals[name]
                overview.append((name, _rgb_array(tensor, mode)))
        _contact_sheet(overview, self.output_dir / "overview.png", columns=4)

        feature_order = [
            "lr_features.f128",
            "lr_features.f64",
            "lr_features.f32",
            "lr_features.f16",
            "latent.denoised",
            "mapper.content",
            "mapper.evidence_confidence",
            "mapper.edit_permission",
            "output.abstention_map",
        ]
        features = []
        for name in feature_order:
            if name in self.visuals:
                tensor, mode = self.visuals[name]
                array = (
                    _feature_grid(tensor)
                    if mode == "features"
                    else _rgb_array(tensor, mode)
                )
                features.append((name, array))
        _contact_sheet(features, self.output_dir / "features.png", columns=4)

        trajectory = []
        for timestep, latent, clean in self.diffusion_snapshots:
            trajectory.append((f"latent t={timestep}", _rgb_array(latent, "heatmap")))
            trajectory.append((f"clean t={timestep}", _rgb_array(clean, "heatmap")))
        _contact_sheet(trajectory, self.output_dir / "diffusion_trajectory.png", columns=4)

        lines = ["GeoDiff-GAN diagnostic summary", ""]
        for name, value in sorted(self.scalars.items()):
            lines.append(f"{name}: {value}")
        lines.extend(["", "Tensor checks:"])
        for name, stats in self.tensors.items():
            lines.append(
                f"{name}: shape={stats['shape']} mean={stats.get('mean')} "
                f"std={stats.get('std')} nan={stats['nan_count']} inf={stats['inf_count']}"
            )
        (self.output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
        return report_path
