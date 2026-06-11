from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.nn import functional as F

from .models.blocks import haar_wavelet
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


def _edge_map(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().float()
    gray = value.mean(dim=1, keepdim=True)
    dx = F.pad(gray[:, :, :, 1:] - gray[:, :, :, :-1], (0, 1, 0, 0))
    dy = F.pad(gray[:, :, 1:, :] - gray[:, :, :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx.square() + dy.square() + 1e-12)


def _spectrum_array(tensor: torch.Tensor) -> np.ndarray:
    value = _first_sample(tensor)
    if value.ndim == 3:
        value = value.mean(dim=0)
    spectrum = torch.fft.fftshift(torch.fft.fft2(value))
    magnitude = torch.log1p(spectrum.abs())
    return _rgb_array(magnitude[None], "heatmap")


def _histogram_array(
    tensor: torch.Tensor,
    bins: int = 64,
    width: int = 640,
    height: int = 360,
) -> np.ndarray:
    value = tensor.detach().float().cpu().flatten()
    value = value[torch.isfinite(value)]
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    if not value.numel():
        draw.text((20, 20), "No finite values", fill=(180, 30, 30), font=font)
        return np.asarray(canvas)
    low = float(value.quantile(0.005))
    high = float(value.quantile(0.995))
    if high <= low:
        high = low + 1e-6
    histogram = torch.histc(value.clamp(low, high), bins=bins, min=low, max=high)
    histogram = histogram / histogram.max().clamp_min(1)
    margin_left, margin_right = 62, 20
    margin_top, margin_bottom = 34, 52
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    draw.line(
        (margin_left, margin_top, margin_left, margin_top + plot_height),
        fill=(70, 85, 105),
        width=2,
    )
    draw.line(
        (
            margin_left,
            margin_top + plot_height,
            margin_left + plot_width,
            margin_top + plot_height,
        ),
        fill=(70, 85, 105),
        width=2,
    )
    bar_width = plot_width / bins
    for index, count in enumerate(histogram.tolist()):
        x0 = margin_left + index * bar_width
        x1 = margin_left + (index + 1) * bar_width
        y1 = margin_top + plot_height
        y0 = y1 - count * plot_height
        draw.rectangle((x0, y0, x1, y1), fill=(45, 108, 172))
    draw.text((margin_left, 8), f"mean={float(value.mean()):.5f}", fill=(30, 50, 75), font=font)
    draw.text(
        (margin_left + 210, 8),
        f"std={float(value.std(unbiased=False)):.5f}",
        fill=(30, 50, 75),
        font=font,
    )
    draw.text(
        (margin_left, height - 34),
        f"{low:.4f}",
        fill=(70, 85, 105),
        font=font,
    )
    high_text = f"{high:.4f}"
    draw.text(
        (width - margin_right - 70, height - 34),
        high_text,
        fill=(70, 85, 105),
        font=font,
    )
    return np.asarray(canvas)


def _overlay_array(image: torch.Tensor, policy: torch.Tensor) -> np.ndarray:
    image_value = image.detach().float()
    policy_value = F.interpolate(
        policy.detach().float(),
        size=image_value.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    rgb = _rgb_array(image_value, "rgb").astype(np.float32)
    heatmap = _rgb_array(policy_value, "heatmap").astype(np.float32)
    return np.clip(0.62 * rgb + 0.38 * heatmap, 0, 255).astype(np.uint8)


def _line_chart(
    points: list[tuple[int, float]],
    width: int = 640,
    height: int = 360,
) -> np.ndarray:
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    margin_left, margin_right = 66, 24
    margin_top, margin_bottom = 32, 50
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    draw.line(
        (margin_left, margin_top, margin_left, margin_top + plot_height),
        fill=(70, 85, 105),
        width=2,
    )
    draw.line(
        (
            margin_left,
            margin_top + plot_height,
            margin_left + plot_width,
            margin_top + plot_height,
        ),
        fill=(70, 85, 105),
        width=2,
    )
    if not points:
        return np.asarray(canvas)
    epochs = [point[0] for point in points]
    values = [point[1] for point in points]
    if len(points) == 1:
        x = margin_left + plot_width / 2
        y = margin_top + plot_height / 2
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(20, 75, 130))
        draw.text(
            (margin_left + 20, margin_top + 20),
            "Single epoch summary",
            fill=(70, 85, 105),
            font=font,
        )
        draw.text(
            (margin_left + 20, margin_top + 52),
            f"value={values[0]:.6f}",
            fill=(30, 50, 75),
            font=font,
        )
        draw.text(
            (margin_left + 20, margin_top + 84),
            "A trend requires at least two epochs.",
            fill=(140, 70, 35),
            font=font,
        )
        draw.text(
            (margin_left, height - 34),
            f"epoch {epochs[0]}",
            fill=(70, 85, 105),
            font=font,
        )
        return np.asarray(canvas)
    x_min, x_max = min(epochs), max(epochs)
    y_min, y_max = min(values), max(values)
    if x_max == x_min:
        x_max += 1
    if y_max == y_min:
        padding = max(abs(y_min) * 0.05, 1e-6)
        y_min -= padding
        y_max += padding
    coordinates = []
    for epoch, value in points:
        x = margin_left + (epoch - x_min) / (x_max - x_min) * plot_width
        y = margin_top + (y_max - value) / (y_max - y_min) * plot_height
        coordinates.append((x, y))
    if len(coordinates) > 1:
        draw.line(coordinates, fill=(45, 108, 172), width=4)
    for coordinate in coordinates:
        x, y = coordinate
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(20, 75, 130))
    draw.text((margin_left, 8), f"latest={values[-1]:.6f}", fill=(30, 50, 75), font=font)
    draw.text((8, margin_top), f"{y_max:.4g}", fill=(70, 85, 105), font=font)
    draw.text(
        (8, margin_top + plot_height - 18),
        f"{y_min:.4g}",
        fill=(70, 85, 105),
        font=font,
    )
    draw.text(
        (margin_left, height - 34),
        f"epoch {x_min}",
        fill=(70, 85, 105),
        font=font,
    )
    draw.text(
        (width - margin_right - 86, height - 34),
        f"epoch {epochs[-1]}",
        fill=(70, 85, 105),
        font=font,
    )
    return np.asarray(canvas)


def _scalar_bar_chart(
    values: list[tuple[str, float]],
    width: int = 960,
    row_height: int = 42,
) -> np.ndarray:
    height = max(120, 48 + len(values) * row_height)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    if not values:
        return np.asarray(canvas)
    maximum = max(abs(value) for _, value in values) or 1.0
    label_width = 260
    zero = label_width + (width - label_width - 50) / 2
    half_width = (width - label_width - 50) / 2
    draw.line((zero, 24, zero, height - 20), fill=(110, 120, 130), width=2)
    for index, (name, value) in enumerate(values):
        y = 34 + index * row_height
        draw.text((12, y - 9), name, fill=(30, 50, 75), font=font)
        extent = value / maximum * half_width
        color = (45, 140, 90) if value >= 0 else (190, 65, 65)
        bar = (min(zero, zero + extent), y - 10, max(zero, zero + extent), y + 10)
        draw.rectangle(bar, fill=color)
        draw.text((width - 142, y - 9), f"{value:.6f}", fill=(30, 50, 75), font=font)
    return np.asarray(canvas)


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
    title_width = max(16, panel_size // 10)
    wrapped_titles = []
    for title, _ in panels:
        display_title = title.replace(".", " / ").replace("_", " ")
        wrapped_titles.append(
            textwrap.wrap(
                display_title,
                width=title_width,
                break_long_words=False,
            )[:3]
        )
    max_title_lines = max(len(lines) for lines in wrapped_titles)
    font_size = 15 if panel_size < 280 else 18
    line_height = font_size + 4
    title_height = 14 + max_title_lines * line_height
    sheet = Image.new(
        "RGB",
        (columns * panel_size, rows * (panel_size + title_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=font_size)
    for index, (title, array) in enumerate(panels):
        row, column = divmod(index, columns)
        x = column * panel_size
        y = row * (panel_size + title_height)
        draw.rectangle((x, y, x + panel_size, y + title_height), fill=(242, 246, 252))
        for line_index, line in enumerate(wrapped_titles[index]):
            draw.text(
                (x + 10, y + 8 + line_index * line_height),
                line,
                fill=(20, 37, 63),
                font=font,
            )
        sheet.paste(_resize(array, panel_size), (x, y + title_height))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


class DiagnosticRecorder:
    def __init__(
        self,
        output_dir: str | Path,
        verbose: bool = True,
        fail_on_nonfinite: bool = True,
        panel_size: int = 320,
        histogram_bins: int = 64,
        max_feature_channels: int = 16,
        save_tensors: bool = False,
        max_saved_tensors: int = 32,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.fail_on_nonfinite = fail_on_nonfinite
        self.panel_size = max(128, int(panel_size))
        self.histogram_bins = max(8, int(histogram_bins))
        self.max_feature_channels = max(1, int(max_feature_channels))
        self.save_tensors = save_tensors
        self.max_saved_tensors = max(1, int(max_saved_tensors))
        self.tensors: dict[str, dict[str, Any]] = {}
        self.scalars: dict[str, float | int | str | bool] = {}
        self.visuals: dict[str, tuple[torch.Tensor, str]] = {}
        self.diffusion_snapshots: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        self.projection_snapshots: list[tuple[int, torch.Tensor, torch.Tensor]] = []

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

    def projection_step(
        self,
        index: int,
        image: torch.Tensor,
        lr_error: torch.Tensor,
    ) -> None:
        self.projection_snapshots.append(
            (
                index,
                image.detach().float().cpu(),
                lr_error.detach().float().cpu(),
            )
        )
        self.capture(f"projection.step_{index:02d}.image", image, visual="rgb")
        self.capture(
            f"projection.step_{index:02d}.lr_error",
            lr_error.abs(),
            visual="heatmap",
        )
        self.scalar(
            f"projection.step_{index:02d}.lr_error_l1",
            lr_error.abs().mean(),
        )

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
                self.capture("target.edges", _edge_map(target), visual="heatmap")
                self.capture("output.edges", _edge_map(output), visual="heatmap")
                target_bands = haar_wavelet(target)[1:]
                output_bands = haar_wavelet(output)[1:]
                for name, target_band, output_band in zip(
                    ("lh", "hl", "hh"),
                    target_bands,
                    output_bands,
                ):
                    self.capture(
                        f"wavelet.target_{name}",
                        target_band,
                        visual="residual",
                    )
                    self.capture(
                        f"wavelet.output_{name}",
                        output_band,
                        visual="residual",
                    )
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
        _contact_sheet(
            overview,
            self.output_dir / "overview.png",
            columns=4,
            panel_size=self.panel_size,
        )

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
                    _feature_grid(tensor, max_channels=self.max_feature_channels)
                    if mode == "features"
                    else _rgb_array(tensor, mode)
                )
                features.append((name, array))
        _contact_sheet(
            features,
            self.output_dir / "features.png",
            columns=4,
            panel_size=self.panel_size,
        )

        trajectory = []
        for timestep, latent, clean in self.diffusion_snapshots:
            trajectory.append((f"latent t={timestep}", _rgb_array(latent, "heatmap")))
            trajectory.append((f"clean t={timestep}", _rgb_array(clean, "heatmap")))
        _contact_sheet(
            trajectory,
            self.output_dir / "diffusion_trajectory.png",
            columns=4,
            panel_size=self.panel_size,
        )

        projection_panels = []
        for index, image, lr_error in self.projection_snapshots:
            projection_panels.append(
                (f"projection {index + 1}: HR", _rgb_array(image, "rgb"))
            )
            projection_panels.append(
                (
                    f"projection {index + 1}: LR error",
                    _rgb_array(lr_error.abs(), "heatmap"),
                )
            )
        _contact_sheet(
            projection_panels,
            self.output_dir / "projection_trajectory.png",
            columns=4,
            panel_size=self.panel_size,
        )

        stage = str((extra or {}).get("stage", (extra or {}).get("mode", "inference")))
        stage_orders = {
            "base": [
                "input.lr",
                "base.bicubic",
                "base.hr",
                "target.hr",
                "target.absolute_error",
                "output.edges",
                "target.edges",
                "consistency.degraded_output",
            ],
            "vae": [
                "target.residual",
                "vae.residual_reconstruction",
                "vae.latent",
                "vae.mean",
                "vae.log_variance",
                "decoder.raw_detail_residual",
                "decoder.residual",
                "output.hr",
                "target.absolute_error",
            ],
            "diffusion": [
                "diffusion.clean_latent",
                "diffusion.noisy_latent",
                "diffusion.target_velocity",
                "diffusion.predicted_velocity",
                "diffusion.velocity_absolute_error",
                "latent.denoised",
                "diffusion.decoded_residual",
                "target.residual",
            ],
            "joint": [
                "base.hr",
                "decoder.raw_detail_residual",
                "decoder.detail_high_pass",
                "decoder.evidence_residual",
                "output.ungated_sr",
                "output.sr_anchor",
                "output.pre_projection",
                "output.projection_update",
                "output.hr",
                "target.hr",
                "target.absolute_error",
                "consistency.lr_error",
            ],
            "edit": [
                "base.hr",
                "decoder.raw_detail_residual",
                "decoder.raw_edit_residual",
                "decoder.evidence_residual",
                "decoder.permission_edit_residual",
                "mapper.edit_permission",
                "output.pre_projection",
                "output.projection_update",
                "output.hr",
                "target.hr",
                "consistency.lr_error",
            ],
            "sr": [
                "base.hr",
                "decoder.raw_detail_residual",
                "decoder.detail_high_pass",
                "decoder.evidence_residual",
                "mapper.evidence_confidence",
                "output.abstention_map",
                "output.pre_projection",
                "output.projection_update",
                "output.hr",
                "target.hr",
                "target.absolute_error",
                "consistency.lr_error",
            ],
        }
        stage_order = stage_orders.get(stage, stage_orders.get("sr", []))
        stage_panels = []
        for name in stage_order:
            if name not in self.visuals:
                continue
            tensor, mode = self.visuals[name]
            array = (
                _feature_grid(tensor, max_channels=self.max_feature_channels)
                if mode == "features"
                else _rgb_array(tensor, mode)
            )
            stage_panels.append((name, array))
        _contact_sheet(
            stage_panels,
            self.output_dir / "stage_intermediates.png",
            columns=4,
            panel_size=self.panel_size,
        )

        histogram_names = [
            "input.lr",
            "base.hr",
            "latent.denoised",
            "diffusion.noisy_latent",
            "diffusion.predicted_velocity",
            "mapper.content",
            "mapper.evidence_confidence",
            "mapper.edit_permission",
            "decoder.raw_detail_residual",
            "decoder.raw_edit_residual",
            "decoder.residual",
            "output.hr",
            "target.hr",
        ]
        histogram_panels = [
            (
                name,
                _histogram_array(
                    self.visuals[name][0],
                    bins=self.histogram_bins,
                ),
            )
            for name in histogram_names
            if name in self.visuals
        ]
        _contact_sheet(
            histogram_panels,
            self.output_dir / "tensor_histograms.png",
            columns=3,
            panel_size=self.panel_size,
        )

        spectrum_names = [
            "input.lr",
            "base.hr",
            "decoder.raw_detail_residual",
            "decoder.detail_high_pass",
            "decoder.residual",
            "output.hr",
            "target.hr",
            "target.absolute_error",
        ]
        spectrum_panels = [
            (name, _spectrum_array(self.visuals[name][0]))
            for name in spectrum_names
            if name in self.visuals
        ]
        _contact_sheet(
            spectrum_panels,
            self.output_dir / "frequency_spectra.png",
            columns=4,
            panel_size=self.panel_size,
        )

        background_name = "output.hr" if "output.hr" in self.visuals else "base.hr"
        policy_panels = []
        if background_name in self.visuals:
            background = self.visuals[background_name][0]
            for name in (
                "mapper.evidence_confidence",
                "mapper.edit_permission",
                "output.abstention_map",
                "consistency.lr_error",
                "target.absolute_error",
            ):
                if name in self.visuals:
                    policy_panels.append(
                        (f"{name} overlay", _overlay_array(background, self.visuals[name][0]))
                    )
        _contact_sheet(
            policy_panels,
            self.output_dir / "policy_overlays.png",
            columns=3,
            panel_size=self.panel_size,
        )

        wavelet_panels = []
        for name in (
            "wavelet.target_lh",
            "wavelet.output_lh",
            "wavelet.target_hl",
            "wavelet.output_hl",
            "wavelet.target_hh",
            "wavelet.output_hh",
            "target.edges",
            "output.edges",
        ):
            if name in self.visuals:
                tensor, mode = self.visuals[name]
                wavelet_panels.append((name, _rgb_array(tensor, mode)))
        _contact_sheet(
            wavelet_panels,
            self.output_dir / "edges_and_wavelets.png",
            columns=4,
            panel_size=self.panel_size,
        )

        loss_values = [
            (name.removeprefix("loss."), float(value))
            for name, value in sorted(self.scalars.items())
            if name.startswith("loss.") and isinstance(value, (int, float))
        ]
        if loss_values:
            Image.fromarray(_scalar_bar_chart(loss_values)).save(
                self.output_dir / "loss_breakdown.png"
            )

        if self.save_tensors:
            archive = {}
            for name, (tensor, _) in list(self.visuals.items())[: self.max_saved_tensors]:
                archive[name.replace(".", "__")] = _first_sample(tensor).numpy()
            if archive:
                np.savez_compressed(self.output_dir / "intermediate_tensors.npz", **archive)

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


def append_training_history(
    history_path: str | Path,
    epoch: int,
    stage: str,
    metrics: dict[str, float],
    panel_size: int = 320,
) -> None:
    destination = Path(history_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = {"epoch": epoch, "stage": stage, "metrics": metrics}
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    records = {}
    for line in destination.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        records[(value["stage"], int(value["epoch"]))] = value
    ordered = sorted(records.values(), key=lambda value: (value["stage"], value["epoch"]))
    metric_names = sorted(
        {
            metric
            for value in ordered
            for metric in value.get("metrics", {})
            if metric != "discriminator"
        }
    )
    panels = []
    for metric in metric_names:
        points = [
            (int(value["epoch"]), float(value["metrics"][metric]))
            for value in ordered
            if metric in value.get("metrics", {})
        ]
        panels.append((metric, _line_chart(points)))
    _contact_sheet(
        panels,
        destination.with_name("training_curves.png"),
        columns=3,
        panel_size=max(200, panel_size),
    )
    destination.with_name("latest_metrics.json").write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )
