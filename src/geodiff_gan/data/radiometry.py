from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import load_manifest


def _channel_first(array: np.ndarray, key: str) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError(f"Patch key {key!r} must be three-dimensional")
    if array.shape[0] <= 16:
        return array.astype(np.float32, copy=False)
    if array.shape[-1] <= 16:
        return np.moveaxis(array, -1, 0).astype(np.float32, copy=False)
    raise ValueError(f"Cannot determine channel axis for {key!r}: {array.shape}")


def area_downsample(image: np.ndarray, scale: int) -> np.ndarray:
    """Area-average a CHW image by an integer scale without interpolation."""
    if scale < 1:
        raise ValueError("scale must be positive")
    channels, height, width = image.shape
    if height % scale or width % scale:
        raise ValueError(
            f"Image shape {(height, width)} is not divisible by scale {scale}"
        )
    return image.reshape(
        channels,
        height // scale,
        scale,
        width // scale,
        scale,
    ).mean(axis=(2, 4))


def estimate_integer_shift(
    source: np.ndarray,
    target: np.ndarray,
    maximum_shift: int = 2,
) -> tuple[int, int, float, float]:
    """Return the LR-pixel shift with the highest normalized correlation."""
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("source and target must be equally shaped 2D arrays")

    def correlation(first: np.ndarray, second: np.ndarray) -> float:
        first = first.astype(np.float64, copy=False).ravel()
        second = second.astype(np.float64, copy=False).ravel()
        finite = np.isfinite(first) & np.isfinite(second)
        if finite.sum() < 16:
            return 0.0
        first = first[finite] - first[finite].mean()
        second = second[finite] - second[finite].mean()
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        return float(first.dot(second) / denominator) if denominator > 1e-12 else 0.0

    zero_correlation = correlation(source, target)
    best = (0, 0, zero_correlation)
    height, width = source.shape
    for row_shift in range(-maximum_shift, maximum_shift + 1):
        for col_shift in range(-maximum_shift, maximum_shift + 1):
            source_rows = slice(max(0, -row_shift), min(height, height - row_shift))
            target_rows = slice(max(0, row_shift), min(height, height + row_shift))
            source_cols = slice(max(0, -col_shift), min(width, width - col_shift))
            target_cols = slice(max(0, col_shift), min(width, width + col_shift))
            value = correlation(
                source[source_rows, source_cols],
                target[target_rows, target_cols],
            )
            if value > best[2]:
                best = (row_shift, col_shift, value)
    return best[0], best[1], best[2], zero_correlation


def _robust_affine(source: np.ndarray, target: np.ndarray) -> dict[str, float]:
    design = np.column_stack((source, np.ones_like(source)))
    slope, offset = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = target - (slope * source + offset)
    low, high = np.quantile(residual, (0.02, 0.98))
    keep = (residual >= low) & (residual <= high)
    if keep.sum() >= 32:
        robust_design = np.column_stack((source[keep], np.ones(keep.sum())))
        slope, offset = np.linalg.lstsq(
            robust_design, target[keep], rcond=None
        )[0]
    slope = float(np.clip(slope, 0.5, 1.5))
    offset = float(np.clip(offset, -0.2, 0.2))
    raw_error = source - target
    calibrated_error = slope * source + offset - target
    return {
        "slope": slope,
        "offset": offset,
        "samples": int(source.size),
        "raw_mae": float(np.mean(np.abs(raw_error))),
        "calibrated_mae": float(np.mean(np.abs(calibrated_error))),
        "raw_rmse": float(np.sqrt(np.mean(raw_error**2))),
        "calibrated_rmse": float(np.sqrt(np.mean(calibrated_error**2))),
    }


def fit_paired_radiometric_calibration(
    manifest: str | Path,
    condition_key: str = "lr",
    scale: int = 3,
    maximum_patches: int = 2000,
    pixels_per_patch: int = 1024,
    maximum_shift: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit Landsat RGB to area-averaged Sentinel RGB using train records only."""
    records = load_manifest(manifest, split="train")
    if not records:
        raise ValueError("The manifest has no training records")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))[:maximum_patches]
    source_values: list[list[np.ndarray]] = [[], [], []]
    target_values: list[list[np.ndarray]] = [[], [], []]
    shifts: Counter[tuple[int, int]] = Counter()
    shift_improvements: list[float] = []
    used_patches = 0
    for record_index in order:
        record = records[int(record_index)]
        with np.load(record.patch) as patch:
            if condition_key not in patch.files:
                raise KeyError(
                    f"Patch {record.patch} does not contain {condition_key!r}"
                )
            lr = _channel_first(patch[condition_key], condition_key)[:3]
            hr = _channel_first(patch["hr"], "hr")[:3]
            target_lr = area_downsample(hr, scale)
            valid_lr = (
                _channel_first(patch["valid_mask_lr"], "valid_mask_lr")[0] >= 0.999
                if "valid_mask_lr" in patch.files
                else np.ones(lr.shape[-2:], dtype=bool)
            )
            if "valid_mask_hr" in patch.files:
                valid_hr = _channel_first(
                    patch["valid_mask_hr"], "valid_mask_hr"
                )
                valid_lr &= area_downsample(valid_hr, scale)[0] >= 0.999
        valid_lr &= np.isfinite(lr).all(axis=0)
        valid_lr &= np.isfinite(target_lr).all(axis=0)
        candidates = np.flatnonzero(valid_lr.ravel())
        if candidates.size < 32:
            continue
        selected = rng.choice(
            candidates,
            size=min(pixels_per_patch, candidates.size),
            replace=False,
        )
        for band in range(3):
            source_values[band].append(lr[band].ravel()[selected])
            target_values[band].append(target_lr[band].ravel()[selected])
        source_gray = lr.mean(axis=0)
        target_gray = target_lr.mean(axis=0)
        row_shift, col_shift, best_corr, zero_corr = estimate_integer_shift(
            source_gray,
            target_gray,
            maximum_shift=maximum_shift,
        )
        shifts[(row_shift, col_shift)] += 1
        shift_improvements.append(best_corr - zero_corr)
        used_patches += 1
    if not used_patches:
        raise ValueError("No valid training patches were available for calibration")
    channel_names = ("red", "green", "blue")
    channels = []
    for band, name in enumerate(channel_names):
        source = np.concatenate(source_values[band]).astype(np.float64)
        target = np.concatenate(target_values[band]).astype(np.float64)
        channels.append({"name": name, **_robust_affine(source, target)})
    nonzero_shifts = sum(count for shift, count in shifts.items() if shift != (0, 0))
    return {
        "version": 1,
        "method": "robust_train_affine_to_area_averaged_sentinel",
        "manifest": str(Path(manifest)),
        "fit_split": "train",
        "condition_key": condition_key,
        "scale": scale,
        "seed": seed,
        "patches": used_patches,
        "channels": channels,
        "slope": [channel["slope"] for channel in channels],
        "offset": [channel["offset"] for channel in channels],
        "registration_audit": {
            "maximum_shift_lr_pixels": maximum_shift,
            "shift_counts": {
                f"row_{row:+d}_col_{col:+d}": count
                for (row, col), count in sorted(shifts.items())
            },
            "nonzero_shift_fraction": nonzero_shifts / used_patches,
            "mean_correlation_improvement": float(np.mean(shift_improvements)),
        },
    }
