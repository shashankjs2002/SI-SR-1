"""Spectral-index diagnostics for matched six-band SR predictions."""
from __future__ import annotations

import torch
from torch.nn import functional as F


INDEX_BANDS = {
    "ndvi": ("nir", "red"),
    "ndwi": ("green", "nir"),
    "ndbi": ("swir1", "nir"),
}


def _band_lookup(band_names) -> dict[str, int]:
    lookup = {str(name).strip().casefold(): index for index, name in enumerate(band_names)}
    missing = sorted({name for pair in INDEX_BANDS.values() for name in pair} - set(lookup))
    if missing:
        raise ValueError(f"Spectral indices require bands {missing}; received {list(band_names)}")
    return lookup


def spectral_index_images(image: torch.Tensor, band_names) -> dict[str, torch.Tensor]:
    """Return NDVI, NDWI and NDBI maps in [-1, 1] for BCHW or CHW input."""
    if image.ndim not in (3, 4):
        raise ValueError(f"Expected CHW or BCHW image, got shape {tuple(image.shape)}")
    lookup = _band_lookup(band_names)
    channel_axis = 0 if image.ndim == 3 else 1
    values = {}
    for name, (positive, negative) in INDEX_BANDS.items():
        first = image.select(channel_axis, lookup[positive])
        second = image.select(channel_axis, lookup[negative])
        denominator = first + second
        values[name] = ((first - second) / denominator.clamp_min(1e-6)).clamp(-1, 1)
    return values


def _masked_values(value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return value.reshape(-1)
    while mask.ndim > value.ndim:
        mask = mask.squeeze(1)
    if mask.shape[-2:] != value.shape[-2:]:
        mask = F.interpolate(mask[:, None].float(), size=value.shape[-2:], mode="nearest").squeeze(1)
    return value[mask > 0.5]


def _correlation(first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor | None) -> float:
    first_values = _masked_values(first, mask).float()
    second_values = _masked_values(second, mask).float()
    if first_values.numel() < 2:
        return 0.0
    first_values = first_values - first_values.mean()
    second_values = second_values - second_values.mean()
    denominator = first_values.square().sum().sqrt() * second_values.square().sum().sqrt()
    if denominator < 1e-12:
        return float(torch.allclose(first_values, second_values))
    return float((first_values * second_values).sum() / denominator)


@torch.no_grad()
def spectral_index_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lr: torch.Tensor,
    band_names,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compare model indices with HR truth and with bicubic LR-index baselines."""
    prediction = prediction.float()
    target = target.float()
    lr_up = F.interpolate(lr.float(), size=target.shape[-2:], mode="bicubic", align_corners=False)
    predicted_indices = spectral_index_images(prediction, band_names)
    target_indices = spectral_index_images(target, band_names)
    lr_indices = spectral_index_images(lr_up, band_names)
    result = {}
    for name in INDEX_BANDS:
        predicted = predicted_indices[name]
        reference = target_indices[name]
        baseline = lr_indices[name]
        predicted_error = _masked_values(predicted - reference, mask)
        baseline_error = _masked_values(baseline - reference, mask)
        model_mae = float(predicted_error.abs().mean())
        lr_mae = float(baseline_error.abs().mean())
        result.update({
            f"{name}_mae": model_mae,
            f"{name}_rmse": float(predicted_error.square().mean().sqrt()),
            f"{name}_bias": float(predicted_error.mean()),
            f"{name}_correlation": _correlation(predicted, reference, mask),
            f"bicubic_{name}_mae": lr_mae,
            f"bicubic_{name}_rmse": float(baseline_error.square().mean().sqrt()),
            f"bicubic_{name}_bias": float(baseline_error.mean()),
            f"bicubic_{name}_correlation": _correlation(baseline, reference, mask),
            f"{name}_mae_improvement_vs_bicubic": lr_mae - model_mae,
        })
    return result
