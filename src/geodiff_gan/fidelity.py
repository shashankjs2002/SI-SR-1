from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .losses import oracle_residual_trust, ssim


def _expanded_mask(value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(value)
    resized = F.interpolate(mask.float(), size=value.shape[-2:], mode="nearest")
    if resized.shape[1] == 1 and value.shape[1] != 1:
        resized = resized.expand(-1, value.shape[1], -1, -1)
    return resized.to(device=value.device, dtype=value.dtype)


def per_image_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return one masked PSNR value per image for paired statistical tests."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    weights = _expanded_mask(prediction, mask)
    denominator = weights.flatten(1).sum(dim=1).clamp_min(1.0)
    mse = ((prediction - target).square() * weights).flatten(1).sum(dim=1)
    mse = mse / denominator
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def per_image_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return masked SSIM per image without coupling samples in the batch."""

    values = []
    for index in range(prediction.shape[0]):
        sample_mask = mask[index : index + 1] if mask is not None else None
        values.append(
            ssim(
                prediction[index : index + 1],
                target[index : index + 1],
                mask=sample_mask,
            )
        )
    return torch.stack(values)


@torch.no_grad()
def per_image_affine_oracle(
    base: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    ridge: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit validation-only per-image, per-band gain and bias coefficients."""

    if base.shape != target.shape:
        raise ValueError("base and target must have the same shape")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    weights = _expanded_mask(base, mask)
    count = weights.sum(dim=(-2, -1)).clamp_min(1.0)
    base_mean = (base * weights).sum(dim=(-2, -1)) / count
    target_mean = (target * weights).sum(dim=(-2, -1)) / count
    centered_base = base - base_mean[..., None, None]
    centered_target = target - target_mean[..., None, None]
    covariance = (
        centered_base * centered_target * weights
    ).sum(dim=(-2, -1))
    variance = (centered_base.square() * weights).sum(dim=(-2, -1))
    gain = covariance / (variance + float(ridge)).clamp_min(1e-12)
    bias = target_mean - gain * base_mean
    corrected = (
        base * gain[..., None, None] + bias[..., None, None]
    ).clamp(0, 1)
    return corrected, gain, bias


@torch.no_grad()
def local_low_frequency_oracle(
    base: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    window: int = 31,
) -> torch.Tensor:
    """Apply a target-derived local radiometric correction for ceiling analysis."""

    if base.shape != target.shape:
        raise ValueError("base and target must have the same shape")
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    weights = _expanded_mask(base, mask)
    padding = window // 2
    weighted_difference = (target - base) * weights
    numerator = F.avg_pool2d(
        F.pad(weighted_difference, (padding,) * 4, mode="reflect"),
        window,
        stride=1,
    )
    denominator = F.avg_pool2d(
        F.pad(weights, (padding,) * 4, mode="reflect"),
        window,
        stride=1,
    )
    correction = numerator / denominator.clamp_min(1e-8)
    return (base + correction).clamp(0, 1)


@torch.no_grad()
def global_residual_oracle(
    base: torch.Tensor,
    candidate_residual: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    maximum_scale: float = 1.0,
    ridge: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit one residual acceptance coefficient per validation image."""

    weights = _expanded_mask(base, mask)
    missing = target - base
    numerator = (candidate_residual * missing * weights).flatten(1).sum(dim=1)
    denominator = (
        candidate_residual.square() * weights
    ).flatten(1).sum(dim=1)
    coefficient = numerator / (denominator + float(ridge)).clamp_min(1e-12)
    coefficient = coefficient.clamp(0, float(maximum_scale))[:, None, None, None]
    return (base + coefficient * candidate_residual).clamp(0, 1), coefficient


@torch.no_grad()
def residual_oracle_variants(
    base: torch.Tensor,
    candidate_residual: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    maximum_scale: float = 1.0,
    smoothing_window: int = 9,
    ridge: float = 1e-8,
) -> dict[str, dict[str, torch.Tensor]]:
    """Return global, local-scalar, and local-per-band oracle residual outputs."""

    global_output, global_trust = global_residual_oracle(
        base,
        candidate_residual,
        target,
        mask=mask,
        maximum_scale=maximum_scale,
        ridge=ridge,
    )
    scalar_trust = oracle_residual_trust(
        candidate_residual,
        base,
        target,
        maximum_scale=maximum_scale,
        smoothing_window=smoothing_window,
        per_band=False,
        ridge=ridge,
        mask=mask,
    )
    per_band_trust = oracle_residual_trust(
        candidate_residual,
        base,
        target,
        maximum_scale=maximum_scale,
        smoothing_window=smoothing_window,
        per_band=True,
        ridge=ridge,
        mask=mask,
    )
    return {
        "global_scalar": {
            "image": global_output,
            "trust": global_trust,
        },
        "local_scalar": {
            "image": (base + scalar_trust * candidate_residual).clamp(0, 1),
            "trust": scalar_trust,
        },
        "local_per_band": {
            "image": (base + per_band_trust * candidate_residual).clamp(0, 1),
            "trust": per_band_trust,
        },
    }


def paired_bootstrap_interval(
    differences: Sequence[float] | np.ndarray,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap a confidence interval over paired per-image differences."""

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("differences must be a non-empty one-dimensional sequence")
    if samples < 1:
        raise ValueError("samples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, values.size, size=(samples, values.size))
    bootstrap_means = values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap_means, [tail, 1.0 - tail])
    return float(lower), float(upper)


def fidelity_acceptance_report(
    previous_base_psnr: float,
    new_base_psnr: float,
    final_psnr: float,
    base_ssim: float,
    final_ssim: float,
    per_image_psnr_delta: Sequence[float] | np.ndarray,
    base_gain_required: float = 0.75,
    residual_gain_required: float = 0.05,
    improvement_fraction_required: float = 0.55,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate all pre-test FidelityTrust V2 acceptance criteria."""

    differences = np.asarray(per_image_psnr_delta, dtype=np.float64)
    if differences.ndim != 1 or differences.size < 1:
        raise ValueError("per_image_psnr_delta must be non-empty")
    interval = paired_bootstrap_interval(
        differences,
        samples=bootstrap_samples,
        seed=seed,
    )
    improvement_fraction = float(np.mean(differences > 0))
    checks = {
        "base_gain": new_base_psnr - previous_base_psnr >= base_gain_required,
        "residual_gain": final_psnr - new_base_psnr >= residual_gain_required,
        "ssim_non_degradation": final_ssim >= base_ssim,
        "improvement_fraction": (
            improvement_fraction >= improvement_fraction_required
        ),
        "bootstrap_lower_non_negative": interval[0] >= 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "previous_base_psnr": float(previous_base_psnr),
        "new_base_psnr": float(new_base_psnr),
        "base_gain_db": float(new_base_psnr - previous_base_psnr),
        "final_psnr": float(final_psnr),
        "residual_gain_db": float(final_psnr - new_base_psnr),
        "base_ssim": float(base_ssim),
        "final_ssim": float(final_ssim),
        "improvement_fraction": improvement_fraction,
        "bootstrap_psnr_delta_ci": interval,
    }


@torch.no_grad()
def radiometric_ceiling_report(
    base: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    local_window: int = 31,
    spatial_head_threshold_db: float = 0.20,
) -> Mapping[str, Any]:
    """Measure validation-only global and local radiometric oracle ceilings."""

    affine, gain, bias = per_image_affine_oracle(base, target, mask=mask)
    local = local_low_frequency_oracle(
        affine,
        target,
        mask=mask,
        window=local_window,
    )
    base_psnr = per_image_psnr(base, target, mask)
    affine_psnr = per_image_psnr(affine, target, mask)
    local_psnr = per_image_psnr(local, target, mask)
    spatial_gain = float((local_psnr - affine_psnr).mean())
    return {
        "base_psnr": base_psnr,
        "affine_psnr": affine_psnr,
        "local_low_frequency_psnr": local_psnr,
        "affine_gain": gain,
        "affine_bias": bias,
        "mean_affine_gain_db": float((affine_psnr - base_psnr).mean()),
        "mean_spatial_gain_beyond_affine_db": spatial_gain,
        "enable_spatial_radiometric_head": (
            spatial_gain >= float(spatial_head_threshold_db)
        ),
    }
