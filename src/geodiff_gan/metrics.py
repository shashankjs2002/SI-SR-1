from __future__ import annotations

import torch
from torch.nn import functional as F

from .losses import charbonnier, ssim
from .models.degradation import sensor_degrade


def _masked_mean(value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return value.mean()
    mask = F.interpolate(mask.float(), size=value.shape[-2:], mode="nearest")
    if mask.shape[1] == 1 and value.shape[1] != 1:
        mask = mask.expand(-1, value.shape[1], -1, -1)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    mse = _masked_mean((prediction - target).square(), mask)
    return -10 * torch.log10(mse.clamp_min(1e-12))


def _expanded_mask(
    value: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(value)
    resized = F.interpolate(mask.float(), size=value.shape[-2:], mode="nearest")
    if resized.shape[1] == 1 and value.shape[1] != 1:
        resized = resized.expand(-1, value.shape[1], -1, -1)
    return resized.to(dtype=value.dtype)


def _masked_channel_moments(
    first: torch.Tensor,
    second: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    weights = _expanded_mask(first, mask)
    count = weights.sum(dim=(-2, -1)).clamp_min(1.0)
    first_mean = (first * weights).sum(dim=(-2, -1)) / count
    second_mean = (second * weights).sum(dim=(-2, -1)) / count
    first_centered = first - first_mean[..., None, None]
    second_centered = second - second_mean[..., None, None]
    first_variance = (first_centered.square() * weights).sum(dim=(-2, -1)) / count
    second_variance = (second_centered.square() * weights).sum(dim=(-2, -1)) / count
    covariance = (
        first_centered * second_centered * weights
    ).sum(dim=(-2, -1)) / count
    return (
        first_mean,
        second_mean,
        first_variance,
        second_variance,
        covariance,
    )


def ergas(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Relative global synthesis error; lower is better."""
    if scale < 1:
        raise ValueError("ERGAS scale must be at least 1")
    weights = _expanded_mask(prediction, mask)
    count = weights.sum(dim=(-2, -1)).clamp_min(1.0)
    rmse = torch.sqrt(
        ((prediction - target).square() * weights).sum(dim=(-2, -1)) / count
    )
    reference_mean = (target * weights).sum(dim=(-2, -1)) / count
    relative_error = rmse / reference_mean.abs().clamp_min(1e-8)
    return (100.0 / float(scale)) * torch.sqrt(
        relative_error.square().mean(dim=1)
    ).mean()


def spectral_angle_mapper(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean per-pixel spectral angle in degrees; lower is better."""
    dot = (prediction * target).sum(dim=1, keepdim=True)
    prediction_norm = torch.linalg.vector_norm(prediction, dim=1, keepdim=True)
    target_norm = torch.linalg.vector_norm(target, dim=1, keepdim=True)
    denominator = prediction_norm * target_norm
    cosine = dot / denominator.clamp_min(1e-12)
    identical = (prediction - target).abs().amax(dim=1, keepdim=True) < 1e-8
    both_zero = (prediction_norm < 1e-8) & (target_norm < 1e-8)
    one_zero = (prediction_norm < 1e-8) ^ (target_norm < 1e-8)
    cosine = torch.where(both_zero | identical, torch.ones_like(cosine), cosine)
    cosine = torch.where(one_zero, torch.zeros_like(cosine), cosine)
    angle = torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))
    return _masked_mean(angle, mask)


def uiqi(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Band-averaged universal image quality index; higher is better."""
    (
        prediction_mean,
        target_mean,
        prediction_variance,
        target_variance,
        covariance,
    ) = _masked_channel_moments(prediction, target, mask)
    numerator = 4.0 * covariance * prediction_mean * target_mean
    denominator = (
        (prediction_variance + target_variance)
        * (prediction_mean.square() + target_mean.square())
    )
    score = numerator / denominator.clamp_min(1e-12)
    weights = _expanded_mask(prediction, mask)
    identical_degenerate = (
        denominator < 1e-12
    ) & (((prediction - target).abs() * weights).amax(dim=(-2, -1)) < 1e-8)
    score = torch.where(identical_degenerate, torch.ones_like(score), score)
    score = torch.where(
        (denominator < 1e-12) & ~identical_degenerate,
        torch.zeros_like(score),
        score,
    )
    return score.clamp(-1.0, 1.0).mean()


def spatial_correlation_coefficient(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Correlation of 3x3 Laplacian high-pass bands; higher is better."""
    channels = prediction.shape[1]
    kernel = prediction.new_tensor(
        [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]
    ).view(1, 1, 3, 3).expand(channels, 1, 3, 3)

    def high_pass(value: torch.Tensor) -> torch.Tensor:
        padded = F.pad(value, (1, 1, 1, 1), mode="reflect")
        return F.conv2d(padded, kernel, groups=channels)

    prediction_high = high_pass(prediction)
    target_high = high_pass(target)
    correlation_mask = mask
    if mask is not None:
        resized = F.interpolate(mask.float(), size=prediction.shape[-2:], mode="nearest")
        correlation_mask = (
            F.avg_pool2d(resized, kernel_size=3, stride=1, padding=1) >= 1.0 - 1e-6
        ).float()
    (
        _,
        _,
        prediction_variance,
        target_variance,
        covariance,
    ) = _masked_channel_moments(
        prediction_high,
        target_high,
        correlation_mask,
    )
    denominator = torch.sqrt(prediction_variance * target_variance)
    score = covariance / denominator.clamp_min(1e-12)
    weights = _expanded_mask(prediction_high, correlation_mask)
    identical_degenerate = (
        denominator < 1e-12
    ) & (
        ((prediction_high - target_high).abs() * weights).amax(dim=(-2, -1))
        < 1e-8
    )
    score = torch.where(identical_degenerate, torch.ones_like(score), score)
    score = torch.where(
        (denominator < 1e-12) & ~identical_degenerate,
        torch.zeros_like(score),
        score,
    )
    return score.clamp(-1.0, 1.0).mean()


def remote_sensing_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale: int,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Dependency-free, full-reference metrics for aligned remote-sensing data."""
    prediction = prediction.float()
    target = target.float()
    return {
        "ergas": float(ergas(prediction, target, scale=scale, mask=mask)),
        "sam_degrees": float(spectral_angle_mapper(prediction, target, mask=mask)),
        "uiqi": float(uiqi(prediction, target, mask=mask)),
        "scc": float(spatial_correlation_coefficient(prediction, target, mask=mask)),
    }


class RemoteSensingMetricSuite:
    """Compatibility wrapper for evaluation paths that use a metric suite."""

    def __init__(
        self,
        device: torch.device | None = None,
        enabled: bool = True,
        scale: int = 4,
    ) -> None:
        del device
        self.enabled = enabled
        self.scale = scale
        self.load_errors: dict[str, str] = {}

    @property
    def available_metrics(self) -> tuple[str, ...]:
        return (
            ("ergas", "sam_degrees", "uiqi", "scc")
            if self.enabled
            else ()
        )

    @torch.no_grad()
    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        if not self.enabled:
            return {}
        return remote_sensing_metrics(
            prediction,
            target,
            scale=self.scale,
            mask=mask,
        )


# Retain the old import name for previously generated notebooks. It no longer
# loads LPIPS or DISTS and now returns dependency-free remote-sensing metrics.
OptionalMetricSuite = RemoteSensingMetricSuite


def edge_f1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float | None = None,
    quantile: float = 0.9,
    tolerance: int = 1,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    def gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        dx = F.pad(gray[:, :, :, 1:] - gray[:, :, :, :-1], (0, 1, 0, 0))
        dy = F.pad(gray[:, :, 1:, :] - gray[:, :, :-1, :], (0, 0, 0, 1))
        return torch.sqrt(dx.square() + dy.square() + 1e-12)

    predicted_gradient = gradient_magnitude(prediction)
    target_gradient = gradient_magnitude(target)
    valid = (
        F.interpolate(mask.float(), size=target_gradient.shape[-2:], mode="nearest")
        if mask is not None
        else torch.ones_like(target_gradient)
    ).bool()
    if threshold is None:
        thresholds = []
        for sample_gradient, sample_valid in zip(target_gradient, valid):
            values = sample_gradient[sample_valid]
            thresholds.append(
                torch.quantile(values, quantile)
                if values.numel()
                else target_gradient.new_tensor(0.005)
            )
        threshold_tensor = torch.stack(thresholds).view(-1, 1, 1, 1)
        threshold_tensor = threshold_tensor.clamp_min(0.005)
    else:
        threshold_tensor = target_gradient.new_full(
            (target_gradient.shape[0], 1, 1, 1),
            threshold,
        )
    predicted = (predicted_gradient > threshold_tensor) & valid
    actual = (target_gradient > threshold_tensor) & valid
    kernel_size = tolerance * 2 + 1
    predicted_near = F.max_pool2d(
        predicted.float(),
        kernel_size,
        stride=1,
        padding=tolerance,
    ).bool()
    actual_near = F.max_pool2d(
        actual.float(),
        kernel_size,
        stride=1,
        padding=tolerance,
    ).bool()
    precision = (predicted & actual_near).sum().float() / predicted.sum().clamp_min(1)
    recall = (actual & predicted_near).sum().float() / actual.sum().clamp_min(1)
    return 2 * precision * recall / (precision + recall).clamp_min(1e-8)


def redegradation_error(
    prediction: torch.Tensor,
    lr: torch.Tensor,
    degradation: torch.Tensor,
    scale: int = 4,
    severity: str = "mild",
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return _masked_mean(
        (
            sensor_degrade(
                prediction,
                degradation,
                scale=scale,
                severity=severity,
            )
            - lr
        ).abs(),
        mask,
    )


def basic_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lr: torch.Tensor,
    degradation: torch.Tensor,
    scale: int = 4,
    severity: str = "mild",
    mask: torch.Tensor | None = None,
    lr_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    values = {
        "l1": float(charbonnier(prediction, target, epsilon=0.0, mask=mask)),
        "psnr": float(psnr(prediction, target, mask=mask)),
        "ssim": float(ssim(prediction, target, mask=mask)),
        "edge_f1": float(edge_f1(prediction, target, mask=mask)),
        "redegradation_l1": float(
            redegradation_error(
                prediction,
                lr,
                degradation,
                scale,
                severity=severity,
                mask=lr_mask,
            )
        ),
    }
    values.update(remote_sensing_metrics(prediction, target, scale=scale, mask=mask))
    return values
