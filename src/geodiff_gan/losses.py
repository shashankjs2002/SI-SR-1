from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .models.blocks import haar_wavelet, high_pass
from .models.degradation import sensor_degrade


def _masked_mean(value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return value.mean()
    mask = F.interpolate(mask.float(), size=value.shape[-2:], mode="nearest")
    if mask.shape[1] == 1 and value.shape[1] != 1:
        mask = mask.expand(-1, value.shape[1], -1, -1)
    denominator = mask.sum().clamp_min(1.0)
    return (value * mask).sum() / denominator


def charbonnier(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-3,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    value = torch.sqrt((prediction - target).square() + epsilon**2)
    return _masked_mean(value, mask)


def mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked MSE, the distortion objective directly optimized by PSNR."""
    return _masked_mean((prediction - target).square(), mask)


def multiscale_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    scales: tuple[int, ...] = (1, 2, 4),
) -> torch.Tensor:
    """Match radiometry and structure at native and progressively coarser scales."""
    losses = []
    for scale in scales:
        if scale < 1:
            raise ValueError("multiscale MSE scales must be positive")
        if scale == 1:
            scaled_prediction = prediction
            scaled_target = target
            scaled_mask = mask
        else:
            scaled_prediction = F.avg_pool2d(prediction, scale, stride=scale)
            scaled_target = F.avg_pool2d(target, scale, stride=scale)
            scaled_mask = (
                (F.avg_pool2d(mask.float(), scale, stride=scale) >= 0.999).float()
                if mask is not None
                else None
            )
        losses.append(mse_loss(scaled_prediction, scaled_target, scaled_mask))
    return torch.stack(losses).mean()


def radiometric_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match per-band mean and standard deviation on valid pixels."""
    if mask is None:
        mask = torch.ones_like(prediction[:, :1])
    mask = F.interpolate(mask.float(), size=prediction.shape[-2:], mode="nearest")
    if mask.shape[1] == 1:
        mask = mask.expand(-1, prediction.shape[1], -1, -1)
    denominator = mask.sum(dim=(-2, -1)).clamp_min(1.0)

    def statistics(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = (value * mask).sum(dim=(-2, -1)) / denominator
        variance = (
            (value - mean[..., None, None]).square() * mask
        ).sum(dim=(-2, -1)) / denominator
        return mean, variance.clamp_min(1e-12).sqrt()

    prediction_mean, prediction_std = statistics(prediction)
    target_mean, target_std = statistics(target)
    return F.l1_loss(prediction_mean, target_mean) + F.l1_loss(
        prediction_std, target_std
    )


def base_guard_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    base: torch.Tensor,
    margin: float = 0.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalize samples whose reconstruction MSE is worse than the frozen base."""
    error = (prediction - target).square().mean(dim=1, keepdim=True)
    with torch.no_grad():
        base_error = (base - target).square().mean(dim=1, keepdim=True)
    if mask is None:
        prediction_mse = error.flatten(1).mean(dim=1)
        base_mse = base_error.flatten(1).mean(dim=1)
    else:
        mask = F.interpolate(mask.float(), size=error.shape[-2:], mode="nearest")
        denominator = mask.flatten(1).sum(dim=1).clamp_min(1.0)
        prediction_mse = (error * mask).flatten(1).sum(dim=1) / denominator
        base_mse = (base_error * mask).flatten(1).sum(dim=1) / denominator
    return F.relu(prediction_mse - base_mse + float(margin)).mean()


def residual_supervision_loss(
    residual: torch.Tensor,
    base: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Supervise the SR branch against the high-frequency target residual."""
    target_residual = high_pass(target - base)
    return charbonnier(residual, target_residual, mask=mask)


def _ssim_statistics(x: torch.Tensor, window: int = 11) -> tuple[torch.Tensor, ...]:
    padding = window // 2
    mean = F.avg_pool2d(x, window, stride=1, padding=padding)
    mean_square = F.avg_pool2d(x.square(), window, stride=1, padding=padding)
    variance = (mean_square - mean.square()).clamp_min(0)
    return mean, variance


def ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window: int = 11,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    mean_x, variance_x = _ssim_statistics(prediction, window)
    mean_y, variance_y = _ssim_statistics(target, window)
    padding = window // 2
    covariance = (
        F.avg_pool2d(prediction * target, window, stride=1, padding=padding)
        - mean_x * mean_y
    )
    c1, c2 = 0.01**2, 0.03**2
    numerator = (2 * mean_x * mean_y + c1) * (2 * covariance + c2)
    denominator = (mean_x.square() + mean_y.square() + c1) * (
        variance_x + variance_y + c2
    )
    similarity = numerator / denominator.clamp_min(1e-8)
    if mask is not None:
        valid_window = F.avg_pool2d(
            mask.float(),
            window,
            stride=1,
            padding=window // 2,
        )
        valid_window = (valid_window >= 0.999).float()
        if not bool(valid_window.any()):
            valid_window = mask.float()
        return _masked_mean(similarity, valid_window)
    return similarity.mean()


def gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    pred_dx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    pred_dy = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    if mask is None:
        return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)
    mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    return _masked_mean((pred_dx - target_dx).abs(), mask_x) + _masked_mean(
        (pred_dy - target_dy).abs(), mask_y
    )


def wavelet_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    pred_bands = haar_wavelet(prediction)[1:]
    target_bands = haar_wavelet(target)[1:]
    wavelet_mask = None
    if mask is not None:
        wavelet_mask = (F.avg_pool2d(mask.float(), 2, stride=2) >= 0.999).float()
    return sum(
        _masked_mean((pred - real).abs(), wavelet_mask)
        for pred, real in zip(pred_bands, target_bands)
    ) / 3


def kl_loss(mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
    return -0.5 * (1 + log_variance - mean.square() - log_variance.exp()).mean()


def degradation_consistency(
    prediction: torch.Tensor,
    lr: torch.Tensor,
    parameters: torch.Tensor,
    scale: int = 4,
    severity: str = "mild",
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return charbonnier(
        sensor_degrade(
            prediction,
            parameters,
            scale=scale,
            severity=severity,
        ),
        lr,
        mask=mask,
    )


def evidence_calibration_loss(
    confidence: torch.Tensor,
    ungated_prediction: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.05,
    smoothing_window: int = 9,
    selectivity_weight: float = 0.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calibrate confidence magnitude and spatial ranking against local accuracy."""
    with torch.no_grad():
        if mask is not None:
            error_prediction = ungated_prediction * mask + target * (1 - mask)
        else:
            error_prediction = ungated_prediction
        error = (error_prediction - target).abs().mean(dim=1, keepdim=True)
        error = F.avg_pool2d(
            error,
            smoothing_window,
            stride=1,
            padding=smoothing_window // 2,
        )
        target_confidence = torch.exp(-error / max(temperature, 1e-8))
        target_confidence = F.interpolate(
            target_confidence,
            size=confidence.shape[-2:],
            mode="area",
        )
    confidence_mask = (
        F.interpolate(mask.float(), size=confidence.shape[-2:], mode="nearest")
        if mask is not None
        else None
    )
    calibration = _masked_mean(
        F.smooth_l1_loss(confidence, target_confidence, reduction="none"),
        confidence_mask,
    )
    if selectivity_weight <= 0:
        return calibration

    if confidence_mask is None:
        confidence_mask = torch.ones_like(confidence)
    flat_mask = confidence_mask.flatten(1)
    normalizer = flat_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    confidence_centered = confidence.flatten(1)
    confidence_centered = (
        confidence_centered
        - (confidence_centered * flat_mask).sum(dim=1, keepdim=True) / normalizer
    ) * flat_mask
    target_centered = target_confidence.flatten(1)
    target_centered = (
        target_centered
        - (target_centered * flat_mask).sum(dim=1, keepdim=True) / normalizer
    ) * flat_mask
    target_norm = target_centered.norm(dim=1)
    valid = target_norm > 1e-6
    if not valid.any():
        return calibration
    correlation = F.cosine_similarity(
        confidence_centered[valid],
        target_centered[valid],
        dim=1,
        eps=1e-6,
    )
    selectivity = (1 - correlation).mean()
    return calibration + float(selectivity_weight) * selectivity


def evidence_improvement_loss(
    confidence: torch.Tensor,
    ungated_prediction: torch.Tensor,
    base: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.01,
    smoothing_window: int = 9,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Teach the gate to trust detail only where it improves on the base."""
    with torch.no_grad():
        candidate_error = (ungated_prediction - target).square().mean(
            dim=1, keepdim=True
        )
        base_error = (base - target).square().mean(dim=1, keepdim=True)
        improvement = base_error - candidate_error
        improvement = F.avg_pool2d(
            improvement,
            smoothing_window,
            stride=1,
            padding=smoothing_window // 2,
        )
        target_confidence = torch.sigmoid(
            improvement / max(float(temperature), 1e-8)
        )
        target_confidence = F.interpolate(
            target_confidence,
            size=confidence.shape[-2:],
            mode="area",
        )
    confidence_mask = (
        F.interpolate(mask.float(), size=confidence.shape[-2:], mode="nearest")
        if mask is not None
        else None
    )
    return _masked_mean(
        F.binary_cross_entropy(
            confidence.clamp(1e-6, 1 - 1e-6),
            target_confidence,
            reduction="none",
        ),
        confidence_mask,
    )


def edit_localization_loss(
    raw_edit_residual: torch.Tensor,
    edit_permission: torch.Tensor,
) -> torch.Tensor:
    permission = F.interpolate(
        edit_permission,
        size=raw_edit_residual.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    return ((1 - permission) * raw_edit_residual.abs()).mean()


def snr_weighted_velocity_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    timesteps: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    gamma: float = 5.0,
) -> torch.Tensor:
    alpha = alphas_cumprod.gather(0, timesteps)
    snr = alpha / (1 - alpha).clamp_min(1e-8)
    weights = torch.minimum(snr, torch.full_like(snr, gamma)) / (snr + 1)
    per_sample = (prediction - target).square().flatten(1).mean(dim=1)
    return (weights * per_sample).mean()


def discriminator_hinge(
    real_outputs: torch.Tensor | Iterable[torch.Tensor],
    fake_outputs: torch.Tensor | Iterable[torch.Tensor],
) -> torch.Tensor:
    real = [real_outputs] if isinstance(real_outputs, torch.Tensor) else list(real_outputs)
    fake = [fake_outputs] if isinstance(fake_outputs, torch.Tensor) else list(fake_outputs)
    return sum(
        F.relu(1 - real_value).mean() + F.relu(1 + fake_value).mean()
        for real_value, fake_value in zip(real, fake)
    ) / len(real)


def generator_hinge(fake_outputs: torch.Tensor | Iterable[torch.Tensor]) -> torch.Tensor:
    fake = [fake_outputs] if isinstance(fake_outputs, torch.Tensor) else list(fake_outputs)
    return -sum(value.mean() for value in fake) / len(fake)


class OptionalPerceptualLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        try:
            import lpips

            self.model: nn.Module | None = lpips.LPIPS(net="alex")
            self.available = True
        except ImportError:
            self.model = None
            self.available = False

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            return prediction.new_zeros(())
        return self.model(prediction * 2 - 1, target * 2 - 1).mean()
