from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .models.blocks import haar_wavelet
from .models.degradation import sensor_degrade


def charbonnier(prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((prediction - target).square() + epsilon**2).mean()


def _ssim_statistics(x: torch.Tensor, window: int = 11) -> tuple[torch.Tensor, ...]:
    padding = window // 2
    mean = F.avg_pool2d(x, window, stride=1, padding=padding)
    mean_square = F.avg_pool2d(x.square(), window, stride=1, padding=padding)
    variance = (mean_square - mean.square()).clamp_min(0)
    return mean, variance


def ssim(prediction: torch.Tensor, target: torch.Tensor, window: int = 11) -> torch.Tensor:
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
    return (numerator / denominator.clamp_min(1e-8)).mean()


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    pred_dy = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def wavelet_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_bands = haar_wavelet(prediction)[1:]
    target_bands = haar_wavelet(target)[1:]
    return sum(F.l1_loss(pred, real) for pred, real in zip(pred_bands, target_bands)) / 3


def kl_loss(mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
    return -0.5 * (1 + log_variance - mean.square() - log_variance.exp()).mean()


def degradation_consistency(
    prediction: torch.Tensor,
    lr: torch.Tensor,
    parameters: torch.Tensor,
    scale: int = 4,
    severity: str = "mild",
) -> torch.Tensor:
    return charbonnier(
        sensor_degrade(
            prediction,
            parameters,
            scale=scale,
            severity=severity,
        ),
        lr,
    )


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
