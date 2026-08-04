from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from geodiff_gan.models.degradation import DEGRADATION_PRESETS

from .outputs import NoiseCalibrationOutput


def _range_value(
    unit_value: torch.Tensor,
    bounds: tuple[float, float],
    *,
    reverse: bool = False,
) -> torch.Tensor:
    low, high = bounds
    fraction = 1.0 - unit_value if reverse else unit_value
    return low + fraction.clamp(0, 1) * (high - low)


def degradation_noise_variance(
    mean_lr: torch.Tensor,
    degradation: torch.Tensor,
    *,
    severity: str = "mild",
    variance_floor: float = 1e-7,
) -> torch.Tensor:
    """Approximate simulator variance from Gaussian, Poisson, and quantization terms."""

    if severity not in DEGRADATION_PRESETS:
        raise ValueError(
            f"unknown degradation severity {severity!r}; "
            f"expected one of {sorted(DEGRADATION_PRESETS)}"
        )
    if degradation.ndim != 2 or degradation.shape[1] < 4:
        raise ValueError("degradation must have shape [batch, at least 4]")
    preset = DEGRADATION_PRESETS[severity]
    gaussian_std = _range_value(
        degradation[:, 1],
        preset["gaussian_std"],
    )[:, None, None, None]
    poisson_peak = _range_value(
        degradation[:, 2],
        preset["poisson_peak"],
    )[:, None, None, None]
    levels = _range_value(
        degradation[:, 3],
        preset["quantization_levels"],
        reverse=True,
    )[:, None, None, None]
    gaussian_variance = gaussian_std.square()
    poisson_variance = mean_lr.clamp_min(0) / poisson_peak.clamp_min(1.0)
    quantization_variance = 1.0 / (
        12.0 * (levels - 1.0).square().clamp_min(1.0)
    )
    return (
        gaussian_variance + poisson_variance + quantization_variance
    ).expand_as(mean_lr).clamp_min(variance_floor)


def gaussian_nll(
    residual: torch.Tensor,
    variance: torch.Tensor,
    *,
    variance_floor: float = 1e-7,
    reduction: str = "mean",
) -> torch.Tensor:
    variance = variance.clamp_min(variance_floor)
    values = 0.5 * (residual.square() / variance + variance.log())
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean()
    raise ValueError("reduction must be one of 'none', 'sum', or 'mean'")


def calibration_training_loss(
    output: NoiseCalibrationOutput,
    observed_lr: torch.Tensor,
    clean_lr: torch.Tensor,
    *,
    mean_weight: float = 0.1,
    variance_floor: float = 1e-7,
) -> dict[str, torch.Tensor]:
    realized_noise = observed_lr - clean_lr
    nll = gaussian_nll(
        realized_noise,
        output.variance,
        variance_floor=variance_floor,
    )
    mean_l1 = F.l1_loss(output.mean_lr, clean_lr)
    return {
        "nll": nll,
        "mean_l1": mean_l1,
        "total": nll + mean_weight * mean_l1,
    }


class _ResidualCalibrationBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + self.body(tensor)


class SensorNoiseCalibrator(nn.Module):
    """Low-capacity band-aware Poisson-Gaussian covariance estimator."""

    def __init__(
        self,
        channels: int = 3,
        degradation_dim: int = 4,
        hidden_channels: int = 24,
        *,
        alpha_bounds: tuple[float, float] = (1e-6, 5e-3),
        beta_bounds: tuple[float, float] = (1e-7, 5e-4),
        variance_floor: float = 1e-7,
        variance_ceiling: float = 2e-2,
    ) -> None:
        super().__init__()
        if channels < 1 or hidden_channels < 1:
            raise ValueError("channels and hidden_channels must be positive")
        if alpha_bounds[0] <= 0 or beta_bounds[0] <= 0:
            raise ValueError("alpha and beta lower bounds must be positive")
        self.channels = int(channels)
        self.alpha_bounds = alpha_bounds
        self.beta_bounds = beta_bounds
        self.variance_floor = float(variance_floor)
        self.variance_ceiling = float(variance_ceiling)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            _ResidualCalibrationBlock(hidden_channels),
            _ResidualCalibrationBlock(hidden_channels),
        )
        self.metadata_encoder = nn.Sequential(
            nn.Linear(degradation_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
        )
        self.parameter_head = nn.Linear(hidden_channels * 2, channels * 2)
        self.mean_residual = nn.Conv2d(hidden_channels, channels, 3, padding=1)
        nn.init.zeros_(self.mean_residual.weight)
        nn.init.zeros_(self.mean_residual.bias)

    @staticmethod
    def _bounded(
        raw: torch.Tensor,
        bounds: tuple[float, float],
    ) -> torch.Tensor:
        low, high = bounds
        return low + torch.sigmoid(raw) * (high - low)

    def forward(
        self,
        observed_lr: torch.Tensor,
        degradation: torch.Tensor,
    ) -> NoiseCalibrationOutput:
        if observed_lr.ndim != 4 or observed_lr.shape[1] != self.channels:
            raise ValueError(
                f"observed_lr must have shape [batch, {self.channels}, height, width]"
            )
        if degradation.ndim != 2 or degradation.shape[0] != observed_lr.shape[0]:
            raise ValueError("degradation must have shape [batch, parameters]")
        image_features = self.image_encoder(observed_lr)
        pooled = image_features.mean(dim=(-2, -1))
        metadata = self.metadata_encoder(degradation)
        raw = self.parameter_head(torch.cat([pooled, metadata], dim=1))
        raw_alpha, raw_beta = raw.chunk(2, dim=1)
        alpha = self._bounded(raw_alpha, self.alpha_bounds)
        beta = self._bounded(raw_beta, self.beta_bounds)

        smooth = F.avg_pool2d(
            F.pad(observed_lr, (1, 1, 1, 1), mode="reflect"),
            kernel_size=3,
            stride=1,
        )
        mean_lr = (
            smooth + 0.05 * torch.tanh(self.mean_residual(image_features))
        ).clamp(0, 1)
        variance = (
            alpha[:, :, None, None] * mean_lr
            + beta[:, :, None, None]
        ).clamp(self.variance_floor, self.variance_ceiling)
        return NoiseCalibrationOutput(
            variance=variance,
            alpha=alpha,
            beta=beta,
            mean_lr=mean_lr,
            raw_alpha_beta=raw,
        )
