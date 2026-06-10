from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def _gaussian_kernel(sigma: torch.Tensor, size: int = 9) -> torch.Tensor:
    coordinates = torch.arange(size, device=sigma.device, dtype=sigma.dtype) - size // 2
    grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    radius = grid_x.square() + grid_y.square()
    kernel = torch.exp(-radius[None] / (2 * sigma[:, None, None].square().clamp_min(1e-6)))
    return kernel / kernel.sum(dim=(-2, -1), keepdim=True)


def sensor_degrade(
    hr: torch.Tensor,
    parameters: torch.Tensor,
    scale: int = 4,
    add_noise: bool = False,
) -> torch.Tensor:
    """Differentiable MTF blur and area downsample.

    parameters columns are normalized sigma, Gaussian noise, Poisson strength,
    and quantization/compression strength.
    """

    batch, channels, _, _ = hr.shape
    sigma = 0.35 + parameters[:, 0].clamp(0, 1) * 1.65
    kernels = _gaussian_kernel(sigma)
    weight = kernels[:, None].repeat_interleave(channels, dim=0)
    flattened = hr.reshape(1, batch * channels, *hr.shape[-2:])
    flattened = F.pad(
        flattened,
        (kernels.shape[-1] // 2,) * 4,
        mode="reflect",
    )
    blurred = F.conv2d(
        flattened,
        weight,
        groups=batch * channels,
    ).reshape_as(hr)
    lr = F.interpolate(blurred, scale_factor=1 / scale, mode="area")
    if add_noise:
        gaussian_std = parameters[:, 1, None, None, None] * 0.03
        lr = lr + torch.randn_like(lr) * gaussian_std
        poisson_strength = parameters[:, 2, None, None, None] * 80 + 20
        lr = torch.poisson((lr.clamp_min(0) * poisson_strength)) / poisson_strength
        levels = (256 - parameters[:, 3, None, None, None] * 192).clamp_min(16)
        lr = torch.round(lr * (levels - 1)) / (levels - 1)
    return lr.clamp(0, 1)


def random_degradation(
    hr: torch.Tensor,
    scale: int = 4,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    parameters = torch.rand(
        hr.shape[0],
        4,
        device=hr.device,
        dtype=hr.dtype,
        generator=generator,
    )
    return sensor_degrade(hr, parameters, scale=scale, add_noise=True), parameters


def back_project(
    estimate: torch.Tensor,
    observed_lr: torch.Tensor,
    parameters: torch.Tensor,
    scale: int = 4,
    iterations: int = 3,
    step_size: float = 0.5,
) -> torch.Tensor:
    result = estimate
    for _ in range(iterations):
        error = observed_lr - sensor_degrade(result, parameters, scale=scale)
        correction = F.interpolate(error, size=result.shape[-2:], mode="bicubic", align_corners=False)
        result = (result + step_size * correction).clamp(0, 1)
    return result


def default_degradation(batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor([0.4, 0.0, 0.0, 0.0], device=device, dtype=dtype).repeat(batch, 1)
