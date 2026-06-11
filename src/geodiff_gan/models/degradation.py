from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch.nn import functional as F


DEGRADATION_PRESETS = {
    "mild": {
        "sigma": (0.50, 1.00),
        "gaussian_std": (0.0002, 0.002),
        "poisson_peak": (5000.0, 20000.0),
        "quantization_levels": (2048.0, 10000.0),
    },
    "moderate": {
        "sigma": (0.55, 1.25),
        "gaussian_std": (0.0005, 0.006),
        "poisson_peak": (1200.0, 6000.0),
        "quantization_levels": (512.0, 4096.0),
    },
    "severe": {
        "sigma": (0.70, 1.80),
        "gaussian_std": (0.002, 0.015),
        "poisson_peak": (200.0, 1500.0),
        "quantization_levels": (64.0, 1024.0),
    },
}


def _range_value(
    unit_value: torch.Tensor,
    bounds: tuple[float, float],
    reverse: bool = False,
) -> torch.Tensor:
    low, high = bounds
    fraction = 1 - unit_value if reverse else unit_value
    return low + fraction.clamp(0, 1) * (high - low)


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
    generator: torch.Generator | None = None,
    severity: str = "mild",
) -> torch.Tensor:
    """Differentiable MTF blur and area downsample.

    parameters columns are normalized sigma, Gaussian noise, Poisson strength,
    and quantization/compression strength.
    """

    if severity not in DEGRADATION_PRESETS:
        raise ValueError(
            f"Unknown degradation severity {severity!r}; "
            f"expected one of {sorted(DEGRADATION_PRESETS)}"
        )
    preset = DEGRADATION_PRESETS[severity]
    batch, channels, _, _ = hr.shape
    sigma = _range_value(parameters[:, 0], preset["sigma"])
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
        gaussian_std = _range_value(
            parameters[:, 1], preset["gaussian_std"]
        )[:, None, None, None]
        gaussian_noise = torch.randn(
            lr.shape,
            device=lr.device,
            dtype=lr.dtype,
            generator=generator,
        )
        lr = lr + gaussian_noise * gaussian_std
        poisson_strength = _range_value(
            parameters[:, 2], preset["poisson_peak"]
        )[:, None, None, None]
        lr = torch.poisson(
            lr.clamp_min(0) * poisson_strength,
            generator=generator,
        ) / poisson_strength
        levels = _range_value(
            parameters[:, 3], preset["quantization_levels"], reverse=True
        )[:, None, None, None]
        lr = torch.round(lr * (levels - 1)) / (levels - 1)
    return lr.clamp(0, 1)


def random_degradation(
    hr: torch.Tensor,
    scale: int = 4,
    generator: torch.Generator | None = None,
    return_clean: bool = False,
    severity: str = "mild",
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    parameters = torch.rand(
        hr.shape[0],
        4,
        device=hr.device,
        dtype=hr.dtype,
        generator=generator,
    )
    clean_lr = sensor_degrade(
        hr,
        parameters,
        scale=scale,
        add_noise=False,
        severity=severity,
    )
    observed_lr = sensor_degrade(
        hr,
        parameters,
        scale=scale,
        add_noise=True,
        generator=generator,
        severity=severity,
    )
    if return_clean:
        return observed_lr, parameters, clean_lr
    return observed_lr, parameters


def back_project(
    estimate: torch.Tensor,
    observed_lr: torch.Tensor,
    parameters: torch.Tensor,
    scale: int = 4,
    iterations: int = 3,
    step_size: float = 0.5,
    severity: str = "mild",
    debug_callback: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
) -> torch.Tensor:
    result = estimate
    for iteration in range(iterations):
        error = observed_lr - sensor_degrade(
            result,
            parameters,
            scale=scale,
            severity=severity,
        )
        correction = F.interpolate(
            error,
            size=result.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        result = (result + step_size * correction).clamp(0, 1)
        if debug_callback is not None:
            debug_callback(iteration, result, error)
    return result


def default_degradation(batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor([0.4, 0.0, 0.0, 0.0], device=device, dtype=dtype).repeat(batch, 1)
