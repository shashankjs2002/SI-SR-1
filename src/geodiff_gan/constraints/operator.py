from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch
from torch.nn import functional as F

from geodiff_gan.models.degradation import DEGRADATION_PRESETS


@runtime_checkable
class LinearSensorOperator(Protocol):
    """Linear HR-to-LR operator used by the proximal solvers."""

    scale: int

    def forward(self, hr: torch.Tensor) -> torch.Tensor:
        ...

    def adjoint(
        self,
        lr: torch.Tensor,
        output_shape: Sequence[int] | None = None,
    ) -> torch.Tensor:
        ...


def _normalized_value(
    value: torch.Tensor,
    bounds: tuple[float, float],
    *,
    reverse: bool = False,
) -> torch.Tensor:
    low, high = bounds
    fraction = 1.0 - value if reverse else value
    return low + fraction.clamp(0, 1) * (high - low)


def _gaussian_kernel(sigma: torch.Tensor, size: int) -> torch.Tensor:
    coordinates = torch.arange(size, device=sigma.device, dtype=sigma.dtype)
    coordinates = coordinates - size // 2
    grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    squared_radius = grid_x.square() + grid_y.square()
    kernel = torch.exp(
        -squared_radius[None]
        / (2.0 * sigma[:, None, None].square().clamp_min(1e-12))
    )
    return kernel / kernel.sum(dim=(-2, -1), keepdim=True)


def _spatial_shape(shape: Sequence[int]) -> tuple[int, int]:
    if len(shape) < 2:
        raise ValueError("output_shape must contain at least height and width")
    return int(shape[-2]), int(shape[-1])


class StridedKernelSensorOperator:
    """Batched depthwise strided-convolution sensor operator.

    Zero padding is deliberate: it gives a simple, exact transpose through
    ``conv_transpose2d``. Boundary handling is therefore part of the operator
    definition and must remain identical in training and evaluation.
    """

    def __init__(
        self,
        kernels: torch.Tensor,
        *,
        scale: int = 4,
        padding: int | None = None,
    ) -> None:
        if kernels.ndim != 3:
            raise ValueError("kernels must have shape [batch, height, width]")
        if kernels.shape[-2] != kernels.shape[-1]:
            raise ValueError("sensor kernels must be square")
        if scale < 1:
            raise ValueError("scale must be positive")
        if kernels.shape[-1] % 2 == 0 and padding is None:
            raise ValueError("automatic padding requires an odd kernel size")
        self.kernels = kernels
        self.scale = int(scale)
        self.padding = kernels.shape[-1] // 2 if padding is None else int(padding)
        self._last_input_shape: tuple[int, int] | None = None

    def _weights(self, channels: int) -> torch.Tensor:
        return self.kernels[:, None].repeat_interleave(channels, dim=0)

    def _validate_batch(self, tensor: torch.Tensor) -> None:
        if tensor.ndim != 4:
            raise ValueError("sensor tensors must have shape [batch, channels, height, width]")
        if tensor.shape[0] != self.kernels.shape[0]:
            raise ValueError(
                f"operator batch {self.kernels.shape[0]} does not match tensor batch "
                f"{tensor.shape[0]}"
            )

    def forward(self, hr: torch.Tensor) -> torch.Tensor:
        self._validate_batch(hr)
        batch, channels, height, width = hr.shape
        weights = self._weights(channels)
        flattened = hr.reshape(1, batch * channels, height, width)
        result = F.conv2d(
            flattened,
            weights,
            stride=self.scale,
            padding=self.padding,
            groups=batch * channels,
        )
        self._last_input_shape = (height, width)
        return result.reshape(batch, channels, *result.shape[-2:])

    def adjoint(
        self,
        lr: torch.Tensor,
        output_shape: Sequence[int] | None = None,
    ) -> torch.Tensor:
        self._validate_batch(lr)
        if output_shape is None:
            if self._last_input_shape is None:
                raise ValueError("output_shape is required before the first forward call")
            target_height, target_width = self._last_input_shape
        else:
            target_height, target_width = _spatial_shape(output_shape)

        batch, channels, height, width = lr.shape
        kernel_size = self.kernels.shape[-1]
        base_height = (
            (height - 1) * self.scale
            - 2 * self.padding
            + kernel_size
        )
        base_width = (
            (width - 1) * self.scale
            - 2 * self.padding
            + kernel_size
        )
        output_padding = (
            target_height - base_height,
            target_width - base_width,
        )
        if any(value < 0 or value >= self.scale for value in output_padding):
            raise ValueError(
                "requested output shape is incompatible with the strided operator: "
                f"target={(target_height, target_width)}, base={(base_height, base_width)}, "
                f"scale={self.scale}"
            )

        weights = self._weights(channels)
        flattened = lr.reshape(1, batch * channels, height, width)
        result = F.conv_transpose2d(
            flattened,
            weights,
            stride=self.scale,
            padding=self.padding,
            output_padding=output_padding,
            groups=batch * channels,
        )
        return result.reshape(batch, channels, target_height, target_width)


class BlockAverageSensorOperator(StridedKernelSensorOperator):
    """Non-overlapping block-average operator with an exact transpose."""

    def __init__(
        self,
        batch_size: int,
        *,
        scale: int = 4,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        kernel = torch.full(
            (batch_size, scale, scale),
            1.0 / (scale * scale),
            device=device,
            dtype=dtype,
        )
        super().__init__(kernel, scale=scale, padding=0)


class GaussianMTFSensorOperator(StridedKernelSensorOperator):
    """Gaussian MTF approximation parameterized by normalized degradation data."""

    def __init__(
        self,
        degradation: torch.Tensor,
        *,
        scale: int = 4,
        severity: str = "mild",
        kernel_size: int = 9,
    ) -> None:
        if degradation.ndim != 2 or degradation.shape[1] < 1:
            raise ValueError("degradation must have shape [batch, parameters]")
        if severity not in DEGRADATION_PRESETS:
            raise ValueError(
                f"unknown degradation severity {severity!r}; "
                f"expected one of {sorted(DEGRADATION_PRESETS)}"
            )
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        sigma = _normalized_value(
            degradation[:, 0],
            DEGRADATION_PRESETS[severity]["sigma"],
        )
        super().__init__(
            _gaussian_kernel(sigma, kernel_size),
            scale=scale,
            padding=kernel_size // 2,
        )


def adjoint_relative_error(
    operator: LinearSensorOperator,
    hr: torch.Tensor,
    lr: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Return per-sample relative error in <Cx,z> = <x,C^Tz>."""

    forward_inner = (operator.forward(hr) * lr).flatten(1).sum(dim=1)
    adjoint_inner = (
        hr * operator.adjoint(lr, hr.shape[-2:])
    ).flatten(1).sum(dim=1)
    denominator = forward_inner.abs() + adjoint_inner.abs() + epsilon
    return (forward_inner - adjoint_inner).abs() / denominator
