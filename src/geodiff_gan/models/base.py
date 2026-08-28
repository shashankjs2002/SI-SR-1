from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import LayerNorm2d, validate_attention_heads


def _window_partition(x: torch.Tensor, window: int) -> tuple[torch.Tensor, tuple[int, int]]:
    batch, channels, height, width = x.shape
    pad_h = (window - height % window) % window
    pad_w = (window - width % window) % window
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    padded_h, padded_w = height + pad_h, width + pad_w
    x = x.view(
        batch,
        channels,
        padded_h // window,
        window,
        padded_w // window,
        window,
    )
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window * window, channels)
    return windows, (padded_h, padded_w)


def _window_reverse(
    windows: torch.Tensor,
    window: int,
    shape: tuple[int, int],
    original: tuple[int, int],
    batch: int,
) -> torch.Tensor:
    padded_h, padded_w = shape
    channels = windows.shape[-1]
    x = windows.view(
        batch,
        padded_h // window,
        padded_w // window,
        window,
        window,
        channels,
    )
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(batch, channels, padded_h, padded_w)
    return x[:, :, : original[0], : original[1]]


class WindowTransformerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        window_size: int = 8,
        heads: int = 6,
        shift: bool = False,
    ) -> None:
        super().__init__()
        heads = validate_attention_heads(channels, heads)
        self.window_size = window_size
        self.shift = window_size // 2 if shift else 0
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm2 = LayerNorm2d(channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        shifted = torch.roll(x, (-self.shift, -self.shift), dims=(-2, -1)) if self.shift else x
        windows, padded_shape = _window_partition(shifted, self.window_size)
        normalized = self.norm1(windows)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        windows = windows + attended
        attended = _window_reverse(
            windows,
            self.window_size,
            padded_shape,
            (height, width),
            batch,
        )
        if self.shift:
            attended = torch.roll(attended, (self.shift, self.shift), dims=(-2, -1))
        return attended + self.mlp(self.norm2(attended))


class ResizeConvUpsampler(nn.Module):
    """Integer-scale interpolation followed by a learned reconstruction convolution."""

    def __init__(self, channels: int, scale: int) -> None:
        super().__init__()
        self.scale = scale
        self.convolution = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(
            value,
            scale_factor=self.scale,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return self.convolution(value)


class ResidualSwinGroup(nn.Module):
    """A residual group of shifted-window transformer blocks."""

    def __init__(
        self,
        channels: int,
        depth: int,
        window_size: int,
        heads: int,
        start_index: int = 0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            WindowTransformerBlock(
                channels,
                window_size=window_size,
                heads=heads,
                shift=bool((start_index + index) % 2),
            )
            for index in range(depth)
        )
        self.projection = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        for block in self.blocks:
            value = block(value)
        return residual + self.projection(value)


class FidelitySwinIRBase(nn.Module):
    """Higher-capacity Swin reconstruction anchor for paired cross-sensor SR.

    The nested residual groups preserve a short optimization path, while the
    zero-initialized radiometric head can learn a scene-level RGB correction
    without changing the bicubic anchor at initialization. Upsampling always
    uses resize-convolution to avoid sub-pixel phase artifacts.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 72,
        depth: int = 12,
        group_size: int = 4,
        window_size: int = 8,
        heads: int = 6,
        scale: int = 3,
        output_channels: int = 3,
        radiometric_calibration: bool = True,
    ) -> None:
        super().__init__()
        if scale < 2:
            raise ValueError("scale must be an integer of at least 2")
        if in_channels < output_channels:
            raise ValueError("in_channels must be >= output_channels")
        if depth < 1 or group_size < 1:
            raise ValueError("depth and group_size must be positive")
        self.scale = scale
        self.output_channels = output_channels
        self.upsample_mode = "resize_conv"
        self.radiometric_calibration = radiometric_calibration
        self.shallow = nn.Conv2d(in_channels, embed_dim, 3, padding=1)

        groups: list[nn.Module] = []
        consumed = 0
        while consumed < depth:
            group_depth = min(group_size, depth - consumed)
            groups.append(
                ResidualSwinGroup(
                    embed_dim,
                    depth=group_depth,
                    window_size=window_size,
                    heads=heads,
                    start_index=consumed,
                )
            )
            consumed += group_depth
        self.groups = nn.ModuleList(groups)
        self.body = nn.Sequential(
            LayerNorm2d(embed_dim),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
        )
        self.upsample = nn.Sequential(
            ResizeConvUpsampler(embed_dim, scale),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.output = nn.Conv2d(embed_dim, output_channels, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.radiometric_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, output_channels * 2),
        )
        nn.init.zeros_(self.radiometric_head[-1].weight)
        nn.init.zeros_(self.radiometric_head[-1].bias)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        shallow = self.shallow(lr)
        features = shallow
        for group in self.groups:
            features = group(features)
        features = shallow + self.body(features)

        anchor = F.interpolate(
            lr[:, : self.output_channels],
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
        )
        if self.radiometric_calibration:
            calibration = self.radiometric_head(shallow)
            gain, bias = calibration.chunk(2, dim=1)
            gain = 1 + 0.1 * torch.tanh(gain)[:, :, None, None]
            bias = 0.1 * torch.tanh(bias)[:, :, None, None]
            anchor = anchor * gain + bias

        detail = self.output(self.upsample(features))
        return (anchor + detail).clamp(0, 1)


class SwinIRBase(nn.Module):
    """Compact SwinIR-style conservative integer-scale reconstruction branch."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 60,
        depth: int = 6,
        window_size: int = 8,
        heads: int = 6,
        scale: int = 4,
        output_channels: int = 3,
        upsample_mode: str = "pixelshuffle",
    ) -> None:
        super().__init__()
        if scale < 2:
            raise ValueError("scale must be an integer of at least 2")
        if upsample_mode not in ("pixelshuffle", "resize_conv"):
            raise ValueError("upsample_mode must be 'pixelshuffle' or 'resize_conv'")
        if upsample_mode == "pixelshuffle" and scale not in (2, 3, 4, 8):
            raise ValueError("Pixel-shuffle scale must be 2, 3, 4, or 8")
        if in_channels < output_channels:
            raise ValueError("in_channels must be >= output_channels")
        self.scale = scale
        self.output_channels = output_channels
        self.upsample_mode = upsample_mode
        self.shallow = nn.Conv2d(in_channels, embed_dim, 3, padding=1)
        self.blocks = nn.ModuleList(
            WindowTransformerBlock(
                embed_dim,
                window_size=window_size,
                heads=heads,
                shift=bool(index % 2),
            )
            for index in range(depth)
        )
        self.body = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
        if upsample_mode == "resize_conv":
            self.upsample = nn.Sequential(
                ResizeConvUpsampler(embed_dim, scale),
                nn.LeakyReLU(0.1, inplace=True),
            )
        elif scale == 3:
            self.upsample = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim * 9, 3, padding=1),
                nn.PixelShuffle(3),
                nn.LeakyReLU(0.1, inplace=True),
            )
        else:
            upsamplers: list[nn.Module] = []
            remaining = scale
            while remaining > 1:
                upsamplers.extend(
                    [
                        nn.Conv2d(embed_dim, embed_dim * 4, 3, padding=1),
                        nn.PixelShuffle(2),
                        nn.LeakyReLU(0.1, inplace=True),
                    ]
                )
                remaining //= 2
            self.upsample = nn.Sequential(*upsamplers)
        self.output = nn.Conv2d(embed_dim, output_channels, 3, padding=1)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        shallow = self.shallow(lr)
        features = shallow
        for block in self.blocks:
            features = block(features)
        features = shallow + self.body(features)
        residual = self.output(self.upsample(features))
        bicubic = F.interpolate(
            lr[:, : self.output_channels],
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
        )
        return (bicubic + residual).clamp(0, 1)
