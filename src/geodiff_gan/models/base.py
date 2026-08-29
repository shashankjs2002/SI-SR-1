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


class _RadiometricAnchor(nn.Module):
    """Identity-initialized cross-sensor anchor shared by V2 base candidates."""

    def __init__(
        self,
        feature_channels: int,
        output_channels: int,
        scale: int,
        global_calibration: bool,
        spatial_calibration: bool,
    ) -> None:
        super().__init__()
        self.output_channels = output_channels
        self.scale = scale
        self.global_calibration = global_calibration
        self.spatial_calibration = spatial_calibration
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_channels, feature_channels),
            nn.GELU(),
            nn.Linear(feature_channels, output_channels * 2),
        )
        nn.init.zeros_(self.global_head[-1].weight)
        nn.init.zeros_(self.global_head[-1].bias)
        self.spatial_head = nn.Conv2d(feature_channels, output_channels, 3, padding=1)
        nn.init.zeros_(self.spatial_head.weight)
        nn.init.zeros_(self.spatial_head.bias)

    def forward(self, lr: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        anchor = F.interpolate(
            lr[:, : self.output_channels],
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
        )
        if self.global_calibration:
            calibration = self.global_head(features)
            gain, bias = calibration.chunk(2, dim=1)
            gain = 1 + 0.1 * torch.tanh(gain)[:, :, None, None]
            bias = 0.1 * torch.tanh(bias)[:, :, None, None]
            anchor = anchor * gain + bias
        if self.spatial_calibration:
            correction = self.spatial_head(features)
            correction = F.avg_pool2d(correction, 9, stride=1, padding=4)
            correction = 0.05 * torch.tanh(correction)
            correction = F.interpolate(
                correction,
                scale_factor=self.scale,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            anchor = anchor + correction
        return anchor


class ResidualDenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_channels: int) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, growth_channels, 3, padding=1)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        value = F.gelu(self.convolution(torch.cat(features, dim=1)))
        return [*features, value]


class ResidualDenseBlock(nn.Module):
    """Residual-dense feature block used by the OLI2MSI fidelity candidate."""

    def __init__(
        self,
        channels: int,
        layers: int = 6,
        growth_channels: int = 32,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            ResidualDenseLayer(channels + index * growth_channels, growth_channels)
            for index in range(layers)
        )
        self.local_fusion = nn.Conv2d(
            channels + layers * growth_channels,
            channels,
            1,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        features = [value]
        for layer in self.layers:
            features = layer(features)
        return value + 0.2 * self.local_fusion(torch.cat(features, dim=1))


class FidelityRDNBase(nn.Module):
    """High-capacity residual-dense 3x anchor with artifact-safe upsampling."""

    def __init__(
        self,
        in_channels: int = 3,
        channels: int = 64,
        blocks: int = 20,
        layers: int = 6,
        growth_channels: int = 32,
        scale: int = 3,
        output_channels: int = 3,
        radiometric_calibration: bool = True,
        spatial_radiometric_calibration: bool = False,
    ) -> None:
        super().__init__()
        if scale < 2:
            raise ValueError("scale must be an integer of at least 2")
        if blocks < 1 or layers < 1 or growth_channels < 1:
            raise ValueError("RDN blocks, layers and growth channels must be positive")
        self.scale = scale
        self.output_channels = output_channels
        self.upsample_mode = "resize_conv"
        self.shallow1 = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.shallow2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.blocks = nn.ModuleList(
            ResidualDenseBlock(channels, layers, growth_channels)
            for _ in range(blocks)
        )
        self.global_fusion = nn.Sequential(
            nn.Conv2d(channels * blocks, channels, 1),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.anchor = _RadiometricAnchor(
            channels,
            output_channels,
            scale,
            radiometric_calibration,
            spatial_radiometric_calibration,
        )
        self.upsample = nn.Sequential(
            ResizeConvUpsampler(channels, scale),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
        )
        self.output = nn.Conv2d(channels, output_channels, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        shallow = self.shallow1(lr)
        value = self.shallow2(shallow)
        block_outputs = []
        for block in self.blocks:
            value = block(value)
            block_outputs.append(value)
        features = shallow + self.global_fusion(torch.cat(block_outputs, dim=1))
        anchor = self.anchor(lr, shallow)
        detail = self.output(self.upsample(features))
        return (anchor + detail).clamp(0, 1)


class RelativeWindowAttention(nn.Module):
    """Window attention with learned relative position bias and shift masking."""

    def __init__(self, channels: int, window_size: int, heads: int) -> None:
        super().__init__()
        heads = validate_attention_heads(channels, heads)
        self.channels = channels
        self.window_size = window_size
        self.heads = heads
        self.head_dim = channels // heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(channels, channels * 3)
        self.projection = nn.Linear(channels, channels)
        table_size = (2 * window_size - 1) ** 2
        self.relative_position_bias = nn.Parameter(torch.zeros(table_size, heads))
        coordinates = torch.stack(
            torch.meshgrid(
                torch.arange(window_size),
                torch.arange(window_size),
                indexing="ij",
            )
        )
        flattened = coordinates.flatten(1)
        relative = flattened[:, :, None] - flattened[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window_size - 1
        relative[:, :, 1] += window_size - 1
        relative[:, :, 0] *= 2 * window_size - 1
        self.register_buffer(
            "relative_position_index",
            relative.sum(-1),
            persistent=False,
        )
        nn.init.trunc_normal_(self.relative_position_bias, std=0.02)

    def forward(
        self,
        windows: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_windows, tokens, channels = windows.shape
        qkv = self.qkv(windows).reshape(
            batch_windows,
            tokens,
            3,
            self.heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = (query * self.scale) @ key.transpose(-2, -1)
        relative = self.relative_position_bias[
            self.relative_position_index.reshape(-1)
        ].reshape(tokens, tokens, self.heads)
        attention = attention + relative.permute(2, 0, 1).unsqueeze(0)
        if mask is not None:
            window_count = mask.shape[0]
            attention = attention.view(
                batch_windows // window_count,
                window_count,
                self.heads,
                tokens,
                tokens,
            )
            attention = attention + mask[None, :, None]
            attention = attention.view(batch_windows, self.heads, tokens, tokens)
        attention = attention.softmax(dim=-1)
        output = (attention @ value).transpose(1, 2).reshape(
            batch_windows,
            tokens,
            channels,
        )
        return self.projection(output)


class SwinIRV2Block(nn.Module):
    def __init__(
        self,
        channels: int,
        window_size: int = 8,
        heads: int = 6,
        shift: bool = False,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift else 0
        self.norm1 = nn.LayerNorm(channels)
        self.attention = RelativeWindowAttention(channels, window_size, heads)
        self.norm2 = nn.LayerNorm(channels)
        hidden = max(channels, int(round(channels * mlp_ratio)))
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def attention_mask(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.shift_size == 0:
            return None
        window = self.window_size
        padded_h = height + (window - height % window) % window
        padded_w = width + (window - width % window) % window
        labels = torch.zeros((1, 1, padded_h, padded_w), device=device, dtype=dtype)
        h_slices = (
            slice(0, -window),
            slice(-window, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = h_slices
        label = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                labels[:, :, h_slice, w_slice] = label
                label += 1
        windows, _ = _window_partition(labels, window)
        windows = windows.squeeze(-1)
        difference = windows[:, None, :] - windows[:, :, None]
        return difference.masked_fill(difference != 0, -100.0).masked_fill(
            difference == 0,
            0.0,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = value.shape
        shortcut = value
        normalized = self.norm1(value.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        normalized = F.pad(normalized, (0, pad_w, 0, pad_h), mode="reflect")
        if self.shift_size:
            normalized = torch.roll(
                normalized,
                (-self.shift_size, -self.shift_size),
                dims=(-2, -1),
            )
        windows, padded_shape = _window_partition(normalized, self.window_size)
        attended = self.attention(
            windows,
            self.attention_mask(height, width, value.device, value.dtype),
        )
        attended = _window_reverse(
            attended,
            self.window_size,
            padded_shape,
            padded_shape,
            batch,
        )
        if self.shift_size:
            attended = torch.roll(
                attended,
                (self.shift_size, self.shift_size),
                dims=(-2, -1),
            )
        attended = attended[:, :, :height, :width]
        value = shortcut + attended
        tokens = value.permute(0, 2, 3, 1)
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.permute(0, 3, 1, 2)


class ResidualSwinV2Group(nn.Module):
    def __init__(
        self,
        channels: int,
        blocks: int,
        window_size: int,
        heads: int,
        mlp_ratio: float,
        start_index: int,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            SwinIRV2Block(
                channels,
                window_size,
                heads,
                shift=bool((start_index + index) % 2),
                mlp_ratio=mlp_ratio,
            )
            for index in range(blocks)
        )
        self.projection = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        for block in self.blocks:
            value = block(value)
        return residual + self.projection(value)


class FidelitySwinIRV2Base(nn.Module):
    """SwinIR-style V2 anchor with proper shifted-window attention semantics."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 120,
        groups: int = 6,
        blocks_per_group: int = 6,
        window_size: int = 8,
        heads: int = 6,
        mlp_ratio: float = 2.0,
        scale: int = 3,
        output_channels: int = 3,
        radiometric_calibration: bool = True,
        spatial_radiometric_calibration: bool = False,
    ) -> None:
        super().__init__()
        if groups < 1 or blocks_per_group < 1:
            raise ValueError("SwinIR groups and blocks per group must be positive")
        self.scale = scale
        self.output_channels = output_channels
        self.upsample_mode = "resize_conv"
        self.shallow = nn.Conv2d(in_channels, embed_dim, 3, padding=1)
        self.groups = nn.ModuleList(
            ResidualSwinV2Group(
                embed_dim,
                blocks_per_group,
                window_size,
                heads,
                mlp_ratio,
                start_index=index * blocks_per_group,
            )
            for index in range(groups)
        )
        self.body = nn.Sequential(
            LayerNorm2d(embed_dim),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
        )
        self.anchor = _RadiometricAnchor(
            embed_dim,
            output_channels,
            scale,
            radiometric_calibration,
            spatial_radiometric_calibration,
        )
        self.upsample = nn.Sequential(
            ResizeConvUpsampler(embed_dim, scale),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.GELU(),
        )
        self.output = nn.Conv2d(embed_dim, output_channels, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        shallow = self.shallow(lr)
        features = shallow
        for group in self.groups:
            features = group(features)
        features = shallow + self.body(features)
        anchor = self.anchor(lr, shallow)
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
