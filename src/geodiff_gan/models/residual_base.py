"""Opt-in compact residual-group base; legacy SwinIR checkpoints stay unchanged."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .base import ResizeConvUpsampler, _window_partition, _window_reverse
from .blocks import validate_attention_heads


class MaskedWindowBlock(nn.Module):
    def __init__(self, channels: int, heads: int, window: int, shift: bool):
        super().__init__()
        self.heads = validate_attention_heads(channels, heads)
        if window < 2:
            raise ValueError("window must be at least two")
        self.window = window
        self.shift = window // 2 if shift else 0
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.projection = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(nn.Linear(channels, channels * 2), nn.GELU(),
                                 nn.Linear(channels * 2, channels))
        coords = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij"))
        relative = coords.flatten(1)[:, :, None] - coords.flatten(1)[:, None, :]
        relative = relative.permute(1, 2, 0) + window - 1
        index = relative[..., 0] * (2 * window - 1) + relative[..., 1]
        self.register_buffer("relative_index", index.long(), persistent=False)
        self.relative_bias = nn.Parameter(torch.zeros((2 * window - 1) ** 2, heads))
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

    def attention_mask(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        w = self.window
        ph, pw = ((height + w - 1) // w) * w, ((width + w - 1) // w) * w
        shift = self.shift if min(height, width) > w else 0
        labels = torch.zeros(1, 1, ph, pw, device=device)
        if shift:
            slices = (slice(0, -w), slice(-w, -shift), slice(-shift, None))
            for i, rows in enumerate(slices):
                for j, cols in enumerate(slices):
                    labels[:, :, rows, cols] = 3 * i + j
        windows, _ = _window_partition(labels, w)
        blocked = windows[..., 0, None] != windows[..., 0].unsqueeze(1)
        valid = torch.zeros_like(labels)
        valid[:, :, :height, :width] = 1
        if shift:
            valid = torch.roll(valid, (-shift, -shift), (-2, -1))
        valid_windows, _ = _window_partition(valid, w)
        blocked = blocked | (valid_windows[..., 0].unsqueeze(1) == 0)
        # Finite negative bias also keeps fully padded query rows numerically defined.
        return blocked.float().mul(-100.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, h, w = x.shape
        size = self.window
        shift = self.shift if min(h, w) > size else 0
        padded = F.pad(x, (0, (-w) % size, 0, (-h) % size), mode="replicate")
        if shift:
            padded = torch.roll(padded, (-shift, -shift), (-2, -1))
        tokens, shape = _window_partition(padded, size)
        n = size * size
        q, k, v = self.qkv(self.norm1(tokens)).reshape(-1, n, 3, self.heads, channels // self.heads).permute(2, 0, 3, 1, 4)
        bias = self.relative_bias[self.relative_index.reshape(-1)].reshape(n, n, self.heads).permute(2, 0, 1)
        mask = self.attention_mask(h, w, x.device).repeat(batch, 1, 1)[:, None]
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=(bias[None] + mask).to(q.dtype))
        tokens = tokens + self.projection(attended.transpose(1, 2).reshape(-1, n, channels))
        tokens = tokens + self.mlp(self.norm2(tokens))
        out = _window_reverse(tokens, size, shape, shape, batch)
        if shift:
            out = torch.roll(out, (shift, shift), (-2, -1))
        return out[:, :, :h, :w]


class ResidualGroup(nn.Module):
    def __init__(self, channels: int, depth: int, heads: int, window: int):
        super().__init__()
        self.blocks = nn.Sequential(*(MaskedWindowBlock(channels, heads, window, bool(i % 2)) for i in range(depth)))
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.1 * self.conv(self.blocks(x))


class ResidualSwinBase(nn.Module):
    """Local/group/global skips with a zero-initialized RGB residual over bicubic."""
    def __init__(self, in_channels=3, embed_dim=48, depth=4, heads=4,
                 window_size=8, scale=3, output_channels=3, groups=2):
        super().__init__()
        if scale < 2 or in_channels < output_channels or depth < 1 or groups < 1:
            raise ValueError("Invalid residual base geometry/depth")
        self.scale, self.output_channels = scale, output_channels
        self.shallow = nn.Conv2d(in_channels, embed_dim, 3, padding=1)
        self.groups = nn.Sequential(*(ResidualGroup(embed_dim, depth, heads, window_size) for _ in range(groups)))
        self.body = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
        self.upsample = ResizeConvUpsampler(embed_dim, scale)
        self.output = nn.Conv2d(embed_dim, output_channels, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        shallow = self.shallow(lr)
        features = shallow + self.body(self.groups(shallow))
        correction = self.output(F.gelu(self.upsample(features)))
        anchor = F.interpolate(lr[:, :self.output_channels], scale_factor=self.scale, mode="bicubic", align_corners=False)
        return (anchor + correction).clamp(0, 1)
