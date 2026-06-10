from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import FiLMResidualBlock, ResidualBlock


class LREncoder(nn.Module):
    def __init__(self, in_channels: int = 3, channels: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1),
            ResidualBlock(channels),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 4, stride=2, padding=1),
            ResidualBlock(channels * 2),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 4, 4, stride=2, padding=1),
            ResidualBlock(channels * 4),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(channels * 4, channels * 4, 4, stride=2, padding=1),
            ResidualBlock(channels * 4),
        )

    def forward(self, lr: torch.Tensor) -> list[torch.Tensor]:
        f128 = self.stem(lr)
        f64 = self.down1(f128)
        f32 = self.down2(f64)
        f16 = self.down3(f32)
        return [f128, f64, f32, f16]


@dataclass
class MapperOutput:
    content: torch.Tensor
    styles: list[torch.Tensor]
    evidence_gate: torch.Tensor


class GeoMapper(nn.Module):
    def __init__(
        self,
        latent_channels: int = 4,
        lr_channels: int = 128,
        content_channels: int = 128,
        context_dim: int = 768,
        style_dim: int = 256,
        stages: int = 4,
        use_evidence_gate: bool = True,
    ) -> None:
        super().__init__()
        self.input = nn.Conv2d(latent_channels + lr_channels, content_channels, 3, padding=1)
        self.blocks = nn.Sequential(*(ResidualBlock(content_channels) for _ in range(4)))
        self.context = nn.Linear(context_dim, content_channels)
        self.gate = nn.Sequential(
            nn.Conv2d(content_channels * 2, content_channels, 1),
            nn.SiLU(),
            nn.Conv2d(content_channels, 1, 1),
            nn.Sigmoid(),
        )
        self.style_heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(content_channels + context_dim, style_dim),
                nn.SiLU(),
                nn.Linear(style_dim, style_dim),
            )
            for _ in range(stages)
        )
        self.use_evidence_gate = use_evidence_gate

    def forward(
        self,
        latent: torch.Tensor,
        lr_feature: torch.Tensor,
        context: torch.Tensor,
        mode: torch.Tensor,
    ) -> MapperOutput:
        lr_feature = F.interpolate(
            lr_feature, size=latent.shape[-2:], mode="bilinear", align_corners=False
        )
        content = self.blocks(self.input(torch.cat((latent, lr_feature), dim=1)))
        pooled_context = context.mean(dim=1)
        context_map = self.context(pooled_context)[:, :, None, None].expand_as(content)
        evidence_gate = self.gate(torch.cat((content, context_map), dim=1))
        if not self.use_evidence_gate:
            evidence_gate = torch.ones_like(evidence_gate)
        mode_strength = mode.float()[:, None, None, None]
        gated_content = content + context_map * evidence_gate * (0.25 + 0.75 * mode_strength)
        pooled_content = gated_content.mean(dim=(-2, -1))
        style_input = torch.cat((pooled_content, pooled_context), dim=1)
        styles = [head(style_input) for head in self.style_heads]
        return MapperOutput(gated_content, styles, evidence_gate)


class ResidualSRDecoder(nn.Module):
    def __init__(
        self,
        content_channels: int = 128,
        style_dim: int = 256,
        lr_channels: int = 64,
        stage_channels: tuple[int, ...] = (128, 96, 64, 48),
    ) -> None:
        super().__init__()
        if len(stage_channels) != 4:
            raise ValueError("Decoder expects four stages at 64, 128, 256, and 512 pixels")
        self.input = nn.Conv2d(content_channels, stage_channels[0], 3, padding=1)
        self.blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.skip_projections = nn.ModuleList()
        current = stage_channels[0]
        skip_dims = (lr_channels * 2, lr_channels, lr_channels, lr_channels)
        for index, channels in enumerate(stage_channels):
            if index:
                self.upsamples.append(
                    nn.Sequential(
                        nn.Conv2d(current, channels * 4, 3, padding=1),
                        nn.PixelShuffle(2),
                    )
                )
                current = channels
            self.skip_projections.append(nn.Conv2d(skip_dims[index], current, 1))
            self.blocks.append(FiLMResidualBlock(current, style_dim))
        self.output = nn.Sequential(
            nn.Conv2d(current, current, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(current, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(
        self, mapped: MapperOutput, lr_features: list[torch.Tensor]
    ) -> torch.Tensor:
        x = self.input(mapped.content)
        source_skips = [lr_features[1], lr_features[0], lr_features[0], lr_features[0]]
        for index, block in enumerate(self.blocks):
            if index:
                x = self.upsamples[index - 1](x)
            skip = F.interpolate(
                source_skips[index], size=x.shape[-2:], mode="bilinear", align_corners=False
            )
            x = x + self.skip_projections[index](skip)
            x = block(x, mapped.styles[index])
        return self.output(x)
