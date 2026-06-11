from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import ResidualBlock


class ResidualVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4]
        self.downsample_factor = 2 ** len(channels)
        encoder: list[nn.Module] = [nn.Conv2d(in_channels, channels[0], 3, padding=1)]
        for index, channel in enumerate(channels):
            if index:
                encoder.append(nn.Conv2d(channels[index - 1], channel, 4, stride=2, padding=1))
            encoder.append(ResidualBlock(channel))
        encoder.append(nn.Conv2d(channels[-1], channels[-1], 4, stride=2, padding=1))
        self.encoder = nn.Sequential(*encoder)
        self.moments = nn.Conv2d(channels[-1], latent_channels * 2, 1)

        self.from_latent = nn.Conv2d(latent_channels, channels[-1], 3, padding=1)
        decoder: list[nn.Module] = []
        current = channels[-1]
        for channel in reversed(channels):
            decoder.extend(
                [
                    ResidualBlock(current),
                    nn.Conv2d(current, channel * 4, 3, padding=1),
                    nn.PixelShuffle(2),
                ]
            )
            current = channel
        self.decoder = nn.Sequential(*decoder)
        self.output = nn.Conv2d(current, in_channels, 3, padding=1)

    def encode(
        self, residual: torch.Tensor, sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_variance = self.moments(self.encoder(residual)).chunk(2, dim=1)
        log_variance = log_variance.clamp(-20, 10)
        if sample:
            latent = mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)
        else:
            latent = mean
        return latent, mean, log_variance

    def decode(
        self,
        latent: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        decoded = self.output(self.decoder(self.from_latent(latent)))
        if output_size is not None and decoded.shape[-2:] != output_size:
            decoded = F.interpolate(decoded, size=output_size, mode="bilinear", align_corners=False)
        return decoded

    def forward(
        self, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, mean, log_variance = self.encode(residual)
        return self.decode(latent, residual.shape[-2:]), latent, mean, log_variance
