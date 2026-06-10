from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import haar_wavelet


def _spectral_conv(in_channels: int, out_channels: int, stride: int = 1) -> nn.Module:
    return nn.utils.parametrizations.spectral_norm(
        nn.Conv2d(in_channels, out_channels, 4, stride=stride, padding=1)
    )


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 6, base_channels: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _spectral_conv(in_channels, base_channels, 2),
            nn.LeakyReLU(0.2, inplace=True),
            _spectral_conv(base_channels, base_channels * 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            _spectral_conv(base_channels * 2, base_channels * 4, 2),
            nn.LeakyReLU(0.2, inplace=True),
            _spectral_conv(base_channels * 4, base_channels * 4, 1),
            nn.LeakyReLU(0.2, inplace=True),
            _spectral_conv(base_channels * 4, 1, 1),
        )

    def forward(self, hr: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        condition = F.interpolate(lr, size=hr.shape[-2:], mode="bilinear", align_corners=False)
        return self.net(torch.cat((hr, condition), dim=1))


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, scales: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            PatchDiscriminator(base_channels=base_channels) for _ in range(scales)
        )

    def forward(self, hr: torch.Tensor, lr: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for discriminator in self.discriminators:
            if min(hr.shape[-2:]) < 32:
                break
            outputs.append(discriminator(hr, lr))
            hr = F.avg_pool2d(hr, 2)
            lr = F.avg_pool2d(lr, 2) if min(lr.shape[-2:]) >= 2 else lr
        if not outputs:
            raise ValueError("Multi-scale discriminator requires inputs of at least 32x32 pixels")
        return outputs


class WaveletDiscriminator(nn.Module):
    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        self.net = PatchDiscriminator(in_channels=18, base_channels=base_channels)

    def forward(self, hr: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
        _, lh, hl, hh = haar_wavelet(hr)
        high_frequency = torch.cat((lh, hl, hh), dim=1)
        lr_condition = F.interpolate(lr, size=high_frequency.shape[-2:], mode="area")
        lr_condition = lr_condition.repeat(1, 3, 1, 1)
        return self.net.net(torch.cat((high_frequency, lr_condition), dim=1))
