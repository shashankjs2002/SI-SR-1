from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .blocks import ConditionedResBlock, CrossAttention2d, SinusoidalEmbedding


def cosine_beta_schedule(steps: int, offset: float = 0.008) -> torch.Tensor:
    x = torch.linspace(0, steps, steps + 1)
    cumulative = torch.cos(((x / steps) + offset) / (1 + offset) * math.pi * 0.5) ** 2
    cumulative = cumulative / cumulative[0]
    betas = 1 - cumulative[1:] / cumulative[:-1]
    return betas.clamp(1e-5, 0.999)


class DiffusionScheduler(nn.Module):
    def __init__(self, steps: int = 1000) -> None:
        super().__init__()
        betas = cosine_beta_schedule(steps)
        alphas = 1 - betas
        cumulative = torch.cumprod(alphas, dim=0)
        self.steps = steps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", cumulative)

    def _extract(self, values: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return values.gather(0, t).view(-1, *((1,) * (x.ndim - 1))).to(x.dtype)

    def q_sample(
        self, clean: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(clean) if noise is None else noise
        alpha = self._extract(self.alphas_cumprod.sqrt(), t, clean)
        sigma = self._extract((1 - self.alphas_cumprod).sqrt(), t, clean)
        return alpha * clean + sigma * noise, noise

    def velocity_target(
        self, clean: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        alpha = self._extract(self.alphas_cumprod.sqrt(), t, clean)
        sigma = self._extract((1 - self.alphas_cumprod).sqrt(), t, clean)
        return alpha * noise - sigma * clean

    def predict_clean(
        self, noisy: torch.Tensor, velocity: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        alpha = self._extract(self.alphas_cumprod.sqrt(), t, noisy)
        sigma = self._extract((1 - self.alphas_cumprod).sqrt(), t, noisy)
        return alpha * noisy - sigma * velocity

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        context: torch.Tensor,
        degradation: torch.Tensor,
        mode: torch.Tensor,
        lr_condition: torch.Tensor,
        sample_steps: int = 20,
        guidance_scale: float = 1.0,
        null_context: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        debug_callback: Callable[
            [int, int, torch.Tensor, torch.Tensor], None
        ]
        | None = None,
        debug_interval: int = 0,
    ) -> torch.Tensor:
        device = context.device
        latent = torch.randn(shape, device=device, generator=generator)
        sequence = torch.linspace(self.steps - 1, 0, sample_steps, device=device).long()
        for index, step in enumerate(sequence):
            t = torch.full((shape[0],), int(step.item()), device=device, dtype=torch.long)
            velocity = model(latent, t, context, degradation, mode, lr_condition)
            if guidance_scale != 1.0 and null_context is not None:
                null_velocity = model(latent, t, null_context, degradation, mode, lr_condition)
                velocity = null_velocity + guidance_scale * (velocity - null_velocity)
            clean = self.predict_clean(latent, velocity, t)
            if debug_callback is not None and (
                index == 0
                or index == len(sequence) - 1
                or (debug_interval > 0 and index % debug_interval == 0)
            ):
                debug_callback(index, int(step.item()), latent, clean)
            if index == len(sequence) - 1:
                latent = clean
                continue
            next_t = sequence[index + 1].expand(shape[0])
            alpha_next = self._extract(self.alphas_cumprod.sqrt(), next_t, latent)
            sigma_next = self._extract((1 - self.alphas_cumprod).sqrt(), next_t, latent)
            alpha_t = self._extract(self.alphas_cumprod.sqrt(), t, latent)
            sigma_t = self._extract((1 - self.alphas_cumprod).sqrt(), t, latent)
            predicted_noise = sigma_t * latent + alpha_t * velocity
            latent = alpha_next * clean + sigma_next * predicted_noise
        return latent


class _UNetLevel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_dim: int,
        context_dim: int,
        attention: bool,
    ) -> None:
        super().__init__()
        self.block1 = ConditionedResBlock(in_channels, out_channels, condition_dim)
        self.block2 = ConditionedResBlock(out_channels, out_channels, condition_dim)
        self.attention = CrossAttention2d(out_channels, context_dim) if attention else nn.Identity()

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        x = self.block2(self.block1(x, condition), condition)
        if isinstance(self.attention, CrossAttention2d):
            x = self.attention(x, context)
        return x


class ConditionalDiffusionUNet(nn.Module):
    def __init__(
        self,
        latent_channels: int = 4,
        widths: tuple[int, ...] = (128, 256, 384, 512),
        context_dim: int = 768,
        degradation_dim: int = 4,
        lr_condition_channels: int = 64,
        attention_levels: tuple[int, ...] = (1, 2),
    ) -> None:
        super().__init__()
        condition_dim = widths[0] * 4
        self.time = nn.Sequential(
            SinusoidalEmbedding(widths[0]),
            nn.Linear(widths[0], condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.degradation = nn.Sequential(
            nn.Linear(degradation_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.mode = nn.Embedding(2, condition_dim)
        self.context_dim = context_dim
        self.gradient_checkpointing = False
        self.input = nn.Conv2d(latent_channels + lr_condition_channels, widths[0], 3, padding=1)

        self.down_levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current = widths[0]
        for index, width in enumerate(widths):
            self.down_levels.append(
                _UNetLevel(
                    current,
                    width,
                    condition_dim,
                    context_dim,
                    attention=index in attention_levels,
                )
            )
            current = width
            if index < len(widths) - 1:
                self.downsamples.append(nn.Conv2d(current, current, 4, stride=2, padding=1))

        self.middle = _UNetLevel(current, current, condition_dim, context_dim, attention=True)
        self.upsamples = nn.ModuleList()
        self.up_levels = nn.ModuleList()
        for index in reversed(range(len(widths) - 1)):
            self.upsamples.append(
                nn.Sequential(
                    nn.Conv2d(current, widths[index] * 4, 3, padding=1),
                    nn.PixelShuffle(2),
                )
            )
            current = widths[index]
            self.up_levels.append(
                _UNetLevel(
                    current + widths[index],
                    current,
                    condition_dim,
                    context_dim,
                    attention=index in attention_levels,
                )
            )
        self.output = nn.Sequential(
            nn.GroupNorm(math.gcd(current, 8), current),
            nn.SiLU(),
            nn.Conv2d(current, latent_channels, 3, padding=1),
        )

    def forward(
        self,
        latent: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        degradation: torch.Tensor,
        mode: torch.Tensor,
        lr_condition: torch.Tensor,
    ) -> torch.Tensor:
        lr_condition = F.interpolate(
            lr_condition, size=latent.shape[-2:], mode="bilinear", align_corners=False
        )
        x = self.input(torch.cat((latent, lr_condition), dim=1))
        condition = self.time(t) + self.degradation(degradation) + self.mode(mode)
        skips: list[torch.Tensor] = []
        for index, level in enumerate(self.down_levels):
            if self.gradient_checkpointing and self.training:
                x = checkpoint(level, x, condition, context, use_reentrant=False)
            else:
                x = level(x, condition, context)
            skips.append(x)
            if index < len(self.downsamples):
                x = self.downsamples[index](x)
        if self.gradient_checkpointing and self.training:
            x = checkpoint(self.middle, x, condition, context, use_reentrant=False)
        else:
            x = self.middle(x, condition, context)
        for upsample, level, skip in zip(self.upsamples, self.up_levels, reversed(skips[:-1])):
            x = upsample(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            merged = torch.cat((x, skip), dim=1)
            if self.gradient_checkpointing and self.training:
                x = checkpoint(level, merged, condition, context, use_reentrant=False)
            else:
                x = level(merged, condition, context)
        return self.output(x)


@dataclass
class DiffusionBatch:
    noisy: torch.Tensor
    noise: torch.Tensor
    target_velocity: torch.Tensor
    timesteps: torch.Tensor
