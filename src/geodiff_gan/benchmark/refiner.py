from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..models.blocks import high_pass
from ..models.degradation import sensor_degrade


class ConditionedResidualBlock(nn.Module):
    def __init__(self, channels: int, condition_dim: int) -> None:
        super().__init__()
        groups = math.gcd(channels, 8)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.film = nn.Linear(condition_dim, channels * 2)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        residual = x
        x = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(condition).chunk(2, dim=1)
        x = self.norm2(x)
        x = x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        x = self.conv2(F.silu(x))
        return residual + x


class ConditionedStage(nn.Module):
    def __init__(self, channels: int, condition_dim: int, blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ConditionedResidualBlock(channels, condition_dim)
            for _ in range(blocks)
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, condition)
        return x


@dataclass
class RefinerOutput:
    base: torch.Tensor
    refined: torch.Tensor
    residual: torch.Tensor
    raw_residual: torch.Tensor
    confidence: torch.Tensor
    sensor_error: torch.Tensor
    degraded_base: torch.Tensor


class SensorNullspaceHighFrequencyRefiner(nn.Module):
    """Shared high-frequency residual wrapper for frozen x4 SR backbones.

    The module does not alter the wrapped backbone. It predicts a bounded,
    confidence-gated residual and removes components visible through the
    differentiable LR sensor model before adding the residual to the base SR.
    """

    def __init__(
        self,
        channels: int = 32,
        condition_dim: int = 64,
        blocks_per_level: tuple[int, int, int] = (2, 2, 3),
        scale: int = 4,
        max_residual: float = 0.12,
        nullspace_iterations: int = 1,
        nullspace_step: float = 0.75,
        degradation_severity: str = "mild",
    ) -> None:
        super().__init__()
        if channels < 8:
            raise ValueError("channels must be at least 8")
        if len(blocks_per_level) != 3:
            raise ValueError("blocks_per_level must contain three values")
        if scale != 4:
            raise ValueError("SN-HFR currently implements the x4 protocol only")
        self.scale = scale
        self.max_residual = float(max_residual)
        self.nullspace_iterations = int(nullspace_iterations)
        self.nullspace_step = float(nullspace_step)
        self.degradation_severity = degradation_severity

        self.condition = nn.Sequential(
            nn.Linear(4, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )

        # base, base high pass, upsampled LR, LR high pass, and LR sensor error
        input_channels = 3 * 5
        self.stem = nn.Conv2d(input_channels, channels, 3, padding=1)
        self.level0 = ConditionedStage(
            channels,
            condition_dim,
            blocks_per_level[0],
        )

        level1_channels = channels * 2
        self.down1 = nn.Conv2d(channels, level1_channels, 3, stride=2, padding=1)
        self.level1 = ConditionedStage(
            level1_channels,
            condition_dim,
            blocks_per_level[1],
        )

        level2_channels = channels * 3
        self.down2 = nn.Conv2d(
            level1_channels,
            level2_channels,
            3,
            stride=2,
            padding=1,
        )
        self.level2 = ConditionedStage(
            level2_channels,
            condition_dim,
            blocks_per_level[2],
        )

        self.up1 = nn.Conv2d(
            level2_channels + level1_channels,
            level1_channels,
            3,
            padding=1,
        )
        self.decode1 = ConditionedStage(
            level1_channels,
            condition_dim,
            blocks_per_level[1],
        )
        self.up0 = nn.Conv2d(
            level1_channels + channels,
            channels,
            3,
            padding=1,
        )
        self.decode0 = ConditionedStage(
            channels,
            condition_dim,
            blocks_per_level[0],
        )
        self.head = nn.Sequential(
            nn.GroupNorm(math.gcd(channels, 8), channels),
            nn.SiLU(),
            nn.Conv2d(channels, 4, 3, padding=1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        with torch.no_grad():
            self.head[-1].bias[3] = -1.0

    def _sensor_response(
        self,
        residual: torch.Tensor,
        degradation: torch.Tensor,
    ) -> torch.Tensor:
        neutral = torch.full_like(residual, 0.5)
        return sensor_degrade(
            neutral + residual,
            degradation,
            scale=self.scale,
            severity=self.degradation_severity,
        ) - sensor_degrade(
            neutral,
            degradation,
            scale=self.scale,
            severity=self.degradation_severity,
        )

    def _project_residual(
        self,
        residual: torch.Tensor,
        degradation: torch.Tensor,
    ) -> torch.Tensor:
        projected = residual
        for _ in range(self.nullspace_iterations):
            observable = self._sensor_response(projected, degradation)
            correction = F.interpolate(
                observable,
                size=projected.shape[-2:],
                mode="bicubic",
                align_corners=False,
            )
            projected = high_pass(
                projected - self.nullspace_step * correction
            )
        return projected.clamp(-self.max_residual, self.max_residual)

    def forward(
        self,
        lr: torch.Tensor,
        base: torch.Tensor,
        degradation: torch.Tensor,
    ) -> RefinerOutput:
        expected_hr = (lr.shape[-2] * self.scale, lr.shape[-1] * self.scale)
        if base.shape[-2:] != expected_hr:
            raise ValueError(
                f"Base output {tuple(base.shape[-2:])} does not match x4 LR "
                f"shape {tuple(lr.shape[-2:])}; expected {expected_hr}"
            )
        if lr.shape[1] != 3 or base.shape[1] != 3:
            raise ValueError("SN-HFR currently expects RGB LR and RGB base output")
        if degradation.shape != (lr.shape[0], 4):
            raise ValueError(
                "degradation must have shape [batch, 4], "
                f"received {tuple(degradation.shape)}"
            )

        degraded_base = sensor_degrade(
            base,
            degradation,
            scale=self.scale,
            severity=self.degradation_severity,
        )
        sensor_error = lr - degraded_base
        lr_up = F.interpolate(
            lr,
            size=base.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        sensor_error_up = F.interpolate(
            sensor_error,
            size=base.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        features = torch.cat(
            (
                base,
                high_pass(base),
                lr_up,
                high_pass(lr_up),
                sensor_error_up,
            ),
            dim=1,
        )
        condition = self.condition(degradation)

        level0 = self.level0(self.stem(features), condition)
        level1 = self.level1(self.down1(level0), condition)
        level2 = self.level2(self.down2(level1), condition)

        decoded1 = F.interpolate(
            level2,
            size=level1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded1 = self.decode1(
            self.up1(torch.cat((decoded1, level1), dim=1)),
            condition,
        )
        decoded0 = F.interpolate(
            decoded1,
            size=level0.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded0 = self.decode0(
            self.up0(torch.cat((decoded0, level0), dim=1)),
            condition,
        )
        raw_residual, gate_logits = self.head(decoded0).split((3, 1), dim=1)
        confidence = torch.sigmoid(gate_logits)
        candidate = self.max_residual * torch.tanh(high_pass(raw_residual))
        residual = self._project_residual(
            confidence * candidate,
            degradation,
        )
        refined = (base + residual).clamp(0, 1)
        return RefinerOutput(
            base=base,
            refined=refined,
            residual=residual,
            raw_residual=raw_residual,
            confidence=confidence,
            sensor_error=sensor_error,
            degraded_base=degraded_base,
        )


class FrozenBackboneRefiner(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        refiner: SensorNullspaceHighFrequencyRefiner,
    ) -> None:
        super().__init__()
        self.backbone = backbone.eval()
        self.refiner = refiner
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> FrozenBackboneRefiner:
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(
        self,
        lr: torch.Tensor,
        degradation: torch.Tensor,
    ) -> RefinerOutput:
        with torch.no_grad():
            base = self.backbone(lr).clamp(0, 1)
        return self.refiner(lr, base, degradation)
