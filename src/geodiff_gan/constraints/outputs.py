from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ProximalOutput:
    """Structured output shared by all constraint-study arms."""

    image: torch.Tensor
    dual: torch.Tensor
    observation_residual: torch.Tensor
    decomposition_residual: torch.Tensor
    iterations: torch.Tensor
    converged: torch.Tensor
    objective: torch.Tensor
    variance: torch.Tensor | None = None
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class NoiseCalibrationOutput:
    """Outputs of the low-capacity sensor-noise calibration head."""

    variance: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    mean_lr: torch.Tensor
    raw_alpha_beta: torch.Tensor
