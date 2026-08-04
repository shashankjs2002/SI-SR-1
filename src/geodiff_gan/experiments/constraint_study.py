from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from geodiff_gan.constraints.arms import OracleData
from geodiff_gan.constraints.noise_calibration import gaussian_nll
from geodiff_gan.constraints.operator import LinearSensorOperator
from geodiff_gan.constraints.outputs import ProximalOutput


@dataclass
class ConstraintStudyResult:
    outputs: dict[str, ProximalOutput]
    metrics: dict[str, dict[str, torch.Tensor]]


def _per_sample_mean(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.flatten(1).mean(dim=1)


def output_metrics(
    output: ProximalOutput,
    observed_lr: torch.Tensor,
    operator: LinearSensorOperator,
    *,
    target_hr: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    projected_lr = operator.forward(output.image)
    values = {
        "lr_l1": _per_sample_mean((projected_lr - observed_lr).abs()),
        "decomposition_l1": _per_sample_mean(
            output.decomposition_residual.abs()
        ),
        "objective": output.objective,
        "iterations": output.iterations.to(output.image.dtype),
        "converged": output.converged.to(output.image.dtype),
    }
    if target_hr is not None:
        error = output.image - target_hr
        mse = _per_sample_mean(error.square())
        values["hr_l1"] = _per_sample_mean(error.abs())
        values["psnr"] = -10.0 * torch.log10(mse.clamp_min(1e-12))
    if output.variance is not None:
        values["gaussian_nll"] = _per_sample_mean(
            gaussian_nll(
                output.observation_residual,
                output.variance,
                reduction="none",
            )
        )
        standardized = output.observation_residual / output.variance.sqrt()
        values["standardized_mean"] = standardized.flatten(1).mean(dim=1)
        values["standardized_variance"] = standardized.flatten(1).var(
            dim=1,
            unbiased=False,
        )
    return values


class ConstraintStudy(nn.Module):
    """Apply every arm to one shared prediction and one shared observation."""

    def __init__(self, arms: nn.ModuleDict) -> None:
        super().__init__()
        self.arms = arms

    def forward(
        self,
        prediction: torch.Tensor,
        observed_lr: torch.Tensor,
        degradation: torch.Tensor,
        operator: LinearSensorOperator,
        *,
        oracle: OracleData = None,
        target_hr: torch.Tensor | None = None,
        selected_arms: list[str] | None = None,
    ) -> ConstraintStudyResult:
        names = selected_arms or list(self.arms.keys())
        unknown = sorted(set(names) - set(self.arms.keys()))
        if unknown:
            raise KeyError(f"unknown constraint arms: {unknown}")

        outputs: dict[str, ProximalOutput] = {}
        metrics: dict[str, dict[str, torch.Tensor]] = {}
        for name in names:
            arm: Any = self.arms[name]
            output = arm(
                prediction,
                observed_lr,
                degradation,
                operator,
                oracle,
            )
            outputs[name] = output
            metrics[name] = output_metrics(
                output,
                observed_lr,
                operator,
                target_hr=target_hr,
            )
        return ConstraintStudyResult(outputs=outputs, metrics=metrics)
