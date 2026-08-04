from __future__ import annotations

import torch
from torch import nn

from .linear_solve import batch_norm, conjugate_gradient
from .operator import LinearSensorOperator
from .outputs import ProximalOutput


class EuclideanProxLayer(nn.Module):
    """Gaussian Euclidean proximal baseline in image space."""

    def __init__(
        self,
        *,
        max_cg_iterations: int = 80,
        relative_tolerance: float = 1e-6,
        variance_floor: float = 1e-7,
        cg_damping: float = 1e-10,
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        self.max_cg_iterations = int(max_cg_iterations)
        self.relative_tolerance = float(relative_tolerance)
        self.variance_floor = float(variance_floor)
        self.cg_damping = float(cg_damping)
        self.clamp_output = bool(clamp_output)

    def forward(
        self,
        prediction: torch.Tensor,
        observed_lr: torch.Tensor,
        variance: torch.Tensor,
        operator: LinearSensorOperator,
    ) -> ProximalOutput:
        if variance.shape != observed_lr.shape:
            variance = torch.broadcast_to(variance, observed_lr.shape)
        variance = variance.clamp_min(self.variance_floor)
        initial_lr = operator.forward(prediction)
        rhs = observed_lr - initial_lr

        def normal_operator(vector: torch.Tensor) -> torch.Tensor:
            lifted = operator.adjoint(vector, prediction.shape[-2:])
            result = operator.forward(lifted) + variance * vector
            if self.cg_damping:
                result = result + self.cg_damping * vector
            return result

        dual, iterations, cg_relative = conjugate_gradient(
            normal_operator,
            rhs,
            max_iterations=self.max_cg_iterations,
            relative_tolerance=self.relative_tolerance,
        )
        unconstrained = prediction + operator.adjoint(
            dual,
            prediction.shape[-2:],
        )
        image = unconstrained.clamp(0, 1) if self.clamp_output else unconstrained
        projected_lr = operator.forward(image)
        observation_residual = observed_lr - projected_lr
        decomposition_residual = projected_lr + variance * dual - observed_lr
        observation_norm = batch_norm(observed_lr).clamp_min(1e-12)
        relative = batch_norm(decomposition_residual) / observation_norm
        clipped_fraction = (
            ((unconstrained < 0) | (unconstrained > 1))
            .float()
            .flatten(1)
            .mean(dim=1)
        )
        objective = (
            0.5 * (image - prediction).square().flatten(1).sum(dim=1)
            + 0.5
            * (
                observation_residual.square() / variance
            ).flatten(1).sum(dim=1)
        )
        return ProximalOutput(
            image=image,
            dual=dual,
            observation_residual=observation_residual,
            decomposition_residual=decomposition_residual,
            iterations=iterations,
            converged=relative <= self.relative_tolerance,
            objective=objective,
            variance=variance,
            diagnostics={
                "initial_relative_residual": batch_norm(rhs) / observation_norm,
                "final_relative_residual": relative,
                "cg_relative_residual": cg_relative,
                "clipped_fraction": clipped_fraction,
            },
        )
