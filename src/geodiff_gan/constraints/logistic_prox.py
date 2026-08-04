from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .linear_solve import batch_inner, batch_norm, conjugate_gradient
from .operator import LinearSensorOperator
from .outputs import ProximalOutput


def safe_logit(image: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must lie in (0, 0.5)")
    return torch.logit(image.clamp(epsilon, 1.0 - epsilon))


def _expand(values: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    return values.view(values.shape[0], *([1] * (tensor.ndim - 1)))


def _dual_objective(
    dual: torch.Tensor,
    logits: torch.Tensor,
    observed_lr: torch.Tensor,
    variance: torch.Tensor,
    operator: LinearSensorOperator,
) -> torch.Tensor:
    shifted_logits = logits + operator.adjoint(dual, logits.shape[-2:])
    entropy_term = F.softplus(shifted_logits).flatten(1).sum(dim=1)
    noise_term = 0.5 * (variance * dual.square()).flatten(1).sum(dim=1)
    observation_term = (observed_lr * dual).flatten(1).sum(dim=1)
    return entropy_term + noise_term - observation_term


class LogisticProxLayer(nn.Module):
    """Noise-aware Bernoulli logistic-proximal output layer.

    The implementation differentiates through the unrolled Newton-CG solve.
    An implicit custom backward is intentionally deferred until finite-
    difference validation is complete.
    """

    def __init__(
        self,
        *,
        max_newton_iterations: int = 20,
        max_cg_iterations: int = 50,
        relative_tolerance: float = 1e-5,
        cg_relative_tolerance: float = 1e-6,
        variance_floor: float = 1e-7,
        logit_epsilon: float = 1e-5,
        cg_damping: float = 1e-8,
        max_line_search_iterations: int = 8,
    ) -> None:
        super().__init__()
        if max_newton_iterations < 1:
            raise ValueError("max_newton_iterations must be positive")
        if max_cg_iterations < 1:
            raise ValueError("max_cg_iterations must be positive")
        if relative_tolerance <= 0 or cg_relative_tolerance <= 0:
            raise ValueError("solver tolerances must be positive")
        if variance_floor <= 0:
            raise ValueError("variance_floor must be positive")
        self.max_newton_iterations = int(max_newton_iterations)
        self.max_cg_iterations = int(max_cg_iterations)
        self.relative_tolerance = float(relative_tolerance)
        self.cg_relative_tolerance = float(cg_relative_tolerance)
        self.variance_floor = float(variance_floor)
        self.logit_epsilon = float(logit_epsilon)
        self.cg_damping = float(cg_damping)
        self.max_line_search_iterations = int(max_line_search_iterations)

    def forward(
        self,
        logits: torch.Tensor,
        observed_lr: torch.Tensor,
        variance: torch.Tensor,
        operator: LinearSensorOperator,
        warm_start: torch.Tensor | None = None,
    ) -> ProximalOutput:
        if logits.ndim != 4 or observed_lr.ndim != 4:
            raise ValueError("logits and observed_lr must be BCHW tensors")
        if variance.shape != observed_lr.shape:
            try:
                variance = torch.broadcast_to(variance, observed_lr.shape)
            except RuntimeError as error:
                raise ValueError(
                    "variance must match or broadcast to observed_lr"
                ) from error
        variance = variance.clamp_min(self.variance_floor)
        if warm_start is None:
            dual = torch.zeros_like(observed_lr)
        else:
            if warm_start.shape != observed_lr.shape:
                raise ValueError("warm_start must match observed_lr")
            dual = warm_start

        observation_norm = batch_norm(observed_lr).clamp_min(1e-12)
        initial_relative = None
        iteration_counts = torch.zeros(
            logits.shape[0],
            device=logits.device,
            dtype=torch.long,
        )

        for iteration in range(self.max_newton_iterations):
            shifted_logits = logits + operator.adjoint(dual, logits.shape[-2:])
            image = torch.sigmoid(shifted_logits)
            gradient = operator.forward(image) + variance * dual - observed_lr
            relative_gradient = batch_norm(gradient) / observation_norm
            if initial_relative is None:
                initial_relative = relative_gradient
            active = relative_gradient.detach() > self.relative_tolerance
            if not bool(torch.any(active)):
                break

            sigmoid_derivative = image * (1.0 - image)

            def hessian_vector(vector: torch.Tensor) -> torch.Tensor:
                lifted = operator.adjoint(vector, logits.shape[-2:])
                result = operator.forward(sigmoid_derivative * lifted)
                result = result + variance * vector
                if self.cg_damping:
                    result = result + self.cg_damping * vector
                return result

            step_direction, _, _ = conjugate_gradient(
                hessian_vector,
                -gradient,
                max_iterations=self.max_cg_iterations,
                relative_tolerance=self.cg_relative_tolerance,
            )
            current_objective = _dual_objective(
                dual,
                logits,
                observed_lr,
                variance,
                operator,
            )
            directional_derivative = batch_inner(gradient, step_direction)
            step_size = torch.ones_like(current_objective)

            for _ in range(self.max_line_search_iterations):
                candidate = dual + _expand(step_size, dual) * step_direction
                candidate_objective = _dual_objective(
                    candidate,
                    logits,
                    observed_lr,
                    variance,
                    operator,
                )
                armijo_limit = (
                    current_objective
                    + 1e-4 * step_size * directional_derivative
                )
                accepted = candidate_objective.detach() <= armijo_limit.detach()
                if bool(torch.all(accepted | ~active)):
                    break
                step_size = torch.where(
                    accepted | ~active,
                    step_size,
                    step_size * 0.5,
                )

            candidate = dual + _expand(step_size, dual) * step_direction
            dual = torch.where(_expand(active, dual), candidate, dual)
            iteration_counts = torch.where(
                active,
                torch.full_like(iteration_counts, iteration + 1),
                iteration_counts,
            )

        shifted_logits = logits + operator.adjoint(dual, logits.shape[-2:])
        image = torch.sigmoid(shifted_logits)
        projected_lr = operator.forward(image)
        observation_residual = observed_lr - projected_lr
        decomposition_residual = projected_lr + variance * dual - observed_lr
        relative_decomposition = batch_norm(decomposition_residual) / observation_norm
        converged = relative_decomposition <= self.relative_tolerance
        objective = _dual_objective(
            dual,
            logits,
            observed_lr,
            variance,
            operator,
        )
        saturation = (
            ((image < self.logit_epsilon) | (image > 1.0 - self.logit_epsilon))
            .float()
            .flatten(1)
            .mean(dim=1)
        )
        if initial_relative is None:
            initial_relative = relative_decomposition

        return ProximalOutput(
            image=image,
            dual=dual,
            observation_residual=observation_residual,
            decomposition_residual=decomposition_residual,
            iterations=iteration_counts,
            converged=converged,
            objective=objective,
            variance=variance,
            diagnostics={
                "initial_relative_residual": initial_relative,
                "final_relative_residual": relative_decomposition,
                "sigmoid_saturation_fraction": saturation,
            },
        )


class HardLogisticProjection(nn.Module):
    """Exact-equality GLinSAT-class Bernoulli projection baseline.

    This class is separate from ``LogisticProxLayer`` because a tiny variance is
    not numerically or scientifically identical to the hard problem.
    """

    def __init__(
        self,
        *,
        max_newton_iterations: int = 30,
        max_cg_iterations: int = 80,
        relative_tolerance: float = 1e-6,
        cg_relative_tolerance: float = 1e-7,
        logit_epsilon: float = 1e-5,
        cg_damping: float = 1e-10,
        max_line_search_iterations: int = 10,
    ) -> None:
        super().__init__()
        self.max_newton_iterations = int(max_newton_iterations)
        self.max_cg_iterations = int(max_cg_iterations)
        self.relative_tolerance = float(relative_tolerance)
        self.cg_relative_tolerance = float(cg_relative_tolerance)
        self.logit_epsilon = float(logit_epsilon)
        self.cg_damping = float(cg_damping)
        self.max_line_search_iterations = int(max_line_search_iterations)

    def forward(
        self,
        logits: torch.Tensor,
        observed_lr: torch.Tensor,
        operator: LinearSensorOperator,
        warm_start: torch.Tensor | None = None,
    ) -> ProximalOutput:
        zero_variance = torch.zeros_like(observed_lr)
        dual = torch.zeros_like(observed_lr) if warm_start is None else warm_start
        observation_norm = batch_norm(observed_lr).clamp_min(1e-12)
        initial_relative = None
        iteration_counts = torch.zeros(
            logits.shape[0],
            device=logits.device,
            dtype=torch.long,
        )

        for iteration in range(self.max_newton_iterations):
            shifted_logits = logits + operator.adjoint(dual, logits.shape[-2:])
            image = torch.sigmoid(shifted_logits)
            gradient = operator.forward(image) - observed_lr
            relative_gradient = batch_norm(gradient) / observation_norm
            if initial_relative is None:
                initial_relative = relative_gradient
            active = relative_gradient.detach() > self.relative_tolerance
            if not bool(torch.any(active)):
                break
            sigmoid_derivative = image * (1.0 - image)

            def hessian_vector(vector: torch.Tensor) -> torch.Tensor:
                lifted = operator.adjoint(vector, logits.shape[-2:])
                result = operator.forward(sigmoid_derivative * lifted)
                if self.cg_damping:
                    result = result + self.cg_damping * vector
                return result

            direction, _, _ = conjugate_gradient(
                hessian_vector,
                -gradient,
                max_iterations=self.max_cg_iterations,
                relative_tolerance=self.cg_relative_tolerance,
            )
            current_objective = _dual_objective(
                dual,
                logits,
                observed_lr,
                zero_variance,
                operator,
            )
            directional_derivative = batch_inner(gradient, direction)
            step_size = torch.ones_like(current_objective)
            for _ in range(self.max_line_search_iterations):
                candidate = dual + _expand(step_size, dual) * direction
                candidate_objective = _dual_objective(
                    candidate,
                    logits,
                    observed_lr,
                    zero_variance,
                    operator,
                )
                limit = current_objective + 1e-4 * step_size * directional_derivative
                accepted = candidate_objective.detach() <= limit.detach()
                if bool(torch.all(accepted | ~active)):
                    break
                step_size = torch.where(
                    accepted | ~active,
                    step_size,
                    step_size * 0.5,
                )
            candidate = dual + _expand(step_size, dual) * direction
            dual = torch.where(_expand(active, dual), candidate, dual)
            iteration_counts = torch.where(
                active,
                torch.full_like(iteration_counts, iteration + 1),
                iteration_counts,
            )

        image = torch.sigmoid(
            logits + operator.adjoint(dual, logits.shape[-2:])
        )
        projected_lr = operator.forward(image)
        residual = projected_lr - observed_lr
        relative = batch_norm(residual) / observation_norm
        saturation = (
            ((image < self.logit_epsilon) | (image > 1.0 - self.logit_epsilon))
            .float()
            .flatten(1)
            .mean(dim=1)
        )
        if initial_relative is None:
            initial_relative = relative
        return ProximalOutput(
            image=image,
            dual=dual,
            observation_residual=observed_lr - projected_lr,
            decomposition_residual=residual,
            iterations=iteration_counts,
            converged=relative <= self.relative_tolerance,
            objective=_dual_objective(
                dual,
                logits,
                observed_lr,
                zero_variance,
                operator,
            ),
            variance=None,
            diagnostics={
                "initial_relative_residual": initial_relative,
                "final_relative_residual": relative,
                "sigmoid_saturation_fraction": saturation,
            },
        )
