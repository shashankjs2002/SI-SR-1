from __future__ import annotations

from collections.abc import Callable

import torch


def batch_inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError("inner-product tensors must have identical shapes")
    return (left * right).flatten(1).sum(dim=1)


def batch_norm(tensor: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
    return (batch_inner(tensor, tensor) + epsilon).sqrt()


def _expand_batch(values: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    return values.view(values.shape[0], *([1] * (tensor.ndim - 1)))


def conjugate_gradient(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    *,
    max_iterations: int = 50,
    relative_tolerance: float = 1e-6,
    epsilon: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable batched conjugate-gradient solve.

    Each batch member is treated as an independent linear system. The loop may
    terminate early only after all members satisfy the detached stopping test;
    tensor operations themselves remain in the autograd graph.
    """

    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    solution = torch.zeros_like(rhs)
    residual = rhs - matvec(solution)
    direction = residual
    residual_squared = batch_inner(residual, residual)
    initial_norm = residual_squared.clamp_min(0).sqrt()
    iterations = torch.zeros(
        rhs.shape[0],
        device=rhs.device,
        dtype=torch.long,
    )

    for iteration in range(max_iterations):
        product = matvec(direction)
        denominator = batch_inner(direction, product)
        safe_denominator = torch.where(
            denominator.abs() > epsilon,
            denominator,
            torch.full_like(denominator, epsilon),
        )
        alpha = residual_squared / safe_denominator
        solution = solution + _expand_batch(alpha, rhs) * direction
        residual = residual - _expand_batch(alpha, rhs) * product
        new_residual_squared = batch_inner(residual, residual).clamp_min(0)
        iterations = torch.full_like(iterations, iteration + 1)

        relative = new_residual_squared.sqrt() / initial_norm.clamp_min(epsilon)
        if bool(torch.all(relative.detach() <= relative_tolerance)):
            residual_squared = new_residual_squared
            break

        beta = new_residual_squared / residual_squared.clamp_min(epsilon)
        direction = residual + _expand_batch(beta, rhs) * direction
        residual_squared = new_residual_squared

    final_relative = residual_squared.sqrt() / initial_norm.clamp_min(epsilon)
    return solution, iterations, final_relative
