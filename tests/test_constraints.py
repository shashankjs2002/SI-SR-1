from __future__ import annotations

import torch

from geodiff_gan.constraints import (
    BlockAverageSensorOperator,
    GaussianMTFSensorOperator,
    HardLogisticProjection,
    LogisticProxLayer,
    SensorNoiseCalibrator,
    adjoint_relative_error,
    build_constraint_arms,
    calibration_training_loss,
    degradation_noise_variance,
    safe_logit,
)
from geodiff_gan.experiments import ConstraintStudy


def test_block_average_operator_has_exact_adjoint() -> None:
    torch.manual_seed(1)
    hr = torch.randn(2, 3, 8, 10, dtype=torch.float64)
    operator = BlockAverageSensorOperator(
        2,
        scale=2,
        dtype=torch.float64,
    )
    lr = torch.randn_like(operator.forward(hr))
    error = adjoint_relative_error(operator, hr, lr)
    assert torch.max(error) < 1e-12


def test_gaussian_mtf_operator_has_exact_adjoint() -> None:
    torch.manual_seed(2)
    degradation = torch.rand(2, 4, dtype=torch.float64)
    operator = GaussianMTFSensorOperator(
        degradation,
        scale=2,
        kernel_size=5,
    )
    hr = torch.randn(2, 2, 12, 14, dtype=torch.float64)
    lr = torch.randn_like(operator.forward(hr))
    error = adjoint_relative_error(operator, hr, lr)
    assert torch.max(error) < 1e-12


def test_logistic_prox_reduces_decomposition_residual() -> None:
    torch.manual_seed(3)
    operator = BlockAverageSensorOperator(1, scale=2)
    prediction = torch.rand(1, 1, 8, 8) * 0.6 + 0.2
    target = torch.rand(1, 1, 8, 8) * 0.6 + 0.2
    observed = operator.forward(target)
    logits = safe_logit(prediction)
    variance = torch.full_like(observed, 5e-4)
    layer = LogisticProxLayer(
        max_newton_iterations=15,
        max_cg_iterations=30,
        relative_tolerance=1e-6,
    )
    initial = (operator.forward(prediction) - observed).norm()
    output = layer(logits, observed, variance, operator)
    final = output.decomposition_residual.norm()
    assert torch.isfinite(output.image).all()
    assert output.image.min() > 0
    assert output.image.max() < 1
    assert final < initial * 1e-3
    assert bool(output.converged.all())


def test_hard_logistic_projection_enforces_block_average() -> None:
    torch.manual_seed(4)
    operator = BlockAverageSensorOperator(1, scale=2, dtype=torch.float64)
    prediction = torch.rand(1, 1, 6, 6, dtype=torch.float64) * 0.5 + 0.25
    target = torch.rand(1, 1, 6, 6, dtype=torch.float64) * 0.5 + 0.25
    observed = operator.forward(target)
    layer = HardLogisticProjection(
        max_newton_iterations=20,
        max_cg_iterations=40,
        relative_tolerance=1e-7,
    )
    output = layer(safe_logit(prediction), observed, operator)
    assert torch.max((operator.forward(output.image) - observed).abs()) < 2e-7
    assert bool(output.converged.all())


def test_unrolled_logistic_prox_propagates_finite_gradients() -> None:
    torch.manual_seed(5)
    operator = BlockAverageSensorOperator(1, scale=2, dtype=torch.float64)
    logits = torch.randn(
        1,
        1,
        4,
        4,
        dtype=torch.float64,
        requires_grad=True,
    )
    target = torch.sigmoid(torch.randn_like(logits))
    observed = operator.forward(target).detach()
    variance = torch.full_like(observed, 1e-2)
    layer = LogisticProxLayer(
        max_newton_iterations=10,
        max_cg_iterations=20,
        relative_tolerance=1e-8,
    )
    loss = layer(logits, observed, variance, operator).image.square().mean()
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0


def test_degradation_variance_is_positive_and_signal_dependent() -> None:
    degradation = torch.tensor([[0.4, 0.5, 0.5, 0.5]])
    dark = torch.full((1, 3, 4, 4), 0.1)
    bright = torch.full((1, 3, 4, 4), 0.8)
    dark_variance = degradation_noise_variance(dark, degradation)
    bright_variance = degradation_noise_variance(bright, degradation)
    assert torch.all(dark_variance > 0)
    assert torch.all(bright_variance > dark_variance)


def test_sensor_calibrator_is_bounded_and_trainable() -> None:
    torch.manual_seed(6)
    calibrator = SensorNoiseCalibrator(channels=3, hidden_channels=8)
    observed = torch.rand(2, 3, 8, 8)
    clean = observed * 0.98
    degradation = torch.rand(2, 4)
    output = calibrator(observed, degradation)
    assert output.variance.shape == observed.shape
    assert output.alpha.shape == (2, 3)
    assert output.beta.shape == (2, 3)
    assert torch.all(output.variance >= calibrator.variance_floor)
    assert torch.all(output.variance <= calibrator.variance_ceiling)
    losses = calibration_training_loss(output, observed, clean)
    losses["total"].backward()
    gradients = [
        parameter.grad
        for parameter in calibrator.parameters()
        if parameter.requires_grad
    ]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_six_arm_study_uses_shared_shapes_and_finite_outputs() -> None:
    torch.manual_seed(7)
    prediction = torch.rand(1, 3, 8, 8) * 0.5 + 0.25
    target = torch.rand(1, 3, 8, 8) * 0.5 + 0.25
    degradation = torch.full((1, 4), 0.5)
    operator = BlockAverageSensorOperator(1, scale=2)
    clean_lr = operator.forward(target)
    observed = clean_lr
    logistic = LogisticProxLayer(
        max_newton_iterations=12,
        max_cg_iterations=24,
        relative_tolerance=1e-5,
    )
    hard = HardLogisticProjection(
        max_newton_iterations=15,
        max_cg_iterations=30,
        relative_tolerance=1e-6,
    )
    arms = build_constraint_arms(
        channels=3,
        calibrator=SensorNoiseCalibrator(channels=3, hidden_channels=8),
        logistic_layer=logistic,
        hard_layer=hard,
    )
    result = ConstraintStudy(arms)(
        prediction,
        observed,
        degradation,
        operator,
        oracle={"clean_lr": clean_lr},
        target_hr=target,
    )
    assert set(result.outputs) == {
        "soft",
        "euclidean",
        "glinsat_hard",
        "logistic_fixed",
        "logistic_oracle",
        "sensorcal_logistic",
    }
    for name, output in result.outputs.items():
        assert output.image.shape == prediction.shape, name
        assert output.observation_residual.shape == observed.shape, name
        assert torch.isfinite(output.image).all(), name
        assert result.metrics[name]["hr_l1"].shape == (1,), name
