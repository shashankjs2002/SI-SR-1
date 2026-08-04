from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .euclidean_prox import EuclideanProxLayer
from .linear_solve import batch_norm
from .logistic_prox import (
    HardLogisticProjection,
    LogisticProxLayer,
    safe_logit,
)
from .noise_calibration import (
    SensorNoiseCalibrator,
    degradation_noise_variance,
)
from .operator import LinearSensorOperator
from .outputs import ProximalOutput


OracleData = Mapping[str, torch.Tensor] | None


def _identity_output(
    prediction: torch.Tensor,
    observed_lr: torch.Tensor,
    operator: LinearSensorOperator,
) -> ProximalOutput:
    projected_lr = operator.forward(prediction)
    residual = observed_lr - projected_lr
    zeros = torch.zeros_like(observed_lr)
    batch = prediction.shape[0]
    return ProximalOutput(
        image=prediction,
        dual=zeros,
        observation_residual=residual,
        decomposition_residual=-residual,
        iterations=torch.zeros(batch, device=prediction.device, dtype=torch.long),
        converged=torch.ones(batch, device=prediction.device, dtype=torch.bool),
        objective=0.5 * residual.square().flatten(1).sum(dim=1),
        variance=None,
        diagnostics={
            "initial_relative_residual": batch_norm(residual)
            / batch_norm(observed_lr).clamp_min(1e-12),
            "final_relative_residual": batch_norm(residual)
            / batch_norm(observed_lr).clamp_min(1e-12),
        },
    )


def _oracle_variance(
    observed_lr: torch.Tensor,
    degradation: torch.Tensor,
    oracle: OracleData,
    *,
    severity: str,
) -> torch.Tensor:
    if oracle is None:
        raise ValueError("oracle data is required for the oracle variance arm")
    if "variance" in oracle:
        return oracle["variance"].to(
            device=observed_lr.device,
            dtype=observed_lr.dtype,
        )
    if "clean_lr" in oracle:
        clean_lr = oracle["clean_lr"].to(
            device=observed_lr.device,
            dtype=observed_lr.dtype,
        )
        return degradation_noise_variance(
            clean_lr,
            degradation,
            severity=severity,
        )
    raise ValueError("oracle data must contain 'variance' or 'clean_lr'")


class ConstraintArm(nn.Module):
    name: str

    def forward(
        self,
        prediction: torch.Tensor,
        observed_lr: torch.Tensor,
        degradation: torch.Tensor,
        operator: LinearSensorOperator,
        oracle: OracleData = None,
    ) -> ProximalOutput:
        raise NotImplementedError


class SoftConsistencyArm(ConstraintArm):
    name = "soft"

    def forward(
        self,
        prediction: torch.Tensor,
        observed_lr: torch.Tensor,
        degradation: torch.Tensor,
        operator: LinearSensorOperator,
        oracle: OracleData = None,
    ) -> ProximalOutput:
        del degradation, oracle
        return _identity_output(prediction, observed_lr, operator)


class EuclideanProxArm(ConstraintArm):
    name = "euclidean"

    def __init__(
        self,
        *,
        severity: str = "mild",
        layer: EuclideanProxLayer | None = None,
    ) -> None:
        super().__init__()
        self.severity = severity
        self.layer = layer or EuclideanProxLayer()

    def forward(
        self,
        prediction: torch.Tensor,
        observed_lr: torch.Tensor,
        degradation: torch.Tensor,
        operator: LinearSensorOperator,
        oracle: OracleData = None,
    ) -> ProximalOutput:
        del oracle
        variance = degradation_noise_variance(
            observed_lr,
            degradation,
            severity=self.severity,
        )
        return self.layer(prediction, observed_lr, variance, operator)


class GLinSATHardArm(ConstraintArm):
    name = "glinsat_hard"

    def __init__(
        self,
        *,
        layer: HardLogisticProjection | None = None,
        logit_epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.layer = layer or HardLogisticProjection()
        self.logit_epsilon = float(logit_epsilon)

    def forward(
        self,
        prediction: torch.Tensor,
        observed_lr: torch.Tensor,
        degradation: torch.Tensor,
        operator: LinearSensorOperator,
        oracle: OracleData = None,
    ) -> ProximalOutput:
        del degradation, oracle
        return self.layer(
            safe_logit(prediction, self.logit_epsilon),
            observed_lr,
            operator,
        )


class LogisticProxArm(ConstraintArm):
    def __init__(
        self,
        *,
        variance_source: str,
        severity: str = "mild",
        layer: LogisticProxLayer | None = None,
        calibrator: SensorNoiseCalibrator | None = None,
        logit_epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        if variance_source not in {"fixed", "oracle", "sensorcal"}:
            raise ValueError("variance_source must be fixed, oracle, or sensorcal")
        if variance_source == "sensorcal" and calibrator is None:
            raise ValueError("sensorcal variance requires a calibrator")
        self.variance_source = variance_source
        self.name = f"logistic_{variance_source}"
        self.severity = severity
        self.layer = layer or LogisticProxLayer()
        self.calibrator = calibrator
        self.logit_epsilon = float(logit_epsilon)

    def _variance(
        self,
        observed_lr: torch.Tensor,
        degradation: torch.Tensor,
        oracle: OracleData,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.variance_source == "fixed":
            variance = degradation_noise_variance(
                observed_lr,
                degradation,
                severity=self.severity,
            )
            return variance, {}
        if self.variance_source == "oracle":
            return (
                _oracle_variance(
                    observed_lr,
                    degradation,
                    oracle,
                    severity=self.severity,
                ),
                {},
            )
        assert self.calibrator is not None
        calibration = self.calibrator(observed_lr, degradation)
        return calibration.variance, {
            "calibration_alpha": calibration.alpha,
            "calibration_beta": calibration.beta,
            "calibration_mean_lr": calibration.mean_lr,
        }

    def forward(
        self,
        prediction: torch.Tensor,
        observed_lr: torch.Tensor,
        degradation: torch.Tensor,
        operator: LinearSensorOperator,
        oracle: OracleData = None,
    ) -> ProximalOutput:
        variance, calibration_diagnostics = self._variance(
            observed_lr,
            degradation,
            oracle,
        )
        output = self.layer(
            safe_logit(prediction, self.logit_epsilon),
            observed_lr,
            variance,
            operator,
        )
        output.diagnostics.update(calibration_diagnostics)
        return output


def build_constraint_arms(
    *,
    channels: int = 3,
    severity: str = "mild",
    calibrator: SensorNoiseCalibrator | None = None,
    logistic_layer: LogisticProxLayer | None = None,
    hard_layer: HardLogisticProjection | None = None,
    euclidean_layer: EuclideanProxLayer | None = None,
) -> nn.ModuleDict:
    calibrator = calibrator or SensorNoiseCalibrator(channels=channels)
    logistic_layer = logistic_layer or LogisticProxLayer()
    return nn.ModuleDict(
        {
            "soft": SoftConsistencyArm(),
            "euclidean": EuclideanProxArm(
                severity=severity,
                layer=euclidean_layer,
            ),
            "glinsat_hard": GLinSATHardArm(layer=hard_layer),
            "logistic_fixed": LogisticProxArm(
                variance_source="fixed",
                severity=severity,
                layer=logistic_layer,
            ),
            "logistic_oracle": LogisticProxArm(
                variance_source="oracle",
                severity=severity,
                layer=logistic_layer,
            ),
            "sensorcal_logistic": LogisticProxArm(
                variance_source="sensorcal",
                severity=severity,
                layer=logistic_layer,
                calibrator=calibrator,
            ),
        }
    )
