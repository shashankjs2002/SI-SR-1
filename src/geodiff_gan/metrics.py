from __future__ import annotations

import math
import sys
import sysconfig
import warnings
from importlib import metadata
from pathlib import Path

import torch
from torch.nn import functional as F

from .losses import charbonnier, ssim
from .models.degradation import sensor_degrade


def _masked_mean(value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return value.mean()
    mask = F.interpolate(mask.float(), size=value.shape[-2:], mode="nearest")
    if mask.shape[1] == 1 and value.shape[1] != 1:
        mask = mask.expand(-1, value.shape[1], -1, -1)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    mse = _masked_mean((prediction - target).square(), mask)
    return -10 * torch.log10(mse.clamp_min(1e-12))


def edge_f1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float | None = None,
    quantile: float = 0.9,
    tolerance: int = 1,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    def gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        dx = F.pad(gray[:, :, :, 1:] - gray[:, :, :, :-1], (0, 1, 0, 0))
        dy = F.pad(gray[:, :, 1:, :] - gray[:, :, :-1, :], (0, 0, 0, 1))
        return torch.sqrt(dx.square() + dy.square() + 1e-12)

    predicted_gradient = gradient_magnitude(prediction)
    target_gradient = gradient_magnitude(target)
    valid = (
        F.interpolate(mask.float(), size=target_gradient.shape[-2:], mode="nearest")
        if mask is not None
        else torch.ones_like(target_gradient)
    ).bool()
    if threshold is None:
        thresholds = []
        for sample_gradient, sample_valid in zip(target_gradient, valid):
            values = sample_gradient[sample_valid]
            thresholds.append(
                torch.quantile(values, quantile)
                if values.numel()
                else target_gradient.new_tensor(0.005)
            )
        threshold_tensor = torch.stack(thresholds).view(-1, 1, 1, 1)
        threshold_tensor = threshold_tensor.clamp_min(0.005)
    else:
        threshold_tensor = target_gradient.new_full(
            (target_gradient.shape[0], 1, 1, 1),
            threshold,
        )
    predicted = (predicted_gradient > threshold_tensor) & valid
    actual = (target_gradient > threshold_tensor) & valid
    kernel_size = tolerance * 2 + 1
    predicted_near = F.max_pool2d(
        predicted.float(),
        kernel_size,
        stride=1,
        padding=tolerance,
    ).bool()
    actual_near = F.max_pool2d(
        actual.float(),
        kernel_size,
        stride=1,
        padding=tolerance,
    ).bool()
    precision = (predicted & actual_near).sum().float() / predicted.sum().clamp_min(1)
    recall = (actual & predicted_near).sum().float() / actual.sum().clamp_min(1)
    return 2 * precision * recall / (precision + recall).clamp_min(1e-8)


def redegradation_error(
    prediction: torch.Tensor,
    lr: torch.Tensor,
    degradation: torch.Tensor,
    scale: int = 4,
    severity: str = "mild",
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return _masked_mean(
        (
            sensor_degrade(
                prediction,
                degradation,
                scale=scale,
                severity=severity,
            )
            - lr
        ).abs(),
        mask,
    )


def basic_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lr: torch.Tensor,
    degradation: torch.Tensor,
    scale: int = 4,
    severity: str = "mild",
    mask: torch.Tensor | None = None,
    lr_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    return {
        "l1": float(charbonnier(prediction, target, epsilon=0.0, mask=mask)),
        "psnr": float(psnr(prediction, target, mask=mask)),
        "ssim": float(ssim(prediction, target, mask=mask)),
        "edge_f1": float(edge_f1(prediction, target, mask=mask)),
        "redegradation_l1": float(
            redegradation_error(
                prediction,
                lr,
                degradation,
                scale,
                severity=severity,
                mask=lr_mask,
            )
        ),
    }


class OptionalMetricSuite:
    def __init__(self, device: torch.device, enabled: bool = True) -> None:
        self.lpips_model = None
        self.dists_model = None
        self.load_errors: dict[str, str] = {}
        if not enabled:
            return
        try:
            import lpips

            self.lpips_model = lpips.LPIPS(net="alex").to(device).eval()
        except Exception as error:
            self._record_load_error("lpips", error)
        try:
            self.dists_model = _load_dists_model(device)
        except Exception as error:
            self._record_load_error("dists", error)

    @property
    def available_metrics(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, model in (
                ("lpips", self.lpips_model),
                ("dists", self.dists_model),
            )
            if model is not None
        )

    def _record_load_error(self, name: str, error: Exception) -> None:
        message = f"{type(error).__name__}: {error}"
        self.load_errors[name] = message
        warnings.warn(
            f"Optional metric {name.upper()} is unavailable ({message}). "
            "Evaluation will continue with the remaining metrics.",
            RuntimeWarning,
            stacklevel=2,
        )

    @torch.no_grad()
    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        values: dict[str, float] = {}
        if mask is not None:
            prediction = prediction * mask + target * (1 - mask)
        if self.lpips_model is not None:
            normalized_prediction = prediction * 2 - 1
            normalized_target = target * 2 - 1
            values["lpips"] = float(
                self.lpips_model(normalized_prediction, normalized_target).mean()
            )
        if self.dists_model is not None:
            values["dists"] = float(
                self.dists_model(prediction, target).mean()
            )
        return values


def _load_dists_model(device: torch.device) -> torch.nn.Module:
    """Load DISTS despite the package's hard-coded ``sys.prefix`` weight path."""
    import DISTS_pytorch

    package_file = Path(DISTS_pytorch.__file__).resolve()
    candidates = [
        Path(sys.prefix) / "weights.pt",
        Path(sysconfig.get_path("data")) / "weights.pt",
        package_file.parent / "weights.pt",
    ]
    try:
        distribution = metadata.distribution("DISTS-pytorch")
    except metadata.PackageNotFoundError:
        distribution = None
    if distribution is not None:
        candidates.extend(
            Path(distribution.locate_file(file))
            for file in distribution.files or ()
            if Path(file).name == "weights.pt"
        )
    candidates = list(dict.fromkeys(candidates))
    weights_path = next((path for path in candidates if path.is_file()), None)
    if weights_path is None:
        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"DISTS weights.pt was not found; searched: {searched}")

    model = DISTS_pytorch.DISTS(load_weights=False)
    weights = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=True,
    )
    missing = {"alpha", "beta"}.difference(weights)
    if missing:
        raise KeyError(f"DISTS weights are missing keys: {sorted(missing)}")
    with torch.no_grad():
        model.alpha.copy_(weights["alpha"])
        model.beta.copy_(weights["beta"])
    return model.to(device).eval()
