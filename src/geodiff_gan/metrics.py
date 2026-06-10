from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from .losses import ssim
from .models.degradation import sensor_degrade


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(prediction, target)
    return -10 * torch.log10(mse.clamp_min(1e-12))


def edge_f1(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.08) -> torch.Tensor:
    def edges(x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        dx = F.pad(gray[:, :, :, 1:] - gray[:, :, :, :-1], (0, 1, 0, 0))
        dy = F.pad(gray[:, :, 1:, :] - gray[:, :, :-1, :], (0, 0, 0, 1))
        return torch.sqrt(dx.square() + dy.square() + 1e-12) > threshold

    predicted = edges(prediction)
    actual = edges(target)
    true_positive = (predicted & actual).sum().float()
    precision = true_positive / predicted.sum().clamp_min(1)
    recall = true_positive / actual.sum().clamp_min(1)
    return 2 * precision * recall / (precision + recall).clamp_min(1e-8)


def redegradation_error(
    prediction: torch.Tensor,
    lr: torch.Tensor,
    degradation: torch.Tensor,
    scale: int = 4,
) -> torch.Tensor:
    return F.l1_loss(sensor_degrade(prediction, degradation, scale=scale), lr)


def basic_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lr: torch.Tensor,
    degradation: torch.Tensor,
    scale: int = 4,
) -> dict[str, float]:
    return {
        "psnr": float(psnr(prediction, target)),
        "ssim": float(ssim(prediction, target)),
        "edge_f1": float(edge_f1(prediction, target)),
        "redegradation_l1": float(redegradation_error(prediction, lr, degradation, scale)),
    }


class OptionalMetricSuite:
    def __init__(self, device: torch.device) -> None:
        self.lpips_model = None
        self.dists_model = None
        try:
            import lpips

            self.lpips_model = lpips.LPIPS(net="alex").to(device).eval()
        except ImportError:
            pass
        try:
            from DISTS_pytorch import DISTS

            self.dists_model = DISTS().to(device).eval()
        except ImportError:
            pass

    @torch.no_grad()
    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        values: dict[str, float] = {}
        normalized_prediction = prediction * 2 - 1
        normalized_target = target * 2 - 1
        if self.lpips_model is not None:
            values["lpips"] = float(
                self.lpips_model(normalized_prediction, normalized_target).mean()
            )
        if self.dists_model is not None:
            values["dists"] = float(
                self.dists_model(normalized_prediction, normalized_target).mean()
            )
        return values
