"""Fidelity and routing objectives for direct residual expert training."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from ..losses import (_masked_mean, charbonnier, gradient_loss, mse_loss,
                      radiometric_loss, ssim, wavelet_loss)


@torch.no_grad()
def local_trust_target(residual, target_error, mask, window=9, ridge=1e-8):
    """Per-band least-squares acceptance; HR is used only as a training label."""
    def pooled(x):
        return F.avg_pool2d(x * mask, window, 1, window // 2)
    return (pooled(residual * target_error) /
            (pooled(residual.square()) + ridge)).clamp(0, 1)


def trust_moe_losses(output, target, mask, tile_size, scale, weights, use_trust=True):
    image, base = output.image.float(), output.base.detach().float()
    residual, mask = output.residual.float(), mask.float()
    active = output.active.float() * mask
    error = (image - target).square()
    base_error = (base - target).square()
    local_excess = F.avg_pool2d((error - base_error) * mask, 9, 1, 4)
    losses = {
        "mse": mse_loss(image, target, mask),
        "charbonnier": charbonnier(image, target, mask=mask),
        "ssim": 1 - ssim(image, target, mask=mask),
        "gradient": gradient_loss(image, target, mask=mask),
        "wavelet": wavelet_loss(image, target, mask=mask),
        "radiometric": radiometric_loss(image, target, mask=mask),
        "guard": _masked_mean(F.relu(local_excess), mask),
        "proposal": charbonnier(base + residual, target, mask=active),
    }
    label = local_trust_target(residual.detach(), target - base, mask)
    losses["trust"] = (_masked_mean((output.trust.float() - label).square(), active)
                       if use_trust else image.new_zeros(()))
    # Masked log-MSE per dispatch tile: harder tiles receive a larger target.
    size = tile_size * scale
    pad = (0, (-target.shape[-1]) % size, 0, (-target.shape[-2]) % size)
    valid = F.avg_pool2d(F.pad(mask, pad), size, size)
    risk_target = F.avg_pool2d(F.pad(base_error.mean(1, keepdim=True) * mask, pad), size, size)
    risk_target = (risk_target / valid.clamp_min(1e-8)).clamp_min(1e-8).log10().flatten(1)
    valid_tokens = valid.flatten(1) > 0
    losses["risk"] = F.smooth_l1_loss(output.difficulty.float()[valid_tokens], risk_target[valid_tokens])
    probabilities = output.probabilities.float()
    load = output.assignments.detach().sum((0, 1))
    load = load / load.sum().clamp_min(1)
    importance = probabilities[valid_tokens].mean(0)
    losses["balance"] = probabilities.shape[-1] * (load * importance).sum()
    total = sum(float(weights.get(key, 0)) * value for key, value in losses.items())
    return total, losses
