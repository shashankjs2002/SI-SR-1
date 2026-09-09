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


@torch.no_grad()
def tile_risk_target(output, target, mask, tile_size, scale, kind="error"):
    """Train-only labels: observable base error or realized fractional proposal benefit.

    A bypassed tile has no observed proposal. It must not become a false zero-gain
    label. Full-coverage warm-up and exploratory dispatch supply those observations.
    """
    if kind not in ("error", "gain"):
        raise ValueError("risk_target must be error or gain")
    base = output.base.detach().float()
    size = tile_size * scale
    pad = (0, (-target.shape[-1]) % size, 0, (-target.shape[-2]) % size)
    def pool(value):
        return F.avg_pool2d(F.pad(value, pad), size, size)
    valid = pool(mask.float())
    base_mse = pool((base - target).square().mean(1, keepdim=True) * mask) / valid.clamp_min(1e-8)
    observed = valid.flatten(1) > 0
    if kind == "error":
        return base_mse.clamp_min(1e-8).log10().flatten(1), observed
    proposal = (base + output.residual.detach().float()).clamp(0, 1)
    proposal_mse = pool((proposal - target).square().mean(1, keepdim=True) * mask) / valid.clamp_min(1e-8)
    benefit = ((base_mse - proposal_mse) / (base_mse + 1e-6)).clamp(-1, 1)
    observed &= pool(output.active.float() * mask).flatten(1) > 0
    return benefit.flatten(1), observed


def trust_moe_losses(output, target, mask, tile_size, scale, weights, use_trust=True,
                     risk_target="error"):
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
        "proposal_mse": mse_loss(residual, target - base, mask=active),
    }
    label = local_trust_target(residual.detach(), target - base, mask)
    losses["trust"] = (_masked_mean((output.trust.float() - label).square(), active)
                       if use_trust else image.new_zeros(()))
    risk_label, valid_tokens = tile_risk_target(output, target, mask, tile_size, scale, risk_target)
    losses["risk"] = (F.smooth_l1_loss(output.difficulty.float()[valid_tokens], risk_label[valid_tokens])
                       if valid_tokens.any() else output.difficulty.float().sum() * 0)
    probabilities = output.probabilities.float()
    load = output.assignments.detach().sum((0, 1))
    load = load / load.sum().clamp_min(1)
    importance = (probabilities[valid_tokens].mean(0) if valid_tokens.any()
                  else probabilities.mean((0, 1)))
    losses["balance"] = probabilities.shape[-1] * (load * importance).sum()
    total = sum(float(weights.get(key, 0)) * value for key, value in losses.items())
    return total, losses
