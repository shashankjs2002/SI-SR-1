"""Image-level routing over small velocity heads, not over the Gaussian noise process."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import SinusoidalEmbedding
from .diffusion import ConditionalDiffusionUNet


class ExpertRouter(nn.Module):
    def __init__(self, feature_channels: int, evidence_channels: int, experts: int, hidden: int):
        super().__init__()
        self.time = SinusoidalEmbedding(hidden)
        self.net = nn.Sequential(nn.Linear(feature_channels + evidence_channels + hidden, hidden), nn.SiLU())
        self.selection = nn.Linear(hidden, experts)
        self.quality = nn.Linear(hidden, experts)
        nn.init.zeros_(self.selection.weight)
        nn.init.zeros_(self.selection.bias)
        nn.init.zeros_(self.quality.weight)
        nn.init.constant_(self.quality.bias, 2.0)

    def forward(self, features, t, evidence):
        x = torch.cat((features.mean((-2, -1)), evidence.mean((-2, -1)), self.time(t)), 1)
        x = self.net(x)
        return self.selection(x), self.quality(x)


class MixtureDiffusionUNet(ConditionalDiffusionUNet):
    """Shared trunk + velocity adapters. All heads train; only routed heads run at inference."""
    def __init__(self, *, num_experts=2, top_k=1, routing_mode="generic",
                 expert_channels=24, evidence_channels=9, **kwargs):
        if not isinstance(num_experts, int) or not 1 <= top_k <= num_experts:
            raise ValueError("Require integer 1 <= top_k <= num_experts")
        if routing_mode not in ("generic", "reliability"):
            raise ValueError("routing_mode must be generic or reliability")
        super().__init__(**kwargs)
        self.num_experts, self.top_k = num_experts, top_k
        self.routing_mode, self.evidence_channels = routing_mode, evidence_channels
        self.routing_warmup = False
        width = kwargs.get("widths", (128,))[0]
        latent_channels = kwargs.get("latent_channels", 4)
        feature_channels = kwargs.get("lr_condition_channels", 64)
        if expert_channels < 2 or expert_channels % 2:
            raise ValueError("expert_channels must be a positive even integer >= 2")
        self.router = ExpertRouter(feature_channels, evidence_channels, num_experts, expert_channels)
        self.experts = nn.ModuleList(nn.Sequential(
            nn.Conv2d(width, expert_channels, 1), nn.SiLU(),
            nn.Conv2d(expert_channels, latent_channels, 3, padding=1),
        ) for _ in range(num_experts))
        for expert in self.experts:
            nn.init.normal_(expert[-1].weight, std=1e-3)
            nn.init.zeros_(expert[-1].bias)

    def routing(self, features, t, evidence=None):
        if evidence is None:
            raise ValueError("MoE routing requires LR/base-derived evidence, never HR targets")
        logits, quality_logits = self.router(features, t, evidence)
        probabilities = logits.float().softmax(1)
        quality = quality_logits.float().sigmoid()
        scores = probabilities * quality if self.routing_mode == "reliability" else probabilities
        scores = scores / scores.sum(1, keepdim=True).clamp_min(1e-8)
        top = scores.topk(self.top_k, dim=1).indices
        selected = torch.zeros_like(scores).scatter(1, top, 1)
        sparse = scores * selected
        sparse = sparse / sparse.sum(1, keepdim=True).clamp_min(1e-8)
        if self.training and self.routing_warmup:
            weights = torch.ones_like(scores) / self.num_experts
        elif self.training:
            # Straight-through selection: hard top-k forward, soft routing gradients.
            weights = scores + (sparse - scores).detach()
        else:
            weights = sparse
        acceptance = (weights * quality).sum(1) if self.routing_mode == "reliability" else torch.ones_like(scores[:, 0])
        return {"weights": weights, "probabilities": probabilities, "logits": logits,
                "quality_logits": quality_logits, "acceptance": acceptance, "selected": selected}

    def forward_with_routing(self, latent, t, context, degradation, mode, lr_condition, routing_context=None):
        features = self.forward_features(latent, t, context, degradation, mode, lr_condition)
        shared = self.output(features)
        route = self.routing(lr_condition, t, routing_context)
        # Training needs every candidate to supervise specialists and router targets.
        candidates = torch.stack([shared + expert(features) for expert in self.experts], 1)
        prediction = (candidates * route["weights"][:, :, None, None, None]).sum(1)
        return prediction, candidates, route

    def forward(self, latent, t, context, degradation, mode, lr_condition, routing_context=None):
        if self.training:
            return self.forward_with_routing(latent, t, context, degradation, mode, lr_condition, routing_context)[0]
        features = self.forward_features(latent, t, context, degradation, mode, lr_condition)
        shared = self.output(features)
        route = self.routing(lr_condition, t, routing_context)
        correction = torch.zeros_like(shared)
        for index, expert in enumerate(self.experts):
            rows = torch.where(route["weights"][:, index] > 0)[0]
            if rows.numel():
                values = expert(features[rows]) * route["weights"][rows, index, None, None, None]
                correction = correction.index_add(0, rows, values.to(correction.dtype))
        return shared + correction


def router_objectives(route, candidates, target_velocity, valid_mask, candidate_mse=None,
                      base_mse=None, temperature=1e-4):
    """Labels are detached training-only HR comparisons; no labels enter router.forward."""
    mask = F.interpolate(valid_mask.float(), size=target_velocity.shape[-2:], mode="area")[:, None]
    squared = (candidates.float() - target_velocity[:, None].float()).square()
    expert_loss = (squared * mask).sum() / (mask.sum() * squared.shape[1] * squared.shape[2]).clamp_min(1)
    importance = route["probabilities"].mean(0)
    usage = route["selected"].mean(0) / route["selected"].sum(1).mean()
    balance = importance.numel() * (importance * usage.detach()).sum()
    losses = {"expert_denoising": expert_loss, "router_balance": balance}
    if candidate_mse is not None:
        valid = (valid_mask.flatten(1).sum(1) > 0).float()
        gains = (base_mse[:, None] - candidate_mse).detach()
        targets = torch.sigmoid(gains / max(float(temperature), 1e-8))
        quality = F.binary_cross_entropy_with_logits(route["quality_logits"].float(), targets, reduction="none").mean(1)
        distribution = (-candidate_mse.detach() / max(float(temperature), 1e-8)).softmax(1)
        rank = -(distribution * route["logits"].float().log_softmax(1)).sum(1)
        losses["router_quality"] = (quality * valid).sum() / valid.sum().clamp_min(1)
        losses["router_ranking"] = (rank * valid).sum() / valid.sum().clamp_min(1)
    return losses
