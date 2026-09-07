"""Diffusion-free, tile-dispatched residual SR with a contextual Transformer router."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .residual_base import ResidualSwinBase


@dataclass
class TrustMoEOutput:
    image: torch.Tensor
    base: torch.Tensor
    residual: torch.Tensor
    trust: torch.Tensor
    active: torch.Tensor
    difficulty: torch.Tensor
    probabilities: torch.Tensor
    assignments: torch.Tensor
    dispatched_tiles: torch.Tensor


class ContextRouter(nn.Module):
    """Attention runs on pooled LR tiles, not on HR pixels."""

    def __init__(self, width, experts, kind="transformer", heads=4, depth=2):
        super().__init__()
        if kind not in ("transformer", "conv") or width % heads:
            raise ValueError("Router must be transformer/conv, with width divisible by heads")
        self.position = nn.Conv2d(width, width, 3, padding=1, groups=width)
        if kind == "transformer":
            # Instantiate independently: cloned encoder layers otherwise start identically.
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(width, heads, width * 2, dropout=0.0,
                                           batch_first=True, norm_first=True)
                for _ in range(depth)
            ])
        else:
            self.layers = nn.ModuleList([
                nn.Sequential(nn.Linear(width, width * 2), nn.GELU(),
                              nn.Linear(width * 2, width)) for _ in range(depth)
            ])
        self.norm = nn.LayerNorm(width)
        self.routes = nn.Linear(width, experts)
        self.risk = nn.Linear(width, 1)

    def forward(self, pooled):
        tokens = (pooled + self.position(pooled)).flatten(2).transpose(1, 2)
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)
        return self.routes(tokens), self.risk(tokens).squeeze(-1)


class ResidualTileExpert(nn.Module):
    def __init__(self, width, scale, residual_bound):
        super().__init__()
        self.scale, self.bound = scale, residual_bound
        self.body = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(width, width, 3, padding=1), nn.GELU())
        self.rgb = nn.Conv2d(width, 3, 3, padding=1)
        # Small independent proposals give the gate a signal without a large initial correction.
        nn.init.normal_(self.rgb.weight, std=1e-3)
        nn.init.zeros_(self.rgb.bias)

    def forward(self, features):
        features = F.interpolate(self.body(features), scale_factor=self.scale,
                                 mode="bilinear", align_corners=False)
        return self.bound * torch.tanh(self.rgb(features))


class TrustMoESR(nn.Module):
    """One base pass + selected tile experts + a per-band trust gate.

    Sparsity is blockwise. Features/router/trust are shared dense overhead; only
    selected feature tiles (including their halos) enter expert networks.
    """

    def __init__(self, scale=3, input_channels=3, width=32, num_experts=5, top_k=2,
                 tile_size=8, halo=4, coverage=0.5, router_kind="transformer",
                 router_depth=2, routing="reliability", use_trust=True,
                 residual_bound=0.1, base_embed_dim=32, base_depth=2,
                 base_groups=2, base_heads=4, window_size=8, dispatch_chunk=128,
                 adaptive_k=False):
        super().__init__()
        if not 1 <= top_k <= num_experts or not 0 <= coverage <= 1:
            raise ValueError("Require 1 <= top_k <= num_experts and 0 <= coverage <= 1")
        if tile_size < 2 or halo < 3 or input_channels < 3 or dispatch_chunk < 1:
            raise ValueError("Invalid tile geometry, input channels or dispatch chunk")
        if routing not in ("reliability", "uniform"):
            raise ValueError("routing must be reliability or uniform")
        self.scale, self.input_channels = scale, input_channels
        self.num_experts, self.top_k = num_experts, top_k
        self.tile_size, self.halo, self.coverage = tile_size, halo, coverage
        self.routing, self.use_trust = routing, use_trust
        self.dispatch_chunk = dispatch_chunk
        self.adaptive_k = adaptive_k
        self.base = ResidualSwinBase(3, base_embed_dim, base_depth, base_heads,
                                    window_size, scale, 3, base_groups)
        # Cues are observable at inference: LR, base on LR grid, HF energy,
        # base-vs-observation discrepancy, and LR local variation.
        self.encoder = nn.Sequential(nn.Conv2d(input_channels + 10, width, 3, padding=1),
                                     nn.GELU(), nn.Conv2d(width, width, 3, padding=1),
                                     nn.GELU())
        self.router = ContextRouter(width, num_experts, router_kind, depth=router_depth)
        self.experts = nn.ModuleList([
            ResidualTileExpert(width, scale, residual_bound) for _ in range(num_experts)
        ])
        self.trust_head = nn.Sequential(nn.Conv2d(width + 6, width, 1), nn.GELU(),
                                        nn.Conv2d(width, 3, 1))
        nn.init.constant_(self.trust_head[-1].bias, 1.0)
        self.freeze_base()

    def freeze_base(self):
        self.base.requires_grad_(False)
        self.base.eval()

    def train(self, mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.base.parameters()):
            self.base.eval()
        return self

    def forward(self, lr, *, coverage=None, top_k=None, residual_scale=1.0,
                explore=0.0, base_only=False):
        coverage = self.coverage if coverage is None else float(coverage)
        top_k = self.top_k if top_k is None else int(top_k)
        if not 0 <= coverage <= 1 or not 1 <= top_k <= self.num_experts:
            raise ValueError("Invalid inference coverage/top_k")
        b, _, h, w = lr.shape
        t, s, halo = self.tile_size, self.scale, self.halo
        gh, gw = math.ceil(h / t), math.ceil(w / t)
        n, ph, pw = gh * gw, gh * t, gw * t
        base = self.base(lr[:, :3])
        if base_only or residual_scale == 0 or coverage == 0:
            zeros = base.new_zeros(b, n)
            return TrustMoEOutput(base, base, torch.zeros_like(base), torch.zeros_like(base),
                                  torch.zeros_like(base[:, :1]), zeros, zeros[..., None].expand(-1, -1, self.num_experts),
                                  zeros[..., None].expand(-1, -1, self.num_experts), zeros.sum().long())
        small_base = F.interpolate(base, size=(h, w), mode="area")
        hf = base - F.avg_pool2d(base, 5, 1, 2)
        small_hf = F.interpolate(hf.abs(), size=(h, w), mode="area")
        variation = (lr[:, :3] - F.avg_pool2d(lr[:, :3], 3, 1, 1)).abs().mean(1, keepdim=True)
        discrepancy = small_base - lr[:, :3]
        cues = torch.cat((lr, small_base, small_hf, discrepancy, variation), 1)
        features = self.encoder(cues)
        padded = F.pad(features, (0, pw - w, 0, ph - h), mode="replicate")
        logits, risk = self.router(F.avg_pool2d(padded, t, t))
        probabilities = logits.float().softmax(-1)
        count = math.ceil(n * coverage)
        if self.routing == "uniform":
            indices = torch.linspace(0, n - 1, count, device=lr.device).round().long()
            selected = indices[None].expand(b, -1)
        else:
            ranking = risk.detach()
            if self.training and explore:
                ranking = torch.where(torch.rand_like(ranking) < explore,
                                      torch.rand_like(ranking) * 6 - 3, ranking)
            selected = ranking.topk(count, dim=1).indices
        active_tokens = torch.zeros_like(risk).scatter(1, selected, 1)
        route_logits = logits.float()
        if self.training and explore:
            random_rows = torch.rand_like(risk)[..., None] < explore
            route_logits = torch.where(random_rows, torch.rand_like(route_logits), route_logits)
        chosen = route_logits.topk(top_k, dim=-1).indices
        slots = torch.ones_like(chosen, dtype=probabilities.dtype)
        if self.adaptive_k and top_k > 1:
            # Risk rank allocates 1..top_k slots within the active region budget.
            # This is an explicit rank policy, not a learned prediction of runtime.
            ranked = risk.detach().masked_fill(active_tokens == 0, -torch.inf)
            risk_rank = ranked.argsort(1).argsort(1).float() - (n - count)
            risk_rank = risk_rank.clamp_min(0) / max(count - 1, 1)
            allowed = (1 + (risk_rank * top_k).floor()).clamp(max=top_k)
            slots = (torch.arange(top_k, device=lr.device)[None, None] < allowed[..., None]).float()
        assignments = torch.zeros_like(probabilities).scatter(-1, chosen, slots) * active_tokens[..., None]
        # Full-softmax gradients survive top-1; detached normalization keeps selected
        # weights summing to one in the forward pass without a zero router gradient.
        gates = probabilities * assignments
        gates = gates / gates.sum(-1, keepdim=True).clamp_min(1e-8).detach()

        # Gather only selected tiles from an unfolded VIEW; do not run experts on
        # full frames and mask them afterward. Halos provide neighboring context.
        patches = F.pad(padded, (halo, halo, halo, halo), mode="replicate")
        patches = patches.unfold(2, t + 2 * halo, t).unfold(3, t + 2 * halo, t)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        merged = base.new_zeros(b * n, 3, t * s, t * s)
        dispatched = assignments.sum().long()
        for k, expert in enumerate(self.experts):
            batch_ids, token_ids = torch.where(assignments[..., k] > 0)
            for start in range(0, len(batch_ids), self.dispatch_chunk):
                bi, ti = batch_ids[start:start + self.dispatch_chunk], token_ids[start:start + self.dispatch_chunk]
                inputs = patches[bi, ti // gw, ti % gw]
                proposal = expert(inputs)
                offset = halo * s
                proposal = proposal[:, :, offset:offset + t * s, offset:offset + t * s]
                weight = gates[bi, ti, k, None, None, None].to(proposal.dtype)
                merged = merged.index_add(0, bi * n + ti, (proposal * weight).to(merged.dtype))
        residual = merged.reshape(b, gh, gw, 3, t * s, t * s)
        residual = residual.permute(0, 3, 1, 4, 2, 5).reshape(b, 3, ph * s, pw * s)
        residual = residual[:, :, :h * s, :w * s]
        active = active_tokens.reshape(b, 1, gh, gw).repeat_interleave(t * s, 2).repeat_interleave(t * s, 3)
        active = active[:, :, :h * s, :w * s]
        trust_input = torch.cat((features, F.interpolate(residual, size=(h, w), mode="area"), discrepancy), 1)
        trust = F.interpolate(self.trust_head(trust_input), size=base.shape[-2:], mode="bilinear", align_corners=False).sigmoid()
        if not self.use_trust:
            trust = torch.ones_like(base)
        image = (base + float(residual_scale) * trust * residual).clamp(0, 1)
        return TrustMoEOutput(image, base, residual, trust, active, risk,
                              probabilities, assignments, dispatched)
