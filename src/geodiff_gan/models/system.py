from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import nn

from .base import SwinIRBase
from .blocks import high_pass
from .degradation import back_project, default_degradation, sensor_degrade
from .diffusion import ConditionalDiffusionUNet, DiffusionBatch, DiffusionScheduler
from .generator import GeoMapper, LREncoder, ResidualSRDecoder
from .vae import ResidualVAE
from .residual_base import ResidualSwinBase
from .moe import MixtureDiffusionUNet

if TYPE_CHECKING:
    from ..diagnostics import DiagnosticRecorder

Mode = Literal["sr", "edit"]


@dataclass
class GeoDiffOutput:
    image: torch.Tensor
    base: torch.Tensor
    residual: torch.Tensor
    latent: torch.Tensor
    evidence_confidence: torch.Tensor
    edit_permission: torch.Tensor
    abstention_map: torch.Tensor
    raw_detail_residual: torch.Tensor
    raw_edit_residual: torch.Tensor
    ungated_sr: torch.Tensor
    sr_anchor: torch.Tensor
    metadata: list[dict[str, Any]]

    @property
    def evidence_gate(self) -> torch.Tensor:
        """Backward-compatible alias for pre-dual-policy callers."""
        return self.evidence_confidence


class GeoDiffGAN(nn.Module):
    def __init__(
        self,
        scale: int = 4,
        input_channels: int = 3,
        base_input_channels: int | None = None,
        output_channels: int = 3,
        base_embed_dim: int = 60,
        base_depth: int = 6,
        base_heads: int = 6,
        window_size: int = 8,
        base_upsample_mode: str = "pixelshuffle",
        latent_channels: int = 4,
        vae_channels: int = 64,
        vae_upsample_mode: str = "pixelshuffle",
        lr_channels: int = 64,
        diffusion_widths: tuple[int, ...] = (128, 256, 384, 512),
        context_dim: int = 768,
        degradation_dim: int = 4,
        mapper_channels: int = 128,
        style_dim: int = 256,
        decoder_channels: tuple[int, ...] = (128, 96, 64, 48),
        decoder_upsample_mode: str = "pixelshuffle",
        diffusion_steps: int = 1000,
        use_text_conditioning: bool = True,
        use_degradation_conditioning: bool = True,
        use_evidence_gate: bool = True,
        use_edit_gate: bool = True,
        use_uncertainty_abstention: bool = True,
        abstention_confidence_floor: float = 0.0,
        uncertainty_scale: float = 0.0025,
        use_back_projection: bool = True,
        degradation_severity: str = "mild",
        base_architecture: str = "swinir",
        base_groups: int = 2,
        diffusion_upsample_mode: str = "pixelshuffle",
        diffusion_experts: int = 0,
        diffusion_top_k: int = 1,
        routing_mode: str = "generic",
        expert_channels: int = 24,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.input_channels = input_channels
        self.base_input_channels = int(
            input_channels if base_input_channels is None else base_input_channels
        )
        if self.base_input_channels < output_channels:
            raise ValueError(
                "base_input_channels must include at least the output RGB channels"
            )
        if self.base_input_channels > input_channels:
            raise ValueError(
                "base_input_channels cannot exceed input_channels"
            )
        self.output_channels = output_channels
        self.latent_channels = latent_channels
        self.context_dim = context_dim
        self.use_text_conditioning = use_text_conditioning
        self.use_degradation_conditioning = use_degradation_conditioning
        self.use_uncertainty_abstention = use_uncertainty_abstention
        self.abstention_confidence_floor = float(abstention_confidence_floor)
        self.uncertainty_scale = float(uncertainty_scale)
        self.use_back_projection = use_back_projection
        self.degradation_severity = degradation_severity
        if base_architecture not in ("swinir", "residual_swin"):
            raise ValueError("base_architecture must be swinir or residual_swin")
        if diffusion_experts < 0:
            raise ValueError("diffusion_experts must be non-negative")
        base_class = ResidualSwinBase if base_architecture == "residual_swin" else SwinIRBase
        self.base = base_class(
            in_channels=self.base_input_channels,
            embed_dim=base_embed_dim,
            depth=base_depth,
            heads=base_heads,
            window_size=window_size,
            scale=scale,
            output_channels=output_channels,
            **({"groups": base_groups} if base_architecture == "residual_swin" else {"upsample_mode": base_upsample_mode}),
        )
        self.vae = ResidualVAE(latent_channels=latent_channels, base_channels=vae_channels, upsample_mode=vae_upsample_mode)
        self.lr_encoder = LREncoder(in_channels=input_channels, channels=lr_channels)
        diffusion_class = MixtureDiffusionUNet if diffusion_experts else ConditionalDiffusionUNet
        self.diffusion = diffusion_class(
            latent_channels=latent_channels,
            widths=diffusion_widths,
            context_dim=context_dim,
            degradation_dim=degradation_dim,
            lr_condition_channels=lr_channels * 2,
            upsample_mode=diffusion_upsample_mode,
            **({"num_experts": diffusion_experts, "top_k": diffusion_top_k,
                "routing_mode": routing_mode, "expert_channels": expert_channels,
                "evidence_channels": input_channels + 2 * output_channels} if diffusion_experts else {}),
        )
        self.scheduler = DiffusionScheduler(diffusion_steps)
        self.mapper = GeoMapper(
            latent_channels=latent_channels,
            lr_channels=lr_channels * 2,
            content_channels=mapper_channels,
            context_dim=context_dim,
            style_dim=style_dim,
            use_evidence_gate=use_evidence_gate,
            use_edit_gate=use_edit_gate,
        )
        self.decoder = ResidualSRDecoder(
            content_channels=mapper_channels,
            style_dim=style_dim,
            lr_channels=lr_channels,
            stage_channels=decoder_channels,
            upsample_mode=decoder_upsample_mode,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GeoDiffGAN":
        model = config.get("model", config)
        return cls(
            scale=model.get("scale", 4),
            base_architecture=model.get("base_architecture", "swinir"),
            base_groups=model.get("base_groups", 2),
            diffusion_upsample_mode=model.get("diffusion_upsample_mode", "pixelshuffle"),
            diffusion_experts=model.get("diffusion_experts", 0),
            diffusion_top_k=model.get("diffusion_top_k", 1),
            routing_mode=model.get("routing_mode", "generic"),
            expert_channels=model.get("expert_channels", 24),
            input_channels=model.get("input_channels", 3),
            base_input_channels=model.get("base_input_channels"),
            output_channels=model.get("output_channels", 3),
            base_embed_dim=model.get("base_embed_dim", 60),
            base_depth=model.get("base_depth", 6),
            base_heads=model.get("base_heads", 6),
            window_size=model.get("window_size", 8),
            base_upsample_mode=model.get("base_upsample_mode", "pixelshuffle"),
            latent_channels=model.get("latent_channels", 4),
            vae_channels=model.get("vae_channels", 64),
            vae_upsample_mode=model.get("vae_upsample_mode", "pixelshuffle"),
            lr_channels=model.get("lr_channels", 64),
            diffusion_widths=tuple(model.get("diffusion_widths", [128, 256, 384, 512])),
            context_dim=model.get("context_dim", 768),
            degradation_dim=model.get("degradation_dim", 4),
            mapper_channels=model.get("mapper_channels", 128),
            style_dim=model.get("style_dim", 256),
            decoder_channels=tuple(model.get("decoder_channels", [128, 96, 64, 48])),
            decoder_upsample_mode=model.get(
                "decoder_upsample_mode",
                "pixelshuffle",
            ),
            diffusion_steps=model.get("diffusion_steps", 1000),
            use_text_conditioning=model.get("use_text_conditioning", True),
            use_degradation_conditioning=model.get("use_degradation_conditioning", True),
            use_evidence_gate=model.get("use_evidence_gate", True),
            use_edit_gate=model.get("use_edit_gate", True),
            use_uncertainty_abstention=model.get(
                "use_uncertainty_abstention", True
            ),
            abstention_confidence_floor=model.get(
                "abstention_confidence_floor", 0.0
            ),
            uncertainty_scale=model.get("uncertainty_scale", 0.0025),
            use_back_projection=model.get("use_back_projection", True),
            degradation_severity=config.get("data", {}).get(
                "degradation_severity", "mild"
            ),
        )

    def output_lr(self, lr: torch.Tensor) -> torch.Tensor:
        if lr.shape[1] < self.output_channels:
            raise ValueError(
                f"Expected at least {self.output_channels} LR channels, got {lr.shape[1]}"
            )
        return lr[:, : self.output_channels]

    def base_lr(self, lr: torch.Tensor) -> torch.Tensor:
        """Select only evidence intended for deterministic reconstruction."""
        if lr.shape[1] < self.base_input_channels:
            raise ValueError(
                f"Expected at least {self.base_input_channels} base input channels, "
                f"got {lr.shape[1]}"
            )
        return lr[:, : self.base_input_channels]

    def predict_base(self, lr: torch.Tensor) -> torch.Tensor:
        """Run the conservative base without leaking auxiliary spectral channels."""
        return self.base(self.base_lr(lr))

    def routing_context(self, lr: torch.Tensor, base: torch.Tensor) -> torch.Tensor | None:
        if not isinstance(self.diffusion, MixtureDiffusionUNet):
            return None
        base_lr = torch.nn.functional.interpolate(base, size=lr.shape[-2:], mode="area")
        energy = torch.nn.functional.interpolate(high_pass(base).abs(), size=lr.shape[-2:], mode="area")
        return torch.cat((lr, energy, base_lr - self.output_lr(lr)), 1).detach()

    def apply_ablation_inputs(
        self, context: torch.Tensor, degradation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_text_conditioning:
            context = torch.zeros_like(context)
        if not self.use_degradation_conditioning:
            degradation = torch.zeros_like(degradation)
        return context, degradation

    def _resize_policy(
        self, policy: torch.Tensor, output_size: tuple[int, int]
    ) -> torch.Tensor:
        return torch.nn.functional.interpolate(
            policy,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        ).clamp(0, 1)

    def _effective_confidence(self, confidence: torch.Tensor) -> torch.Tensor:
        floor = min(max(self.abstention_confidence_floor, 0.0), 1.0)
        return floor + (1 - floor) * confidence

    def apply_uncertainty_abstention(
        self,
        image: torch.Tensor,
        base: torch.Tensor,
        evidence_confidence: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Blend stochastic SR toward the deterministic base when support is weak."""
        evidence = self._resize_policy(evidence_confidence, image.shape[-2:])
        if uncertainty.ndim == 3:
            uncertainty = uncertainty[:, None]
        uncertainty = torch.nn.functional.interpolate(
            uncertainty,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp_min(0)
        if self.use_uncertainty_abstention:
            agreement = torch.exp(
                -uncertainty / max(self.uncertainty_scale, 1e-8)
            )
            confidence = evidence * agreement
            blend_strength = agreement
        else:
            confidence = evidence
            blend_strength = torch.ones_like(evidence)
        effective = self._effective_confidence(blend_strength)
        abstained = base + effective * (image - base)
        return abstained.clamp(0, 1), confidence, 1 - confidence

    @staticmethod
    def mode_tensor(mode: Mode, batch: int, device: torch.device) -> torch.Tensor:
        return torch.full((batch,), 0 if mode == "sr" else 1, device=device, dtype=torch.long)

    def prepare_diffusion_batch(
        self, latent: torch.Tensor, timesteps: torch.Tensor | None = None
    ) -> DiffusionBatch:
        if timesteps is None:
            timesteps = torch.randint(
                0, self.scheduler.steps, (latent.shape[0],), device=latent.device
            )
        noisy, noise = self.scheduler.q_sample(latent, timesteps)
        target = self.scheduler.velocity_target(latent, noise, timesteps)
        return DiffusionBatch(noisy, noise, target, timesteps)

    def predict_velocity(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        degradation: torch.Tensor,
        mode: Mode,
        lr_features: list[torch.Tensor],
        routing_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mode_values = self.mode_tensor(mode, noisy_latent.shape[0], noisy_latent.device)
        context, degradation = self.apply_ablation_inputs(context, degradation)
        return self.diffusion(
            noisy_latent,
            timesteps,
            context,
            degradation,
            mode_values,
            lr_features[1],
            **({"routing_context": routing_context} if routing_context is not None else {}),
        )

    def decode_latent(
        self,
        latent: torch.Tensor,
        lr: torch.Tensor,
        context: torch.Tensor,
        degradation: torch.Tensor,
        mode: Mode,
        base: torch.Tensor | None = None,
        back_projection_steps: int | None = None,
        diagnostics: DiagnosticRecorder | None = None,
        projection_lr: torch.Tensor | None = None,
        conditioning_prepared: bool = False,
        lr_features: list[torch.Tensor] | None = None,
        apply_router_acceptance: bool = True,
    ) -> GeoDiffOutput:
        base = self.predict_base(lr) if base is None else base
        consistency_lr = self.output_lr(lr) if projection_lr is None else projection_lr
        if not conditioning_prepared:
            context, degradation = self.apply_ablation_inputs(context, degradation)
        lr_features = self.lr_encoder(lr) if lr_features is None else lr_features
        if diagnostics is not None:
            diagnostics.capture("input.lr", lr, visual="rgb")
            diagnostics.capture("conditioning.text", context)
            diagnostics.capture("conditioning.degradation", degradation)
            diagnostics.capture(
                "base.bicubic",
                torch.nn.functional.interpolate(
                    self.output_lr(lr),
                    scale_factor=self.scale,
                    mode="bicubic",
                    align_corners=False,
                ),
                visual="rgb",
            )
            diagnostics.capture("base.hr", base, visual="rgb")
            for name, feature in zip(("f128", "f64", "f32", "f16"), lr_features):
                diagnostics.capture(f"lr_features.{name}", feature, visual="features")
            diagnostics.capture("latent.denoised", latent, visual="features")
        mode_values = self.mode_tensor(mode, lr.shape[0], lr.device)
        mapped = self.mapper(latent, lr_features[1], context, mode_values)
        decoded = self.decoder(mapped, lr_features)
        raw_detail = decoded.detail_residual
        raw_edit = decoded.edit_residual
        evidence_hr = self._resize_policy(
            mapped.evidence_confidence, base.shape[-2:]
        )
        edit_hr = self._resize_policy(mapped.edit_permission, base.shape[-2:])
        effective_evidence = self._effective_confidence(evidence_hr)
        route = None
        if isinstance(self.diffusion, MixtureDiffusionUNet) and apply_router_acceptance:
            route = self.diffusion.routing(lr_features[1], torch.zeros(lr.shape[0], device=lr.device, dtype=torch.long), self.routing_context(lr, base))
            effective_evidence = effective_evidence * route["acceptance"][:, None, None, None]
        detail_residual = high_pass(raw_detail)
        evidence_residual = detail_residual * effective_evidence
        edit_residual = raw_edit * edit_hr
        ungated_sr = (base + detail_residual).clamp(0, 1)
        sr_anchor = (base + evidence_residual).clamp(0, 1)
        residual = (
            evidence_residual
            if mode == "sr"
            else evidence_residual + edit_residual
        )
        if diagnostics is not None:
            diagnostics.capture("mapper.content", mapped.content, visual="features")
            diagnostics.capture(
                "mapper.evidence_confidence",
                mapped.evidence_confidence,
                visual="heatmap",
            )
            diagnostics.capture(
                "mapper.edit_permission",
                mapped.edit_permission,
                visual="heatmap",
            )
            for index, style in enumerate(mapped.styles):
                diagnostics.capture(f"mapper.style_{index}", style)
            diagnostics.capture(
                "decoder.raw_detail_residual", raw_detail, visual="residual"
            )
            diagnostics.capture(
                "decoder.raw_edit_residual", raw_edit, visual="residual"
            )
            diagnostics.capture(
                "decoder.detail_high_pass",
                detail_residual,
                visual="residual",
            )
            diagnostics.capture(
                "decoder.evidence_residual",
                evidence_residual,
                visual="residual",
            )
            diagnostics.capture(
                "decoder.permission_edit_residual",
                edit_residual,
                visual="residual",
            )
            diagnostics.capture("output.ungated_sr", ungated_sr, visual="rgb")
            diagnostics.capture("output.sr_anchor", sr_anchor, visual="rgb")
            diagnostics.scalar(
                "mapper.evidence_mean", mapped.evidence_confidence.mean()
            )
            diagnostics.scalar(
                "mapper.evidence_std",
                mapped.evidence_confidence.std(unbiased=False),
            )
            diagnostics.scalar(
                "mapper.evidence_saturated_fraction",
                (
                    (mapped.evidence_confidence < 0.05)
                    | (mapped.evidence_confidence > 0.95)
                )
                .float()
                .mean(),
            )
            diagnostics.scalar(
                "mapper.edit_permission_mean", mapped.edit_permission.mean()
            )
            diagnostics.capture(
                "output.abstention_map", 1 - evidence_hr, visual="heatmap"
            )
        if mode == "sr":
            if diagnostics is not None:
                diagnostics.capture("decoder.residual", residual, visual="residual")
                raw_low = raw_detail - detail_residual
                diagnostics.scalar(
                    "residual.raw_low_frequency_fraction",
                    raw_low.abs().mean() / raw_detail.abs().mean().clamp_min(1e-8),
                )
            estimate = sr_anchor
            steps = 3 if back_projection_steps is None else back_projection_steps
            steps = steps if self.use_back_projection else 0
            image = back_project(
                estimate,
                consistency_lr,
                degradation,
                scale=self.scale,
                iterations=steps,
                step_size=0.5,
                severity=self.degradation_severity,
                debug_callback=(
                    diagnostics.projection_step
                    if diagnostics is not None
                    else None
                ),
            )
        else:
            estimate = (base + residual).clamp(0, 1)
            if diagnostics is not None:
                diagnostics.capture("decoder.residual", residual, visual="residual")
            steps = 1 if back_projection_steps is None else back_projection_steps
            steps = steps if self.use_back_projection else 0
            image = back_project(
                estimate,
                consistency_lr,
                degradation,
                scale=self.scale,
                iterations=steps,
                step_size=0.15,
                severity=self.degradation_severity,
                debug_callback=(
                    diagnostics.projection_step
                    if diagnostics is not None
                    else None
                ),
            )
        if diagnostics is not None:
            diagnostics.capture("output.pre_projection", estimate, visual="rgb")
            diagnostics.capture(
                "output.projection_update",
                image - estimate,
                visual="residual",
            )
            diagnostics.capture("output.hr", image, visual="rgb")
            diagnostics.scalar("output.back_projection_steps", steps)
            diagnostics.scalar("output.mode", mode)
            pre_degraded = sensor_degrade(
                estimate,
                degradation,
                scale=self.scale,
                add_noise=False,
                severity=self.degradation_severity,
            )
            post_degraded = sensor_degrade(
                image,
                degradation,
                scale=self.scale,
                add_noise=False,
                severity=self.degradation_severity,
            )
            diagnostics.scalar(
                "spatial.lr_error_before_projection",
                (pre_degraded - consistency_lr).abs().mean(),
            )
            diagnostics.scalar(
                "spatial.lr_error_after_projection",
                (post_degraded - consistency_lr).abs().mean(),
            )
            diagnostics.scalar(
                "spatial.projection_update_abs_mean",
                (image - estimate).abs().mean(),
            )
        metadata = []
        evidence_means = mapped.evidence_confidence.flatten(1).mean(dim=1)
        permission_means = mapped.edit_permission.flatten(1).mean(dim=1)
        for index in range(lr.shape[0]):
            metadata.append(
                {
                    "mode": mode,
                    "synthetic_edit": mode == "edit",
                    "scale": self.scale,
                    "input_channels": self.input_channels,
                    "base_input_channels": self.base_input_channels,
                    "output_channels": self.output_channels,
                    "back_projection_steps": steps,
                    "dual_policy_gating": True,
                    "uncertainty_abstention": self.use_uncertainty_abstention,
                    "evidence_confidence_mean": float(
                        evidence_means[index].detach()
                    ),
                    "edit_permission_mean": float(
                        permission_means[index].detach()
                    ),
                }
            )
            if route is not None:
                metadata[-1]["expert_weights"] = route["weights"][index].detach().cpu().tolist()
                metadata[-1]["router_acceptance"] = float(route["acceptance"][index].detach())
        return GeoDiffOutput(
            image=image,
            base=base,
            residual=residual,
            latent=latent,
            evidence_confidence=mapped.evidence_confidence,
            edit_permission=mapped.edit_permission,
            abstention_map=1 - evidence_hr,
            raw_detail_residual=raw_detail,
            raw_edit_residual=raw_edit,
            ungated_sr=ungated_sr,
            sr_anchor=sr_anchor,
            metadata=metadata,
        )

    @torch.no_grad()
    def sample(
        self,
        lr: torch.Tensor,
        context: torch.Tensor,
        degradation: torch.Tensor | None = None,
        mode: Mode = "sr",
        sample_steps: int = 20,
        guidance_scale: float = 1.0,
        null_context: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        diagnostics: DiagnosticRecorder | None = None,
        diffusion_debug_interval: int = 0,
        projection_lr: torch.Tensor | None = None,
        back_projection_steps: int | None = None,
        base: torch.Tensor | None = None,
        lr_features: list[torch.Tensor] | None = None,
    ) -> GeoDiffOutput:
        degradation = (
            default_degradation(lr.shape[0], lr.device, lr.dtype)
            if degradation is None
            else degradation
        )
        context, degradation = self.apply_ablation_inputs(context, degradation)
        if null_context is not None and not self.use_text_conditioning:
            null_context = torch.zeros_like(null_context)
        base = self.predict_base(lr) if base is None else base
        lr_features = self.lr_encoder(lr) if lr_features is None else lr_features
        latent_height = base.shape[-2] // self.vae.downsample_factor
        latent_width = base.shape[-1] // self.vae.downsample_factor
        mode_values = self.mode_tensor(mode, lr.shape[0], lr.device)
        latent = self.scheduler.ddim_sample(
            self.diffusion,
            (lr.shape[0], self.latent_channels, latent_height, latent_width),
            context,
            degradation,
            mode_values,
            lr_features[1],
            sample_steps=sample_steps,
            guidance_scale=guidance_scale,
            null_context=null_context,
            generator=generator,
            debug_callback=diagnostics.diffusion_step if diagnostics is not None else None,
            debug_interval=diffusion_debug_interval,
            routing_context=self.routing_context(lr, base),
        )
        return self.decode_latent(
            latent,
            lr,
            context,
            degradation,
            mode,
            base=base,
            projection_lr=projection_lr,
            back_projection_steps=back_projection_steps,
            diagnostics=diagnostics,
            conditioning_prepared=True,
            lr_features=lr_features,
        )
