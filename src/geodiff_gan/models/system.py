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
        base_embed_dim: int = 60,
        base_depth: int = 6,
        base_heads: int = 6,
        window_size: int = 8,
        latent_channels: int = 4,
        vae_channels: int = 64,
        lr_channels: int = 64,
        diffusion_widths: tuple[int, ...] = (128, 256, 384, 512),
        context_dim: int = 768,
        degradation_dim: int = 4,
        mapper_channels: int = 128,
        style_dim: int = 256,
        decoder_channels: tuple[int, ...] = (128, 96, 64, 48),
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
    ) -> None:
        super().__init__()
        self.scale = scale
        self.latent_channels = latent_channels
        self.context_dim = context_dim
        self.use_text_conditioning = use_text_conditioning
        self.use_degradation_conditioning = use_degradation_conditioning
        self.use_uncertainty_abstention = use_uncertainty_abstention
        self.abstention_confidence_floor = float(abstention_confidence_floor)
        self.uncertainty_scale = float(uncertainty_scale)
        self.use_back_projection = use_back_projection
        self.degradation_severity = degradation_severity
        self.base = SwinIRBase(
            embed_dim=base_embed_dim,
            depth=base_depth,
            heads=base_heads,
            window_size=window_size,
            scale=scale,
        )
        self.vae = ResidualVAE(latent_channels=latent_channels, base_channels=vae_channels)
        self.lr_encoder = LREncoder(channels=lr_channels)
        self.diffusion = ConditionalDiffusionUNet(
            latent_channels=latent_channels,
            widths=diffusion_widths,
            context_dim=context_dim,
            degradation_dim=degradation_dim,
            lr_condition_channels=lr_channels * 2,
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
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GeoDiffGAN":
        model = config.get("model", config)
        return cls(
            scale=model.get("scale", 4),
            base_embed_dim=model.get("base_embed_dim", 60),
            base_depth=model.get("base_depth", 6),
            base_heads=model.get("base_heads", 6),
            window_size=model.get("window_size", 8),
            latent_channels=model.get("latent_channels", 4),
            vae_channels=model.get("vae_channels", 64),
            lr_channels=model.get("lr_channels", 64),
            diffusion_widths=tuple(model.get("diffusion_widths", [128, 256, 384, 512])),
            context_dim=model.get("context_dim", 768),
            degradation_dim=model.get("degradation_dim", 4),
            mapper_channels=model.get("mapper_channels", 128),
            style_dim=model.get("style_dim", 256),
            decoder_channels=tuple(model.get("decoder_channels", [128, 96, 64, 48])),
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
    ) -> GeoDiffOutput:
        base = self.base(lr) if base is None else base
        consistency_lr = lr if projection_lr is None else projection_lr
        if not conditioning_prepared:
            context, degradation = self.apply_ablation_inputs(context, degradation)
        lr_features = self.lr_encoder(lr)
        if diagnostics is not None:
            diagnostics.capture("input.lr", lr, visual="rgb")
            diagnostics.capture("conditioning.text", context)
            diagnostics.capture("conditioning.degradation", degradation)
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
            )
        if diagnostics is not None:
            diagnostics.capture("output.pre_projection", estimate, visual="rgb")
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
    ) -> GeoDiffOutput:
        degradation = (
            default_degradation(lr.shape[0], lr.device, lr.dtype)
            if degradation is None
            else degradation
        )
        context, degradation = self.apply_ablation_inputs(context, degradation)
        if null_context is not None and not self.use_text_conditioning:
            null_context = torch.zeros_like(null_context)
        base = self.base(lr)
        lr_features = self.lr_encoder(lr)
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
        )
        return self.decode_latent(
            latent,
            lr,
            context,
            degradation,
            mode,
            base=base,
            projection_lr=projection_lr,
            diagnostics=diagnostics,
            conditioning_prepared=True,
        )
