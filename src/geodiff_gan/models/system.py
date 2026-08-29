from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import nn

from .base import (
    FidelityRDNBase,
    FidelitySwinIRBase,
    FidelitySwinIRV2Base,
    SwinIRBase,
)
from .blocks import high_pass
from .degradation import back_project, default_degradation, sensor_degrade
from .diffusion import ConditionalDiffusionUNet, DiffusionBatch, DiffusionScheduler
from .generator import (
    BaseReferencedTrustController,
    GeoMapper,
    LREncoder,
    MapperOutput,
    ResidualSRDecoder,
)
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
    evidence_residual: torch.Tensor
    trust_map: torch.Tensor
    trust_content: torch.Tensor
    ungated_sr: torch.Tensor
    pretrust_sr: torch.Tensor
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
        base_architecture: str = "swinir",
        base_group_size: int = 4,
        base_radiometric_calibration: bool = True,
        base_spatial_radiometric_calibration: bool = False,
        base_rdn_blocks: int = 20,
        base_rdn_layers: int = 6,
        base_rdn_growth: int = 32,
        base_swin_groups: int = 6,
        base_swin_blocks_per_group: int = 6,
        base_mlp_ratio: float = 2.0,
        latent_channels: int = 4,
        vae_channels: int = 64,
        vae_upsample_mode: str = "pixelshuffle",
        lr_channels: int = 64,
        diffusion_widths: tuple[int, ...] = (128, 256, 384, 512),
        diffusion_upsample_mode: str = "pixelshuffle",
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
        use_base_referenced_trust: bool = False,
        trust_channels: int = 32,
        trust_blocks: int = 2,
        trust_initial_scale: float = 0.25,
        trust_maximum_scale: float = 1.0,
        trust_mode: str = "scalar",
        abstention_confidence_floor: float = 0.0,
        uncertainty_scale: float = 0.0025,
        use_back_projection: bool = True,
        degradation_severity: str = "mild",
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
        if base_architecture == "swinir":
            self.base = SwinIRBase(
                in_channels=self.base_input_channels,
                embed_dim=base_embed_dim,
                depth=base_depth,
                heads=base_heads,
                window_size=window_size,
                scale=scale,
                output_channels=output_channels,
                upsample_mode=base_upsample_mode,
            )
        elif base_architecture == "fidelity_swinir":
            if base_upsample_mode != "resize_conv":
                raise ValueError(
                    "fidelity_swinir requires base_upsample_mode='resize_conv'"
                )
            self.base = FidelitySwinIRBase(
                in_channels=self.base_input_channels,
                embed_dim=base_embed_dim,
                depth=base_depth,
                group_size=base_group_size,
                heads=base_heads,
                window_size=window_size,
                scale=scale,
                output_channels=output_channels,
                radiometric_calibration=base_radiometric_calibration,
            )
        elif base_architecture == "fidelity_rdn":
            if base_upsample_mode != "resize_conv":
                raise ValueError(
                    "fidelity_rdn requires base_upsample_mode='resize_conv'"
                )
            self.base = FidelityRDNBase(
                in_channels=self.base_input_channels,
                channels=base_embed_dim,
                blocks=base_rdn_blocks,
                layers=base_rdn_layers,
                growth_channels=base_rdn_growth,
                scale=scale,
                output_channels=output_channels,
                radiometric_calibration=base_radiometric_calibration,
                spatial_radiometric_calibration=(
                    base_spatial_radiometric_calibration
                ),
            )
        elif base_architecture == "fidelity_swinir_v2":
            if base_upsample_mode != "resize_conv":
                raise ValueError(
                    "fidelity_swinir_v2 requires base_upsample_mode='resize_conv'"
                )
            self.base = FidelitySwinIRV2Base(
                in_channels=self.base_input_channels,
                embed_dim=base_embed_dim,
                groups=base_swin_groups,
                blocks_per_group=base_swin_blocks_per_group,
                window_size=window_size,
                heads=base_heads,
                mlp_ratio=base_mlp_ratio,
                scale=scale,
                output_channels=output_channels,
                radiometric_calibration=base_radiometric_calibration,
                spatial_radiometric_calibration=(
                    base_spatial_radiometric_calibration
                ),
            )
        else:
            raise ValueError(
                "base_architecture must be one of: swinir, fidelity_swinir, "
                "fidelity_rdn, fidelity_swinir_v2"
            )
        self.base_architecture = base_architecture
        self.vae = ResidualVAE(
            latent_channels=latent_channels,
            base_channels=vae_channels,
            upsample_mode=vae_upsample_mode,
        )
        self.lr_encoder = LREncoder(in_channels=input_channels, channels=lr_channels)
        self.diffusion = ConditionalDiffusionUNet(
            latent_channels=latent_channels,
            widths=diffusion_widths,
            context_dim=context_dim,
            degradation_dim=degradation_dim,
            lr_condition_channels=lr_channels * 2,
            upsample_mode=diffusion_upsample_mode,
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
        self.use_base_referenced_trust = use_base_referenced_trust
        self.trust_maximum_scale = float(trust_maximum_scale)
        if trust_mode not in ("scalar", "per_band"):
            raise ValueError("trust_mode must be 'scalar' or 'per_band'")
        self.trust_mode = trust_mode
        self.trust_controller = (
            BaseReferencedTrustController(
                content_channels=mapper_channels,
                lr_channels=lr_channels * 2,
                hidden_channels=trust_channels,
                initial_scale=trust_initial_scale,
                maximum_scale=trust_maximum_scale,
                blocks=trust_blocks,
                mode=trust_mode,
                output_channels=output_channels,
            )
            if use_base_referenced_trust
            else nn.Identity()
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GeoDiffGAN":
        model = config.get("model", config)
        return cls(
            scale=model.get("scale", 4),
            input_channels=model.get("input_channels", 3),
            base_input_channels=model.get("base_input_channels"),
            output_channels=model.get("output_channels", 3),
            base_embed_dim=model.get("base_embed_dim", 60),
            base_depth=model.get("base_depth", 6),
            base_heads=model.get("base_heads", 6),
            window_size=model.get("window_size", 8),
            base_upsample_mode=model.get("base_upsample_mode", "pixelshuffle"),
            base_architecture=model.get("base_architecture", "swinir"),
            base_group_size=model.get("base_group_size", 4),
            base_radiometric_calibration=model.get(
                "base_radiometric_calibration", True
            ),
            base_spatial_radiometric_calibration=model.get(
                "base_spatial_radiometric_calibration", False
            ),
            base_rdn_blocks=model.get("base_rdn_blocks", 20),
            base_rdn_layers=model.get("base_rdn_layers", 6),
            base_rdn_growth=model.get("base_rdn_growth", 32),
            base_swin_groups=model.get("base_swin_groups", 6),
            base_swin_blocks_per_group=model.get(
                "base_swin_blocks_per_group", 6
            ),
            base_mlp_ratio=model.get("base_mlp_ratio", 2.0),
            latent_channels=model.get("latent_channels", 4),
            vae_channels=model.get("vae_channels", 64),
            vae_upsample_mode=model.get("vae_upsample_mode", "pixelshuffle"),
            lr_channels=model.get("lr_channels", 64),
            diffusion_widths=tuple(model.get("diffusion_widths", [128, 256, 384, 512])),
            diffusion_upsample_mode=model.get(
                "diffusion_upsample_mode", "pixelshuffle"
            ),
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
            use_base_referenced_trust=model.get(
                "use_base_referenced_trust", False
            ),
            trust_channels=model.get("trust_channels", 32),
            trust_blocks=model.get("trust_blocks", 2),
            trust_initial_scale=model.get("trust_initial_scale", 0.25),
            trust_maximum_scale=model.get("trust_maximum_scale", 1.0),
            trust_mode=model.get("trust_mode", "scalar"),
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
        trust_map: torch.Tensor | None = None,
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
        normalized_trust = (
            torch.ones_like(evidence)
            if trust_map is None
            else torch.nn.functional.interpolate(
                trust_map,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).clamp(0, self.trust_maximum_scale)
            / max(self.trust_maximum_scale, 1e-8)
        )
        if self.use_uncertainty_abstention:
            agreement = torch.exp(
                -uncertainty / max(self.uncertainty_scale, 1e-8)
            )
            confidence = evidence * normalized_trust * agreement
            blend_strength = agreement
        else:
            confidence = evidence * normalized_trust
            blend_strength = torch.ones_like(evidence)
        effective = self._effective_confidence(blend_strength)
        abstained = base + effective * (image - base)
        if confidence.shape[1] != 1:
            confidence = confidence.mean(dim=1, keepdim=True)
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

    def fuse_sr_detail(
        self,
        raw_detail: torch.Tensor,
        mapped: MapperOutput,
        lr_features: list[torch.Tensor],
        base: torch.Tensor,
        consistency_lr: torch.Tensor | None = None,
        degradation: torch.Tensor | None = None,
        sample_variance: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Apply high-pass, evidence and base-referenced trust constraints."""
        detail_residual = high_pass(raw_detail)
        evidence_hr = self._resize_policy(
            mapped.evidence_confidence, base.shape[-2:]
        )
        evidence_residual = detail_residual * self._effective_confidence(
            evidence_hr
        )
        consistency_error = None
        if consistency_lr is not None and degradation is not None:
            candidate = (base + evidence_residual).clamp(0, 1)
            candidate_lr = sensor_degrade(
                candidate,
                degradation,
                scale=self.scale,
                add_noise=False,
                severity=self.degradation_severity,
            )
            consistency_error = candidate_lr - consistency_lr
        if self.use_base_referenced_trust:
            trust_map = self.trust_controller(
                mapped.content,
                lr_features[1],
                mapped.evidence_confidence,
                base,
                evidence_residual,
                consistency_error=consistency_error,
                sample_variance=sample_variance,
            )
        else:
            trust_map = torch.ones_like(evidence_hr)
        residual = evidence_residual * trust_map
        return {
            "detail_residual": detail_residual,
            "evidence_hr": evidence_hr,
            "evidence_residual": evidence_residual,
            "trust_map": trust_map,
            "residual": residual,
        }

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

    @torch.no_grad()
    def sample_latent(
        self,
        lr: torch.Tensor,
        context: torch.Tensor,
        degradation: torch.Tensor | None = None,
        mode: Mode = "sr",
        sample_steps: int = 20,
        guidance_scale: float = 1.0,
        null_context: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        base: torch.Tensor | None = None,
        lr_features: list[torch.Tensor] | None = None,
        conditioning_prepared: bool = False,
        debug_callback: Any | None = None,
        debug_interval: int = 0,
    ) -> torch.Tensor:
        """Sample only the residual latent while keeping diffusion frozen."""
        degradation = (
            default_degradation(lr.shape[0], lr.device, lr.dtype)
            if degradation is None
            else degradation
        )
        if not conditioning_prepared:
            context, degradation = self.apply_ablation_inputs(context, degradation)
        if null_context is not None and not self.use_text_conditioning:
            null_context = torch.zeros_like(null_context)
        base = self.predict_base(lr) if base is None else base
        lr_features = self.lr_encoder(lr) if lr_features is None else lr_features
        latent_height = base.shape[-2] // self.vae.downsample_factor
        latent_width = base.shape[-1] // self.vae.downsample_factor
        mode_values = self.mode_tensor(mode, lr.shape[0], lr.device)
        return self.scheduler.ddim_sample(
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
            debug_callback=debug_callback,
            debug_interval=debug_interval,
        )

    def aggregate_sr_outputs(
        self,
        outputs: list[GeoDiffOutput],
        lr_features: list[torch.Tensor],
        projection_lr: torch.Tensor,
        degradation: torch.Tensor,
        back_projection_steps: int = 0,
    ) -> GeoDiffOutput:
        """Fuse sampled proposals once using their mean and spatial variance."""
        if not outputs:
            raise ValueError("aggregate_sr_outputs requires at least one proposal")
        if back_projection_steps < 0:
            raise ValueError("back_projection_steps must be non-negative")
        base = outputs[0].base
        evidence = torch.stack(
            [value.evidence_confidence for value in outputs]
        ).mean(0)
        content = torch.stack([value.trust_content for value in outputs]).mean(0)
        candidate_residual = torch.stack(
            [value.evidence_residual for value in outputs]
        ).mean(0)
        candidate_images = torch.stack(
            [(base + value.evidence_residual).clamp(0, 1) for value in outputs]
        )
        sample_variance = candidate_images.var(0, unbiased=False).mean(
            dim=1,
            keepdim=True,
        )
        candidate_lr = sensor_degrade(
            (base + candidate_residual).clamp(0, 1),
            degradation,
            scale=self.scale,
            add_noise=False,
            severity=self.degradation_severity,
        )
        consistency_error = candidate_lr - projection_lr
        if self.use_base_referenced_trust:
            trust_map = self.trust_controller(
                content,
                lr_features[1],
                evidence,
                base,
                candidate_residual,
                consistency_error=consistency_error,
                sample_variance=sample_variance,
            )
        else:
            trust_map = torch.ones_like(evidence)
        trusted_residual = candidate_residual * trust_map
        estimate = (base + trusted_residual).clamp(0, 1)
        image = back_project(
            estimate,
            projection_lr,
            degradation,
            scale=self.scale,
            iterations=(back_projection_steps if self.use_back_projection else 0),
            step_size=0.5,
            severity=self.degradation_severity,
        )
        raw_detail = torch.stack(
            [value.raw_detail_residual for value in outputs]
        ).mean(0)
        raw_edit = torch.stack([value.raw_edit_residual for value in outputs]).mean(0)
        ungated = (base + high_pass(raw_detail)).clamp(0, 1)
        pretrust = (base + candidate_residual).clamp(0, 1)
        normalized_trust = (
            trust_map / max(self.trust_maximum_scale, 1e-8)
        ).clamp(0, 1)
        metadata = [dict(value) for value in outputs[0].metadata]
        for value in metadata:
            value["ensemble_samples"] = len(outputs)
            value["trust_mode"] = self.trust_mode
        return GeoDiffOutput(
            image=image,
            base=base,
            residual=trusted_residual,
            latent=torch.stack([value.latent for value in outputs]).mean(0),
            evidence_confidence=evidence,
            edit_permission=torch.stack(
                [value.edit_permission for value in outputs]
            ).mean(0),
            abstention_map=1
            - (self._resize_policy(evidence, base.shape[-2:]) * normalized_trust)
            .mean(dim=1, keepdim=True),
            raw_detail_residual=raw_detail,
            raw_edit_residual=raw_edit,
            evidence_residual=candidate_residual,
            trust_map=trust_map,
            trust_content=content,
            ungated_sr=ungated,
            pretrust_sr=pretrust,
            sr_anchor=estimate,
            metadata=metadata,
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
        edit_hr = self._resize_policy(mapped.edit_permission, base.shape[-2:])
        fusion = self.fuse_sr_detail(
            raw_detail,
            mapped,
            lr_features,
            base,
            consistency_lr=consistency_lr,
            degradation=degradation,
        )
        detail_residual = fusion["detail_residual"]
        evidence_hr = fusion["evidence_hr"]
        evidence_residual = fusion["evidence_residual"]
        trust_map = fusion["trust_map"]
        trusted_residual = fusion["residual"]
        edit_residual = raw_edit * edit_hr
        ungated_sr = (base + detail_residual).clamp(0, 1)
        pretrust_sr = (base + evidence_residual).clamp(0, 1)
        sr_anchor = (base + trusted_residual).clamp(0, 1)
        residual = (
            trusted_residual
            if mode == "sr"
            else trusted_residual + edit_residual
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
                "mapper.base_referenced_trust",
                trust_map,
                visual="heatmap",
            )
            diagnostics.capture(
                "decoder.trusted_residual",
                trusted_residual,
                visual="residual",
            )
            diagnostics.capture(
                "decoder.permission_edit_residual",
                edit_residual,
                visual="residual",
            )
            diagnostics.capture("output.ungated_sr", ungated_sr, visual="rgb")
            diagnostics.capture("output.pretrust_sr", pretrust_sr, visual="rgb")
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
            diagnostics.scalar("mapper.trust_mean", trust_map.mean())
            diagnostics.scalar(
                "mapper.trust_std", trust_map.std(unbiased=False)
            )
            normalized_trust = (
                trust_map / max(self.trust_maximum_scale, 1e-8)
            ).clamp(0, 1)
            diagnostics.capture(
                "output.abstention_map",
                1 - (evidence_hr * normalized_trust).mean(dim=1, keepdim=True),
                visual="heatmap",
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
        trust_means = trust_map.flatten(1).mean(dim=1)
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
                    "base_referenced_trust": self.use_base_referenced_trust,
                    "evidence_confidence_mean": float(
                        evidence_means[index].detach()
                    ),
                    "edit_permission_mean": float(
                        permission_means[index].detach()
                    ),
                    "trust_mean": float(trust_means[index].detach()),
                }
            )
        normalized_trust = (
            trust_map / max(self.trust_maximum_scale, 1e-8)
        ).clamp(0, 1)
        return GeoDiffOutput(
            image=image,
            base=base,
            residual=residual,
            latent=latent,
            evidence_confidence=mapped.evidence_confidence,
            edit_permission=mapped.edit_permission,
            abstention_map=1
            - (evidence_hr * normalized_trust).mean(dim=1, keepdim=True),
            raw_detail_residual=raw_detail,
            raw_edit_residual=raw_edit,
            evidence_residual=evidence_residual,
            trust_map=trust_map,
            trust_content=mapped.content,
            ungated_sr=ungated_sr,
            pretrust_sr=pretrust_sr,
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
        latent = self.sample_latent(
            lr,
            context,
            degradation,
            mode=mode,
            sample_steps=sample_steps,
            guidance_scale=guidance_scale,
            null_context=null_context,
            generator=generator,
            base=base,
            lr_features=lr_features,
            conditioning_prepared=True,
            debug_callback=(
                diagnostics.diffusion_step if diagnostics is not None else None
            ),
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
            back_projection_steps=back_projection_steps,
            diagnostics=diagnostics,
            conditioning_prepared=True,
            lr_features=lr_features,
        )
