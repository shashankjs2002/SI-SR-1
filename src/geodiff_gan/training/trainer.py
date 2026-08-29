from __future__ import annotations

import json
import math
import os
import random
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import replace
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from ..data import SentinelPatchDataset
from ..diagnostics import DiagnosticRecorder, append_training_history
from ..losses import (
    OptionalPerceptualLoss,
    base_guard_loss,
    charbonnier,
    degradation_consistency,
    discriminator_hinge,
    edit_localization_loss,
    evidence_calibration_loss,
    evidence_improvement_loss,
    generator_hinge,
    gradient_loss,
    kl_loss,
    local_excess_mse_loss,
    mse_loss,
    multiscale_mse_loss,
    radiometric_loss,
    residual_trust_projection_loss,
    residual_supervision_loss,
    snr_weighted_velocity_loss,
    spatial_base_guard_loss,
    ssim,
    wavelet_loss,
)
from ..models.degradation import sensor_degrade
from ..models.discriminators import MultiScaleDiscriminator, WaveletDiscriminator
from ..models.system import GeoDiffGAN
from ..metrics import basic_metrics
from ..text import PromptBatch, TextEncoder, augment_prompts, build_text_encoder
from .checkpoint import (
    best_stage_checkpoint,
    checkpoint_sha256,
    copy_checkpoint,
    latest_stage_checkpoint,
    load_checkpoint,
    prune_stage_epoch_checkpoints,
    save_checkpoint,
    unwrap,
)
from .ema import ModelEMA
from .stages import STAGES, configure_stage_trainability


class Trainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.stage = config["training"]["stage"]
        if self.stage not in STAGES:
            raise ValueError(f"Unknown stage {self.stage!r}; expected one of {STAGES}")
        self.train_back_projection_steps = int(
            config["training"].get("train_back_projection_steps", 1)
        )
        if self.train_back_projection_steps < 0:
            raise ValueError("training.train_back_projection_steps must be non-negative")
        self.distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if self.distributed:
            torch.distributed.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)
        self.device = torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        )
        seed = int(config.get("seed", 42)) + self.local_rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.is_main = not self.distributed or torch.distributed.get_rank() == 0
        training = config["training"]
        output_dir = Path(training.get("output_dir", "runs/default"))
        resume = training.get("resume")
        if not resume and bool(training.get("auto_resume", False)):
            resume = latest_stage_checkpoint(output_dir, self.stage)
            if resume is not None:
                training["resume"] = str(resume)
        self.model = GeoDiffGAN.from_config(config).to(self.device)
        init_checkpoint = training.get("init_checkpoint")
        self.init_checkpoint = init_checkpoint
        self.parent_checkpoint_sha256 = (
            checkpoint_sha256(init_checkpoint) if init_checkpoint else None
        )
        if init_checkpoint and not resume:
            load_checkpoint(
                init_checkpoint,
                self.model,
                strict=False,
                prefer_ema=bool(training.get("init_use_ema", True)),
            )
        self.text_encoder: TextEncoder | None = None
        if self.stage != "base" and self.model.use_text_conditioning:
            self.text_encoder = build_text_encoder(config).to(self.device).eval()
            self.text_encoder.requires_grad_(False)
            if self.text_encoder.context_dim != self.model.context_dim:
                raise ValueError(
                    f"Text encoder dimension {self.text_encoder.context_dim} does not match "
                    f"model context dimension {self.model.context_dim}"
                )
        self._configure_stage()
        self.model.diffusion.gradient_checkpointing = bool(
            config["training"].get("gradient_checkpointing", True)
        )
        self.patch_discriminator = MultiScaleDiscriminator(
            base_channels=config["training"].get("discriminator_channels", 64),
            output_channels=self.model.output_channels,
            condition_channels=self.model.output_channels,
        ).to(self.device)
        self.wavelet_discriminator = WaveletDiscriminator(
            base_channels=config["training"].get("discriminator_channels", 64),
            output_channels=self.model.output_channels,
            condition_channels=self.model.output_channels,
        ).to(self.device)
        if self.distributed:
            self.patch_discriminator = DistributedDataParallel(
                self.patch_discriminator, device_ids=[self.local_rank]
            )
            self.wavelet_discriminator = DistributedDataParallel(
                self.wavelet_discriminator, device_ids=[self.local_rank]
            )
        self.perceptual = (
            OptionalPerceptualLoss().to(self.device)
            if float(training.get("loss_weights", {}).get("perceptual", 0.1)) > 0
            else None
        )
        parameters = self._optimizer_parameter_groups()
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=float(config["training"]["learning_rate"]),
            betas=(0.9, 0.99),
            weight_decay=float(config["training"].get("weight_decay", 1e-4)),
        )
        self.max_optimizer_steps = max(
            0, int(training.get("max_optimizer_steps", 0))
        )
        scheduler_factor = float(training.get("lr_scheduler_factor", 1.0))
        scheduler_type = str(
            training.get(
                "lr_scheduler_type",
                "plateau" if 0 < scheduler_factor < 1 else "none",
            )
        ).lower()
        if scheduler_type not in ("none", "plateau", "cosine_warmup"):
            raise ValueError(
                "training.lr_scheduler_type must be none, plateau, or cosine_warmup"
            )
        self.lr_scheduler_step_per_update = scheduler_type == "cosine_warmup"
        if scheduler_type == "plateau":
            if not 0 < scheduler_factor < 1:
                raise ValueError(
                    "training.lr_scheduler_factor must be in (0, 1) for plateau"
                )
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=str(training.get("lr_scheduler_mode", "max")),
                factor=scheduler_factor,
                patience=max(0, int(training.get("lr_scheduler_patience", 2))),
                threshold=max(
                    0.0, float(training.get("lr_scheduler_threshold", 1e-4))
                ),
                min_lr=max(0.0, float(training.get("lr_scheduler_min_lr", 1e-7))),
            )
        elif scheduler_type == "cosine_warmup":
            if self.max_optimizer_steps < 1:
                raise ValueError(
                    "training.max_optimizer_steps is required for cosine_warmup"
                )
            warmup_steps = max(0, int(training.get("warmup_steps", 0)))
            minimum_lr = max(
                0.0, float(training.get("lr_scheduler_min_lr", 0.0))
            )
            base_lr = float(training["learning_rate"])
            minimum_factor = min(1.0, minimum_lr / max(base_lr, 1e-12))

            def learning_rate_factor(step: int) -> float:
                if warmup_steps > 0 and step < warmup_steps:
                    return max(1, step + 1) / warmup_steps
                span = max(1, self.max_optimizer_steps - warmup_steps)
                progress = min(1.0, max(0.0, (step - warmup_steps) / span))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return minimum_factor + (1.0 - minimum_factor) * cosine

            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=learning_rate_factor,
            )
        else:
            self.lr_scheduler = None
        discriminator_parameters = list(self.patch_discriminator.parameters()) + list(
            self.wavelet_discriminator.parameters()
        )
        self.discriminator_optimizer = torch.optim.AdamW(
            discriminator_parameters,
            lr=float(config["training"].get("discriminator_learning_rate", 1e-4)),
            betas=(0.0, 0.99),
        )
        amp_enabled = bool(config["training"].get("amp", True) and self.device.type == "cuda")
        self.amp_enabled = amp_enabled
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        self.start_epoch = 0
        self.optimizer_step = 0
        self.resume_extra: dict[str, Any] = {}
        if resume:
            payload = load_checkpoint(resume, self.model, self.optimizer, strict=False)
            checkpoint_stage = payload.get("stage")
            if checkpoint_stage != self.stage:
                raise ValueError(
                    f"Cannot resume stage {self.stage!r} from checkpoint stage "
                    f"{checkpoint_stage!r}: {resume}"
                )
            self.start_epoch = int(payload["epoch"]) + 1
            extra = payload.get("extra", {})
            self.resume_extra = extra
            self.optimizer_step = int(extra.get("optimizer_step", 0))
            if (
                bool(training.get("enforce_init_checkpoint_lineage", False))
                and self.parent_checkpoint_sha256 is not None
            ):
                stored_parent = extra.get("parent_checkpoint_sha256")
                if stored_parent is None:
                    raise RuntimeError(
                        "Resume checkpoint has no parent lineage fingerprint. "
                        "Use a fresh output directory for this downstream stage."
                    )
                if stored_parent != self.parent_checkpoint_sha256:
                    raise RuntimeError(
                        "Resume checkpoint was initialized from a different parent. "
                        "Use a new stage output directory; do not mix checkpoint lineages."
                    )
            if "patch_discriminator" in extra:
                unwrap(self.patch_discriminator).load_state_dict(extra["patch_discriminator"])
            if "wavelet_discriminator" in extra:
                unwrap(self.wavelet_discriminator).load_state_dict(
                    extra["wavelet_discriminator"]
                )
            if "discriminator_optimizer" in extra:
                self.discriminator_optimizer.load_state_dict(extra["discriminator_optimizer"])
            if "scaler" in extra:
                scaler_state = extra["scaler"]
                if scaler_state:
                    self.scaler.load_state_dict(scaler_state)
            if self.lr_scheduler is not None and extra.get("lr_scheduler"):
                self.lr_scheduler.load_state_dict(extra["lr_scheduler"])
            if self.is_main:
                print(
                    f"[{self.stage}] resuming from {resume} at epoch "
                    f"{self.start_epoch + 1}",
                    flush=True,
                )
        ema_decay = float(training.get("ema_decay", 0.0))
        self.ema = ModelEMA(self.model, ema_decay) if ema_decay > 0 else None
        if self.ema is not None and self.resume_extra.get("ema"):
            self.ema.load_state_dict(self.resume_extra["ema"])
        self.use_ema_for_evaluation = bool(
            training.get("use_ema_for_evaluation", self.ema is not None)
        )

    @staticmethod
    def _duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:d}h{minutes:02d}m"
        if minutes:
            return f"{minutes:d}m{seconds:02d}s"
        return f"{seconds:d}s"

    @staticmethod
    def _checkpoint_improved(
        value: float,
        best: float | None,
        mode: str,
    ) -> bool:
        if best is None:
            return True
        if mode == "min":
            return value < best
        if mode == "max":
            return value > best
        raise ValueError("training.checkpoint_mode must be 'min' or 'max'")

    @contextmanager
    def _evaluation_parameters(self):
        if self.ema is None or not self.use_ema_for_evaluation:
            yield
            return
        with self.ema.average_parameters(self.model):
            yield

    def _synchronize_model_gradients(self) -> None:
        if not self.distributed:
            return
        for parameter in self.model.parameters():
            if parameter.requires_grad and parameter.grad is not None:
                torch.distributed.all_reduce(
                    parameter.grad, op=torch.distributed.ReduceOp.SUM
                )
                parameter.grad.div_(self.world_size)

    def _reduce_metrics(self, metrics: defaultdict[str, float]) -> dict[str, float]:
        if not self.distributed:
            return dict(metrics)
        reduced: dict[str, float] = {}
        for name, value in metrics.items():
            tensor = torch.tensor(value, device=self.device, dtype=torch.float64)
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
            reduced[name] = float(tensor / self.world_size)
        return reduced

    def _configure_stage(self) -> None:
        trainable_modules = self.config["training"].get("trainable_modules")
        if trainable_modules is None:
            configure_stage_trainability(self.model, self.stage)
            return
        if not isinstance(trainable_modules, list) or not all(
            isinstance(name, str) for name in trainable_modules
        ):
            raise ValueError("training.trainable_modules must be a list of module names")
        self.model.requires_grad_(False)
        available = dict(self.model.named_children())
        unknown = sorted(set(trainable_modules) - set(available))
        if unknown:
            raise ValueError(
                "training.trainable_modules contains unknown modules: "
                + ", ".join(unknown)
            )
        for name in trainable_modules:
            available[name].requires_grad_(True)

    def _optimizer_parameter_groups(self) -> list[dict[str, Any]]:
        """Build optional per-module learning-rate groups for delicate joint tuning."""
        training = self.config["training"]
        base_learning_rate = float(training["learning_rate"])
        multipliers = training.get("module_learning_rate_multipliers", {})
        if not multipliers:
            return [
                {
                    "params": [
                        parameter
                        for parameter in self.model.parameters()
                        if parameter.requires_grad
                    ],
                    "lr": base_learning_rate,
                }
            ]
        if not isinstance(multipliers, dict):
            raise ValueError(
                "training.module_learning_rate_multipliers must be a mapping"
            )
        named_children = dict(self.model.named_children())
        unknown = sorted(set(multipliers) - set(named_children))
        if unknown:
            raise ValueError(
                "Unknown module learning-rate multipliers: " + ", ".join(unknown)
            )
        groups: list[dict[str, Any]] = []
        grouped_ids: set[int] = set()
        for name, multiplier in multipliers.items():
            module_parameters = [
                parameter
                for parameter in named_children[name].parameters()
                if parameter.requires_grad
            ]
            if not module_parameters:
                continue
            groups.append(
                {
                    "params": module_parameters,
                    "lr": base_learning_rate * float(multiplier),
                    "name": name,
                }
            )
            grouped_ids.update(id(parameter) for parameter in module_parameters)
        remaining = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad and id(parameter) not in grouped_ids
        ]
        if remaining:
            groups.append(
                {"params": remaining, "lr": base_learning_rate, "name": "default"}
            )
        return groups

    def _loader(self, split: str) -> DataLoader:
        data = self.config["data"]
        is_train = split == "train"
        train_degradation_sampling = str(
            data.get("train_degradation_sampling", "random")
        )
        if train_degradation_sampling not in ("random", "fixed"):
            raise ValueError("data.train_degradation_sampling must be 'random' or 'fixed'")
        dataset = SentinelPatchDataset(
            data["manifest"],
            split=split,
            scale=self.config["model"].get("scale", 4),
            caption_file=data.get("captions"),
            caption_field=data.get("caption_field", "caption"),
            caption_sampling=(
                data.get("caption_sampling", "fixed") if is_train else "fixed"
            ),
            random_caption_fields=tuple(
                data.get(
                    "random_caption_fields",
                    ("brief", "descriptive", "analytical", "positional"),
                )
            ),
            augment=is_train,
            random_degradation=(
                is_train and train_degradation_sampling == "random"
            ),
            degradation_seed=int(data.get("degradation_seed", 0)),
            degradation_severity=data.get("degradation_severity", "mild"),
            target_key=data.get("target_key", "hr"),
            condition_key=data.get("condition_key"),
            radiometric_calibration=data.get("radiometric_calibration"),
            output_channels=self.config["model"].get("output_channels", 3),
            input_mode=data.get("input_mode", "synthetic"),
            paired_lr_crop_size=(
                data.get("paired_lr_crop_size") if is_train else None
            ),
        )
        sampler = (
            DistributedSampler(dataset, shuffle=split == "train")
            if self.distributed and len(dataset) > 0
            else None
        )
        return DataLoader(
            dataset,
            batch_size=int(self.config["training"]["batch_size"]),
            shuffle=sampler is None and split == "train",
            sampler=sampler,
            num_workers=int(self.config["training"].get("num_workers", 4)),
            pin_memory=self.device.type == "cuda",
            persistent_workers=int(self.config["training"].get("num_workers", 4)) > 0,
            drop_last=split == "train",
        )

    def _contexts(
        self, captions: list[str], training: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
        prompt_config = self.config.get("prompts", {})
        prompt_kinds = ["original"] * len(captions)
        if training:
            augmented = augment_prompts(
                captions,
                null_probability=float(prompt_config.get("null_probability", 0.4)),
                paraphrase_probability=float(
                    prompt_config.get("paraphrase_probability", 0.2)
                ),
                mismatch_probability=float(prompt_config.get("mismatch_probability", 0.1)),
                return_metadata=True,
            )
            assert isinstance(augmented, PromptBatch)
            captions = augmented.prompts
            prompt_kinds = augmented.kinds
        if self.text_encoder is None:
            context = torch.zeros(
                len(captions),
                1,
                self.model.context_dim,
                device=self.device,
            )
            return context, context.clone(), captions, prompt_kinds
        context = self.text_encoder(captions)
        null_context = self.text_encoder([""] * len(captions))
        return (
            context.to(self.device),
            null_context.to(self.device),
            captions,
            prompt_kinds,
        )

    def _sampled_joint_output(
        self,
        model: GeoDiffGAN,
        lr: torch.Tensor,
        context: torch.Tensor,
        degradation: torch.Tensor,
        base: torch.Tensor,
        lr_features: list[torch.Tensor],
        consistency_lr: torch.Tensor,
        diagnostics: DiagnosticRecorder | None,
    ) -> Any:
        training = self.config["training"]
        sample_steps = int(training.get("joint_sample_steps", 8))
        sample_count = int(training.get("trust_samples", 1))
        if sample_steps < 1:
            raise ValueError("training.joint_sample_steps must be positive")
        if sample_count < 1:
            raise ValueError("training.trust_samples must be positive")
        prepared_context, prepared_degradation = model.apply_ablation_inputs(
            context,
            degradation,
        )
        outputs = []
        for sample_index in range(sample_count):
            random_seed = int(
                torch.randint(0, 2**31 - 1, (), device="cpu").item()
            )
            generator = torch.Generator(device=self.device).manual_seed(random_seed)
            latent = model.sample_latent(
                lr,
                prepared_context,
                prepared_degradation,
                mode="sr",
                sample_steps=sample_steps,
                generator=generator,
                base=base,
                lr_features=lr_features,
                conditioning_prepared=True,
            )
            outputs.append(
                model.decode_latent(
                    latent,
                    lr,
                    prepared_context,
                    prepared_degradation,
                    mode="sr",
                    base=base,
                    projection_lr=consistency_lr,
                    back_projection_steps=self.train_back_projection_steps,
                    diagnostics=diagnostics if sample_index == 0 else None,
                    conditioning_prepared=True,
                    lr_features=lr_features,
                )
            )
        if len(outputs) == 1:
            return outputs[0]
        aggregated = model.aggregate_sr_outputs(
            outputs,
            lr_features,
            consistency_lr,
            prepared_degradation,
            back_projection_steps=self.train_back_projection_steps,
        )
        uncertainty = torch.stack(
            [value.pretrust_sr for value in outputs]
        ).var(dim=0, unbiased=False).mean(dim=1)
        image, _, _ = model.apply_uncertainty_abstention(
            aggregated.image,
            base,
            aggregated.evidence_confidence,
            uncertainty,
            trust_map=aggregated.trust_map,
        )
        return replace(aggregated, image=image)

    def _forward_stage(
        self,
        batch: dict[str, Any],
        diagnostics: DiagnosticRecorder | None = None,
        training: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        model: GeoDiffGAN = unwrap(self.model)  # type: ignore[assignment]
        hr = batch["hr"].to(self.device, non_blocking=True)
        lr = batch["lr"].to(self.device, non_blocking=True)
        lr_rgb_value = batch.get("lr_rgb")
        lr_rgb = (
            lr_rgb_value.to(self.device, non_blocking=True)
            if lr_rgb_value is not None
            else model.output_lr(lr)
        )
        consistency_lr_value = batch.get("clean_lr")
        consistency_lr = (
            consistency_lr_value.to(self.device, non_blocking=True)
            if consistency_lr_value is not None
            else lr_rgb
        )
        degradation = batch["degradation"].to(self.device, non_blocking=True)
        valid_mask_value = batch.get("valid_mask")
        valid_mask = (
            valid_mask_value.to(self.device, non_blocking=True).float()
            if valid_mask_value is not None
            else torch.ones_like(hr[:, :1])
        )
        valid_lr_mask_value = batch.get("valid_mask_lr")
        if valid_lr_mask_value is not None:
            valid_lr_mask = valid_lr_mask_value.to(
                self.device,
                non_blocking=True,
            ).float()
        else:
            valid_lr_mask = 1 - F.max_pool2d(
                1 - valid_mask,
                kernel_size=model.scale,
                stride=model.scale,
            )
        losses: dict[str, torch.Tensor] = {}
        if diagnostics is not None:
            diagnostics.capture("input.lr", lr, visual="rgb")
            if "lr_raw_rgb" in batch:
                diagnostics.capture(
                    "input.lr_raw_rgb",
                    batch["lr_raw_rgb"].to(self.device, non_blocking=True),
                    visual="rgb",
                )
            diagnostics.capture("target.hr", hr, visual="rgb")
            diagnostics.capture("target.valid_mask", valid_mask, visual="heatmap")
            diagnostics.capture("conditioning.degradation", degradation)

        if self.stage == "base":
            prediction = model.predict_base(lr)
            if diagnostics is not None:
                diagnostics.capture(
                    "base.bicubic",
                    F.interpolate(
                        lr_rgb,
                        scale_factor=model.scale,
                        mode="bicubic",
                        align_corners=False,
                    ),
                    visual="rgb",
                )
                diagnostics.capture("base.hr", prediction, visual="rgb")
                diagnostics.capture("output.hr", prediction, visual="rgb")
            losses["charbonnier"] = charbonnier(prediction, hr, mask=valid_mask)
            losses["mse"] = mse_loss(prediction, hr, mask=valid_mask)
            losses["multiscale_mse"] = multiscale_mse_loss(
                prediction, hr, mask=valid_mask
            )
            losses["ssim"] = 1 - ssim(prediction, hr, mask=valid_mask)
            losses["radiometric"] = radiometric_loss(
                prediction, hr, mask=valid_mask
            )
            losses["gradient"] = gradient_loss(prediction, hr, mask=valid_mask)
            losses["consistency"] = degradation_consistency(
                prediction,
                consistency_lr,
                degradation,
                scale=model.scale,
                severity=model.degradation_severity,
                mask=valid_lr_mask,
            )
            return prediction, losses

        context, _, used_prompts, prompt_kinds = self._contexts(
            list(batch["caption"]),
            training=training,
        )
        with torch.no_grad():
            base = model.predict_base(lr)
            target_residual = (hr - base) * valid_mask
        lr_features = model.lr_encoder(lr)
        if diagnostics is not None:
            diagnostics.capture("conditioning.text", context)
            diagnostics.capture("base.hr", base, visual="rgb")
            diagnostics.capture("target.residual", target_residual, visual="residual")
            for name, feature in zip(("f128", "f64", "f32", "f16"), lr_features):
                diagnostics.capture(f"lr_features.{name}", feature, visual="features")

        if self.stage == "vae":
            reconstruction, latent, mean, log_variance = model.vae(target_residual)
            mode_values = model.mode_tensor("sr", hr.shape[0], hr.device)
            mapped = model.mapper(latent, lr_features[1], context, mode_values)
            decoded = model.decoder(mapped, lr_features)
            fusion = model.fuse_sr_detail(
                decoded.detail_residual,
                mapped,
                lr_features,
                base,
                consistency_lr=consistency_lr,
                degradation=degradation,
            )
            detail = fusion["residual"]
            prediction = (base + detail).clamp(0, 1)
            ungated_prediction = (
                base + fusion["detail_residual"]
            ).clamp(0, 1)
            if diagnostics is not None:
                diagnostics.capture("vae.latent", latent, visual="features")
                diagnostics.capture("vae.mean", mean, visual="features")
                diagnostics.capture("vae.log_variance", log_variance, visual="features")
                diagnostics.capture(
                    "vae.residual_reconstruction", reconstruction, visual="residual"
                )
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
                diagnostics.capture(
                    "decoder.raw_detail_residual",
                    decoded.detail_residual,
                    visual="residual",
                )
                diagnostics.capture(
                    "decoder.raw_edit_residual",
                    decoded.edit_residual,
                    visual="residual",
                )
                diagnostics.capture("decoder.residual", detail, visual="residual")
                diagnostics.capture(
                    "mapper.base_referenced_trust",
                    fusion["trust_map"],
                    visual="heatmap",
                )
                diagnostics.capture("output.hr", prediction, visual="rgb")
            losses["vae_reconstruction"] = charbonnier(
                reconstruction,
                target_residual,
                mask=valid_mask,
            )
            losses["kl"] = kl_loss(mean, log_variance)
            losses["charbonnier"] = charbonnier(prediction, hr, mask=valid_mask)
            losses["mse"] = mse_loss(prediction, hr, mask=valid_mask)
            losses["multiscale_mse"] = multiscale_mse_loss(
                prediction, hr, mask=valid_mask
            )
            losses["gradient"] = gradient_loss(prediction, hr, mask=valid_mask)
            losses["wavelet"] = wavelet_loss(prediction, hr, mask=valid_mask)
            losses["ssim"] = 1 - ssim(prediction, hr, mask=valid_mask)
            losses["radiometric"] = radiometric_loss(
                prediction, hr, mask=valid_mask
            )
            losses["residual_supervision"] = residual_supervision_loss(
                (
                    fusion["evidence_residual"]
                    if model.use_base_referenced_trust
                    else detail
                ),
                base,
                hr,
                mask=valid_mask,
            )
            losses["base_guard"] = base_guard_loss(
                prediction,
                hr,
                base,
                margin=float(
                    self.config["training"].get("base_guard_margin", 0.0)
                ),
                mask=valid_mask,
            )
            fidelity_weights = self.config["training"].get("loss_weights", {})
            losses["spatial_base_guard"] = (
                spatial_base_guard_loss(
                    prediction,
                    hr,
                    base,
                    margin=float(
                        self.config["training"].get(
                            "spatial_base_guard_margin", 0.0
                        )
                    ),
                    smoothing_window=int(
                        self.config["training"].get("trust_smoothing_window", 9)
                    ),
                    mask=valid_mask,
                )
                if float(fidelity_weights.get("spatial_base_guard", 0.0)) > 0
                else prediction.new_zeros(())
            )
            losses["trust_projection"] = (
                residual_trust_projection_loss(
                    fusion["trust_map"],
                    fusion["evidence_residual"],
                    base,
                    hr,
                    maximum_scale=model.trust_maximum_scale,
                    smoothing_window=int(
                        self.config["training"].get("trust_smoothing_window", 9)
                    ),
                    ridge=float(
                        self.config["training"].get("trust_ridge", 1e-8)
                    ),
                    mask=valid_mask,
                )
                if model.use_base_referenced_trust
                and float(fidelity_weights.get("trust_projection", 0.0)) > 0
                else prediction.new_zeros(())
            )
            losses["evidence_calibration"] = evidence_calibration_loss(
                mapped.evidence_confidence,
                ungated_prediction,
                hr,
                temperature=float(
                    self.config["training"].get(
                        "evidence_calibration_temperature", 0.05
                    )
                ),
                selectivity_weight=float(
                    self.config["training"].get(
                        "evidence_selectivity_weight",
                        0.0,
                    )
                ),
                mask=valid_mask,
            )
            losses["evidence_improvement"] = evidence_improvement_loss(
                mapped.evidence_confidence,
                ungated_prediction,
                base,
                hr,
                temperature=float(
                    self.config["training"].get(
                        "evidence_improvement_temperature", 0.01
                    )
                ),
                mask=valid_mask,
            )
            return prediction, losses

        sampled_joint = (
            self.stage == "joint"
            and str(
                self.config["training"].get("joint_latent_source", "noisy")
            )
            == "sampled"
        )
        if sampled_joint:
            output = self._sampled_joint_output(
                model,
                lr,
                context,
                degradation,
                base,
                lr_features,
                consistency_lr,
                diagnostics,
            )
            losses["diffusion"] = output.image.new_zeros(())
        else:
            with torch.no_grad():
                latent, _, _ = model.vae.encode(target_residual, sample=False)
            diffusion_timesteps = None
            if self.stage in ("joint", "edit"):
                maximum_fraction = float(
                    self.config["training"].get(
                        "joint_max_timestep_fraction", 1.0
                    )
                )
                if not 0 < maximum_fraction <= 1:
                    raise ValueError(
                        "training.joint_max_timestep_fraction must be in (0, 1]"
                    )
                maximum_timestep = max(
                    1,
                    min(
                        model.scheduler.steps,
                        int(round(model.scheduler.steps * maximum_fraction)),
                    ),
                )
                diffusion_timesteps = torch.randint(
                    0,
                    maximum_timestep,
                    (latent.shape[0],),
                    device=latent.device,
                )
            diffusion_batch = model.prepare_diffusion_batch(
                latent,
                timesteps=diffusion_timesteps,
            )
            mode = "edit" if self.stage == "edit" else "sr"
            velocity = model.predict_velocity(
                diffusion_batch.noisy,
                diffusion_batch.timesteps,
                context,
                degradation,
                mode,
                lr_features,
            )
            if diagnostics is not None:
                diagnostics.capture("diffusion.clean_latent", latent, visual="features")
                diagnostics.capture("diffusion.noisy_latent", diffusion_batch.noisy, visual="features")
                diagnostics.capture(
                    "diffusion.target_velocity",
                    diffusion_batch.target_velocity,
                    visual="features",
                )
                diagnostics.capture("diffusion.predicted_velocity", velocity, visual="features")
                diagnostics.capture(
                    "diffusion.velocity_absolute_error",
                    (velocity - diffusion_batch.target_velocity).abs(),
                    visual="heatmap",
                )
                diagnostics.scalar(
                    "diffusion.timestep_mean", diffusion_batch.timesteps.float().mean()
                )
            losses["diffusion"] = snr_weighted_velocity_loss(
                velocity,
                diffusion_batch.target_velocity,
                diffusion_batch.timesteps,
                model.scheduler.alphas_cumprod,
            )
            if self.stage == "diffusion":
                clean = model.scheduler.predict_clean(
                    diffusion_batch.noisy, velocity, diffusion_batch.timesteps
                )
                prediction = model.vae.decode(clean, hr.shape[-2:])
                if diagnostics is not None:
                    diagnostics.capture("latent.denoised", clean, visual="features")
                    diagnostics.capture(
                        "diffusion.decoded_residual", prediction, visual="residual"
                    )
                    diagnostics.capture(
                        "diffusion.decoded_absolute_error",
                        (prediction - target_residual).abs(),
                        visual="heatmap",
                    )
                return prediction, losses

            clean = model.scheduler.predict_clean(
                diffusion_batch.noisy, velocity, diffusion_batch.timesteps
            )
            output = model.decode_latent(
                clean,
                lr,
                context,
                degradation,
                mode=mode,
                base=base,
                projection_lr=consistency_lr,
                back_projection_steps=self.train_back_projection_steps,
                diagnostics=diagnostics,
            )
        prediction = output.image
        counterfactual = torch.tensor(
            [kind == "mismatch" for kind in prompt_kinds],
            device=prediction.device,
            dtype=torch.bool,
        )
        reconstruction_samples = (
            ~counterfactual
            if self.stage == "edit"
            else torch.ones_like(counterfactual)
        )
        if reconstruction_samples.any():
            reconstruction = prediction[reconstruction_samples]
            reconstruction_target = hr[reconstruction_samples]
            reconstruction_valid = valid_mask[reconstruction_samples]
            losses["charbonnier"] = charbonnier(
                reconstruction,
                reconstruction_target,
                mask=reconstruction_valid,
            )
            losses["mse"] = mse_loss(
                reconstruction,
                reconstruction_target,
                mask=reconstruction_valid,
            )
            losses["multiscale_mse"] = multiscale_mse_loss(
                reconstruction,
                reconstruction_target,
                mask=reconstruction_valid,
            )
            losses["ssim"] = 1 - ssim(
                reconstruction,
                reconstruction_target,
                mask=reconstruction_valid,
            )
            if self.perceptual is not None:
                perceptual_reconstruction = (
                    reconstruction * reconstruction_valid
                    + reconstruction_target * (1 - reconstruction_valid)
                )
                losses["perceptual"] = self.perceptual(
                    perceptual_reconstruction,
                    reconstruction_target,
                )
            else:
                losses["perceptual"] = reconstruction.new_zeros(())
            losses["gradient"] = gradient_loss(
                reconstruction,
                reconstruction_target,
                mask=reconstruction_valid,
            )
            losses["wavelet"] = wavelet_loss(
                reconstruction,
                reconstruction_target,
                mask=reconstruction_valid,
            )
            losses["radiometric"] = radiometric_loss(
                reconstruction,
                reconstruction_target,
                mask=reconstruction_valid,
            )
            losses["residual_supervision"] = residual_supervision_loss(
                (
                    output.evidence_residual[reconstruction_samples]
                    if model.use_base_referenced_trust
                    else output.residual[reconstruction_samples]
                ),
                base[reconstruction_samples],
                reconstruction_target,
                mask=reconstruction_valid,
            )
            losses["base_guard"] = base_guard_loss(
                reconstruction,
                reconstruction_target,
                base[reconstruction_samples],
                margin=float(
                    self.config["training"].get("base_guard_margin", 0.0)
                ),
                mask=reconstruction_valid,
            )
            fidelity_weights = self.config["training"].get("loss_weights", {})
            losses["spatial_base_guard"] = (
                spatial_base_guard_loss(
                    reconstruction,
                    reconstruction_target,
                    base[reconstruction_samples],
                    margin=float(
                        self.config["training"].get(
                            "spatial_base_guard_margin", 0.0
                        )
                    ),
                    smoothing_window=int(
                        self.config["training"].get("trust_smoothing_window", 9)
                    ),
                    mask=reconstruction_valid,
                )
                if float(fidelity_weights.get("spatial_base_guard", 0.0)) > 0
                else reconstruction.new_zeros(())
            )
            losses["local_excess_mse"] = (
                local_excess_mse_loss(
                    reconstruction,
                    reconstruction_target,
                    base[reconstruction_samples],
                    margin=float(
                        self.config["training"].get(
                            "local_excess_mse_margin", 0.0
                        )
                    ),
                    smoothing_window=int(
                        self.config["training"].get("trust_smoothing_window", 9)
                    ),
                    mask=reconstruction_valid,
                )
                if float(fidelity_weights.get("local_excess_mse", 0.0)) > 0
                else reconstruction.new_zeros(())
            )
            losses["trust_projection"] = (
                residual_trust_projection_loss(
                    output.trust_map[reconstruction_samples],
                    output.evidence_residual[reconstruction_samples],
                    base[reconstruction_samples],
                    reconstruction_target,
                    maximum_scale=model.trust_maximum_scale,
                    smoothing_window=int(
                        self.config["training"].get("trust_smoothing_window", 9)
                    ),
                    ridge=float(
                        self.config["training"].get("trust_ridge", 1e-8)
                    ),
                    mask=reconstruction_valid,
                )
                if model.use_base_referenced_trust
                and float(fidelity_weights.get("trust_projection", 0.0)) > 0
                else reconstruction.new_zeros(())
            )
        else:
            zero = prediction.new_zeros(())
            losses["charbonnier"] = zero
            losses["mse"] = zero
            losses["multiscale_mse"] = zero
            losses["ssim"] = zero
            losses["perceptual"] = zero
            losses["gradient"] = zero
            losses["wavelet"] = zero
            losses["radiometric"] = zero
            losses["residual_supervision"] = zero
            losses["base_guard"] = zero
            losses["spatial_base_guard"] = zero
            losses["local_excess_mse"] = zero
            losses["trust_projection"] = zero
        losses["consistency"] = degradation_consistency(
            prediction,
            consistency_lr,
            degradation,
            scale=model.scale,
            severity=model.degradation_severity,
            mask=valid_lr_mask,
        )
        losses["evidence_calibration"] = evidence_calibration_loss(
            output.evidence_confidence,
            output.ungated_sr,
            hr,
            temperature=float(
                self.config["training"].get(
                    "evidence_calibration_temperature", 0.05
                )
            ),
            selectivity_weight=float(
                self.config["training"].get(
                    "evidence_selectivity_weight",
                    0.0,
                )
            ),
            mask=valid_mask,
        )
        losses["evidence_improvement"] = evidence_improvement_loss(
            output.evidence_confidence,
            output.ungated_sr,
            base,
            hr,
            temperature=float(
                self.config["training"].get(
                    "evidence_improvement_temperature", 0.01
                )
            ),
            mask=valid_mask,
        )
        if self.stage == "edit":
            losses["edit_localization"] = edit_localization_loss(
                output.raw_edit_residual,
                output.edit_permission,
            )
            permission_mean = output.edit_permission.flatten(1).mean(dim=1)
            permission_loss = permission_mean.new_zeros(())
            if (~counterfactual).any():
                permission_loss = permission_loss + permission_mean[
                    ~counterfactual
                ].mean()
            if counterfactual.any():
                target_coverage = float(
                    self.config["training"].get(
                        "counterfactual_edit_coverage", 0.15
                    )
                )
                permission_loss = permission_loss + (
                    permission_mean[counterfactual] - target_coverage
                ).square().mean()
            losses["edit_permission"] = permission_loss
            assert self.text_encoder is not None
            losses["prompt_alignment"] = self.text_encoder.alignment_loss(
                prediction, used_prompts
            )
        return prediction, losses

    @torch.no_grad()
    def _validate(
        self,
        loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        if len(loader.dataset) == 0:
            return {}
        training = self.config["training"]
        configured_limit = training.get("validation_limit")
        limit = (
            min(len(loader), int(configured_limit))
            if configured_limit is not None
            else len(loader)
        )
        if limit < 1:
            return {}
        self.model.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        totals: defaultdict[str, float] = defaultdict(float)
        seed = int(self.config.get("seed", 42)) + 10_000
        if not bool(training.get("validation_fixed_seed", False)):
            seed += epoch
        cuda_devices = (
            [self.device.index or 0] if self.device.type == "cuda" else []
        )
        progress = tqdm(
            enumerate(islice(loader, limit)),
            total=limit,
            desc=f"{self.stage} validation {epoch + 1}",
            unit="batch",
            leave=False,
            disable=(
                not self.is_main
                or training.get("progress_mode", "compact") != "tqdm"
            ),
        )
        validation_sample_steps = max(
            0, int(training.get("validation_sample_steps", 0))
        )
        validation_samples = max(
            1, int(training.get("validation_samples", 1))
        )
        validation_back_projection_steps = max(
            0, int(training.get("validation_back_projection_steps", 0))
        )
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            for step, batch in progress:
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.amp_enabled,
                ):
                    prediction, losses = self._forward_stage(
                        batch,
                        training=False,
                    )
                    total_loss = self._weighted_loss(losses)
                    if (
                        self.stage in ("joint", "edit")
                        and validation_sample_steps > 0
                    ):
                        model: GeoDiffGAN = unwrap(self.model)  # type: ignore[assignment]
                        lr = batch["lr"].to(self.device, non_blocking=True)
                        degradation = batch["degradation"].to(
                            self.device, non_blocking=True
                        )
                        projection_lr_value = batch.get("clean_lr")
                        projection_lr = (
                            projection_lr_value.to(
                                self.device, non_blocking=True
                            )
                            if projection_lr_value is not None
                            else model.output_lr(lr)
                        )
                        context, _, _, _ = self._contexts(
                            list(batch["caption"]), training=False
                        )
                        base = model.predict_base(lr)
                        lr_features = model.lr_encoder(lr)
                        sampled_outputs = []
                        for sample_index in range(validation_samples):
                            generator = torch.Generator(
                                device=self.device
                            ).manual_seed(seed + step * validation_samples + sample_index)
                            sampled_outputs.append(
                                model.sample(
                                    lr,
                                    context,
                                    degradation=degradation,
                                    projection_lr=projection_lr,
                                    mode=("edit" if self.stage == "edit" else "sr"),
                                    sample_steps=validation_sample_steps,
                                    back_projection_steps=(
                                        validation_back_projection_steps
                                    ),
                                    generator=generator,
                                    base=base,
                                    lr_features=lr_features,
                                )
                            )
                        if (
                            self.stage == "joint"
                            and model.trust_mode == "per_band"
                        ):
                            aggregated = model.aggregate_sr_outputs(
                                sampled_outputs,
                                lr_features,
                                projection_lr,
                                degradation,
                                back_projection_steps=(
                                    validation_back_projection_steps
                                ),
                            )
                            prediction = aggregated.image
                        else:
                            prediction = torch.stack(
                                [output.image for output in sampled_outputs]
                            ).mean(dim=0)
                        if self.stage == "joint" and validation_samples > 1:
                            stack = torch.stack(
                                [
                                    output.pretrust_sr
                                    if model.trust_mode == "per_band"
                                    else output.image
                                    for output in sampled_outputs
                                ]
                            )
                            uncertainty = stack.var(
                                dim=0, unbiased=False
                            ).mean(dim=1)
                            if model.trust_mode == "per_band":
                                evidence = aggregated.evidence_confidence
                                trust = aggregated.trust_map
                            else:
                                evidence = torch.stack(
                                    [
                                        output.evidence_confidence
                                        for output in sampled_outputs
                                    ]
                                ).mean(dim=0)
                                trust = torch.stack(
                                    [output.trust_map for output in sampled_outputs]
                                ).mean(dim=0)
                            prediction, _, _ = model.apply_uncertainty_abstention(
                                prediction,
                                base,
                                evidence,
                                uncertainty,
                                trust_map=trust,
                            )
                for name, value in losses.items():
                    totals[f"loss_{name}"] += float(value.detach())
                totals["loss_total"] += float(total_loss.detach())
                if self.stage != "diffusion":
                    model: GeoDiffGAN = unwrap(self.model)  # type: ignore[assignment]
                    metric_lr_value = batch.get("clean_lr")
                    if metric_lr_value is None:
                        metric_lr = model.output_lr(
                            batch["lr"].to(self.device, non_blocking=True)
                        )
                    else:
                        metric_lr = metric_lr_value.to(
                            self.device,
                            non_blocking=True,
                        )
                    values = basic_metrics(
                        prediction,
                        batch["hr"].to(self.device, non_blocking=True),
                        metric_lr,
                        batch["degradation"].to(
                            self.device,
                            non_blocking=True,
                        ),
                        scale=model.scale,
                        severity=model.degradation_severity,
                        mask=(
                            batch["valid_mask"].to(
                                self.device,
                                non_blocking=True,
                            )
                            if "valid_mask" in batch
                            else None
                        ),
                        lr_mask=(
                            batch["valid_mask_lr"].to(
                                self.device,
                                non_blocking=True,
                            )
                            if "valid_mask_lr" in batch
                            else None
                        ),
                    )
                    for name, value in values.items():
                        totals[name] += value
                    if self.stage not in ("base", "diffusion"):
                        validation_base = model.predict_base(
                            batch["lr"].to(self.device, non_blocking=True)
                        )
                        base_values = basic_metrics(
                            validation_base,
                            batch["hr"].to(self.device, non_blocking=True),
                            metric_lr,
                            batch["degradation"].to(
                                self.device, non_blocking=True
                            ),
                            scale=model.scale,
                            severity=model.degradation_severity,
                            mask=(
                                batch["valid_mask"].to(
                                    self.device, non_blocking=True
                                )
                                if "valid_mask" in batch
                                else None
                            ),
                            lr_mask=(
                                batch["valid_mask_lr"].to(
                                    self.device, non_blocking=True
                                )
                                if "valid_mask_lr" in batch
                                else None
                            ),
                        )
                        for name, value in base_values.items():
                            totals[f"base_{name}"] += value
                        totals["psnr_gain_vs_base"] += (
                            values["psnr"] - base_values["psnr"]
                        )
                        totals["ssim_gain_vs_base"] += (
                            values["ssim"] - base_values["ssim"]
                        )
                        totals["base_improvement_rate"] += float(
                            values["l1"] < base_values["l1"]
                        )
                completed = step + 1
                progress.set_postfix(
                    loss=f"{totals['loss_total'] / completed:.4f}",
                    psnr=(
                        f"{totals['psnr'] / completed:.2f}"
                        if "psnr" in totals
                        else "n/a"
                    ),
                )
        reduced = self._reduce_metrics(totals)
        denominator = max(limit, 1)
        return {
            f"val_{name}": value / denominator
            for name, value in reduced.items()
        }

    def _weighted_loss(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        weights = self.config["training"].get("loss_weights", {})
        default_weights = {
            "charbonnier": 1.0,
            "mse": 0.0,
            "multiscale_mse": 0.0,
            "consistency": 1.0,
            "ssim": 0.2,
            "gradient": 0.1,
            "perceptual": 0.1,
            "wavelet": 0.05,
            "kl": 1e-4,
            "vae_reconstruction": 1.0,
            "diffusion": 1.0,
            "evidence_calibration": 0.1,
            "evidence_improvement": 0.0,
            "radiometric": 0.0,
            "residual_supervision": 0.0,
            "base_guard": 0.0,
            "spatial_base_guard": 0.0,
            "local_excess_mse": 0.0,
            "trust_projection": 0.0,
            "edit_localization": 0.05,
            "edit_permission": 0.05,
            "prompt_alignment": 0.05,
            "adversarial": 0.01,
        }
        return sum(
            losses[name] * float(weights.get(name, default_weights.get(name, 1.0)))
            for name in losses
        )

    @contextmanager
    def _frozen_discriminators(self):
        modules = (
            unwrap(self.patch_discriminator),
            unwrap(self.wavelet_discriminator),
        )
        training_states = [module.training for module in modules]
        parameters = [
            parameter for module in modules for parameter in module.parameters()
        ]
        states = [parameter.requires_grad for parameter in parameters]
        try:
            for module in modules:
                module.eval()
            for parameter in parameters:
                parameter.requires_grad_(False)
            yield modules
        finally:
            for parameter, state in zip(parameters, states):
                parameter.requires_grad_(state)
            for module, state in zip(modules, training_states):
                module.train(state)

    def _generator_adversarial_loss(
        self, prediction: torch.Tensor, lr: torch.Tensor
    ) -> torch.Tensor:
        # Bypass DDP wrappers here: only the gradient with respect to prediction is needed.
        with self._frozen_discriminators() as (patch, wavelet):
            return generator_hinge(patch(prediction, lr)) + generator_hinge(
                wavelet(prediction, lr)
            )

    def _discriminator_loss(
        self, prediction: torch.Tensor, hr: torch.Tensor, lr: torch.Tensor
    ) -> torch.Tensor:
        real_outputs = self.patch_discriminator(hr, lr)
        fake_outputs = self.patch_discriminator(prediction.detach(), lr)
        real_wavelet = self.wavelet_discriminator(hr, lr)
        fake_wavelet = self.wavelet_discriminator(prediction.detach(), lr)
        loss = discriminator_hinge(real_outputs, fake_outputs)
        return loss + discriminator_hinge(real_wavelet, fake_wavelet)

    def train(self) -> None:
        loader = self._loader("train")
        if len(loader.dataset) == 0:
            raise ValueError(
                "The training split contains no patches. Check SAFE prefix rules "
                "and UNMATCHED_SAFE_SPLIT before starting training."
            )
        validation_loader = self._loader("val")
        training = self.config["training"]
        adversarial_weight = float(
            training.get("loss_weights", {}).get("adversarial", 0.01)
        )
        adversarial_enabled = (
            self.stage in ("joint", "edit") and adversarial_weight > 0
        )
        accumulation = int(training.get("gradient_accumulation", 1))
        if accumulation < 1:
            raise ValueError("training.gradient_accumulation must be at least 1")
        epochs = int(training["epochs"])
        if self.max_optimizer_steps > self.optimizer_step:
            updates_per_epoch = max(1, math.ceil(len(loader) / accumulation))
            remaining_updates = self.max_optimizer_steps - self.optimizer_step
            required_epochs = self.start_epoch + math.ceil(
                remaining_updates / updates_per_epoch
            )
            epochs = max(epochs, required_epochs)
            training["epochs"] = epochs
        output_dir = Path(training.get("output_dir", "runs/default"))
        if self.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
                json.dump(self.config, handle, indent=2)
        validate_every = max(1, int(training.get("validate_every", 1)))
        keep_best_and_latest = bool(
            training.get("keep_best_and_latest", True)
        )
        checkpoint_metric = str(
            training.get("checkpoint_metric", "val_l1")
        )
        checkpoint_mode = str(
            training.get("checkpoint_mode", "min")
        ).lower()
        if checkpoint_mode not in ("min", "max"):
            raise ValueError("training.checkpoint_mode must be 'min' or 'max'")
        best_checkpoint_path = output_dir / f"{self.stage}_best.pt"
        best_checkpoint_value: float | None = None
        best_checkpoint_metric: str | None = None
        checkpoint_selection_mode = checkpoint_mode
        existing_best = best_stage_checkpoint(output_dir, self.stage)
        if existing_best is not None:
            best_payload = torch.load(
                existing_best,
                map_location="cpu",
                weights_only=False,
            )
            selection = best_payload.get("extra", {}).get(
                "checkpoint_selection",
                {},
            )
            if (
                selection.get("mode") in ("min", "max")
                and selection.get("value") is not None
            ):
                best_checkpoint_metric = str(selection["metric"])
                checkpoint_selection_mode = str(selection["mode"])
                best_checkpoint_value = float(selection["value"])
        early_stopping_patience = max(
            0,
            int(training.get("early_stopping_patience", 0)),
        )
        early_stopping_min_epochs = max(
            0,
            int(training.get("early_stopping_min_epochs", 0)),
        )
        early_stopping_min_delta = max(
            0.0,
            float(training.get("early_stopping_min_delta", 0.0)),
        )
        early_stopping_metric = str(
            training.get("early_stopping_metric", checkpoint_metric)
        )
        early_stopping_mode = str(
            training.get("early_stopping_mode", checkpoint_mode)
        ).lower()
        if early_stopping_mode not in ("min", "max"):
            raise ValueError(
                "training.early_stopping_mode must be 'min' or 'max'"
            )
        active_early_stopping_metric = early_stopping_metric
        active_early_stopping_mode = early_stopping_mode
        resumed_early_state = self.resume_extra.get("early_stopping", {})
        if (
            resumed_early_state.get("mode") in ("min", "max")
            and resumed_early_state.get("best_value") is not None
        ):
            active_early_stopping_metric = str(resumed_early_state["metric"])
            active_early_stopping_mode = str(resumed_early_state["mode"])
            early_best_value = float(resumed_early_state["best_value"])
        else:
            early_best_value = None
        early_bad_epochs = (
            int(resumed_early_state.get("bad_epochs", 0))
            if early_best_value is not None
            else 0
        )
        progress_mode = str(training.get("progress_mode", "compact")).lower()
        if progress_mode not in ("compact", "tqdm", "quiet"):
            raise ValueError(
                "training.progress_mode must be compact, tqdm, or quiet"
            )
        progress_updates = max(
            0,
            int(training.get("progress_updates_per_epoch", 2)),
        )
        use_tqdm = progress_mode == "tqdm" and self.is_main
        if (
            self.max_optimizer_steps > 0
            and self.optimizer_step >= self.max_optimizer_steps
        ):
            if self.is_main:
                print(
                    f"[{self.stage}] already complete: optimizer_steps="
                    f"{self.optimizer_step}/{self.max_optimizer_steps}",
                    flush=True,
                )
            return
        if self.start_epoch >= epochs:
            if self.is_main:
                checkpoint = latest_stage_checkpoint(output_dir, self.stage)
                print(
                    f"[{self.stage}] already complete: {self.start_epoch}/{epochs} "
                    f"epochs; checkpoint={checkpoint}",
                    flush=True,
                )
            return
        stage_started = time.monotonic()
        completed_epoch_durations: list[float] = []
        epoch_progress = tqdm(
            range(self.start_epoch, epochs),
            desc=f"{self.stage} epochs",
            unit="epoch",
            disable=not use_tqdm,
        )
        for epoch in epoch_progress:
            epoch_started = time.monotonic()
            if isinstance(loader.sampler, DistributedSampler):
                loader.sampler.set_epoch(epoch)
            metrics: defaultdict[str, float] = defaultdict(float)
            completed_batches = 0
            reached_max_optimizer_steps = False
            debug_exports = 0
            self.model.train()
            if self.text_encoder is not None:
                self.text_encoder.eval()
            self.optimizer.zero_grad(set_to_none=True)
            if adversarial_enabled:
                self.patch_discriminator.train()
                self.wavelet_discriminator.train()
                self.discriminator_optimizer.zero_grad(set_to_none=True)
            batch_progress = tqdm(
                loader,
                desc=f"{self.stage} train {epoch + 1}/{epochs}",
                unit="batch",
                leave=False,
                disable=not use_tqdm,
            )
            compact_interval = (
                max(1, math.ceil(len(loader) / progress_updates))
                if progress_mode == "compact" and progress_updates > 0
                else None
            )
            for step, batch in enumerate(batch_progress):
                group_start = (step // accumulation) * accumulation
                group_size = min(accumulation, len(loader) - group_start)
                should_step = (step + 1) % accumulation == 0 or step + 1 == len(loader)
                debug_config = self.config.get("debug", {})
                debug_enabled = bool(debug_config.get("enabled", False)) and self.is_main
                debug_every = max(1, int(debug_config.get("every_n_steps", 100)))
                max_debug_exports = max(
                    1,
                    int(debug_config.get("max_exports_per_epoch", 5)),
                )
                diagnostics = (
                    DiagnosticRecorder(
                        Path(debug_config.get("output_dir", output_dir / "debug"))
                        / self.stage
                        / f"epoch_{epoch:04d}"
                        / f"step_{step:06d}",
                        verbose=bool(debug_config.get("print_tensor_stats", True)),
                        fail_on_nonfinite=bool(
                            debug_config.get("fail_on_nonfinite", True)
                        ),
                        panel_size=int(debug_config.get("panel_size", 320)),
                        histogram_bins=int(
                            debug_config.get("histogram_bins", 64)
                        ),
                        max_feature_channels=int(
                            debug_config.get("max_feature_channels", 16)
                        ),
                        save_tensors=bool(
                            debug_config.get("save_tensors", False)
                        ),
                        max_saved_tensors=int(
                            debug_config.get("max_saved_tensors", 32)
                        ),
                    )
                    if (
                        debug_enabled
                        and step % debug_every == 0
                        and debug_exports < max_debug_exports
                    )
                    else None
                )
                if diagnostics is not None:
                    debug_exports += 1
                hr = batch["hr"].to(self.device, non_blocking=True)
                lr = batch["lr"].to(self.device, non_blocking=True)
                valid_mask = (
                    batch["valid_mask"].to(self.device, non_blocking=True).float()
                    if "valid_mask" in batch
                    else torch.ones_like(hr[:, :1])
                )
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.amp_enabled,
                ):
                    prediction, losses = self._forward_stage(batch, diagnostics)
                    generator_loss = self._weighted_loss(losses)
                    discriminator_loss = None
                    if adversarial_enabled:
                        adversarial_prediction = (
                            prediction * valid_mask + hr * (1 - valid_mask)
                        )
                        adversarial = self._generator_adversarial_loss(
                            adversarial_prediction,
                            lr,
                        )
                        losses["adversarial"] = adversarial
                        generator_loss = generator_loss + adversarial * adversarial_weight
                        discriminator_loss = self._discriminator_loss(
                            adversarial_prediction,
                            hr,
                            lr,
                        )
                    scaled_loss = generator_loss / group_size
                    if diagnostics is not None:
                        for name, value in losses.items():
                            diagnostics.scalar(f"loss.{name}", value)
                        diagnostics.scalar("loss.total", generator_loss)
                        if discriminator_loss is not None:
                            diagnostics.scalar("loss.discriminator", discriminator_loss)
                self.scaler.scale(scaled_loss).backward()
                if discriminator_loss is not None:
                    self.scaler.scale(discriminator_loss / group_size).backward()
                    metrics["discriminator"] += float(discriminator_loss.detach())
                if should_step:
                    self.scaler.unscale_(self.optimizer)
                    self._synchronize_model_gradients()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        float(training.get("gradient_clip", 1.0)),
                    )
                    if diagnostics is not None:
                        diagnostics.scalar("gradient.global_l2_before_clip", gradient_norm)
                    self.scaler.step(self.optimizer)
                    if discriminator_loss is not None:
                        self.scaler.unscale_(self.discriminator_optimizer)
                        self.scaler.step(self.discriminator_optimizer)
                    self.scaler.update()
                    self.optimizer_step += 1
                    if self.ema is not None:
                        self.ema.update(self.model)
                    if (
                        self.lr_scheduler is not None
                        and self.lr_scheduler_step_per_update
                    ):
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if adversarial_enabled:
                        self.discriminator_optimizer.zero_grad(set_to_none=True)
                    reached_max_optimizer_steps = (
                        self.max_optimizer_steps > 0
                        and self.optimizer_step >= self.max_optimizer_steps
                    )
                if diagnostics is not None:
                    if self.stage != "diffusion":
                        if self.stage == "base":
                            debug_base = prediction
                            debug_residual = prediction - torch.nn.functional.interpolate(
                                batch.get("lr_rgb", batch["lr"]).to(self.device)[
                                    :, : unwrap(self.model).output_channels
                                ],
                                size=prediction.shape[-2:],
                                mode="bicubic",
                                align_corners=False,
                            )
                        else:
                            debug_base = unwrap(self.model).predict_base(lr)
                            debug_residual = prediction - debug_base
                        debug_lr_value = batch.get("lr_rgb")
                        debug_lr = (
                            debug_lr_value.to(self.device)
                            if debug_lr_value is not None
                            else unwrap(self.model).output_lr(lr)
                        )
                        diagnostics.add_spatial_metrics(
                            debug_lr,
                            debug_base,
                            debug_residual,
                            prediction,
                            batch["degradation"].to(self.device),
                            scale=unwrap(self.model).scale,
                            target=hr,
                            consistency_lr=batch.get(
                                "clean_lr", debug_lr
                            ).to(self.device),
                            degradation_severity=unwrap(
                                self.model
                            ).degradation_severity,
                            valid_mask=valid_mask,
                            valid_lr_mask=(
                                batch["valid_mask_lr"].to(self.device)
                                if "valid_mask_lr" in batch
                                else None
                            ),
                        )
                    diagnostics.export(
                        {
                            "stage": self.stage,
                            "epoch": epoch,
                            "step": step,
                            "patch": list(batch["patch"]),
                            "tile_id": list(batch["tile_id"]),
                        }
                    )
                for name, value in losses.items():
                    metrics[name] += float(value.detach())
                metrics["total"] += float(generator_loss.detach())
                completed = step + 1
                completed_batches = completed
                if use_tqdm:
                    batch_progress.set_postfix(
                        loss=f"{float(generator_loss.detach()):.4f}",
                        avg=f"{metrics['total'] / completed:.4f}",
                        lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    )
                elif (
                    self.is_main
                    and compact_interval is not None
                    and (
                        completed % compact_interval == 0
                        or completed == len(loader)
                    )
                ):
                    elapsed = time.monotonic() - epoch_started
                    batch_eta = elapsed / completed * (len(loader) - completed)
                    print(
                        f"[{self.stage}] epoch {epoch + 1}/{epochs} "
                        f"batch {completed}/{len(loader)} "
                        f"({completed / len(loader):.0%}) "
                        f"loss={metrics['total'] / completed:.4f} "
                        f"epoch_eta={self._duration(batch_eta)}",
                        flush=True,
                    )
                if reached_max_optimizer_steps:
                    break
            reduced_metrics = self._reduce_metrics(metrics)
            validation_metrics = {}
            if (
                len(validation_loader.dataset) > 0
                and (epoch + 1) % validate_every == 0
            ):
                with self._evaluation_parameters():
                    validation_metrics = self._validate(validation_loader, epoch)
            stop_training = reached_max_optimizer_steps
            if self.is_main:
                denominator = max(completed_batches, 1)
                epoch_metrics = {
                    key: value / denominator for key, value in reduced_metrics.items()
                }
                epoch_metrics.update(validation_metrics)
                checkpoint_path = output_dir / f"{self.stage}_epoch_{epoch:04d}.pt"
                selection_metric = checkpoint_metric
                selection_mode = checkpoint_mode
                selection_value = epoch_metrics.get(selection_metric)
                if selection_value is None:
                    if "val_loss_total" in epoch_metrics:
                        selection_metric = "val_loss_total"
                        selection_mode = "min"
                        selection_value = epoch_metrics[selection_metric]
                    elif len(validation_loader.dataset) == 0:
                        selection_metric = "total"
                        selection_mode = "min"
                        selection_value = epoch_metrics.get(selection_metric)
                if (
                    selection_value is not None
                    and (
                        best_checkpoint_metric != selection_metric
                        or checkpoint_selection_mode != selection_mode
                    )
                ):
                    best_checkpoint_metric = selection_metric
                    checkpoint_selection_mode = selection_mode
                    best_checkpoint_value = None
                is_best = (
                    selection_value is not None
                    and self._checkpoint_improved(
                        float(selection_value),
                        best_checkpoint_value,
                        selection_mode,
                    )
                )
                checkpoint_selection = {
                    "metric": selection_metric,
                    "mode": selection_mode,
                    "value": (
                        float(selection_value)
                        if selection_value is not None
                        else None
                    ),
                    "is_best": is_best,
                }
                early_metric_used = active_early_stopping_metric
                early_mode_used = active_early_stopping_mode
                early_value = epoch_metrics.get(early_metric_used)
                if early_value is None:
                    if "val_loss_total" in epoch_metrics:
                        early_metric_used = "val_loss_total"
                        early_mode_used = "min"
                        early_value = epoch_metrics[early_metric_used]
                    elif len(validation_loader.dataset) == 0:
                        early_metric_used = "total"
                        early_mode_used = "min"
                        early_value = epoch_metrics.get(early_metric_used)
                if (
                    early_value is not None
                    and (
                        early_metric_used != active_early_stopping_metric
                        or early_mode_used != active_early_stopping_mode
                    )
                ):
                    active_early_stopping_metric = early_metric_used
                    active_early_stopping_mode = early_mode_used
                    early_best_value = None
                    early_bad_epochs = 0
                if early_value is not None:
                    early_value = float(early_value)
                    if early_best_value is None:
                        early_improved = True
                    elif early_mode_used == "min":
                        early_improved = (
                            early_value
                            < early_best_value - early_stopping_min_delta
                        )
                    else:
                        early_improved = (
                            early_value
                            > early_best_value + early_stopping_min_delta
                        )
                    if early_improved:
                        early_best_value = early_value
                        early_bad_epochs = 0
                    else:
                        early_bad_epochs += 1
                    stop_training = stop_training or (
                        early_stopping_patience > 0
                        and epoch + 1 >= early_stopping_min_epochs
                        and early_bad_epochs >= early_stopping_patience
                    )
                early_stopping_state = {
                    "metric": early_metric_used,
                    "mode": early_mode_used,
                    "best_value": early_best_value,
                    "bad_epochs": early_bad_epochs,
                    "patience": early_stopping_patience,
                    "min_delta": early_stopping_min_delta,
                    "stopped": stop_training,
                }
                scheduler_metric = str(
                    training.get("lr_scheduler_metric", checkpoint_metric)
                )
                scheduler_value = epoch_metrics.get(scheduler_metric)
                if (
                    self.lr_scheduler is not None
                    and not self.lr_scheduler_step_per_update
                    and scheduler_value is not None
                ):
                    self.lr_scheduler.step(float(scheduler_value))
                if use_tqdm:
                    epoch_progress.set_postfix(
                        train_loss=f"{epoch_metrics.get('total', float('nan')):.4f}",
                        val_psnr=(
                            f"{epoch_metrics['val_psnr']:.2f}"
                            if "val_psnr" in epoch_metrics
                            else "n/a"
                        ),
                    )
                save_checkpoint(
                    checkpoint_path,
                    self.model,
                    self.optimizer,
                    epoch,
                    self.stage,
                    self.config,
                    extra={
                        "metrics": epoch_metrics,
                        "checkpoint_selection": checkpoint_selection,
                        "early_stopping": early_stopping_state,
                        "patch_discriminator": unwrap(self.patch_discriminator).state_dict(),
                        "wavelet_discriminator": unwrap(
                            self.wavelet_discriminator
                        ).state_dict(),
                        "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
                        "scaler": self.scaler.state_dict(),
                        "ema": self.ema.state_dict() if self.ema is not None else None,
                        "optimizer_step": self.optimizer_step,
                        "lr_scheduler": (
                            self.lr_scheduler.state_dict()
                            if self.lr_scheduler is not None
                            else None
                        ),
                        "parent_checkpoint": (
                            str(self.init_checkpoint)
                            if self.init_checkpoint
                            else None
                        ),
                        "parent_checkpoint_sha256": self.parent_checkpoint_sha256,
                    },
                )
                if keep_best_and_latest:
                    if is_best:
                        copy_checkpoint(checkpoint_path, best_checkpoint_path)
                        best_checkpoint_value = float(selection_value)
                    prune_stage_epoch_checkpoints(
                        output_dir,
                        self.stage,
                        keep=checkpoint_path,
                    )
                append_training_history(
                    output_dir / "training_history.jsonl",
                    epoch=epoch,
                    stage=self.stage,
                    metrics=epoch_metrics,
                    panel_size=int(
                        self.config.get("debug", {}).get("panel_size", 320)
                    ),
                )
                epoch_duration = time.monotonic() - epoch_started
                completed_epoch_durations.append(epoch_duration)
                remaining_epochs = epochs - epoch - 1
                stage_eta = (
                    sum(completed_epoch_durations)
                    / len(completed_epoch_durations)
                    * remaining_epochs
                )
                if progress_mode == "compact":
                    summary = (
                        f"[{self.stage}] epoch {epoch + 1}/{epochs} complete "
                        f"loss={epoch_metrics.get('total', float('nan')):.4f}"
                    )
                    if "val_psnr" in epoch_metrics:
                        summary += (
                            f" val_psnr={epoch_metrics['val_psnr']:.2f}"
                            f" val_ssim={epoch_metrics['val_ssim']:.4f}"
                        )
                    if keep_best_and_latest and is_best:
                        summary += (
                            f" best_{selection_metric}="
                            f"{float(selection_value):.6f}"
                        )
                    if early_stopping_patience > 0:
                        summary += (
                            f" early_stop={early_bad_epochs}/"
                            f"{early_stopping_patience}"
                        )
                    if reached_max_optimizer_steps:
                        summary += (
                            f" optimizer_steps={self.optimizer_step}/"
                            f"{self.max_optimizer_steps}"
                        )
                    summary += (
                        f" elapsed={self._duration(time.monotonic() - stage_started)} "
                        f"stage_eta={self._duration(stage_eta)}"
                    )
                    print(summary, flush=True)
                if stop_training:
                    if reached_max_optimizer_steps:
                        print(
                            f"[{self.stage}] reached max optimizer steps: "
                            f"{self.optimizer_step}/{self.max_optimizer_steps}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[{self.stage}] early stopping after epoch {epoch + 1}: "
                            f"{early_metric_used} did not improve by "
                            f"{early_stopping_min_delta:g} for "
                            f"{early_bad_epochs} validation checks",
                            flush=True,
                        )
            if self.distributed:
                stop_tensor = torch.tensor(
                    int(stop_training),
                    device=self.device,
                    dtype=torch.int32,
                )
                torch.distributed.broadcast(stop_tensor, src=0)
                stop_training = bool(stop_tensor.item())
            if stop_training:
                break
