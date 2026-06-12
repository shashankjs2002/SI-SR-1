from __future__ import annotations

import json
import math
import os
import random
import time
from collections import defaultdict
from contextlib import contextmanager
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
    charbonnier,
    degradation_consistency,
    discriminator_hinge,
    edit_localization_loss,
    evidence_calibration_loss,
    generator_hinge,
    gradient_loss,
    kl_loss,
    snr_weighted_velocity_loss,
    ssim,
    wavelet_loss,
)
from ..models.blocks import high_pass
from ..models.degradation import sensor_degrade
from ..models.discriminators import MultiScaleDiscriminator, WaveletDiscriminator
from ..models.system import GeoDiffGAN
from ..metrics import basic_metrics
from ..text import PromptBatch, TextEncoder, augment_prompts, build_text_encoder
from .checkpoint import (
    latest_stage_checkpoint,
    load_checkpoint,
    save_checkpoint,
    unwrap,
)
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
        if init_checkpoint and not resume:
            load_checkpoint(init_checkpoint, self.model, strict=False)
        self.text_encoder: TextEncoder | None = None
        if self.stage != "base":
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
            base_channels=config["training"].get("discriminator_channels", 64)
        ).to(self.device)
        self.wavelet_discriminator = WaveletDiscriminator(
            base_channels=config["training"].get("discriminator_channels", 64)
        ).to(self.device)
        if self.distributed:
            self.patch_discriminator = DistributedDataParallel(
                self.patch_discriminator, device_ids=[self.local_rank]
            )
            self.wavelet_discriminator = DistributedDataParallel(
                self.wavelet_discriminator, device_ids=[self.local_rank]
            )
        self.perceptual = OptionalPerceptualLoss().to(self.device)
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=float(config["training"]["learning_rate"]),
            betas=(0.9, 0.99),
            weight_decay=float(config["training"].get("weight_decay", 1e-4)),
        )
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
            if "patch_discriminator" in extra:
                unwrap(self.patch_discriminator).load_state_dict(extra["patch_discriminator"])
            if "wavelet_discriminator" in extra:
                unwrap(self.wavelet_discriminator).load_state_dict(
                    extra["wavelet_discriminator"]
                )
            if "discriminator_optimizer" in extra:
                self.discriminator_optimizer.load_state_dict(extra["discriminator_optimizer"])
            if "scaler" in extra:
                self.scaler.load_state_dict(extra["scaler"])
            if self.is_main:
                print(
                    f"[{self.stage}] resuming from {resume} at epoch "
                    f"{self.start_epoch + 1}",
                    flush=True,
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
        configure_stage_trainability(self.model, self.stage)

    def _loader(self, split: str) -> DataLoader:
        data = self.config["data"]
        dataset = SentinelPatchDataset(
            data["manifest"],
            split=split,
            scale=self.config["model"].get("scale", 4),
            caption_file=data.get("captions"),
            augment=split == "train",
            degradation_seed=int(data.get("degradation_seed", 0)),
            degradation_severity=data.get("degradation_severity", "mild"),
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
            raise RuntimeError(f"Stage {self.stage} requires a configured text encoder")
        context = self.text_encoder(captions)
        null_context = self.text_encoder([""] * len(captions))
        return (
            context.to(self.device),
            null_context.to(self.device),
            captions,
            prompt_kinds,
        )

    def _forward_stage(
        self,
        batch: dict[str, Any],
        diagnostics: DiagnosticRecorder | None = None,
        training: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        model: GeoDiffGAN = unwrap(self.model)  # type: ignore[assignment]
        hr = batch["hr"].to(self.device, non_blocking=True)
        lr = batch["lr"].to(self.device, non_blocking=True)
        consistency_lr = batch.get("clean_lr", batch["lr"]).to(
            self.device, non_blocking=True
        )
        degradation = batch["degradation"].to(self.device, non_blocking=True)
        losses: dict[str, torch.Tensor] = {}
        if diagnostics is not None:
            diagnostics.capture("input.lr", lr, visual="rgb")
            diagnostics.capture("target.hr", hr, visual="rgb")
            diagnostics.capture("conditioning.degradation", degradation)

        if self.stage == "base":
            prediction = model.base(lr)
            if diagnostics is not None:
                diagnostics.capture(
                    "base.bicubic",
                    F.interpolate(
                        lr,
                        scale_factor=model.scale,
                        mode="bicubic",
                        align_corners=False,
                    ),
                    visual="rgb",
                )
                diagnostics.capture("base.hr", prediction, visual="rgb")
                diagnostics.capture("output.hr", prediction, visual="rgb")
            losses["charbonnier"] = charbonnier(prediction, hr)
            losses["ssim"] = 1 - ssim(prediction, hr)
            losses["gradient"] = gradient_loss(prediction, hr)
            losses["consistency"] = degradation_consistency(
                prediction,
                consistency_lr,
                degradation,
                scale=model.scale,
                severity=model.degradation_severity,
            )
            return prediction, losses

        context, _, used_prompts, prompt_kinds = self._contexts(
            list(batch["caption"]),
            training=training,
        )
        with torch.no_grad():
            base = model.base(lr)
            target_residual = hr - base
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
            detail = high_pass(decoded.detail_residual)
            evidence = model._resize_policy(
                mapped.evidence_confidence, base.shape[-2:]
            )
            detail = detail * model._effective_confidence(evidence)
            prediction = (base + detail).clamp(0, 1)
            ungated_prediction = (base + high_pass(decoded.detail_residual)).clamp(
                0, 1
            )
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
                diagnostics.capture("output.hr", prediction, visual="rgb")
            losses["vae_reconstruction"] = charbonnier(reconstruction, target_residual)
            losses["kl"] = kl_loss(mean, log_variance)
            losses["charbonnier"] = charbonnier(prediction, hr)
            losses["wavelet"] = wavelet_loss(prediction, hr)
            losses["evidence_calibration"] = evidence_calibration_loss(
                mapped.evidence_confidence,
                ungated_prediction,
                hr,
                temperature=float(
                    self.config["training"].get(
                        "evidence_calibration_temperature", 0.05
                    )
                ),
            )
            return prediction, losses

        with torch.no_grad():
            latent, _, _ = model.vae.encode(target_residual, sample=False)
        diffusion_batch = model.prepare_diffusion_batch(latent)
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
        reconstruction_mask = (
            ~counterfactual
            if self.stage == "edit"
            else torch.ones_like(counterfactual)
        )
        if reconstruction_mask.any():
            reconstruction = prediction[reconstruction_mask]
            reconstruction_target = hr[reconstruction_mask]
            losses["charbonnier"] = charbonnier(
                reconstruction, reconstruction_target
            )
            losses["ssim"] = 1 - ssim(reconstruction, reconstruction_target)
            losses["perceptual"] = self.perceptual(
                reconstruction, reconstruction_target
            )
            losses["wavelet"] = wavelet_loss(
                reconstruction, reconstruction_target
            )
        else:
            zero = prediction.new_zeros(())
            losses["charbonnier"] = zero
            losses["ssim"] = zero
            losses["perceptual"] = zero
            losses["wavelet"] = zero
        losses["consistency"] = degradation_consistency(
            prediction,
            consistency_lr,
            degradation,
            scale=model.scale,
            severity=model.degradation_severity,
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
        seed = int(self.config.get("seed", 42)) + 10_000 + epoch
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
                for name, value in losses.items():
                    totals[f"loss_{name}"] += float(value.detach())
                totals["loss_total"] += float(total_loss.detach())
                if self.stage != "diffusion":
                    model: GeoDiffGAN = unwrap(self.model)  # type: ignore[assignment]
                    values = basic_metrics(
                        prediction,
                        batch["hr"].to(self.device, non_blocking=True),
                        batch.get("clean_lr", batch["lr"]).to(
                            self.device,
                            non_blocking=True,
                        ),
                        batch["degradation"].to(
                            self.device,
                            non_blocking=True,
                        ),
                        scale=model.scale,
                        severity=model.degradation_severity,
                    )
                    for name, value in values.items():
                        totals[name] += value
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
            "consistency": 1.0,
            "ssim": 0.2,
            "gradient": 0.1,
            "perceptual": 0.1,
            "wavelet": 0.05,
            "kl": 1e-4,
            "vae_reconstruction": 1.0,
            "diffusion": 1.0,
            "evidence_calibration": 0.1,
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
        accumulation = int(training.get("gradient_accumulation", 1))
        if accumulation < 1:
            raise ValueError("training.gradient_accumulation must be at least 1")
        epochs = int(training["epochs"])
        output_dir = Path(training.get("output_dir", "runs/default"))
        if self.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
                json.dump(self.config, handle, indent=2)
        validate_every = max(1, int(training.get("validate_every", 1)))
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
            debug_exports = 0
            self.model.train()
            if self.text_encoder is not None:
                self.text_encoder.eval()
            self.optimizer.zero_grad(set_to_none=True)
            if self.stage in ("joint", "edit"):
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
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.amp_enabled,
                ):
                    prediction, losses = self._forward_stage(batch, diagnostics)
                    generator_loss = self._weighted_loss(losses)
                    discriminator_loss = None
                    if self.stage in ("joint", "edit"):
                        adversarial = self._generator_adversarial_loss(prediction, lr)
                        losses["adversarial"] = adversarial
                        generator_loss = generator_loss + adversarial * float(
                            training.get("loss_weights", {}).get("adversarial", 0.01)
                        )
                        discriminator_loss = self._discriminator_loss(prediction, hr, lr)
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
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.stage in ("joint", "edit"):
                        self.discriminator_optimizer.zero_grad(set_to_none=True)
                if diagnostics is not None:
                    if self.stage != "diffusion":
                        if self.stage == "base":
                            debug_base = prediction
                            debug_residual = prediction - torch.nn.functional.interpolate(
                                lr,
                                size=prediction.shape[-2:],
                                mode="bicubic",
                                align_corners=False,
                            )
                        else:
                            debug_base = unwrap(self.model).base(lr)
                            debug_residual = prediction - debug_base
                        diagnostics.add_spatial_metrics(
                            lr,
                            debug_base,
                            debug_residual,
                            prediction,
                            batch["degradation"].to(self.device),
                            scale=unwrap(self.model).scale,
                            target=hr,
                            consistency_lr=batch.get(
                                "clean_lr", batch["lr"]
                            ).to(self.device),
                            degradation_severity=unwrap(
                                self.model
                            ).degradation_severity,
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
            reduced_metrics = self._reduce_metrics(metrics)
            validation_metrics = {}
            if (
                len(validation_loader.dataset) > 0
                and (epoch + 1) % validate_every == 0
            ):
                validation_metrics = self._validate(validation_loader, epoch)
            if self.is_main:
                denominator = max(len(loader), 1)
                epoch_metrics = {
                    key: value / denominator for key, value in reduced_metrics.items()
                }
                epoch_metrics.update(validation_metrics)
                checkpoint_path = output_dir / f"{self.stage}_epoch_{epoch:04d}.pt"
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
                        "patch_discriminator": unwrap(self.patch_discriminator).state_dict(),
                        "wavelet_discriminator": unwrap(
                            self.wavelet_discriminator
                        ).state_dict(),
                        "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
                        "scaler": self.scaler.state_dict(),
                    },
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
                    summary += (
                        f" elapsed={self._duration(time.monotonic() - stage_started)} "
                        f"stage_eta={self._duration(stage_eta)}"
                    )
                    print(summary, flush=True)
