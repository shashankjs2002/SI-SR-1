from __future__ import annotations

import math
from typing import Any

from torch import nn

from .models.discriminators import MultiScaleDiscriminator, WaveletDiscriminator
from .models.system import GeoDiffGAN
from .training.stages import STAGES, configure_stage_trainability, stage_uses_discriminators


def count_parameters(module: nn.Module) -> dict[str, int]:
    parameters = list(module.parameters())
    return {
        "scalar_parameters": sum(parameter.numel() for parameter in parameters),
        "trainable_scalar_parameters": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
        "parameter_tensors": len(parameters),
        "trainable_parameter_tensors": sum(
            parameter.requires_grad for parameter in parameters
        ),
    }


def _memory_megabytes(parameters: int, bytes_per_parameter: int) -> float:
    return parameters * bytes_per_parameter / (1024**2)


def build_parameter_report(
    config: dict[str, Any],
    patches: int | None = None,
    world_size: int = 1,
) -> dict[str, Any]:
    if world_size < 1:
        raise ValueError("world_size must be at least 1")
    if patches is not None and patches < 1:
        raise ValueError("patches must be at least 1")

    model = GeoDiffGAN.from_config(config)
    module_counts = {
        name: count_parameters(module)
        for name, module in model.named_children()
    }
    core_count = count_parameters(model)

    discriminator_channels = int(
        config.get("training", {}).get("discriminator_channels", 64)
    )
    patch_discriminator = MultiScaleDiscriminator(
        base_channels=discriminator_channels
    )
    wavelet_discriminator = WaveletDiscriminator(
        base_channels=discriminator_channels
    )
    discriminator_counts = {
        "conditional_multiscale_patchgan": count_parameters(patch_discriminator),
        "haar_wavelet_discriminator": count_parameters(wavelet_discriminator),
    }
    discriminator_total = sum(
        value["scalar_parameters"] for value in discriminator_counts.values()
    )

    stage_counts = {}
    for stage in STAGES:
        names = configure_stage_trainability(model, stage)
        model_count = count_parameters(model)
        discriminator_trainable = (
            discriminator_total if stage_uses_discriminators(stage) else 0
        )
        stage_counts[stage] = {
            "trainable_modules": list(names),
            "core_trainable_parameters": model_count[
                "trainable_scalar_parameters"
            ],
            "discriminator_trainable_parameters": discriminator_trainable,
            "total_optimized_parameters": (
                model_count["trainable_scalar_parameters"]
                + discriminator_trainable
            ),
            "uses_discriminators": stage_uses_discriminators(stage),
        }

    training = config.get("training", {})
    batch_size = int(training.get("batch_size", 1))
    accumulation = int(training.get("gradient_accumulation", 1))
    effective_batch = batch_size * accumulation * world_size
    schedule = {
        "epochs": int(training.get("epochs", 1)),
        "batch_size_per_gpu": batch_size,
        "gradient_accumulation": accumulation,
        "world_size": world_size,
        "effective_batch_size": effective_batch,
        "generator_learning_rate": float(training.get("learning_rate", 1e-4)),
        "discriminator_learning_rate": float(
            training.get("discriminator_learning_rate", 1e-4)
        ),
        "weight_decay": float(training.get("weight_decay", 1e-4)),
        "generator_adamw_betas": [0.9, 0.99],
        "discriminator_adamw_betas": [0.0, 0.99],
        "gradient_clip": float(training.get("gradient_clip", 1.0)),
        "amp": bool(training.get("amp", True)),
        "gradient_checkpointing": bool(
            training.get("gradient_checkpointing", True)
        ),
        "loss_weights": training.get("loss_weights", {}),
    }
    if patches is not None:
        samples_per_rank = math.ceil(patches / world_size)
        microbatches_per_rank = samples_per_rank // batch_size
        updates_per_epoch = math.ceil(microbatches_per_rank / accumulation)
        schedule.update(
            {
                "patches": patches,
                "samples_per_rank": samples_per_rank,
                "microbatches_per_rank_per_epoch": microbatches_per_rank,
                "optimizer_updates_per_epoch": updates_per_epoch,
                "optimizer_updates_for_configured_epochs": (
                    updates_per_epoch * schedule["epochs"]
                ),
            }
        )

    return {
        "core_model": {
            **core_count,
            "fp32_parameter_memory_mb": _memory_megabytes(
                core_count["scalar_parameters"], 4
            ),
            "fp16_parameter_memory_mb": _memory_megabytes(
                core_count["scalar_parameters"], 2
            ),
        },
        "core_modules": module_counts,
        "discriminators": {
            "modules": discriminator_counts,
            "scalar_parameters": discriminator_total,
        },
        "training_stages": stage_counts,
        "training_configuration": schedule,
        "excluded_external_models": {
            "training_text_encoder": (
                "Frozen and model-dependent; use --include-text-encoder to count it."
            ),
            "qwen_captioner": (
                "Offline data-preparation model; never loaded with GeoDiff-GAN training."
            ),
        },
    }


def verify_parameter_report(report: dict[str, Any]) -> None:
    core = report["core_model"]["scalar_parameters"]
    module_total = sum(
        value["scalar_parameters"]
        for value in report["core_modules"].values()
    )
    if core != module_total:
        raise AssertionError(
            f"Core module sum {module_total:,} does not equal model total {core:,}"
        )

    discriminator_total = report["discriminators"]["scalar_parameters"]
    discriminator_sum = sum(
        value["scalar_parameters"]
        for value in report["discriminators"]["modules"].values()
    )
    if discriminator_total != discriminator_sum:
        raise AssertionError("Discriminator module counts do not sum to their total")

    for stage, values in report["training_stages"].items():
        expected = (
            values["core_trainable_parameters"]
            + values["discriminator_trainable_parameters"]
        )
        if values["total_optimized_parameters"] != expected:
            raise AssertionError(f"Optimized parameter count is inconsistent for {stage}")
