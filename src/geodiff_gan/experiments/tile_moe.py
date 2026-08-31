"""Isolated, resumable Landsat/Sentinel ablations for the 3x-continued branch."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from pathlib import Path
import subprocess
import sys

import torch
import yaml

from ..config import load_config
from ..models.system import GeoDiffGAN
from ..training.checkpoint import best_stage_checkpoint, latest_stage_checkpoint


LEGACY_PROFILES = ("rgb_standard", "rgb_fidelity", "multispectral_fidelity",
                   "rgb_harmonized_fidelity", "multispectral_guided_fidelity")
NEW_PROFILES = ("residual_base", "single_expert", "generic_moe", "reliability_moe")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lock_json(path, value):
    """Never silently reuse a directory for another configuration or parent checkpoint."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Existing run differs: {path}. Use a NEW run directory; old artifacts were preserved.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def snapshot_manifest(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise ValueError("Prepared data changed after experiment creation. Start a new SUITE_ROOT.")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    return destination


def balanced_manifest(source, destination, seed=42):
    """Interleave tiles within each split so capped validation is not one city's prefix."""
    rows = [json.loads(line) for line in Path(source).read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(seed)
    ordered = []
    for split in sorted({row["split"] for row in rows}):
        groups = {}
        for row in rows:
            if row["split"] == split:
                groups.setdefault(row["tile_id"], []).append(row)
        for tile in sorted(groups):
            rng.shuffle(groups[tile])
        for index in range(max(map(len, groups.values()))):
            for tile in sorted(groups):
                if index < len(groups[tile]):
                    ordered.append(groups[tile][index])
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered)
    destination = Path(destination)
    if destination.exists() and destination.read_text(encoding="utf-8") != text:
        raise ValueError("Balanced manifest changed; use a new suite root")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.write_text(text, encoding="utf-8")
    return destination


def experiment_config(repository, profile, manifest, output, *, experts=2, top_k=1,
                      crop_size=32, calibration=None, multispectral=False, fast=False):
    if profile not in (*LEGACY_PROFILES, *NEW_PROFILES):
        raise ValueError(f"Unknown experiment: {profile}")
    if not isinstance(experts, int) or not isinstance(top_k, int) or not 1 <= top_k <= experts:
        raise ValueError("Set integer 1 <= TOP_K <= NUM_EXPERTS before training")
    if crop_size < 32 or crop_size % 8:
        raise ValueError("LR crop must be >=32 and divisible by 8")
    repository, output = Path(repository), Path(output)
    config = load_config(repository / "configs/landsat_sentinel_3x_small.yaml")
    new = profile in NEW_PROFILES
    ms = multispectral if new else profile.startswith("multispectral")
    harmonized = profile in ("rgb_harmonized_fidelity", "multispectral_guided_fidelity")
    if harmonized and not calibration:
        raise ValueError("Harmonized profiles require a train-only calibration file")
    config["seed"] = 42
    config["data"].update(manifest=str(snapshot_manifest(manifest, output / "dataset/manifest.jsonl")),
                          input_mode="paired", paired_lr_crop_size=crop_size,
                          condition_key="lr_ms" if ms else "lr", captions=None,
                          radiometric_calibration=str(calibration) if harmonized else None)
    config["model"].update(
        base_architecture="residual_swin" if new else "swinir",
        base_groups=2, base_depth=2 if new else 4, base_embed_dim=32,
        base_heads=4, input_channels=6 if ms else 3,
        base_input_channels=3 if new or harmonized else (6 if ms else 3),
        context_dim=64, diffusion_widths=[32, 64, 96],
        base_upsample_mode="resize_conv", decoder_upsample_mode="resize_conv",
        vae_upsample_mode="resize_conv", diffusion_upsample_mode="resize_conv",
        diffusion_experts=0 if not new or profile == "residual_base" else (1 if profile == "single_expert" else experts),
        diffusion_top_k=1 if profile == "single_expert" else top_k,
        routing_mode="reliability" if profile == "reliability_moe" else "generic",
        expert_channels=24, use_back_projection=False, use_text_conditioning=False,
        use_uncertainty_abstention=False,
    )
    train = config["training"]
    train.update(batch_size=4, gradient_accumulation=2, validation_batch_size=1,
                 num_workers=2, persistent_workers=False, reseed_each_epoch=True,
                 weight_decay=0.0, gradient_checkpointing=False,
                 auto_resume=True, init_checkpoint=None, resume=None,
                 max_batches_per_epoch=2 if fast else 120,
                 validation_limit=2 if fast else 16, validation_seed=10042,
                 validation_sample_steps=2 if fast else 8, validation_samples=1,
                 early_stopping_patience=3, early_stopping_min_epochs=3,
                 router_warmup_epochs=1, router_temperature=1e-4)
    losses = train["loss_weights"]
    losses.update(perceptual=0.0, adversarial=0.0, gradient=0.0, wavelet=0.0,
                  expert_denoising=0.1, router_balance=0.01,
                  router_quality=0.05, router_ranking=0.05)
    if profile == "rgb_standard":
        losses.update(mse=0.0, multiscale_mse=0.0, ssim=0.2, gradient=0.1,
                      wavelet=0.05, radiometric=0.0, residual_supervision=0.0,
                      base_guard=0.0, diffusion=1.0, evidence_calibration=0.1,
                      evidence_improvement=0.0)
    if harmonized:
        losses.update(charbonnier=0.5, mse=100.0, radiometric=0.25,
                      residual_supervision=0.1, base_guard=200.0, diffusion=0.1)
    config["experiment"] = {"profile": profile, "manifest_sha256": sha256(manifest),
                            "new_architecture": new, "multispectral": ms,
                            "protocol": "real_landsat30_sentinel10_no_text_3x",
                            "historical_checkpoint_compatible": False}
    return config


def assert_base_lineage(parent, child):
    before = torch.load(parent, map_location="cpu", weights_only=False)["model"]
    after = torch.load(child, map_location="cpu", weights_only=False)["model"]
    keys = [key for key in before if key.startswith("base.")]
    if not keys or any(key not in after or not torch.equal(before[key], after[key]) for key in keys):
        raise RuntimeError("Frozen base lineage changed. Do not compare this run with the shared base.")


def run_stage(repository, config, root, stage, *, parent=None, epochs=6, minutes=30, fast=False):
    repository, root = Path(repository), Path(root)
    config = copy.deepcopy(config)
    train = config["training"]
    train.update(stage=stage, epochs=1 if fast else epochs,
                 output_dir=str(root / "runs" / stage),
                 init_checkpoint=str(parent) if parent else None, resume=None,
                 max_stage_seconds=60 if fast else minutes * 60,
                 learning_rate=1e-5 if stage == "joint" else 1e-4)
    if stage == "diffusion":
        train.update(checkpoint_metric="val_loss_diffusion", checkpoint_mode="min",
                     early_stopping_metric="val_loss_diffusion", early_stopping_mode="min",
                     lr_scheduler_metric="val_loss_diffusion", lr_scheduler_mode="min")
        train["loss_weights"]["diffusion"] = 1.0
    if stage == "joint" and config["experiment"]["new_architecture"]:
        # Keep the router's LR features stable. The final decoder sees actual DDIM samples.
        train.update(trainable_modules=["mapper", "decoder"], joint_latent_source="sampled",
                     joint_sample_steps=2 if fast else 8)
        train["loss_weights"]["diffusion"] = 0.0
    elif stage == "joint" and config["experiment"]["profile"] in ("rgb_harmonized_fidelity", "multispectral_guided_fidelity"):
        train["trainable_modules"] = ["lr_encoder", "mapper", "decoder"]
    directory = root / "configs"
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / f"{stage}.yaml"
    receipt = {"config": config, "parent_sha256": sha256(parent) if parent else None}
    lock_json(directory / f"{stage}.lock.json", receipt)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    done = root / "runs" / stage / "completed.json"
    if done.exists():
        saved = json.loads(done.read_text(encoding="utf-8"))
        checkpoint = Path(saved["checkpoint"])
        if not checkpoint.is_file() or sha256(checkpoint) != saved["sha256"]:
            raise RuntimeError(f"Completed checkpoint missing or modified: {checkpoint}")
        return config_path, checkpoint
    model = GeoDiffGAN.from_config(config)
    if any(isinstance(module, torch.nn.PixelShuffle) for module in model.modules()):
        raise RuntimeError("New experiment contains PixelShuffle")
    del model
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTHONUNBUFFERED"] = "1"
    subprocess.run([sys.executable, "-m", "geodiff_gan.cli.train", "--config", str(config_path)],
                   cwd=repository, env=environment, check=True)
    checkpoint = best_stage_checkpoint(train["output_dir"], stage) or latest_stage_checkpoint(train["output_dir"], stage)
    if checkpoint is None:
        raise RuntimeError(f"No completed {stage} checkpoint")
    if parent and stage != "base":
        assert_base_lineage(parent, checkpoint)
    lock_json(done, {"checkpoint": str(checkpoint), "sha256": sha256(checkpoint),
                     "parent_sha256": receipt["parent_sha256"]})
    return config_path, checkpoint
