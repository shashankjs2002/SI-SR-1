"""Residual recovery curriculum and evidence for the OLI2MSI TrustMoE study."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from ..losses import mse_loss
from ..metrics import psnr
from .trust_moe import (dataset_for, digest_file, load_model, make_config, state_of,
                        write_json)


RECOVERY_VERSION = "trust-recovery-v2"


def recovery_config(manifest, root, *, parent, seed=42, profile="single_expert",
                    initializer=None, epochs=15, batch_size=4, crop_size=128,
                    experts=5, top_k=2, dataset_id="", base_model=None,
                    risk_target="gain", coverage=0.5, hr_conditioning=True):
    config = make_config(manifest, root, profile=profile, parent=parent, seed=seed,
                         epochs=epochs, batch_size=batch_size, crop_size=crop_size,
                         experts=experts, top_k=top_k, dataset_id=dataset_id,
                         base_model=base_model)
    config["format"] = RECOVERY_VERSION
    config["model"].update(expert_kind="hr_residual" if hr_conditioning else "legacy",
                           expert_blocks=2, halo=6, detach_router_features=True)
    if profile == "single_expert":
        config["model"]["use_trust"] = False
    elif profile != "dense_transformer":
        config["model"]["coverage"] = coverage
    config["residual_initializer"] = str(initializer) if initializer else None
    config["training"].update(
        learning_rate=2e-4, ema_decay=0.99, ema_warmup=True,
        residual_warmup_epochs=0 if profile == "single_expert" else 2,
        guard_ramp_epochs=3, risk_target=risk_target,
        exploration_epochs=3, progress="compact", lr_epoch_decay=0.97,
    )
    config["losses"].update(mse=100.0, proposal_mse=100.0, proposal=2.0,
                            guard=0.0 if profile == "single_expert" else 5.0,
                            gradient=0.2, wavelet=0.1, trust=0.01,
                            risk=0.0 if profile == "single_expert" else 0.02,
                            balance=0.0 if profile == "single_expert" else 0.005)
    return config


def initialize_residual(model, path, *, dataset_id):
    """Clone a learned single expert; never silently replace the parent base."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["dataset_id"] != dataset_id or config["profile"] != "single_expert":
        raise ValueError("Residual initialization needs a single expert from the same dataset")
    source = state_of(checkpoint)
    for key, value in model.base.state_dict().items():
        if not torch.equal(value.cpu(), source[f"base.{key}"].cpu()):
            raise ValueError("Residual initializer and parent have different base weights")
    if config["model"].get("expert_kind", "legacy") != model.expert_kind:
        raise ValueError("Initializer expert architecture differs from the requested model")
    def section(prefix):
        return {k[len(prefix):]: v for k, v in source.items() if k.startswith(prefix)}
    model.encoder.load_state_dict(section("encoder."), strict=True)
    for expert in model.experts:
        expert.load_state_dict(section("experts.0."), strict=True)
    if model.num_experts > 1:
        # Break identical-expert symmetry at the RGB projection only.
        with torch.no_grad():
            for expert in model.experts:
                expert.rgb.weight.add_(torch.randn_like(expert.rgb.weight) * 1e-4)
    model.freeze_base()


def curriculum(config, epoch):
    """Epoch-addressed curriculum remains stable when an interrupted run resumes."""
    training = config["training"]
    warmup = int(training.get("residual_warmup_epochs", 0))
    if warmup < 0 or int(training.get("guard_ramp_epochs", 1)) < 1:
        raise ValueError("Invalid residual curriculum")
    weights = copy.deepcopy(config["losses"])
    is_warmup = epoch < warmup
    fraction = min(1.0, max(0.0, (epoch - warmup + 1) / training.get("guard_ramp_epochs", 1)))
    if "residual_warmup_epochs" in training:
        for key in ("guard", "trust", "adversarial"):
            weights[key] = weights.get(key, 0.0) * fraction
    return weights, ({"coverage": 1.0, "bypass_trust": True} if is_warmup else {}), is_warmup


@torch.no_grad()
def correction_diagnostics(output, target, mask):
    residual = output.residual.detach().float()
    effective = output.image.float() - output.base.float()
    valid = mask.expand_as(residual) > 0
    if not valid.any():
        raise ValueError("Diagnostics need valid pixels")
    base_error = (target - output.base.float())[valid]
    correction = effective[valid]
    base_mse = base_error.square().mean().clamp_min(1e-12)
    correlation = (base_error * correction).mean() / (
        base_mse * correction.square().mean()).sqrt().clamp_min(1e-12)
    return {
        "residual_abs_mean": float(residual[valid].abs().mean()),
        "correction_abs_mean": float(correction.abs().mean()),
        "correction_abs_max": float(correction.abs().max()),
        "correction_rms_to_target_error": float((correction.square().mean() / base_mse).sqrt()),
        "correction_target_cosine": float(correlation),
        "trust_mean": float(output.trust.float()[valid].mean()),
        "active_fraction": float(output.active.float().mean()),
        "pixel_fraction_changed": float((correction.abs() > 1e-7).float().mean()),
        "target_error_p99": float(base_error.abs().quantile(0.99)),
    }


def module_gradient_norms(model):
    modules = {"encoder": model.encoder, "router": model.router, "trust": model.trust_head}
    modules.update({f"expert_{i}": expert for i, expert in enumerate(model.experts)})
    result = {}
    for name, module in modules.items():
        terms = [p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
        norm = float(torch.stack(terms).sum().sqrt()) if terms else 0.0
        result[name] = norm if np.isfinite(norm) else None
    return result


def audit_checkpoint(path, manifest, output_path, *, indices=(0, 1, 2, 3), device=None):
    """Inspect raw and EMA outputs and backpropagate one validation diagnostic batch.

    No optimizer step is taken and no checkpoint is changed. Labels are used only
    in this offline audit, never as inference inputs.
    """
    from ..training.trust_losses import trust_moe_losses
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, config = load_model(path, device)
    config["manifest"] = str(manifest)
    dataset = dataset_for(config, "val")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    amp = bool(config["training"]["amp"] and device.type == "cuda")
    report = {"checkpoint": str(path), "sha256": digest_file(path),
              "epoch": saved.get("epoch"), "config": config, "rows": [],
              "parameters": {name: sum(p.numel() for p in module.parameters())
                             for name, module in model.named_children()},
              "has_ema": "ema" in saved}
    indices = [i for i in indices if 0 <= i < len(dataset)]
    if not indices:
        raise ValueError("No valid validation indices")
    for state_name in ("model", "ema") if "ema" in saved else ("model",):
        model.load_state_dict(saved[state_name], strict=True)
        model.eval()
        for index in indices:
            sample = dataset[index]
            lr, hr, mask = (sample[k][None].to(device) for k in ("lr", "hr", "valid_mask"))
            with torch.inference_mode(), torch.autocast(device.type, enabled=amp):
                out = model(lr, base_only=config["profile"] == "base")
            values = correction_diagnostics(out, hr, mask)
            values.update(index=index, state=state_name,
                          psnr=float(psnr(out.image.float(), hr, mask=mask)),
                          base_psnr=float(psnr(out.base.float(), hr, mask=mask)),
                          expert_load=out.assignments.sum((0, 1)).cpu().tolist())
            report["rows"].append(values)
    model.load_state_dict(state_of(saved), strict=True)
    if config["profile"] != "base":
        model.freeze_base()
        with torch.autocast(device.type, enabled=amp):
            out = model(lr)
        # One direct objective isolates reconstruction gradient flow from auxiliary losses.
        loss = mse_loss(out.image.float(), hr, mask)
        loss.backward()
        report["reconstruction_gradients"] = module_gradient_norms(model)
        report["base_frozen"] = all(p.grad is None for p in model.base.parameters())
        model.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=amp):
            out = model(lr)
        total, parts = trust_moe_losses(out, hr, mask, model.tile_size, model.scale,
                                       config["losses"], model.use_trust,
                                       config["training"].get("risk_target", "error"))
        total.backward()
        report["full_loss_gradients"] = module_gradient_norms(model)
        report["weighted_losses"] = {k: float(v.detach()) * config["losses"].get(k, 0)
                                     for k, v in parts.items()}
    write_json(output_path, report)
    return report


def fixed_training_subset(config, count):
    """Materialize reproducible TRAIN crops for a memorization diagnostic only."""
    from .trust_moe import seed_all
    seed_all(config["seed"])
    dataset = dataset_for(config, "train")
    return [dataset[i] for i in range(min(count, len(dataset)))]


@torch.inference_mode()
def memorization_report(checkpoint, output_path, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, config = load_model(checkpoint, device)
    count = int(config["training"].get("diagnostic_overfit_pairs", 0))
    if count < 1:
        raise ValueError("Expected a diagnostic checkpoint")
    rows = []
    for index, sample in enumerate(fixed_training_subset(config, count)):
        lr, hr, mask = (sample[k][None].to(device) for k in ("lr", "hr", "valid_mask"))
        with torch.autocast(device.type, enabled=device.type == "cuda" and config["training"]["amp"]):
            out = model(lr)
        rows.append(dict(index=index, psnr=float(psnr(out.image.float(), hr, mask=mask)),
                         base_psnr=float(psnr(out.base.float(), hr, mask=mask)),
                         **correction_diagnostics(out, hr, mask)))
    result = {"scope": "fixed training crops; NOT validation or test", "rows": rows,
              **validation_screen(rows, minimum_psnr_gain=0.05)}
    write_json(output_path, result)
    return result


def validation_screen(rows, *, minimum_psnr_gain=0.01):
    """A declared development screen, not a publication significance claim."""
    base = np.array([r["base_psnr"] for r in rows])
    delta = np.array([r["psnr"] for r in rows]) - base
    return {"mean_psnr_gain": float(delta.mean()),
            "fraction_beating_base": float((delta > 0).mean()),
            "passed": bool(delta.mean() >= minimum_psnr_gain and (delta > 0).mean() > 0.5)}
