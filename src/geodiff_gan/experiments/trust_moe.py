"""Training and evaluation of the diffusion-free TrustMoE experiments."""
from __future__ import annotations

from collections import Counter
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..data.dataset import SentinelPatchDataset
from ..data.manifest import (load_manifest, write_manifest, validate_within_tile_spatial_isolation,
                             validate_tile_split_isolation)
from ..losses import charbonnier, mse_loss, ssim, discriminator_hinge, generator_hinge
from ..metrics import basic_metrics
from ..models.discriminators import PatchDiscriminator
from ..models.trust_moe import TrustMoESR
from ..training.trust_losses import trust_moe_losses


PROFILES = ("single_expert", "dense_transformer", "sparse_uniform", "sparse_conv",
            "sparse_transformer", "sparse_no_trust", "sparse_adversarial", "adaptive_k")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_manifest(source, destination, *, spatial_audit=True, minimum_test_fraction=0.1,
                     disjoint_tiles=False):
    """Validate an already prepared paired dataset; retain its original splits."""
    records = load_manifest(source, resolve_paths=True)
    records = [r for r in records if r.split in ("train", "val", "test")]
    counts = Counter(r.split for r in records)
    if any(counts[s] == 0 for s in ("train", "val", "test")):
        raise ValueError(f"Need nonempty train/val/test: {counts}")
    if counts["test"] / len(records) < minimum_test_fraction:
        raise ValueError("Prepared tile dataset needs at least 10% test pairs")
    seen, sizes, fingerprint = {}, set(), hashlib.sha256()
    for record in records:
        path = Path(record.patch)
        content_hash = digest_file(path)
        if content_hash in seen:
            raise ValueError(f"Duplicate pair content: {path}; previously {seen[content_hash]}")
        seen[content_hash] = (str(path), record.split)
        with np.load(path, allow_pickle=False) as data:
            for key in ("lr", "hr", "valid_mask_lr", "valid_mask_hr"):
                if key not in data:
                    raise ValueError(f"{path}: missing {key}")
            lr, hr = data["lr"], data["hr"]
            if lr.ndim != 3 or hr.ndim != 3 or lr.shape[0] != 3 or hr.shape[0] != 3:
                raise ValueError(f"{path}: expected CHW RGB arrays")
            if hr.shape[1:] != tuple(3 * v for v in lr.shape[1:]):
                raise ValueError(f"{path}: not a real 3x pair: {lr.shape}, {hr.shape}")
            for key, array, spatial in (("lr", lr, lr.shape[-2:]), ("hr", hr, hr.shape[-2:])):
                mask = data[f"valid_mask_{key}"]
                if mask.shape != (1, *spatial) or not np.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1 or mask.sum() == 0:
                    raise ValueError(f"{path}: invalid {key} mask")
                if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
                    raise ValueError(f"{path}: {key} must be finite and already scaled to [0,1]")
            sizes.add(hr.shape[-1])
            if hr.shape[-2] != hr.shape[-1]:
                raise ValueError("Dataset audit currently expects square prepared frames")
        identity = asdict(record)
        identity.pop("patch")
        fingerprint.update((content_hash + json.dumps(identity, sort_keys=True)).encode())
    if spatial_audit and disjoint_tiles:
        raise ValueError("Choose within-tile spatial or disjoint-block auditing, not both")
    if disjoint_tiles:
        validate_tile_split_isolation(records)
    if spatial_audit:
        if len(sizes) != 1:
            raise ValueError("Use one frame size for the spatial split audit")
        validate_within_tile_spatial_isolation(records, patch_size=next(iter(sizes)))
    write_manifest(destination, records)
    report = {"dataset_id": fingerprint.hexdigest(), "counts": dict(counts),
              "categories": dict(Counter(r.scene_class for r in records)),
              "scale": 3, "frames_hr": sorted(sizes), "spatial_audit": spatial_audit,
              "disjoint_blocks": disjoint_tiles,
              "split_interpretation": ("disjoint geographic block IDs; footprint guards audited during preparation" if disjoint_tiles
                                        else "within-tile held-out regions" if spatial_audit
                                        else "provided benchmark split; geographic independence not established")}
    write_json(Path(destination).with_suffix(".audit.json"), report)
    return report


def make_config(manifest, root, *, profile="sparse_transformer", seed=42,
                epochs=15, batch_size=4, crop_size=64, experts=5, top_k=2,
                dataset_id="", parent=None, base_model=None):
    if profile not in (*PROFILES, "base"):
        raise ValueError(profile)
    model = {"num_experts": experts, "top_k": top_k, "coverage": 0.5,
             "router_kind": "transformer", "routing": "reliability", "use_trust": True}
    model.update(base_model or {})
    if profile == "single_expert":
        model.update(num_experts=1, top_k=1, coverage=1.0)
    if profile == "dense_transformer":
        model.update(coverage=1.0)
    if profile == "sparse_uniform":
        model.update(routing="uniform")
    if profile == "sparse_conv":
        model.update(router_kind="conv")
    if profile == "sparse_no_trust":
        model.update(use_trust=False)
    if profile == "adaptive_k":
        model.update(adaptive_k=True)
    return {"format": "trust-moe-v1", "profile": profile, "seed": seed,
            "manifest": str(manifest), "dataset_id": dataset_id, "root": str(root),
            "model": model, "parent": str(parent) if parent else None,
            "training": {"epochs": epochs, "batch_size": batch_size, "crop_size": crop_size,
                         "num_workers": 2, "learning_rate": 1e-4, "amp": True,
                         "ema_decay": 0.999, "exploration_epochs": 3},
            "losses": {"mse": 100.0, "charbonnier": 0.5, "ssim": 0.1,
                       "gradient": 0.2, "wavelet": 0.1, "radiometric": 0.05,
                       "proposal": 0.25, "guard": 100.0, "trust": 0.01,
                       "risk": 0.02, "balance": 0.01,
                       "adversarial": 0.002 if profile == "sparse_adversarial" else 0.0}}


def dataset_for(config, split):
    return SentinelPatchDataset(config["manifest"], split, scale=3, input_mode="paired",
                                condition_key="lr", augment=split == "train",
                                random_degradation=False,
                                paired_lr_crop_size=config["training"]["crop_size"] if split == "train" else None)


def state_of(checkpoint):
    return checkpoint.get("ema", checkpoint["model"])


def import_base(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = state_of(checkpoint)
    base = {k.removeprefix("base."): v for k, v in state.items() if k.startswith("base.")}
    if not base:
        base = state
    model.base.load_state_dict(base, strict=True)
    model.freeze_base()


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = TrustMoESR(**checkpoint["config"]["model"])
    model.load_state_dict(state_of(checkpoint), strict=True)
    return model.to(device).eval(), checkpoint["config"]


def adopt_base(config, source):
    """Wrap an existing compatible residual-Swin base without loading diffusion modules."""
    model = TrustMoESR(**config["model"])
    import_base(model, source)
    path = Path(config["root"]) / "best.pt"
    identity = digest_file(source)
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("adopted_source_sha256") != identity or saved["config"]["dataset_id"] != config["dataset_id"]:
            raise ValueError("An imported base already exists for another source/data; use a new root")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_checkpoint(path, {"model": model.state_dict(), "config": config,
                            "adopted_source_sha256": identity})
    return path


def _save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


@torch.no_grad()
def validate(model, dataset, device, *, base_only=False, amp=False):
    from ..metrics import psnr
    values = []
    for sample in dataset:
        lr, target, mask = (sample[key][None].to(device) for key in ("lr", "hr", "valid_mask"))
        with torch.autocast(device.type, enabled=amp):
            output = model(lr, base_only=base_only)
        prediction = output.image.float()
        values.append((float(psnr(prediction, target, mask=mask)),
                       float(ssim(prediction, target, mask=mask))))
    if not values:
        raise ValueError("Validation split is empty")
    return np.mean(values, axis=0).tolist()


def train(config, device=None):
    from .trust_recovery import (correction_diagnostics, curriculum, fixed_training_subset,
                                 initialize_residual, module_gradient_norms)
    root = Path(config["root"])
    root.mkdir(parents=True, exist_ok=True)
    train_cfg = config["training"]
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = bool(train_cfg["amp"] and device.type == "cuda")
    seed_all(config["seed"])
    model = TrustMoESR(**config["model"]).to(device)
    base_only = config["profile"] == "base"
    if base_only:
        model.requires_grad_(False)
        model.base.requires_grad_(True)
    elif not config.get("parent"):
        raise ValueError("Residual training requires the shared trained base checkpoint")
    else:
        import_base(model, config["parent"])
    initializer = config.get("residual_initializer")
    if initializer and not base_only:
        initialize_residual(model, initializer, dataset_id=config["dataset_id"])
    frozen_base = {k: v.detach().cpu().clone() for k, v in model.base.state_dict().items()}
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=train_cfg["learning_rate"], weight_decay=0)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    adversarial = float(config["losses"].get("adversarial", 0))
    discriminator = PatchDiscriminator(base_channels=24).to(device) if adversarial and not base_only else None
    d_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=2e-5, weight_decay=0) if discriminator else None
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    start, best, optimizer_steps = 0, -float("inf"), 0
    last = root / "last.pt"
    # Epoch count and runtime paths may change on resume. Model/data/loss/batch/crop
    # changes require another experiment, not silently altered optimizer history.
    signature = {k: copy.deepcopy(config[k]) for k in ("profile", "seed", "model", "dataset_id", "losses")}
    signature["training"] = {k: v for k, v in train_cfg.items() if k not in ("epochs", "num_workers")}
    if initializer:
        signature["residual_initializer_sha256"] = digest_file(initializer)
    lineage = {"parent_sha256": digest_file(config["parent"]) if config.get("parent") else None,
               "initializer_sha256": digest_file(initializer) if initializer else None}
    if last.exists():
        saved = torch.load(last, map_location=device, weights_only=False)
        if saved["signature"] != signature:
            raise ValueError("Architecture, data, loss or batch/crop changed; choose another experiment root. Epoch increases are supported.")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        if amp and saved.get("scaler"):
            scaler.load_state_dict(saved["scaler"])
        ema = saved["ema"]
        start, best = saved["epoch"], saved["best_psnr"]
        optimizer_steps = saved.get("optimizer_steps", 0)
        if discriminator:
            discriminator.load_state_dict(saved["discriminator"])
            d_optimizer.load_state_dict(saved["d_optimizer"])
        if not base_only and any(not torch.equal(v.cpu(), frozen_base[k]) for k, v in model.base.state_dict().items()):
            raise ValueError("The parent base changed since this residual run. Use the original parent or a new root.")
    write_json(root / "config.json", config)
    train_data, val_data = dataset_for(config, "train"), dataset_for(config, "val")
    diagnostic_count = int(train_cfg.get("diagnostic_overfit_pairs", 0))
    if diagnostic_count:
        train_data = fixed_training_subset(config, diagnostic_count)
        val_data = train_data
        print("Diagnostic only: validation below measures memorization of fixed TRAIN crops.", flush=True)
    if not len(train_data):
        raise ValueError("Training split is empty")
    print(f"[{config['profile']}] train={len(train_data)} val={len(val_data)} epochs={start}->{train_cfg['epochs']} batch={train_cfg['batch_size']} device={device}", flush=True)
    for epoch in range(start, int(train_cfg["epochs"])):
        weights, forward_options, warming = curriculum(config, epoch)
        for group in optimizer.param_groups:
            group["lr"] = train_cfg["learning_rate"] * train_cfg.get("lr_epoch_decay", 1.0) ** epoch
        # Epoch-addressed RNG makes an interrupted epoch reproducible on restart.
        seed_all(config["seed"] + epoch)
        generator = torch.Generator().manual_seed(config["seed"] + epoch)
        loader = DataLoader(train_data, batch_size=train_cfg["batch_size"], shuffle=True,
                            num_workers=train_cfg["num_workers"], generator=generator,
                            pin_memory=device.type == "cuda", drop_last=False)
        model.train()
        totals, tick, gradient_report = Counter(), time.perf_counter(), {}
        for step, batch in enumerate(loader, 1):
            lr, target, mask = (batch[key].to(device) for key in ("lr", "hr", "valid_mask"))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                output = model(lr, base_only=base_only,
                               explore=0.3 if epoch < train_cfg["exploration_epochs"] else 0.05,
                               **forward_options)
            if base_only:
                loss = 100 * mse_loss(output.image.float(), target, mask) + 0.5 * charbonnier(output.image.float(), target, mask=mask)
                losses = {"reconstruction": loss}
            else:
                loss, losses = trust_moe_losses(output, target, mask, model.tile_size,
                                               model.scale, weights, model.use_trust and not warming,
                                               train_cfg.get("risk_target", "error"))
            if discriminator and weights.get("adversarial", 0):
                discriminator.train().requires_grad_(True)
                d_optimizer.zero_grad(set_to_none=True)
                # Invalid pixels are replaced with the same base in real/fake inputs.
                fake = output.image.float() * mask + output.base.detach().float() * (1 - mask)
                real = target * mask + output.base.detach().float() * (1 - mask)
                d_loss = discriminator_hinge(discriminator(real, lr), discriminator(fake.detach(), lr))
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                d_optimizer.step()
                discriminator.eval().requires_grad_(False)
                losses["adversarial"] = generator_hinge(discriminator(fake, lr))
                loss = loss + weights["adversarial"] * losses["adversarial"]
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite loss in epoch {epoch + 1}, batch {step}; last checkpoint preserved")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if step == 1 and not base_only:
                gradient_report = module_gradient_norms(model)
                diagnostic_report = correction_diagnostics(output, target, mask)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            step_applied = not amp or scaler.get_scale() >= previous_scale
            optimizer_steps += int(step_applied)
            with torch.no_grad():
                decay = train_cfg["ema_decay"]
                if train_cfg.get("ema_warmup"):
                    decay = min(decay, (1 + optimizer_steps) / (10 + optimizer_steps))
                for key, value in model.state_dict().items():
                    if value.is_floating_point() and not (key.startswith("base.") and not base_only):
                        if step_applied:
                            ema[key].lerp_(value.detach(), 1 - decay)
                    else:
                        ema[key].copy_(value)
            for key, value in {**losses, "total": loss}.items():
                totals[key] += float(value.detach())
            if train_cfg.get("progress") != "compact" and (step == len(loader) or step == max(1, len(loader) // 2)):
                print(f"[{config['profile']}] epoch {epoch+1}/{train_cfg['epochs']} batch {step}/{len(loader)} loss={totals['total']/step:.5f}", flush=True)
        raw_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema)
        model.eval()
        val_psnr, val_ssim = validate(model, val_data, device, base_only=base_only, amp=amp)
        model.load_state_dict(raw_state)
        # Warm-up uses a different dispatch policy; select only deployed-policy epochs.
        improved = not warming and val_psnr > best
        if improved:
            best = val_psnr
        saved = {"config": config, "signature": signature, "epoch": epoch + 1,
                 "model": raw_state, "ema": ema, "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict(), "best_psnr": best,
                 "validation": {"psnr": val_psnr, "ssim": val_ssim},
                 "optimizer_steps": optimizer_steps, "lineage": lineage}
        if discriminator:
            saved.update(discriminator=discriminator.state_dict(), d_optimizer=d_optimizer.state_dict())
        _save_checkpoint(last, saved)
        if improved:
            _save_checkpoint(root / "best.pt", saved)
        row = {"epoch": epoch + 1, "val_psnr": val_psnr, "val_ssim": val_ssim,
               "seconds": time.perf_counter() - tick, "warmup": warming,
               "optimizer_steps": optimizer_steps, "learning_rate": optimizer.param_groups[0]["lr"],
               "weighted_objective": weights,
               **{f"loss_{k}": v / step for k, v in totals.items()}}
        if not base_only:
            row.update(first_batch_diagnostics=diagnostic_report, first_batch_gradient_norms=gradient_report)
        # One receipt per completed epoch also makes summary rebuilding idempotent.
        write_json(root / "history" / f"epoch_{epoch+1:04d}.json", row)
        suffix = ""
        if not base_only:
            suffix = f" correction={diagnostic_report['correction_abs_mean']:.2e}"
        print(f"[{config['profile']}] epoch {epoch+1}/{train_cfg['epochs']} loss={totals['total']/step:.5f} val_psnr={val_psnr:.4f} val_ssim={val_ssim:.5f} best={best:.4f}{suffix} warmup={warming} elapsed={row['seconds']:.0f}s", flush=True)
    if not (root / "best.pt").exists():
        raise RuntimeError("Only warm-up epochs completed. Increase epochs beyond residual_warmup_epochs and resume.")
    return root / "best.pt"


@torch.inference_mode()
def evaluate(checkpoint, output_root, split="val", *, device=None, coverage=None,
             top_k=None, save_images=True, manifest=None):
    from .trust_recovery import correction_diagnostics
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, config = load_model(checkpoint, device)
    if manifest:
        config["manifest"] = str(manifest)
    if config["training"].get("diagnostic_overfit_pairs"):
        raise ValueError("Memorization checkpoints are diagnostic only; do not evaluate them on validation/test.")
    dataset = dataset_for(config, split)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    request = {"checkpoint_sha256": digest_file(checkpoint), "split": split,
               "dataset_id": config["dataset_id"], "coverage": coverage,
               "top_k": top_k, "save_images": save_images, "device": str(device),
               "torch": torch.__version__, "amp": bool(config["training"]["amp"] and device.type == "cuda")}
    # V2 explicitly versions its evaluator and identifies the actual manifest used.
    if config.get("format") == "trust-recovery-v2":
        request.update(evaluator="trust-recovery-v2", manifest_sha256=digest_file(config["manifest"]))
    receipt = root / "request.json"
    if receipt.exists() and json.loads(receipt.read_text()) != request:
        raise ValueError("Evaluation settings/checkpoint changed. Use another output subfolder.")
    write_json(receipt, request)
    amp = request["amp"]
    rows = []
    for index, sample in enumerate(dataset):
        record_path = root / "records" / f"{index:06d}.json"
        image_path = root / "images" / f"{index:06d}.npz"
        if record_path.exists() and (not save_images or image_path.exists()):
            rows.append(json.loads(record_path.read_text()))
            continue
        tensors = {k: sample[k][None].to(device) for k in ("lr", "hr", "valid_mask", "valid_mask_lr", "degradation")}
        with torch.autocast(device.type, enabled=amp):
            result = model(tensors["lr"], coverage=coverage, top_k=top_k, base_only=config["profile"] == "base")
        target, mask, lr = tensors["hr"], tensors["valid_mask"], tensors["lr"]
        predictions = {"model": result.image.float(), "base": result.base.float(),
                       "bicubic": F.interpolate(lr, size=target.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)}
        record = dataset.records[index]
        row = {"index": index, "patch": record.patch, "tile_id": record.tile_id,
               "source_product": record.source_product, "scene_class": record.scene_class}
        for name, prediction in predictions.items():
            metrics = basic_metrics(prediction, target, lr, tensors["degradation"], scale=3,
                                    mask=mask, lr_mask=tensors["valid_mask_lr"])
            row.update({key if name == "model" else f"{name}_{key}": value for key, value in metrics.items()})
        row.update(psnr_delta_vs_base=row["psnr"] - row["base_psnr"],
                   active_fraction=float(result.active.mean()),
                   expert_tile_calls=int(result.dispatched_tiles),
                   expert_load=result.assignments.sum((0, 1)).cpu().tolist())
        row.update(correction_diagnostics(result, target, mask))
        if save_images:
            image_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = image_path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, prediction=result.image[0].float().cpu().numpy(),
                                base=result.base[0].float().cpu().numpy(),
                                residual=result.residual[0].float().cpu().numpy(),
                                trust=result.trust[0].float().cpu().numpy(),
                                active=result.active[0].float().cpu().numpy(),
                                difficulty=result.difficulty[0].float().cpu().numpy(),
                                probabilities=result.probabilities[0].float().cpu().numpy(),
                                token_grid=np.array([(lr.shape[-2] + model.tile_size - 1) // model.tile_size,
                                                     (lr.shape[-1] + model.tile_size - 1) // model.tile_size]),
                                assignments=result.assignments[0].cpu().numpy())
            temporary.replace(image_path)
        write_json(record_path, row)
        rows.append(row)
        if (index + 1) % 10 == 0 or index == 0 or index + 1 == len(dataset):
            print(f"[{config['profile']}/{split}] {index+1}/{len(dataset)} PSNR={row['psnr']:.3f}", flush=True)
    if not rows:
        raise ValueError(f"Empty split: {split}")
    keys = [k for k, v in rows[0].items() if isinstance(v, (float, int)) and k != "index"]
    summary = {key: float(np.mean([r[key] for r in rows])) for key in keys}
    summary.update(count=len(rows), experiment=config["profile"], split=split,
                   fraction_beating_base_psnr=float(np.mean([r["psnr_delta_vs_base"] > 0 for r in rows])))
    write_json(root / "metrics.json", summary)
    write_json(root / "per_image.json", rows)
    return summary
