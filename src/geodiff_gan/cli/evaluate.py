from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..config import load_config
from ..data import SentinelPatchDataset
from ..metrics import OptionalMetricSuite, basic_metrics
from ..models.system import GeoDiffGAN
from ..text import build_text_encoder
from ..training.checkpoint import load_checkpoint


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA evaluation was requested, but torch.cuda.is_available() is False. "
            "Check the Kaggle accelerator and restart the session."
        )
    return torch.device(requested)


def _device_summary(device: torch.device) -> str:
    if device.type == "cuda":
        return (
            f"{device} ({torch.cuda.get_device_name(device)}) "
            f"torch={torch.__version__} cuda={torch.version.cuda}"
        )
    return f"{device} torch={torch.__version__}"


def _safe_correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.float().flatten()
    second = second.float().flatten()
    if first.numel() < 2 or first.std(unbiased=False) < 1e-8:
        return 0.0
    if second.std(unbiased=False) < 1e-8:
        return 0.0
    value = torch.corrcoef(torch.stack((first, second)))[0, 1]
    return float(value) if torch.isfinite(value) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GeoDiff-GAN with uncertainty sampling")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--mode", choices=("sr", "edit"), default="sr")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--progress",
        choices=("compact", "tqdm", "quiet"),
        default="compact",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Use zero text context and do not load the external text encoder.",
    )
    parser.add_argument(
        "--optional-metrics",
        action="store_true",
        help="Enable LPIPS/DISTS. Their first use may download external weights.",
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    config = load_config(args.config)
    device = _resolve_device(args.device)
    amp_enabled = bool(
        device.type == "cuda"
        and config.get("training", {}).get("amp", True)
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"[evaluate] device={_device_summary(device)}", flush=True)
    print(f"[evaluate] amp={amp_enabled}", flush=True)
    print("[evaluate] building GeoDiff-GAN", flush=True)
    model = GeoDiffGAN.from_config(config).to(device).eval()
    print(f"[evaluate] loading checkpoint {args.checkpoint}", flush=True)
    load_checkpoint(args.checkpoint, model, strict=False)
    text_encoder = None
    if args.no_text:
        print("[evaluate] text conditioning disabled; using zero context", flush=True)
    else:
        print("[evaluate] loading frozen text encoder", flush=True)
        text_encoder = build_text_encoder(config).to(device).eval()
        print("[evaluate] text encoder ready", flush=True)
    if args.optional_metrics:
        print(
            "[evaluate] loading LPIPS/DISTS; first use may download weights",
            flush=True,
        )
    else:
        print("[evaluate] LPIPS/DISTS disabled", flush=True)
    optional_metrics = OptionalMetricSuite(device, enabled=args.optional_metrics)
    print(f"[evaluate] loading {args.split} dataset", flush=True)
    dataset = SentinelPatchDataset(
        config["data"]["manifest"],
        split=args.split,
        scale=config["model"].get("scale", 4),
        caption_file=config["data"].get("captions"),
        augment=False,
        degradation_seed=int(config["data"].get("degradation_seed", 0)),
        degradation_severity=config["data"].get("degradation_severity", "mild"),
    )
    if len(dataset) == 0:
        raise SystemExit(
            f"No patches found for split {args.split!r}. "
            "Check the manifest SAFE-prefix assignments."
        )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    totals: defaultdict[str, float] = defaultdict(float)
    count = 0
    total = min(len(loader), args.limit) if args.limit is not None else len(loader)
    total_passes = total * args.samples * args.steps
    print(
        f"[evaluate] plan patches={total} samples={args.samples} "
        f"steps={args.steps} diffusion_unet_passes={total_passes:,}",
        flush=True,
    )
    use_tqdm = args.progress == "tqdm"
    progress = tqdm(
        islice(loader, total),
        total=total,
        desc=f"evaluate {args.split}",
        unit="patch",
        disable=not use_tqdm,
    )
    started = time.monotonic()
    for index, batch in enumerate(progress):
        patch_started = time.monotonic()
        lr = batch["lr"].to(device)
        clean_lr = batch["clean_lr"].to(device)
        hr = batch["hr"].to(device)
        degradation = batch["degradation"].to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            if text_encoder is None:
                context = torch.zeros(
                    lr.shape[0],
                    1,
                    int(config["model"].get("context_dim", 768)),
                    device=device,
                    dtype=lr.dtype,
                )
            else:
                context = text_encoder(list(batch["caption"]))
            with torch.no_grad():
                base = model.base(lr)
                lr_features = model.lr_encoder(lr)
        outputs = []
        sample_progress = tqdm(
            range(args.samples),
            desc=f"patch {index + 1}/{total} samples",
            unit="sample",
            leave=False,
            disable=not use_tqdm,
        )
        for seed in sample_progress:
            generator = torch.Generator(device=device).manual_seed(seed)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                outputs.append(
                    model.sample(
                        lr,
                        context,
                        degradation=degradation,
                        projection_lr=clean_lr,
                        mode=args.mode,
                        sample_steps=args.steps,
                        generator=generator,
                        base=base,
                        lr_features=lr_features,
                    )
                )
            if args.progress == "compact":
                sample_elapsed = time.monotonic() - patch_started
                samples_done = seed + 1
                sample_eta = (
                    sample_elapsed / samples_done * (args.samples - samples_done)
                )
                print(
                    f"[evaluate] patch {index + 1}/{total} "
                    f"sample {samples_done}/{args.samples} "
                    f"sample_eta={_duration(sample_eta)}",
                    flush=True,
                )
        predictions = [output.image for output in outputs]
        stack = torch.stack(predictions).float()
        raw_mean = stack.mean(dim=0)
        uncertainty = stack.var(dim=0, unbiased=False).mean(dim=1)
        evidence = torch.stack(
            [output.evidence_confidence for output in outputs]
        ).float().mean(dim=0)
        edit_permission = torch.stack(
            [output.edit_permission for output in outputs]
        ).float().mean(dim=0)
        if args.mode == "sr":
            mean, combined_confidence, abstention = (
                model.apply_uncertainty_abstention(
                    raw_mean,
                    outputs[0].base,
                    evidence,
                    uncertainty,
                )
            )
        else:
            mean = raw_mean
            combined_confidence = model._resize_policy(
                evidence, mean.shape[-2:]
            )
            abstention = 1 - combined_confidence
        values = basic_metrics(
            mean,
            hr,
            clean_lr,
            degradation,
            scale=model.scale,
            severity=model.degradation_severity,
        )
        values["observed_lr_noise_l1"] = float((lr - clean_lr).abs().mean())
        values["observed_lr_noise_to_signal"] = float(
            (lr - clean_lr).abs().mean() / clean_lr.abs().mean().clamp_min(1e-8)
        )
        values.update(optional_metrics(mean, hr))
        error_map = (mean - hr).abs().mean(dim=1, keepdim=True)
        confidence_flat = combined_confidence.flatten()
        uncertainty_flat = uncertainty[:, None].flatten()
        error_flat = error_map.flatten()
        if confidence_flat.numel() > 1:
            values["confidence_error_correlation"] = _safe_correlation(
                confidence_flat, error_flat
            )
            values["uncertainty_error_correlation"] = _safe_correlation(
                uncertainty_flat, error_flat
            )
            keep = max(1, int(confidence_flat.numel() * 0.8))
            selected = torch.topk(confidence_flat, keep).indices
            values["selective_l1_at_80pct_coverage"] = float(
                error_flat[selected].mean()
            )
        if args.mode == "edit" and text_encoder is not None:
            values["prompt_alignment"] = float(
                1 - text_encoder.alignment_loss(mean, list(batch["caption"]))
            )
        for name, value in values.items():
            totals[name] += value
        progress.set_postfix(
            psnr=f"{totals['psnr'] / (count + 1):.2f}",
            ssim=f"{totals['ssim'] / (count + 1):.4f}",
        )
        patch_name = Path(batch["patch"][0]).stem
        np.savez_compressed(
            output_dir / f"{patch_name}_uncertainty.npz",
            mean=mean[0].detach().cpu().numpy(),
            raw_mean=raw_mean[0].detach().cpu().numpy(),
            variance=uncertainty[0].detach().cpu().numpy(),
            evidence_confidence=evidence[0].detach().cpu().numpy(),
            edit_permission=edit_permission[0].detach().cpu().numpy(),
            abstention_map=abstention[0].detach().cpu().numpy(),
        )
        count += 1
        if args.progress == "compact":
            elapsed = time.monotonic() - started
            remaining = elapsed / count * (total - count)
            print(
                f"[evaluate] patch {count}/{total} complete "
                f"psnr={totals['psnr'] / count:.2f} "
                f"ssim={totals['ssim'] / count:.4f} "
                f"elapsed={_duration(elapsed)} eta={_duration(remaining)}",
                flush=True,
            )
    summary = {name: value / max(count, 1) for name, value in totals.items()}
    summary["count"] = count
    summary["samples_per_patch"] = args.samples
    summary["diffusion_steps"] = args.steps
    summary["device"] = str(device)
    summary["amp"] = amp_enabled
    summary["optional_metrics"] = args.optional_metrics
    summary["text_conditioning"] = not args.no_text
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[evaluate] complete in {_duration(time.monotonic() - started)}", flush=True)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
