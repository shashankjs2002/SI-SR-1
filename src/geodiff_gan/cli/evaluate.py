from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import load_config
from ..data import SentinelPatchDataset
from ..metrics import OptionalMetricSuite, basic_metrics
from ..models.system import GeoDiffGAN
from ..text import build_text_encoder
from ..training.checkpoint import load_checkpoint


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
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeoDiffGAN.from_config(config).to(device).eval()
    load_checkpoint(args.checkpoint, model, strict=False)
    text_encoder = build_text_encoder(config).to(device).eval()
    optional_metrics = OptionalMetricSuite(device)
    dataset = SentinelPatchDataset(
        config["data"]["manifest"],
        split=args.split,
        scale=config["model"].get("scale", 4),
        caption_file=config["data"].get("captions"),
        augment=False,
        degradation_seed=int(config["data"].get("degradation_seed", 0)),
        degradation_severity=config["data"].get("degradation_severity", "mild"),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    totals: defaultdict[str, float] = defaultdict(float)
    count = 0
    for index, batch in enumerate(loader):
        if args.limit is not None and index >= args.limit:
            break
        lr = batch["lr"].to(device)
        clean_lr = batch["clean_lr"].to(device)
        hr = batch["hr"].to(device)
        degradation = batch["degradation"].to(device)
        context = text_encoder(list(batch["caption"]))
        null_context = text_encoder([""])
        outputs = []
        for seed in range(args.samples):
            generator = torch.Generator(device=device).manual_seed(seed)
            outputs.append(
                model.sample(
                    lr,
                    context,
                    degradation=degradation,
                    projection_lr=clean_lr,
                    mode=args.mode,
                    sample_steps=args.steps,
                    null_context=null_context,
                    generator=generator,
                )
            )
        predictions = [output.image for output in outputs]
        stack = torch.stack(predictions)
        raw_mean = stack.mean(dim=0)
        uncertainty = stack.var(dim=0, unbiased=False).mean(dim=1)
        evidence = torch.stack(
            [output.evidence_confidence for output in outputs]
        ).mean(dim=0)
        edit_permission = torch.stack(
            [output.edit_permission for output in outputs]
        ).mean(dim=0)
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
        if args.mode == "edit":
            values["prompt_alignment"] = float(
                1 - text_encoder.alignment_loss(mean, list(batch["caption"]))
            )
        for name, value in values.items():
            totals[name] += value
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
    summary = {name: value / max(count, 1) for name, value in totals.items()}
    summary["count"] = count
    summary["samples_per_patch"] = args.samples
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(summary)


if __name__ == "__main__":
    main()
