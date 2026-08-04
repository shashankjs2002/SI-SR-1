from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import islice
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..config import load_config
from ..data import SentinelPatchDataset
from ..io import save_rgb
from ..metrics import OptionalMetricSuite, basic_metrics
from ..models.degradation import sensor_degrade
from ..models.system import GeoDiffGAN
from ..text import build_text_encoder, controlled_prompt_variants
from ..training.checkpoint import load_checkpoint


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def _mean(values: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(count, 1) for key, value in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlled caption ablation for evidence-aware GeoDiff-GAN"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--defaults")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--save-images", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--optional-metrics",
        action="store_true",
        help="Enable LPIPS/DISTS. Their first use may download external weights.",
    )
    args = parser.parse_args()
    if args.limit < 1 or args.steps < 1:
        parser.error("--limit and --steps must be positive")

    config = load_config(args.config, args.defaults)
    device = _device(args.device)
    model = GeoDiffGAN.from_config(config).to(device).eval()
    load_checkpoint(args.checkpoint, model, strict=False)
    text_encoder = build_text_encoder(config).to(device).eval()
    optional_metrics = OptionalMetricSuite(
        device,
        enabled=args.optional_metrics,
    )
    dataset = SentinelPatchDataset(
        config["data"]["manifest"],
        split=args.split,
        scale=int(config["model"].get("scale", 4)),
        caption_file=config["data"].get("captions"),
        caption_field=config["data"].get("caption_field", "caption"),
        caption_sampling="fixed",
        augment=False,
        random_degradation=False,
        degradation_seed=int(config["data"].get("degradation_seed", 0)),
        degradation_severity=config["data"].get("degradation_severity", "mild"),
        target_key=config["data"].get("target_key", "hr"),
        condition_key=config["data"].get("condition_key"),
        output_channels=int(config["model"].get("output_channels", 3)),
    )
    if len(dataset) == 0:
        raise SystemExit(f"No patches found for split {args.split!r}")
    total = min(len(dataset), args.limit)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    output_dir = Path(args.output)
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    totals: dict[str, defaultdict[str, float]] = {
        name: defaultdict(float)
        for name in ("matched", "null", "paraphrase", "mismatch")
    }
    rows: list[dict[str, object]] = []
    amp = device.type == "cuda" and bool(config.get("training", {}).get("amp", True))
    print(
        f"[prompt-ablation] split={args.split} patches={total} steps={args.steps} "
        f"seed={args.seed} device={device}",
        flush=True,
    )

    for index, batch in enumerate(
        tqdm(islice(loader, total), total=total, desc="prompt ablation", unit="patch")
    ):
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        degradation = batch["degradation"].to(device)
        clean_lr = batch["clean_lr"].to(device)
        caption = str(batch["caption"][0])
        variants = controlled_prompt_variants(caption)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp,
        ):
            base = model.base(lr)
            lr_features = model.lr_encoder(lr)
            contexts = {
                name: text_encoder([prompt]).to(device)
                for name, prompt in variants.items()
            }
            null_context = contexts["null"]
            outputs = {}
            for name in ("null", "matched", "paraphrase", "mismatch"):
                generator = torch.Generator(device=device).manual_seed(args.seed)
                outputs[name] = model.sample(
                    lr,
                    contexts[name],
                    degradation=degradation,
                    projection_lr=clean_lr,
                    mode="sr",
                    sample_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    null_context=null_context,
                    generator=generator,
                    base=base,
                    lr_features=lr_features,
                )

        null_image = outputs["null"].image
        null_lr = sensor_degrade(
            null_image,
            degradation,
            scale=model.scale,
            severity=model.degradation_severity,
        )
        matched_delta = (outputs["matched"].image - null_image).abs().mean()
        row: dict[str, object] = {
            "index": index,
            "patch": str(batch.get("patch", [index])[0]),
            "tile_id": str(batch.get("tile_id", [""])[0]),
            "captions": variants,
        }
        for name, output in outputs.items():
            values = basic_metrics(
                output.image,
                hr,
                clean_lr,
                degradation,
                scale=model.scale,
                severity=model.degradation_severity,
            )
            values.update(optional_metrics(output.image, hr))
            output_lr = sensor_degrade(
                output.image,
                degradation,
                scale=model.scale,
                severity=model.degradation_severity,
            )
            values.update(
                {
                    "prompt_support_mean": float(output.prompt_support.mean()),
                    "prompt_permission_mean": float(output.prompt_permission.mean()),
                    "image_delta_from_null": float(
                        (output.image - null_image).abs().mean()
                    ),
                    "anchor_delta_from_null": float(
                        (output.sr_anchor - outputs["null"].sr_anchor).abs().mean()
                    ),
                    "lr_delta_from_null": float((output_lr - null_lr).abs().mean()),
                }
            )
            for key, value in values.items():
                totals[name][key] += float(value)
            row[name] = values
            if index < args.save_images:
                save_rgb(image_dir / f"{index:04d}_{name}.png", output.image)
        mismatch_delta = (outputs["mismatch"].image - null_image).abs().mean()
        row["contradiction_ratio"] = float(
            mismatch_delta / matched_delta.clamp_min(1e-8)
        )
        rows.append(row)
        if index < args.save_images:
            save_rgb(image_dir / f"{index:04d}_target.png", hr)
            save_rgb(image_dir / f"{index:04d}_lr.png", lr[:, :3])

    summary = {
        "count": total,
        "split": args.split,
        "steps": args.steps,
        "seed": args.seed,
        "prompt_evidence_controller": bool(
            config["model"].get("use_prompt_evidence_controller", False)
        ),
        "variants": {
            name: _mean(dict(values), total)
            for name, values in totals.items()
        },
        "mean_contradiction_ratio": sum(
            float(row["contradiction_ratio"]) for row in rows
        ) / max(total, 1),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "per_patch.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
