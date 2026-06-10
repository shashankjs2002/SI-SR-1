from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..config import load_config
from ..data import SentinelPatchDataset
from ..diagnostics import DiagnosticRecorder
from ..models.system import GeoDiffGAN
from ..text import build_text_encoder
from ..training.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GeoDiff-GAN tensor and visual diagnostics")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--mode", choices=("sr", "edit"), default="sr")
    parser.add_argument("--prompt")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diffusion-every", type=int, default=5)
    parser.add_argument("--allow-nonfinite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeoDiffGAN.from_config(config).to(device).eval()
    load_checkpoint(args.checkpoint, model, strict=False)
    text_encoder = build_text_encoder(config).to(device).eval()
    dataset = SentinelPatchDataset(
        config["data"]["manifest"],
        split=args.split,
        scale=model.scale,
        caption_file=config["data"].get("captions"),
        augment=False,
        degradation_seed=int(config["data"].get("degradation_seed", 0)),
        degradation_severity=config["data"].get("degradation_severity", "mild"),
    )
    if not dataset:
        raise RuntimeError(f"No samples found in split {args.split!r}")
    sample = dataset[args.index % len(dataset)]
    lr = sample["lr"].unsqueeze(0).to(device)
    clean_lr = sample["clean_lr"].unsqueeze(0).to(device)
    target = sample["hr"].unsqueeze(0).to(device)
    degradation = sample["degradation"].unsqueeze(0).to(device)
    prompt = args.prompt if args.prompt is not None else str(sample["caption"])
    context = text_encoder([prompt])
    null_context = text_encoder([""])
    recorder = DiagnosticRecorder(
        args.output,
        verbose=True,
        fail_on_nonfinite=not args.allow_nonfinite,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.no_grad():
        output = model.sample(
            lr,
            context,
            degradation=degradation,
            projection_lr=clean_lr,
            mode=args.mode,
            sample_steps=args.steps,
            guidance_scale=args.guidance,
            null_context=null_context,
            generator=generator,
            diagnostics=recorder,
            diffusion_debug_interval=args.diffusion_every,
        )
    recorder.add_spatial_metrics(
        lr,
        output.base,
        output.residual,
        output.image,
        degradation,
        scale=model.scale,
        target=target,
        consistency_lr=clean_lr,
        degradation_severity=model.degradation_severity,
    )
    report = recorder.export(
        {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "config": str(Path(args.config).resolve()),
            "patch": str(sample["patch"]),
            "tile_id": str(sample["tile_id"]),
            "split": args.split,
            "index": args.index,
            "mode": args.mode,
            "prompt": prompt,
            "seed": args.seed,
            "steps": args.steps,
            "guidance": args.guidance,
            "architecture_notes": {
                "f32_and_f16_lr_features_are_recorded_but_not_consumed": True,
                "training_joint_path_uses_loss_consistency_without_back_projection": True,
                "inference_sr_mode_uses_three_back_projection_steps": True,
            },
        }
    )
    print(f"Diagnostic report: {report}")
    print(f"Share the complete directory: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
