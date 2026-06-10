from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..config import load_config
from ..diagnostics import DiagnosticRecorder
from ..io import load_rgb, save_metadata, save_rgb
from ..models.system import GeoDiffGAN
from ..text import build_text_encoder
from ..training.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GeoDiff-GAN patch inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--mode", choices=("sr", "edit"), default="sr")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug-dir")
    parser.add_argument("--debug-diffusion-every", type=int, default=5)
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeoDiffGAN.from_config(config).to(device).eval()
    load_checkpoint(args.checkpoint, model, strict=False)
    text_encoder = build_text_encoder(config).to(device).eval()
    lr = load_rgb(args.input).unsqueeze(0).to(device)
    context = text_encoder([args.prompt])
    null_context = text_encoder([""])
    generator = torch.Generator(device=device).manual_seed(args.seed)
    diagnostics = (
        DiagnosticRecorder(args.debug_dir, verbose=True)
        if args.debug_dir
        else None
    )
    output = model.sample(
        lr,
        context,
        mode=args.mode,
        sample_steps=args.steps,
        guidance_scale=args.guidance,
        null_context=null_context,
        generator=generator,
        diagnostics=diagnostics,
        diffusion_debug_interval=args.debug_diffusion_every,
    )
    save_rgb(args.output, output.image)
    metadata_path = Path(args.output).with_suffix(".json")
    metadata = output.metadata[0] | {
        "prompt": args.prompt,
        "guidance_scale": args.guidance,
        "diffusion_steps": args.steps,
        "seed": args.seed,
        "checkpoint": str(Path(args.checkpoint).resolve()),
    }
    save_metadata(metadata_path, metadata)
    if diagnostics is not None:
        diagnostics.add_spatial_metrics(
            lr,
            output.base,
            output.residual,
            output.image,
            torch.tensor(
                [[0.4, 0.0, 0.0, 0.0]],
                device=device,
                dtype=lr.dtype,
            ),
            scale=model.scale,
        )
        diagnostics.export(metadata)
    print(f"wrote {args.output} and {metadata_path}")


if __name__ == "__main__":
    main()
