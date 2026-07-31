from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..benchmark.models import MODEL_SPECS, build_benchmark_model, pixelshuffle_modules
from ..benchmark.runner import BenchmarkConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train faithful or explicitly adapted SR backbones on Sentinel-2 patches"
    )
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--architecture-mode",
        choices=("official", "resize_conv_ablation"),
        default="official",
        help="Use the paper's released x4 head by default. Adapted heads are ablations only.",
    )
    parser.add_argument("--max-updates", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--lr-crop",
        type=int,
        default=None,
        help=(
            "Audit override only. It must equal the selected model's official "
            "training patch size; omit it to use the registry value."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--validation-limit", type=int, default=64)
    parser.add_argument("--test-limit", type=int, default=40)
    parser.add_argument("--validate-every", type=int, default=2000)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--degradation-seed", type=int, default=42)
    parser.add_argument(
        "--degradation-severity",
        choices=("mild", "moderate", "severe"),
        default="mild",
    )
    parser.add_argument("--optional-metrics", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Build the model, run an x4 forward pass, and print architecture metadata.",
    )
    args = parser.parse_args()
    spec = MODEL_SPECS[args.model]
    if args.lr_crop is not None and args.lr_crop != spec.native_lr_size:
        parser.error(
            f"{spec.name} uses {spec.native_lr_size}x{spec.native_lr_size} LR "
            f"patches in this protocol; received --lr-crop {args.lr_crop}."
        )
    if args.probe_only:
        model = build_benchmark_model(
            args.model,
            args.source_root,
            architecture_mode=args.architecture_mode,
        )
        model.eval()
        probe_size = spec.native_lr_size
        with torch.no_grad():
            output = model(torch.rand(1, 3, probe_size, probe_size))
        expected_output = [
            1,
            3,
            probe_size * spec.scale,
            probe_size * spec.scale,
        ]
        if list(output.shape) != expected_output:
            raise RuntimeError(
                f"{spec.name} produced {list(output.shape)} for its native input; "
                f"expected {expected_output}."
            )
        print(
            json.dumps(
                {
                    "model": args.model,
                    "architecture_mode": args.architecture_mode,
                    "parameters": sum(value.numel() for value in model.parameters()),
                    "pixelshuffle_modules": pixelshuffle_modules(model),
                    "probe_input": [1, 3, probe_size, probe_size],
                    "probe_output": list(output.shape),
                },
                indent=2,
            )
        )
        return
    config = BenchmarkConfig(
        model=args.model,
        source_root=Path(args.source_root),
        manifest=Path(args.manifest),
        output=Path(args.output),
        architecture_mode=args.architecture_mode,
        max_updates=args.max_updates,
        batch_size=args.batch_size,
        accumulation=args.accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_crop=args.lr_crop,
        num_workers=args.num_workers,
        validation_limit=args.validation_limit,
        test_limit=args.test_limit,
        validate_every=args.validate_every,
        early_stopping_patience=args.early_stopping_patience,
        degradation_seed=args.degradation_seed,
        degradation_severity=args.degradation_severity,
        amp=not args.no_amp,
        optional_metrics=args.optional_metrics,
        seed=args.seed,
    )
    print(json.dumps(train(config), indent=2))


if __name__ == "__main__":
    main()
