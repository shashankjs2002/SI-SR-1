from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..benchmark.models import MODEL_SPECS
from ..benchmark.refiner_runner import (
    RefinerConfig,
    evaluate_refiner,
    select_refiner_checkpoint,
    train_refiner,
)


def _config(args: argparse.Namespace) -> RefinerConfig:
    return RefinerConfig(
        model=args.model,
        source_root=Path(args.source_root),
        manifest=Path(args.manifest),
        base_checkpoint=Path(args.base_checkpoint),
        output=Path(args.output),
        max_updates=args.max_updates,
        batch_size=args.batch_size,
        accumulation=args.accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        validation_limit=args.validation_limit,
        test_limit=args.test_limit,
        validate_every=args.validate_every,
        early_stopping_patience=args.early_stopping_patience,
        degradation_seed=args.degradation_seed,
        degradation_severity=args.degradation_severity,
        channels=args.channels,
        condition_dim=args.condition_dim,
        blocks_per_level=tuple(args.blocks_per_level),
        max_residual=args.max_residual,
        nullspace_iterations=args.nullspace_iterations,
        nullspace_step=args.nullspace_step,
        amp=not args.no_amp,
        optional_metrics=args.optional_metrics,
        save_images=args.save_images,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the shared Sensor-Nullspace High-Frequency "
            "Refiner on a frozen official x4 SR backbone."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("train-evaluate", "train", "evaluate"),
        default="train-evaluate",
    )
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-updates", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--validation-limit", type=int, default=64)
    parser.add_argument("--test-limit", type=int, default=80)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--degradation-seed", type=int, default=42)
    parser.add_argument(
        "--degradation-severity",
        choices=("mild", "moderate", "severe"),
        default="mild",
    )
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--condition-dim", type=int, default=64)
    parser.add_argument(
        "--blocks-per-level",
        type=int,
        nargs=3,
        default=(2, 2, 3),
        metavar=("HR", "HALF", "QUARTER"),
    )
    parser.add_argument("--max-residual", type=float, default=0.12)
    parser.add_argument("--nullspace-iterations", type=int, default=1)
    parser.add_argument("--nullspace-step", type=float, default=0.75)
    parser.add_argument("--save-images", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--optional-metrics", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.max_updates <= 0:
        parser.error("--max-updates must be positive")
    if args.validate_every <= 0:
        parser.error("--validate-every must be positive")
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    if args.nullspace_iterations < 0:
        parser.error("--nullspace-iterations cannot be negative")
    if args.max_residual <= 0:
        parser.error("--max-residual must be positive")

    config = _config(args)
    if args.mode == "evaluate":
        result = evaluate_refiner(
            config,
            checkpoint=select_refiner_checkpoint(config.output),
            split="test",
        )
    else:
        result = train_refiner(
            config,
            evaluate_test=args.mode == "train-evaluate",
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
