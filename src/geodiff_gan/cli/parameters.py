from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_config
from ..parameters import build_parameter_report, count_parameters, verify_parameter_report
from ..text import build_text_encoder


def _format_count(value: int) -> str:
    return f"{value:,}"


def _print_report(report: dict) -> None:
    core = report["core_model"]
    print("GeoDiff-GAN parameter audit")
    print("=" * 72)
    print(f"Core model: {_format_count(core['scalar_parameters'])}")
    print(f"Parameter tensors: {_format_count(core['parameter_tensors'])}")
    print(
        "Parameter storage: "
        f"{core['fp32_parameter_memory_mb']:.1f} MiB FP32, "
        f"{core['fp16_parameter_memory_mb']:.1f} MiB FP16"
    )
    print("\nCore modules")
    for name, values in report["core_modules"].items():
        print(f"  {name:18s} {_format_count(values['scalar_parameters']):>16s}")

    print("\nTraining-only discriminators")
    for name, values in report["discriminators"]["modules"].items():
        print(f"  {name:36s} {_format_count(values['scalar_parameters']):>16s}")
    print(
        f"  {'combined':36s} "
        f"{_format_count(report['discriminators']['scalar_parameters']):>16s}"
    )

    print("\nStage-wise optimized parameters")
    for stage, values in report["training_stages"].items():
        modules = ", ".join(values["trainable_modules"])
        print(
            f"  {stage:10s} "
            f"core={_format_count(values['core_trainable_parameters']):>14s} "
            f"D={_format_count(values['discriminator_trainable_parameters']):>14s} "
            f"total={_format_count(values['total_optimized_parameters']):>14s}"
        )
        print(f"             modules: {modules}")

    training = report["training_configuration"]
    print("\nTraining configuration")
    for name, value in training.items():
        if name != "loss_weights":
            print(f"  {name}: {value}")
    print("  loss_weights:")
    for name, value in training["loss_weights"].items():
        print(f"    {name}: {value}")

    if "text_encoder" in report:
        text = report["text_encoder"]
        print(
            "\nFrozen text encoder: "
            f"{text['name']} ({_format_count(text['scalar_parameters'])} parameters)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit GeoDiff-GAN model and stage-wise trainable parameters"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--defaults",
        help="Optional base YAML merged below --config, for example configs/default.yaml",
    )
    parser.add_argument("--patches", type=int, help="Estimate optimizer updates per epoch")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--json", dest="json_path", help="Write the report as JSON")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--include-text-encoder",
        action="store_true",
        help="Load and count the frozen text encoder; this may download model weights",
    )
    args = parser.parse_args()

    config = load_config(args.config, args.defaults)
    report = build_parameter_report(
        config,
        patches=args.patches,
        world_size=args.world_size,
    )
    if args.include_text_encoder:
        encoder = build_text_encoder(config)
        text_count = count_parameters(encoder)
        report["text_encoder"] = {
            "name": config.get("text_encoder", {}).get("model_name", "hash"),
            **text_count,
        }
    if args.verify:
        verify_parameter_report(report)
    _print_report(report)
    if args.json_path:
        destination = Path(args.json_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON report: {destination.resolve()}")
    if args.verify:
        print("\nVerification: PASS")


if __name__ == "__main__":
    main()
