from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..config import load_config
from ..data import SentinelPatchDataset
from ..metrics import OptionalMetricSuite, basic_metrics
from ..models.system import GeoDiffGAN
from ..training.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate bicubic and base-branch baselines")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SentinelPatchDataset(
        config["data"]["manifest"],
        split=args.split,
        scale=config["model"].get("scale", 4),
        caption_file=config["data"].get("captions"),
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = None
    if args.base_checkpoint:
        model = GeoDiffGAN.from_config(config).to(device).eval()
        load_checkpoint(args.base_checkpoint, model, strict=False)
    optional_metrics = OptionalMetricSuite(device)
    totals: dict[str, defaultdict[str, float]] = {
        "bicubic": defaultdict(float),
        "base": defaultdict(float),
    }
    count = 0

    with torch.no_grad():
        for index, batch in enumerate(loader):
            if args.limit is not None and index >= args.limit:
                break
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            degradation = batch["degradation"].to(device)
            predictions = {
                "bicubic": F.interpolate(
                    lr, size=hr.shape[-2:], mode="bicubic", align_corners=False
                ).clamp(0, 1)
            }
            if model is not None:
                predictions["base"] = model.base(lr)
            for name, prediction in predictions.items():
                values = basic_metrics(
                    prediction,
                    hr,
                    lr,
                    degradation,
                    scale=config["model"].get("scale", 4),
                )
                values.update(optional_metrics(prediction, hr))
                for metric, value in values.items():
                    totals[name][metric] += value
            count += 1

    summary = {
        name: {metric: value / max(count, 1) for metric, value in metrics.items()}
        for name, metrics in totals.items()
        if metrics
    }
    summary["count"] = count
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(summary)


if __name__ == "__main__":
    main()
