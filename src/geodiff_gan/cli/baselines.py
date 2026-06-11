from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import islice
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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
        degradation_seed=int(config["data"].get("degradation_seed", 0)),
        degradation_severity=config["data"].get("degradation_severity", "mild"),
    )
    if len(dataset) == 0:
        raise SystemExit(
            f"No patches found for split {args.split!r}. "
            "Check the manifest SAFE-prefix assignments."
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
    total = min(len(loader), args.limit) if args.limit is not None else len(loader)
    progress = tqdm(
        islice(loader, total),
        total=total,
        desc=f"baselines {args.split}",
        unit="patch",
    )

    with torch.no_grad():
        for batch in progress:
            lr = batch["lr"].to(device)
            clean_lr = batch["clean_lr"].to(device)
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
                    clean_lr,
                    degradation,
                    scale=config["model"].get("scale", 4),
                    severity=config["data"].get("degradation_severity", "mild"),
                )
                values.update(optional_metrics(prediction, hr))
                values["observed_lr_noise_l1"] = float((lr - clean_lr).abs().mean())
                values["observed_lr_noise_to_signal"] = float(
                    (lr - clean_lr).abs().mean()
                    / clean_lr.abs().mean().clamp_min(1e-8)
                )
                for metric, value in values.items():
                    totals[name][metric] += value
            count += 1
            progress.set_postfix(
                bicubic_psnr=f"{totals['bicubic']['psnr'] / count:.2f}",
                base_psnr=(
                    f"{totals['base']['psnr'] / count:.2f}"
                    if totals["base"]
                    else "n/a"
                ),
            )

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
