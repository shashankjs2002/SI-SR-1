from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data.radiometry import fit_paired_radiometric_calibration


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit train-only Landsat-to-Sentinel RGB harmonization"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--condition-key", default="lr")
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--maximum-patches", type=int, default=2000)
    parser.add_argument("--pixels-per-patch", type=int, default=1024)
    parser.add_argument("--maximum-shift", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    calibration = fit_paired_radiometric_calibration(
        args.manifest,
        condition_key=args.condition_key,
        scale=args.scale,
        maximum_patches=args.maximum_patches,
        pixels_per_patch=args.pixels_per_patch,
        maximum_shift=args.maximum_shift,
        seed=args.seed,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    print(json.dumps(calibration, indent=2))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
