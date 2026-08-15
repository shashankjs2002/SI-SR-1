from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from ..data.landsat_sentinel import (
    ScenePair,
    discover_landsat_products,
    extract_pair_patches,
    pair_scenes,
)
from ..data.manifest import (
    ManifestRecord,
    assign_within_tile_spatial_splits,
    load_manifest,
    validate_tile_split_isolation,
    write_manifest,
)
from ..data.sentinel import discover_safe_products, split_for_product

STATE_VERSION = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create aligned real-pair 30 m Landsat to 10 m Sentinel-2 patches"
        )
    )
    parser.add_argument("--sentinel-input", required=True)
    parser.add_argument("--landsat-input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--state")
    parser.add_argument("--patch-size", type=int, default=384)
    parser.add_argument("--stride", type=int, default=288)
    parser.add_argument("--max-day-gap", type=int, default=3)
    parser.add_argument("--minimum-overlap-fraction", type=float, default=0.1)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.95)
    parser.add_argument(
        "--bandpass-adjustment",
        choices=("none", "hls-oli"),
        default="none",
        help=(
            "Optionally adjust S2A/B targets to the Landsat OLI spectral reference. "
            "Published HLS coefficients are not applied to S2C."
        ),
    )
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--val-prefix", action="append", default=[])
    parser.add_argument("--test-prefix", action="append", default=[])
    parser.add_argument(
        "--split-strategy",
        choices=("tile-holdout", "within-tile-spatial"),
        default="tile-holdout",
        help=(
            "Use complete held-out MGRS tiles, or create spatially isolated "
            "train/val/test regions inside every tile."
        ),
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--unmatched-split",
        choices=("hash", "train", "val", "test"),
        default="hash",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Ignore the active manifest/state and rebuild selected pairs.",
    )
    return parser


def _state_path(manifest: Path, configured: str | None) -> Path:
    return Path(configured) if configured else manifest.with_suffix(
        manifest.suffix + ".pairing.json"
    )


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "patch_size": args.patch_size,
        "stride": args.stride,
        "max_day_gap": args.max_day_gap,
        "minimum_overlap_fraction": args.minimum_overlap_fraction,
        "minimum_valid_fraction": args.minimum_valid_fraction,
        "bandpass_adjustment": args.bandpass_adjustment,
    }


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("version") != STATE_VERSION:
        raise SystemExit(f"Unsupported paired preparation state in {path}")
    return state


def _write_state(
    path: Path,
    settings: dict[str, Any],
    completed_pairs: set[str],
    record_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "version": STATE_VERSION,
                "settings": settings,
                "completed_pairs": sorted(completed_pairs),
                "record_count": record_count,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _record_pair_id(record: ManifestRecord) -> str:
    if not record.sentinel_product or not record.landsat_product:
        return ""
    return f"{record.sentinel_product}::{record.landsat_product}"


def _merge_records(records: list[ManifestRecord]) -> list[ManifestRecord]:
    merged: dict[tuple[str, str, int, int], ManifestRecord] = {}
    for record in records:
        merged[
            (
                record.sentinel_product.casefold(),
                record.landsat_product.casefold(),
                record.row,
                record.col,
            )
        ] = record
    return sorted(
        merged.values(),
        key=lambda record: (
            record.sentinel_product.casefold(),
            record.landsat_product.casefold(),
            record.row,
            record.col,
        ),
    )


def _assign_splits(
    records: list[ManifestRecord],
    args: argparse.Namespace,
) -> dict[str, dict[str, object]]:
    if args.split_strategy == "within-tile-spatial":
        return assign_within_tile_spatial_splits(
            records,
            patch_size=args.patch_size,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
        )
    for record in records:
        record.split = split_for_product(
            record.sentinel_product or record.source_product,
            record.tile_id,
            validation_prefixes=args.val_prefix,
            test_prefixes=args.test_prefix,
            unmatched_split=args.unmatched_split,
        )
    validate_tile_split_isolation(records)
    return {}


def _print_pair(pair: ScenePair) -> None:
    print(
        f"  gap={pair.day_gap}d overlap={pair.overlap_fraction:.1%} "
        f"S2={pair.sentinel.name} L8/9={pair.landsat.product_id}",
        flush=True,
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.patch_size <= 0 or args.patch_size % 24:
        raise SystemExit("--patch-size must be positive and divisible by 24")
    if args.stride <= 0 or args.stride % 3:
        raise SystemExit("--stride must be positive and divisible by 3")
    if args.max_day_gap < 0:
        raise SystemExit("--max-day-gap must be non-negative")
    if not 0 < args.minimum_overlap_fraction <= 1:
        raise SystemExit("--minimum-overlap-fraction must be in (0, 1]")
    if not 0 < args.minimum_valid_fraction <= 1:
        raise SystemExit("--minimum-valid-fraction must be in (0, 1]")
    if not 0 < args.train_fraction < 1:
        raise SystemExit("--train-fraction must be in (0, 1)")
    if not 0 < args.validation_fraction < 1:
        raise SystemExit("--validation-fraction must be in (0, 1)")
    if args.train_fraction + args.validation_fraction >= 1:
        raise SystemExit(
            "--train-fraction + --validation-fraction must be below 1"
        )
    if args.max_pairs is not None and args.max_pairs < 1:
        raise SystemExit("--max-pairs must be at least 1")
    if args.split_strategy == "within-tile-spatial" and (
        args.val_prefix or args.test_prefix
    ):
        print(
            "WARNING: --val-prefix/--test-prefix are ignored by "
            "--split-strategy within-tile-spatial",
            flush=True,
        )

    manifest_path = Path(args.manifest)
    state_path = _state_path(manifest_path, args.state)
    settings = _settings(args)
    sentinel_products = discover_safe_products(args.sentinel_input)
    landsat_products = discover_landsat_products(args.landsat_input)
    if not sentinel_products:
        raise SystemExit(f"No canonical Sentinel-2 SAFE products below {args.sentinel_input}")
    if not landsat_products:
        raise SystemExit(
            f"No complete Landsat Collection 2 Level-2 products below {args.landsat_input}"
        )
    print(
        f"discovered Sentinel={len(sentinel_products)} Landsat={len(landsat_products)}",
        flush=True,
    )
    pairs, unmatched = pair_scenes(
        sentinel_products,
        landsat_products,
        max_day_gap=args.max_day_gap,
        minimum_overlap_fraction=args.minimum_overlap_fraction,
    )
    print(f"paired={len(pairs)} unmatched Sentinel={len(unmatched)}", flush=True)
    for pair in pairs:
        _print_pair(pair)
    for product in unmatched:
        print(f"  no pair: {product.name}", flush=True)

    records = [] if args.rebuild or not manifest_path.exists() else load_manifest(manifest_path)
    completed_pairs = {
        pair_id for record in records if (pair_id := _record_pair_id(record))
    }
    if not args.rebuild:
        state = _load_state(state_path)
        if state:
            if state.get("settings") != settings:
                raise SystemExit(
                    "Paired preparation settings changed. Use a new output/manifest "
                    "or pass --rebuild."
                )
            completed_pairs.update(state.get("completed_pairs", []))
    pending = [pair for pair in pairs if pair.pair_id not in completed_pairs]
    if args.max_pairs is not None:
        pending = pending[: args.max_pairs]
    print(f"pending pairs={len(pending)}", flush=True)

    from tqdm.auto import tqdm

    for pair in tqdm(pending, desc="Landsat-Sentinel pairs", unit="pair"):
        print(f"extracting {pair.pair_id}", flush=True)
        additions = extract_pair_patches(
            pair,
            args.output,
            patch_size=args.patch_size,
            stride=args.stride,
            minimum_valid_fraction=args.minimum_valid_fraction,
            max_day_gap=args.max_day_gap,
            bandpass_adjustment=args.bandpass_adjustment,
            validation_prefixes=args.val_prefix,
            test_prefixes=args.test_prefix,
            unmatched_split=args.unmatched_split,
            show_progress=True,
        )
        records = _merge_records([*records, *additions])
        _assign_splits(records, args)
        write_manifest(manifest_path, records)
        completed_pairs.add(pair.pair_id)
        _write_state(state_path, settings, completed_pairs, len(records))
        print(
            f"completed pair: {len(additions)} patches; total={len(records)}",
            flush=True,
        )

    spatial_reports = _assign_splits(records, args)
    if records:
        write_manifest(manifest_path, records)
    _write_state(state_path, settings, completed_pairs, len(records))
    counts = Counter(record.split for record in records)
    print(f"wrote {len(records)} records to {manifest_path}: {dict(counts)}")
    for tile_id, report in spatial_reports.items():
        print(
            f"  tile={tile_id} axis={report['axis']} "
            f"counts={report['counts']} boundaries={report['boundaries']}",
            flush=True,
        )
    print(f"state: {state_path}")
    if unmatched:
        print(
            "WARNING: unmatched Sentinel products need an overlapping Landsat scene "
            f"within {args.max_day_gap} day(s).",
            flush=True,
        )


if __name__ == "__main__":
    main()
