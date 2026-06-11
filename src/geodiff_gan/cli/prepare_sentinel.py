from __future__ import annotations

import argparse
from pathlib import Path

from ..data.manifest import write_manifest
from ..data.sentinel import discover_safe_products, extract_product_patches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract cloud-filtered Sentinel-2 RGB patches")
    parser.add_argument("--input", required=True, help="Directory containing .SAFE products")
    parser.add_argument("--output", required=True, help="Patch output directory")
    parser.add_argument("--manifest", required=True, help="JSONL manifest path")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.95)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    products = discover_safe_products(args.input)
    if not products:
        raise SystemExit(f"No .SAFE products found below {args.input}")
    records = []
    for product in products:
        print(f"extracting {product}", flush=True)
        records.extend(
            extract_product_patches(
                product,
                args.output,
                patch_size=args.patch_size,
                stride=args.stride,
                minimum_valid_fraction=args.minimum_valid_fraction,
            )
        )
    write_manifest(Path(args.manifest), records)
    counts = {
        split: sum(record.split == split for record in records)
        for split in ("train", "val", "test")
    }
    print(f"wrote {len(records)} patches to {args.manifest}: {counts}")
    tile_ids = sorted({record.tile_id for record in records})
    missing = [split for split, count in counts.items() if count == 0]
    if len(tile_ids) < 3 or missing:
        print(
            "WARNING: tile-level geographic evaluation is incomplete. "
            f"Found {len(tile_ids)} unique tile(s); missing splits: {missing or 'none'}. "
            "Use this manifest only for development until geographically separated "
            "train, validation, and test tiles are present.",
            flush=True,
        )


if __name__ == "__main__":
    main()
