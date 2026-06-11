from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from ..data.manifest import validate_tile_split_isolation, write_manifest
from ..data.sentinel import (
    discover_safe_products,
    extract_product_patches,
    product_matches_prefix,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract cloud-filtered Sentinel-2 RGB patches")
    parser.add_argument("--input", required=True, help="Directory containing .SAFE products")
    parser.add_argument("--output", required=True, help="Patch output directory")
    parser.add_argument("--manifest", required=True, help="JSONL manifest path")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.95)
    parser.add_argument(
        "--max-products",
        type=int,
        help=(
            "Process only the first N unmatched SAFE products; explicitly assigned "
            "validation/test products are always included"
        ),
    )
    parser.add_argument(
        "--val-prefix",
        action="append",
        default=[],
        help="SAFE filename prefix assigned to validation; repeat for multiple prefixes",
    )
    parser.add_argument(
        "--test-prefix",
        action="append",
        default=[],
        help="SAFE filename prefix assigned to test; repeat for multiple prefixes",
    )
    parser.add_argument(
        "--unmatched-split",
        choices=("hash", "train", "val", "test"),
        default="hash",
        help="Split rule for SAFE products not matching an explicit prefix",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    products = discover_safe_products(args.input)
    if not products:
        raise SystemExit(f"No .SAFE products found below {args.input}")
    if args.max_products is not None:
        if args.max_products < 1:
            raise SystemExit("--max-products must be at least 1")
        forced = [
            product
            for product in products
            if product_matches_prefix(
                product,
                [*args.val_prefix, *args.test_prefix],
            )
        ]
        forced_set = set(forced)
        remaining = [
            product for product in products if product not in forced_set
        ]
        products = sorted([*forced, *remaining[: args.max_products]])
        print(
            f"limiting preparation to {len(products)} SAFE product(s)",
            flush=True,
        )
    records = []
    from tqdm.auto import tqdm

    for product in tqdm(products, desc="SAFE products"):
        print(f"extracting {product}", flush=True)
        records.extend(
            extract_product_patches(
                product,
                args.output,
                patch_size=args.patch_size,
                stride=args.stride,
                minimum_valid_fraction=args.minimum_valid_fraction,
                validation_prefixes=args.val_prefix,
                test_prefixes=args.test_prefix,
                unmatched_split=args.unmatched_split,
                show_progress=True,
            )
        )
    validate_tile_split_isolation(records)
    write_manifest(Path(args.manifest), records)
    counts = {
        split: sum(record.split == split for record in records)
        for split in ("train", "val", "test")
    }
    print(f"wrote {len(records)} patches to {args.manifest}: {counts}")
    product_counts = Counter(
        (record.split, record.source_product) for record in records
    )
    print("SAFE product assignments:", flush=True)
    for (split, product), count in sorted(product_counts.items()):
        print(f"  {split:5s} {count:5d} patches  {product}", flush=True)
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
