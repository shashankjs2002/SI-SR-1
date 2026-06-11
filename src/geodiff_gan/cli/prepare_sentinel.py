from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..data.manifest import (
    ManifestRecord,
    load_manifest,
    validate_tile_split_isolation,
    write_manifest,
)
from ..data.sentinel import (
    discover_safe_products,
    extract_product_patches,
    product_matches_prefix,
    reassign_product_splits,
)

STATE_VERSION = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract cloud-filtered Sentinel-2 RGB patches")
    parser.add_argument("--input", required=True, help="Directory containing .SAFE products")
    parser.add_argument("--output", required=True, help="Patch output directory")
    parser.add_argument("--manifest", required=True, help="JSONL manifest path")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.95)
    parser.add_argument(
        "--state",
        help=(
            "Incremental preparation state JSON. Defaults to "
            "<manifest>.preparation.json"
        ),
    )
    parser.add_argument(
        "--completed-manifest",
        action="append",
        default=[],
        help=(
            "Historical JSONL manifest used only to infer completed source_product "
            "names; repeat for multiple audit/quarantine manifests"
        ),
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Ignore existing manifest/state and reprocess selected SAFE products",
    )
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


def _product_key(product: str | Path) -> str:
    return Path(product).name.casefold()


def _state_path(manifest: Path, configured: str | None) -> Path:
    if configured:
        return Path(configured)
    return manifest.with_suffix(manifest.suffix + ".preparation.json")


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "patch_size": int(args.patch_size),
        "stride": int(args.stride),
        "minimum_valid_fraction": float(args.minimum_valid_fraction),
    }


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    if int(state.get("version", 0)) != STATE_VERSION:
        raise SystemExit(
            f"Unsupported preparation state version in {path}. "
            "Use --rebuild or remove the state file."
        )
    return state


def _write_state(
    path: Path,
    settings: dict[str, Any],
    completed_products: dict[str, str],
    record_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "version": STATE_VERSION,
                "settings": settings,
                "completed_products": [
                    completed_products[key] for key in sorted(completed_products)
                ],
                "record_count": record_count,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _completed_products_from_jsonl(path: Path) -> dict[str, str]:
    completed: dict[str, str] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            product = str(value.get("source_product", "")).strip()
            if product:
                completed[_product_key(product)] = product
    return completed


def _merge_records(
    existing: list[ManifestRecord],
    additions: list[ManifestRecord],
) -> list[ManifestRecord]:
    merged = {record.patch: record for record in existing}
    merged.update({record.patch: record for record in additions})
    return sorted(
        merged.values(),
        key=lambda record: (
            record.source_product.casefold(),
            record.tile_id,
            record.row,
            record.col,
            record.patch,
        ),
    )


def _select_pending_products(
    products: list[Path],
    completed_product_keys: set[str],
    validation_prefixes: list[str],
    test_prefixes: list[str],
    max_products: int | None,
) -> list[Path]:
    pending = [
        product
        for product in products
        if _product_key(product) not in completed_product_keys
    ]
    forced = [
        product
        for product in pending
        if product_matches_prefix(
            product,
            [*validation_prefixes, *test_prefixes],
        )
    ]
    forced_keys = {_product_key(product) for product in forced}
    ordinary = [
        product for product in pending if _product_key(product) not in forced_keys
    ]
    if max_products is not None:
        ordinary = ordinary[:max_products]
    return sorted([*forced, *ordinary])


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = Path(args.manifest)
    state_path = _state_path(manifest_path, args.state)
    current_settings = _settings(args)
    products = discover_safe_products(args.input)
    if not products:
        raise SystemExit(f"No .SAFE products found below {args.input}")
    product_names = Counter(_product_key(product) for product in products)
    duplicate_names = sorted(
        name for name, count in product_names.items() if count > 1
    )
    if duplicate_names:
        raise SystemExit(
            "Duplicate SAFE directory names were found below the input root: "
            f"{duplicate_names}. Product names must be unique for incremental tracking."
        )
    existing_records: list[ManifestRecord] = []
    completed_products: dict[str, str] = {}
    if not args.rebuild and manifest_path.exists():
        existing_records = load_manifest(manifest_path)
        missing_source = [
            record.patch for record in existing_records if not record.source_product
        ]
        if missing_source:
            raise SystemExit(
                "The existing manifest predates source_product metadata. "
                "Use --rebuild once before incremental preparation."
            )
        completed_products.update(
            {
                _product_key(record.source_product): record.source_product
                for record in existing_records
            }
        )
    if not args.rebuild:
        for historical_manifest in args.completed_manifest:
            completed_products.update(
                _completed_products_from_jsonl(Path(historical_manifest))
            )
    state = {} if args.rebuild or not manifest_path.exists() else _load_state(state_path)
    if state:
        state_settings = state.get("settings", {})
        if state_settings != current_settings:
            raise SystemExit(
                "Preparation settings changed since the previous run. Existing "
                f"settings={state_settings}, requested={current_settings}. "
                "Use a new output directory or --rebuild."
            )
        completed_products.update(
            {
                _product_key(name): name
                for name in state.get("completed_products", [])
            }
        )
    if args.max_products is not None:
        if args.max_products < 1:
            raise SystemExit("--max-products must be at least 1")
    pending_products = _select_pending_products(
        products,
        set(completed_products),
        validation_prefixes=args.val_prefix,
        test_prefixes=args.test_prefix,
        max_products=args.max_products,
    )
    attached_completed = sum(
        _product_key(product) in completed_products for product in products
    )
    print(
        f"discovered {len(products)} SAFE product(s): "
        f"{attached_completed} attached completed, "
        f"{len(pending_products)} selected new",
        flush=True,
    )
    if args.max_products is not None:
        print(
            "--max-products limits only new unmatched products; explicit "
            "validation/test products remain selected",
            flush=True,
        )
    from tqdm.auto import tqdm

    records = existing_records
    for product in tqdm(pending_products, desc="new SAFE products"):
        print(f"extracting {product}", flush=True)
        additions = extract_product_patches(
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
        records = _merge_records(records, additions)
        records = reassign_product_splits(
            records,
            validation_prefixes=args.val_prefix,
            test_prefixes=args.test_prefix,
            unmatched_split=args.unmatched_split,
        )
        validate_tile_split_isolation(records)
        write_manifest(manifest_path, records)
        completed_products[_product_key(product)] = product.name
        _write_state(
            state_path,
            current_settings,
            completed_products,
            len(records),
        )
        print(
            f"completed {product.name}: {len(additions)} new patches, "
            f"{len(records)} manifest records total",
            flush=True,
        )
    if not pending_products:
        if records:
            records = reassign_product_splits(
                records,
                validation_prefixes=args.val_prefix,
                test_prefixes=args.test_prefix,
                unmatched_split=args.unmatched_split,
            )
            validate_tile_split_isolation(records)
            write_manifest(manifest_path, records)
        _write_state(
            state_path,
            current_settings,
            completed_products,
            len(records),
        )
        print("No new SAFE products require extraction.", flush=True)
    counts = {
        split: sum(record.split == split for record in records)
        for split in ("train", "val", "test")
    }
    print(f"wrote {len(records)} patches to {manifest_path}: {counts}")
    print(f"preparation state: {state_path}", flush=True)
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
