from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .manifest import (
    ManifestRecord,
    deterministic_split,
    validate_tile_split_isolation,
)

INVALID_SCL_CLASSES = {0, 1, 3, 8, 9, 10, 11}
SENTINEL_PRODUCT_PATTERN = re.compile(
    r"(S2[A-Z]_MSIL2A_[A-Z0-9_]+\.SAFE)$",
    re.IGNORECASE,
)


def canonical_product_id(product: str | Path) -> str:
    name = Path(product).name
    match = SENTINEL_PRODUCT_PATTERN.search(name)
    return match.group(1) if match else name


def is_safe_product_root(product: str | Path) -> bool:
    product = Path(product)
    return (
        product.is_dir()
        and (product / "manifest.safe").is_file()
        and (product / "GRANULE").is_dir()
    )


def source_product_name(product: str | Path) -> str:
    product = Path(product)
    name = product.name
    parent = product.parent
    while parent.suffix.casefold() == ".safe":
        name = parent.name
        parent = parent.parent
    return name


def discover_safe_products(root: str | Path) -> list[Path]:
    root = Path(root)
    candidates = list(root.rglob("*.SAFE"))
    if root.suffix.casefold() == ".safe":
        candidates.insert(0, root)
    products: dict[str, Path] = {}
    for candidate in sorted(set(candidates)):
        if not is_safe_product_root(candidate):
            continue
        identity = canonical_product_id(candidate).casefold()
        products.setdefault(identity, candidate)
    return [products[key] for key in sorted(products)]


def resolve_safe_product_root(product: str | Path) -> Path:
    product = Path(product)
    if is_safe_product_root(product):
        return product
    nested = discover_safe_products(product)
    if len(nested) == 1:
        return nested[0]
    if not nested:
        raise ValueError(
            f"{product} is not a Sentinel-2 SAFE product root: expected direct "
            "manifest.safe and GRANULE entries"
        )
    raise ValueError(
        f"{product} is a wrapper containing {len(nested)} SAFE products; "
        "select a canonical inner product"
    )


def _find_band(product: Path, pattern: str) -> Path:
    matches = sorted(product.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find {pattern} below {product}")
    return matches[0]


def tile_id_from_product(product: Path) -> str:
    match = re.search(r"_T([0-9]{2}[A-Z]{3})_", product.name)
    if match:
        return match.group(1)
    granules = list((product / "GRANULE").glob("*")) if (product / "GRANULE").exists() else []
    for granule in granules:
        match = re.search(r"_T([0-9]{2}[A-Z]{3})_", granule.name)
        if match:
            return match.group(1)
    return product.stem


def product_matches_prefix(product: str | Path, prefixes: list[str]) -> bool:
    path = Path(product)
    aliases = [path.name]
    aliases.extend(
        parent.name
        for parent in path.parents
        if parent.suffix.casefold() == ".safe"
    )
    return any(
        alias.casefold().startswith(prefix.casefold())
        for alias in aliases
        for prefix in prefixes
    )


def split_for_product(
    product: str | Path,
    tile_id: str,
    validation_prefixes: list[str] | None = None,
    test_prefixes: list[str] | None = None,
    unmatched_split: str = "hash",
) -> str:
    validation_prefixes = validation_prefixes or []
    test_prefixes = test_prefixes or []
    is_validation = product_matches_prefix(product, validation_prefixes)
    is_test = product_matches_prefix(product, test_prefixes)
    if is_validation and is_test:
        raise ValueError(
            f"{Path(product).name} matches both validation and test prefixes"
        )
    if is_validation:
        return "val"
    if is_test:
        return "test"
    if unmatched_split == "hash":
        return deterministic_split(tile_id)
    if unmatched_split not in ("train", "val", "test"):
        raise ValueError(f"Unsupported unmatched split {unmatched_split!r}")
    return unmatched_split


def reassign_product_splits(
    records: list[ManifestRecord],
    validation_prefixes: list[str] | None = None,
    test_prefixes: list[str] | None = None,
    unmatched_split: str = "hash",
) -> list[ManifestRecord]:
    missing = [record.patch for record in records if not record.source_product]
    if missing:
        raise ValueError(
            "Manifest records do not contain source_product metadata. "
            "Re-run Sentinel preparation before applying SAFE prefix splits."
        )
    for record in records:
        record.split = split_for_product(
            record.source_product,
            record.tile_id,
            validation_prefixes=validation_prefixes,
            test_prefixes=test_prefixes,
            unmatched_split=unmatched_split,
        )
    validate_tile_split_isolation(records)
    return records


def extract_product_patches(
    product: str | Path,
    output_dir: str | Path,
    patch_size: int = 512,
    stride: int = 384,
    minimum_valid_fraction: float = 0.95,
    reflectance_scale: float = 10000.0,
    saturation_value: float = 1.0,
    validation_prefixes: list[str] | None = None,
    test_prefixes: list[str] | None = None,
    unmatched_split: str = "hash",
    show_progress: bool = False,
) -> list[ManifestRecord]:
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.windows import Window, bounds, from_bounds
    except ImportError as error:
        raise RuntimeError("Install rasterio to prepare Sentinel-2 products") from error

    product = resolve_safe_product_root(product)
    output_dir = Path(output_dir)
    source_product = source_product_name(product)
    tile_id = tile_id_from_product(product)
    split = split_for_product(
        source_product,
        tile_id,
        validation_prefixes=validation_prefixes,
        test_prefixes=test_prefixes,
        unmatched_split=unmatched_split,
    )
    destination = output_dir / tile_id / Path(source_product).stem
    destination.mkdir(parents=True, exist_ok=True)
    band_paths = [
        _find_band(product, "*_B04_10m.jp2"),
        _find_band(product, "*_B03_10m.jp2"),
        _find_band(product, "*_B02_10m.jp2"),
    ]
    scl_path = _find_band(product, "*_SCL_20m.jp2")
    records: list[ManifestRecord] = []

    with (
        rasterio.open(band_paths[0]) as red,
        rasterio.open(band_paths[1]) as green,
        rasterio.open(band_paths[2]) as blue,
        rasterio.open(scl_path) as scl,
    ):
        height, width = red.height, red.width
        rows = range(0, max(height - patch_size + 1, 1), stride)
        if show_progress:
            from tqdm.auto import tqdm

            rows = tqdm(
                rows,
                desc=f"windows {product.name[:36]}",
                leave=False,
            )
        for row in rows:
            for col in range(0, max(width - patch_size + 1, 1), stride):
                if row + patch_size > height or col + patch_size > width:
                    continue
                window = Window(col, row, patch_size, patch_size)
                rgb = np.stack(
                    [dataset.read(1, window=window) for dataset in (red, green, blue)]
                ).astype(np.float32)
                scl_window = from_bounds(
                    *bounds(window, red.transform),
                    transform=scl.transform,
                )
                scl_values = scl.read(
                    1,
                    window=scl_window,
                    out_shape=(patch_size, patch_size),
                    resampling=Resampling.nearest,
                    boundless=True,
                    fill_value=0,
                )
                valid = ~np.isin(scl_values, list(INVALID_SCL_CLASSES))
                valid &= np.isfinite(rgb).all(axis=0)
                valid &= (rgb > 0).all(axis=0)
                valid &= (rgb < reflectance_scale * saturation_value).all(axis=0)
                valid_fraction = float(valid.mean())
                if valid_fraction < minimum_valid_fraction:
                    continue
                hr = np.clip(rgb / reflectance_scale, 0, 1).astype(np.float32)
                patch_path = destination / f"{tile_id}_r{row:05d}_c{col:05d}.npz"
                np.savez_compressed(
                    patch_path,
                    hr=hr,
                    valid_mask=valid.astype(np.uint8),
                    transform=np.asarray(red.window_transform(window))[:2].reshape(-1),
                    crs=str(red.crs),
                )
                records.append(
                    ManifestRecord(
                        patch=str(patch_path.resolve()),
                        tile_id=tile_id,
                        split=split,
                        row=row,
                        col=col,
                        valid_fraction=valid_fraction,
                        source_product=source_product,
                    )
                )
    return records
