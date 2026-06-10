from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .manifest import ManifestRecord, deterministic_split

INVALID_SCL_CLASSES = {0, 1, 3, 8, 9, 10, 11}


def discover_safe_products(root: str | Path) -> list[Path]:
    root = Path(root)
    products = sorted(root.rglob("*.SAFE"))
    if root.suffix == ".SAFE":
        products.insert(0, root)
    return sorted(set(products))


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


def extract_product_patches(
    product: str | Path,
    output_dir: str | Path,
    patch_size: int = 512,
    stride: int = 384,
    minimum_valid_fraction: float = 0.95,
    reflectance_scale: float = 10000.0,
    saturation_value: float = 1.0,
) -> list[ManifestRecord]:
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.windows import Window
    except ImportError as error:
        raise RuntimeError("Install rasterio to prepare Sentinel-2 products") from error

    product = Path(product)
    output_dir = Path(output_dir)
    tile_id = tile_id_from_product(product)
    destination = output_dir / tile_id
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
        for row in range(0, max(height - patch_size + 1, 1), stride):
            for col in range(0, max(width - patch_size + 1, 1), stride):
                if row + patch_size > height or col + patch_size > width:
                    continue
                window = Window(col, row, patch_size, patch_size)
                rgb = np.stack(
                    [dataset.read(1, window=window) for dataset in (red, green, blue)]
                ).astype(np.float32)
                scl_values = scl.read(
                    1,
                    window=window,
                    out_shape=(patch_size, patch_size),
                    resampling=Resampling.nearest,
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
                        split=deterministic_split(tile_id),
                        row=row,
                        col=col,
                        valid_fraction=valid_fraction,
                    )
                )
    return records

