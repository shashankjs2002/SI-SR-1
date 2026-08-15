from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import re
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from .manifest import ManifestRecord
from .sentinel import (
    INVALID_SCL_CLASSES,
    SENTINEL_BAND_PATTERNS,
    _find_band,
    source_product_name,
    split_for_product,
    tile_id_from_product,
)

LANDSAT_REQUIRED_LAYERS = (
    "SR_B2",
    "SR_B3",
    "SR_B4",
    "QA_PIXEL",
    "QA_RADSAT",
)
LANDSAT_OPTIONAL_LAYERS = (
    "SR_B5",
    "SR_B6",
    "SR_B7",
    "SR_QA_AEROSOL",
    "MTL.txt",
    "MTL.xml",
    "ANG.txt",
)
LANDSAT_RGB_LAYERS = ("SR_B4", "SR_B3", "SR_B2")
LANDSAT_MULTISPECTRAL_LAYERS = (
    "SR_B4",
    "SR_B3",
    "SR_B2",
    "SR_B5",
    "SR_B6",
    "SR_B7",
)
LANDSAT_MULTISPECTRAL_BAND_NUMBERS = (4, 3, 2, 5, 6, 7)
SENTINEL_RGB_BANDS = ("B04", "B03", "B02")

# HLS v2 coefficients adjust Sentinel-2 MSI reflectance to the Landsat OLI
# spectral reference. NASA currently publishes these coefficients for S2A/B.
HLS_OLI_COEFFICIENTS = {
    "S2A": {
        "B02": (0.9778, -0.0040),
        "B03": (1.0053, -0.0009),
        "B04": (0.9765, 0.0009),
    },
    "S2B": {
        "B02": (0.9778, -0.0040),
        "B03": (1.0075, -0.0008),
        "B04": (0.9761, 0.0010),
    },
}


@dataclass(frozen=True)
class LandsatProduct:
    product_id: str
    directory: Path
    files: dict[str, Path]

    @property
    def acquisition_date(self) -> date:
        return landsat_acquisition_date(self.product_id)

    @property
    def platform(self) -> str:
        return self.product_id[:4].upper()


@dataclass(frozen=True)
class ScenePair:
    sentinel: Path
    landsat: LandsatProduct
    day_gap: int
    overlap_fraction: float

    @property
    def pair_id(self) -> str:
        return f"{source_product_name(self.sentinel)}::{self.landsat.product_id}"


def _casefold_files(directory: Path) -> dict[str, Path]:
    return {
        path.name.casefold(): path
        for path in directory.iterdir()
        if path.is_file()
    }


def _product_id_from_blue(path: Path) -> str:
    suffix = "_SR_B2.TIF"
    if not path.name.upper().endswith(suffix):
        raise ValueError(f"Not a Landsat SR_B2 file: {path}")
    return path.name[: -len(suffix)]


def discover_landsat_products(root: str | Path) -> list[LandsatProduct]:
    """Discover extracted Landsat C2 L2 products, including flat directories."""

    root = Path(root)
    blue_files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.name.upper().endswith("_SR_B2.TIF")
    ]
    products: dict[str, LandsatProduct] = {}
    for blue in sorted(blue_files):
        product_id = _product_id_from_blue(blue)
        siblings = _casefold_files(blue.parent)
        files: dict[str, Path] = {}
        missing: list[str] = []
        for layer in LANDSAT_REQUIRED_LAYERS:
            name = f"{product_id}_{layer}.TIF".casefold()
            if name not in siblings:
                missing.append(layer)
            else:
                files[layer] = siblings[name]
        if missing:
            raise ValueError(
                f"Incomplete Landsat product {product_id}: missing {missing}. "
                "Download the RGB and QA files from the same Collection 2 Level-2 product."
            )
        for layer in LANDSAT_OPTIONAL_LAYERS:
            suffix = layer if "." in layer else f"{layer}.TIF"
            name = f"{product_id}_{suffix}".casefold()
            if name in siblings:
                files[layer] = siblings[name]
        key = product_id.casefold()
        if key in products and products[key].directory != blue.parent:
            raise ValueError(
                f"Duplicate Landsat product {product_id} below {root}: "
                f"{products[key].directory} and {blue.parent}"
            )
        products[key] = LandsatProduct(product_id, blue.parent, files)
    return [products[key] for key in sorted(products)]


def landsat_acquisition_date(product_id: str) -> date:
    match = re.search(r"_L2S[A-Z]_\d{6}_(\d{8})_", product_id, re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not parse Landsat acquisition date from {product_id}")
    return datetime.strptime(match.group(1), "%Y%m%d").date()


def sentinel_acquisition_date(product: str | Path) -> date:
    match = re.search(r"MSIL2A_(\d{8})T", Path(product).name, re.IGNORECASE)
    if not match:
        match = re.search(r"MSIL2A_(\d{8})T", str(product), re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not parse Sentinel acquisition date from {product}")
    return datetime.strptime(match.group(1), "%Y%m%d").date()


def sentinel_platform(product: str | Path) -> str:
    match = re.search(r"(S2[ABC])_MSIL2A_", str(product), re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not parse Sentinel platform from {product}")
    return match.group(1).upper()


def _intersection_fraction(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    bottom = max(first[1], second[1])
    right = min(first[2], second[2])
    top = min(first[3], second[3])
    if left >= right or bottom >= top:
        return 0.0
    intersection = (right - left) * (top - bottom)
    first_area = max((first[2] - first[0]) * (first[3] - first[1]), 1e-12)
    return float(intersection / first_area)


def pair_scenes(
    sentinel_products: list[Path],
    landsat_products: list[LandsatProduct],
    max_day_gap: int = 3,
    minimum_overlap_fraction: float = 0.1,
) -> tuple[list[ScenePair], list[Path]]:
    """Choose the nearest-date, highest-overlap Landsat scene per Sentinel tile."""

    try:
        import rasterio
        from rasterio.warp import transform_bounds
    except ImportError as error:
        raise RuntimeError("Install rasterio to pair Landsat and Sentinel scenes") from error

    landsat_bounds: dict[str, tuple[object, tuple[float, float, float, float]]] = {}
    for product in landsat_products:
        with rasterio.open(product.files["SR_B4"]) as dataset:
            landsat_bounds[product.product_id] = (dataset.crs, dataset.bounds)

    pairs: list[ScenePair] = []
    unmatched: list[Path] = []
    for sentinel in sentinel_products:
        sentinel_date = sentinel_acquisition_date(sentinel)
        with rasterio.open(_find_band(sentinel, SENTINEL_BAND_PATTERNS["B04"])) as dataset:
            sentinel_crs = dataset.crs
            sentinel_bounds = tuple(dataset.bounds)
        candidates: list[tuple[int, float, LandsatProduct]] = []
        for landsat in landsat_products:
            gap = abs((sentinel_date - landsat.acquisition_date).days)
            if gap > max_day_gap:
                continue
            landsat_crs, raw_bounds = landsat_bounds[landsat.product_id]
            projected = transform_bounds(
                landsat_crs,
                sentinel_crs,
                *raw_bounds,
                densify_pts=21,
            )
            overlap = _intersection_fraction(sentinel_bounds, projected)
            if overlap >= minimum_overlap_fraction:
                candidates.append((gap, -overlap, landsat))
        if not candidates:
            unmatched.append(sentinel)
            continue
        gap, negative_overlap, landsat = min(
            candidates,
            key=lambda value: (value[0], value[1], value[2].product_id),
        )
        pairs.append(ScenePair(sentinel, landsat, gap, -negative_overlap))
    return pairs, unmatched


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def sentinel_radiometry(product: Path) -> tuple[float, dict[str, float]]:
    metadata_candidates = sorted(product.rglob("MTD_MSIL2A.xml"))
    if not metadata_candidates:
        raise FileNotFoundError(f"MTD_MSIL2A.xml not found below {product}")
    root = ElementTree.parse(metadata_candidates[0]).getroot()
    quantification = 10000.0
    indexed_offsets: dict[int, float] = {}
    fallback_offsets: list[float] = []
    for element in root.iter():
        name = _xml_local_name(element.tag)
        text = (element.text or "").strip()
        if name == "BOA_QUANTIFICATION_VALUE" and text:
            quantification = float(text)
        elif name == "BOA_ADD_OFFSET" and text:
            band_id = element.attrib.get("band_id") or element.attrib.get("bandId")
            if band_id is None:
                fallback_offsets.append(float(text))
            else:
                indexed_offsets[int(band_id)] = float(text)
    band_indices = {"B02": 1, "B03": 2, "B04": 3}
    offsets: dict[str, float] = {}
    for band, index in band_indices.items():
        if index in indexed_offsets:
            offsets[band] = indexed_offsets[index]
        elif len(fallback_offsets) > index:
            offsets[band] = fallback_offsets[index]
        else:
            offsets[band] = 0.0
    return quantification, offsets


def _metadata_number(product: LandsatProduct, key: str, default: float) -> float:
    path = product.files.get("MTL.txt")
    if path is None:
        return default
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*([^\s]+)", re.MULTILINE)
    match = pattern.search(path.read_text(encoding="utf-8", errors="ignore"))
    return float(match.group(1).strip('"')) if match else default


def landsat_reflectance(
    digital_numbers: np.ndarray,
    product: LandsatProduct,
    band_number: int,
) -> np.ndarray:
    multiplier = _metadata_number(
        product,
        f"REFLECTANCE_MULT_BAND_{band_number}",
        0.0000275,
    )
    offset = _metadata_number(
        product,
        f"REFLECTANCE_ADD_BAND_{band_number}",
        -0.2,
    )
    return digital_numbers.astype(np.float32) * multiplier + offset


def sentinel_reflectance(
    digital_numbers: np.ndarray,
    band: str,
    quantification: float,
    offsets: dict[str, float],
    platform: str,
    bandpass_adjustment: str,
) -> np.ndarray:
    reflectance = (
        digital_numbers.astype(np.float32) + float(offsets.get(band, 0.0))
    ) / quantification
    if bandpass_adjustment == "hls-oli":
        if platform not in HLS_OLI_COEFFICIENTS:
            raise ValueError(
                f"HLS OLI coefficients are not configured for {platform}. "
                "Use --bandpass-adjustment none or pair S2A/S2B products."
            )
        slope, offset = HLS_OLI_COEFFICIENTS[platform][band]
        reflectance = reflectance * slope + offset
    elif bandpass_adjustment != "none":
        raise ValueError(f"Unknown bandpass adjustment {bandpass_adjustment!r}")
    return reflectance


def landsat_valid_mask(
    qa_pixel: np.ndarray,
    qa_radsat: np.ndarray,
    aerosol: np.ndarray | None = None,
) -> np.ndarray:
    qa_pixel = qa_pixel.astype(np.uint16)
    invalid_bits = sum(1 << bit for bit in (0, 1, 2, 3, 4, 5))
    valid = (qa_pixel & invalid_bits) == 0
    valid &= qa_radsat.astype(np.uint16) == 0
    if aerosol is not None:
        aerosol = aerosol.astype(np.uint8)
        aerosol_fill = (aerosol & 1) != 0
        aerosol_level = (aerosol >> 6) & 0b11
        valid &= ~aerosol_fill
        valid &= aerosol_level < 3
    return valid


def _reproject_band(
    source: object,
    destination_shape: tuple[int, int],
    destination_transform: object,
    destination_crs: object,
    resampling: object,
    dtype: np.dtype,
) -> np.ndarray:
    from rasterio.warp import reproject

    destination = np.zeros(destination_shape, dtype=dtype)
    reproject(
        source=source,
        destination=destination,
        dst_transform=destination_transform,
        dst_crs=destination_crs,
        resampling=resampling,
        dst_nodata=0,
    )
    return destination


def _affine_values(transform: object) -> list[float]:
    return [
        float(transform.a),
        float(transform.b),
        float(transform.c),
        float(transform.d),
        float(transform.e),
        float(transform.f),
    ]


def extract_pair_patches(
    pair: ScenePair,
    output_dir: str | Path,
    patch_size: int = 384,
    stride: int = 288,
    minimum_valid_fraction: float = 0.95,
    max_day_gap: int = 3,
    bandpass_adjustment: str = "none",
    validation_prefixes: list[str] | None = None,
    test_prefixes: list[str] | None = None,
    unmatched_split: str = "hash",
    include_multispectral: bool = False,
    show_progress: bool = False,
) -> list[ManifestRecord]:
    try:
        import rasterio
        from affine import Affine
        from rasterio.enums import Resampling
        from rasterio.windows import Window, transform as window_transform
    except ImportError as error:
        raise RuntimeError("Install rasterio to extract paired patches") from error

    if patch_size % 3 or stride % 3:
        raise ValueError("patch_size and stride must be divisible by 3")
    if patch_size % 8:
        raise ValueError("patch_size must be divisible by the VAE factor 8")
    lr_size = patch_size // 3
    sentinel = pair.sentinel
    landsat = pair.landsat
    tile_id = tile_id_from_product(sentinel)
    sentinel_name = source_product_name(sentinel)
    split = split_for_product(
        sentinel_name,
        tile_id,
        validation_prefixes=validation_prefixes,
        test_prefixes=test_prefixes,
        unmatched_split=unmatched_split,
    )
    pair_hash = hashlib.sha256(pair.pair_id.encode("utf-8")).hexdigest()[:16]
    destination = Path(output_dir) / tile_id / pair_hash
    destination.mkdir(parents=True, exist_ok=True)
    quantification, sentinel_offsets = sentinel_radiometry(sentinel)
    platform = sentinel_platform(sentinel)
    sentinel_paths = [
        _find_band(sentinel, SENTINEL_BAND_PATTERNS[band])
        for band in SENTINEL_RGB_BANDS
    ]
    landsat_signal_layers = (
        LANDSAT_MULTISPECTRAL_LAYERS
        if include_multispectral
        else LANDSAT_RGB_LAYERS
    )
    missing_multispectral = [
        layer for layer in landsat_signal_layers if layer not in landsat.files
    ]
    if missing_multispectral:
        raise ValueError(
            f"Landsat product {landsat.product_id} is missing multispectral "
            f"layers {missing_multispectral}. Download SR_B2 through SR_B7."
        )
    scl_path = _find_band(sentinel, "*_SCL_20m.jp2")
    records: list[ManifestRecord] = []

    with ExitStack() as stack:
        sentinel_datasets = [
            stack.enter_context(rasterio.open(path)) for path in sentinel_paths
        ]
        scl = stack.enter_context(rasterio.open(scl_path))
        landsat_datasets = {
            layer: stack.enter_context(rasterio.open(landsat.files[layer]))
            for layer in (*landsat_signal_layers, "QA_PIXEL", "QA_RADSAT")
        }
        aerosol = (
            stack.enter_context(rasterio.open(landsat.files["SR_QA_AEROSOL"]))
            if "SR_QA_AEROSOL" in landsat.files
            else None
        )
        reference = sentinel_datasets[0]
        rows: object = range(0, max(reference.height - patch_size + 1, 1), stride)
        if show_progress:
            from tqdm.auto import tqdm

            rows = tqdm(rows, desc=f"windows {tile_id} {landsat.product_id[:18]}", leave=False)
        for row in rows:
            for col in range(0, max(reference.width - patch_size + 1, 1), stride):
                if row + patch_size > reference.height or col + patch_size > reference.width:
                    continue
                window = Window(col, row, patch_size, patch_size)
                hr_transform = window_transform(window, reference.transform)
                lr_transform = hr_transform * Affine.scale(3, 3)
                sentinel_dn = np.stack(
                    [dataset.read(1, window=window) for dataset in sentinel_datasets]
                )
                sentinel_valid = (sentinel_dn != 0).all(axis=0)
                scl_values = _reproject_band(
                    rasterio.band(scl, 1),
                    (patch_size, patch_size),
                    hr_transform,
                    reference.crs,
                    Resampling.nearest,
                    np.uint8,
                )
                sentinel_valid &= ~np.isin(scl_values, list(INVALID_SCL_CLASSES))
                hr = np.stack(
                    [
                        sentinel_reflectance(
                            values,
                            band,
                            quantification,
                            sentinel_offsets,
                            platform,
                            bandpass_adjustment,
                        )
                        for values, band in zip(sentinel_dn, SENTINEL_RGB_BANDS)
                    ]
                )

                landsat_dn = []
                for layer in landsat_signal_layers:
                    landsat_dn.append(
                        _reproject_band(
                            rasterio.band(landsat_datasets[layer], 1),
                            (lr_size, lr_size),
                            lr_transform,
                            reference.crs,
                            Resampling.bilinear,
                            np.float32,
                        )
                    )
                landsat_dn_array = np.stack(landsat_dn)
                qa_pixel = _reproject_band(
                    rasterio.band(landsat_datasets["QA_PIXEL"], 1),
                    (lr_size, lr_size),
                    lr_transform,
                    reference.crs,
                    Resampling.nearest,
                    np.uint16,
                )
                qa_radsat = _reproject_band(
                    rasterio.band(landsat_datasets["QA_RADSAT"], 1),
                    (lr_size, lr_size),
                    lr_transform,
                    reference.crs,
                    Resampling.nearest,
                    np.uint16,
                )
                aerosol_values = (
                    _reproject_band(
                        rasterio.band(aerosol, 1),
                        (lr_size, lr_size),
                        lr_transform,
                        reference.crs,
                        Resampling.nearest,
                        np.uint8,
                    )
                    if aerosol is not None
                    else None
                )
                landsat_valid = (landsat_dn_array != 0).all(axis=0)
                landsat_valid &= landsat_valid_mask(
                    qa_pixel,
                    qa_radsat,
                    aerosol_values,
                )
                expanded_landsat_valid = np.repeat(
                    np.repeat(landsat_valid, 3, axis=0), 3, axis=1
                )
                valid_hr = sentinel_valid & expanded_landsat_valid
                valid_fraction = float(valid_hr.mean())
                if valid_fraction < minimum_valid_fraction:
                    continue

                lr_all = np.stack(
                    [
                        landsat_reflectance(values, landsat, band_number)
                        for values, band_number in zip(
                            landsat_dn_array,
                            LANDSAT_MULTISPECTRAL_BAND_NUMBERS
                            if include_multispectral
                            else (4, 3, 2),
                        )
                    ]
                )
                hr = np.clip(hr, 0, 1).astype(np.float32)
                lr_all = np.clip(lr_all, 0, 1).astype(np.float32)
                lr = lr_all[:3]
                platform_condition = 1.0 if landsat.platform == "LC09" else 0.0
                sensor_condition = np.array(
                    [
                        0.5,
                        platform_condition,
                        min(pair.day_gap / max(max_day_gap, 1), 1.0),
                        1.0 - valid_fraction,
                    ],
                    dtype=np.float32,
                )
                patch_path = destination / f"r{row:05d}_c{col:05d}.npz"
                arrays = {
                    "hr": hr,
                    "lr": lr,
                    "clean_lr": lr,
                    "degradation": sensor_condition,
                    "valid_mask_hr": valid_hr[None].astype(np.float32),
                    "valid_mask_lr": landsat_valid[None].astype(np.float32),
                }
                if include_multispectral:
                    arrays["lr_ms"] = lr_all
                    arrays["clean_lr_ms"] = lr_all
                np.savez_compressed(patch_path, **arrays)
                records.append(
                    ManifestRecord(
                        patch=str(patch_path),
                        tile_id=tile_id,
                        split=split,
                        row=row,
                        col=col,
                        valid_fraction=valid_fraction,
                        source="landsat8_9_c2_l2_to_sentinel2_l2a",
                        license_id="usgs-public-domain+copernicus-free-full-open",
                        source_product=sentinel_name,
                        sentinel_product=sentinel_name,
                        landsat_product=landsat.product_id,
                        sentinel_acquisition=sentinel_acquisition_date(sentinel).isoformat(),
                        landsat_acquisition=landsat.acquisition_date.isoformat(),
                        day_gap=pair.day_gap,
                        scale=3,
                        target_crs=str(reference.crs),
                        target_transform=_affine_values(hr_transform),
                    )
                )
    return records
