"""Earth Engine India pairs: native Landsat grid, fixed spatial splits, nested sizes.

Earth Engine is imported only by the downloader. Selection/export helpers work offline.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np

from .manifest import ManifestRecord


CLASSES = {10: "forest", 40: "agriculture", 50: "urban", 60: "barren", 80: "water_wetland"}


@dataclass(frozen=True)
class IndiaPairConfig:
    start: str = "2023-01-01"
    end: str = "2026-01-01"  # Exclusive; a completed acquisition period.
    seed: int = 42
    sizes: tuple[int, ...] = (2000, 4000, 6000)
    test_count: int = 600
    val_count: int = 200
    lr_size: int = 128
    block_m: int = 40000
    guard_m: int = 500
    candidates_per_class: int = 4000
    max_pairs_per_location: int = 2
    landsat_candidates: int = 16
    sentinel_candidates: int = 2
    max_gap_days: int = 3
    max_scene_cloud: int = 60
    minimum_valid: float = 0.95
    minimum_class_fraction: float = 0.50

    def validate(self):
        if self.lr_size < 8 or not self.sizes or tuple(sorted(set(self.sizes))) != self.sizes:
            raise ValueError("Use increasing unique dataset sizes and a positive patch size")
        if min(self.sizes) <= self.test_count + self.val_count:
            raise ValueError("Each dataset must leave training pairs")
        if self.test_count < math.ceil(max(self.sizes) * 0.1) or self.val_count < 1:
            raise ValueError("Need fixed test >=10% of largest dataset and nonempty validation")
        if any(n % len(CLASSES) for n in (*self.sizes, self.test_count, self.val_count)):
            raise ValueError("Counts must be divisible by five for exact category quotas")
        if self.guard_m < 0 or self.block_m <= self.lr_size * 60 + 2 * self.guard_m:
            raise ValueError("Spatial blocks are too small for guarded image footprints")
        if not 0 < self.minimum_valid <= 1 or not 0 < self.minimum_class_fraction <= 1:
            raise ValueError("Quality fractions must be in (0,1]")


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def block_assignment(x, y, config):
    bx, by = math.floor(x / config.block_m), math.floor(y / config.block_m)
    block = f"india_6933_{bx}_{by}"
    bucket = int(identity([config.seed, block])[:12], 16) % 100
    split = "train" if bucket < 80 else "val" if bucket < 90 else "test"
    return block, split, bx, by


def aligned_grids(lon, lat, crs, transform, config):
    """Anchor the crop on the ORIGINAL Landsat 30 m pixel grid, not a scale-only export."""
    from pyproj import Transformer
    a, b, c, d, e, f = transform
    if not (abs(a - 30) < 1e-6 and abs(e + 30) < 1e-6 and b == 0 and d == 0):
        raise ValueError(f"Expected north-up native 30 m Landsat projection: {transform}")
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    col = math.floor((x - c) / a) - config.lr_size // 2
    row = math.floor((y - f) / e) - config.lr_size // 2
    x0, y0 = c + col * a, f + row * e
    lr = [30, 0, x0, 0, -30, y0]
    hr = [10, 0, x0, 0, -10, y0]
    side = config.lr_size * 30
    # Densify edges before checking the equal-area block guard.
    q = np.linspace(0, side, 33)
    xx = np.r_[x0 + q, x0 + q, np.full_like(q, x0), np.full_like(q, x0 + side)]
    yy = np.r_[np.full_like(q, y0), np.full_like(q, y0 - side), y0 - q, y0 - q]
    gx, gy = Transformer.from_crs(crs, "EPSG:6933", always_xy=True).transform(xx, yy)
    cx, cy = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True).transform(lon, lat)
    block, split, bx, by = block_assignment(cx, cy, config)
    bounds = [float(min(gx)), float(min(gy)), float(max(gx)), float(max(gy))]
    lo_x, lo_y = bx * config.block_m + config.guard_m, by * config.block_m + config.guard_m
    hi_x, hi_y = (bx + 1) * config.block_m - config.guard_m, (by + 1) * config.block_m - config.guard_m
    if not (bounds[0] >= lo_x and bounds[1] >= lo_y and bounds[2] <= hi_x and bounds[3] <= hi_y):
        raise ValueError("patch_crosses_spatial_guard")
    return dict(crs=crs, lr_transform=lr, hr_transform=hr, row=row, col=col,
                bounds=[x0, y0 - side, x0 + side, y0], bounds_6933=bounds, block=block, split=split)


def make_selection(records, config):
    """Exact nested training sets with identical val/test IDs; never duplicate to fill a quota."""
    config.validate()
    if len({r["pair_id"] for r in records}) != len(records):
        raise ValueError("Duplicate pair IDs")
    locations, blocks = {}, {}
    for record in records:
        for table, key in ((locations, record["location_id"]), (blocks, record["block"])):
            if key in table and table[key] != record["split"]:
                raise ValueError("Spatial split leakage")
            table[key] = record["split"]
    result = {}
    for size in config.sizes:
        chosen = []
        quotas = dict(train=size - config.val_count - config.test_count,
                      val=config.val_count, test=config.test_count)
        for split, count in quotas.items():
            for label in CLASSES.values():
                pool = sorted((r for r in records if r["split"] == split and r["scene_class"] == label),
                              key=lambda r: identity([config.seed, r["pair_id"]]))
                needed = count // len(CLASSES)
                if len(pool) < needed:
                    raise ValueError(f"Insufficient valid {split}/{label}: {len(pool)}/{needed}. Resume collection; no duplication or split relaxation is allowed.")
                chosen.extend(r["pair_id"] for r in pool[:needed])
        result[str(size)] = chosen
    return result


class IndiaPairDownloader:
    def __init__(self, root, config=IndiaPairConfig()):
        import ee
        config.validate()
        self.ee, self.root, self.config = ee, Path(root), config
        self.root.mkdir(parents=True, exist_ok=True)
        conf = json.loads(json.dumps(asdict(config)))
        lock = self.root / "collection_config.json"
        if lock.exists() and json.loads(lock.read_text()) != conf:
            raise ValueError("Collection configuration changed; use another output root to retain fixed selections")
        save_json(lock, conf)
        self.india = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.eq("ADM0_NAME", "India")).geometry()
        self.cover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")

    @staticmethod
    def retry(call):
        for attempt in range(5):
            try:
                return call()
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(60, 2 ** (attempt + 1)))

    def candidates(self):
        """Persist the sampled coordinates so later catalog changes cannot reshuffle sites."""
        all_points = []
        ee, cfg = self.ee, self.config
        from pyproj import Transformer
        converter = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
        for code, label in CLASSES.items():
            path = self.root / "candidate_sites" / f"{code}.json"
            if not path.exists():
                print(f"Sampling {cfg.candidates_per_class} candidate sites for {label}", flush=True)
                sample = self.cover.updateMask(self.cover.eq(code)).stratifiedSample(
                    numPoints=cfg.candidates_per_class, classBand="Map", region=self.india,
                    scale=1000, seed=cfg.seed + code, geometries=True, tileScale=4)
                features = self.retry(sample.getInfo)["features"]
                rows = []
                for feature in features:
                    lon, lat = feature["geometry"]["coordinates"][:2]
                    x, y = converter.transform(lon, lat)
                    block, split, _, _ = block_assignment(x, y, cfg)
                    rows.append(dict(location_id=identity([round(lon, 7), round(lat, 7)])[:20],
                        lon=lon, lat=lat, code=code, scene_class=label, block=block, split=split))
                save_json(path, sorted(rows, key=lambda r: identity([cfg.seed, r["location_id"]])))
            all_points.extend(json.loads(path.read_text()))
        return sorted(all_points, key=lambda r: identity([cfg.seed, r["location_id"]]))

    def scene_pairs(self, site):
        ee, cfg = self.ee, self.config
        path = self.root / "scene_lists" / f"{site['location_id']}.json"
        if path.exists():
            return json.loads(path.read_text())
        point = ee.Geometry.Point([site["lon"], site["lat"]])
        collections = []
        for sensor in ("LC08", "LC09"):
            prefix = f"LANDSAT/{sensor}/C02/T1_L2/"
            collection = ee.ImageCollection(prefix[:-1]).filterBounds(point).filterDate(cfg.start, cfg.end)
            collection = collection.filter(ee.Filter.lte("CLOUD_COVER", cfg.max_scene_cloud))
            collection = collection.map(lambda im, prefix=prefix: im.set("asset", ee.String(prefix).cat(im.get("system:index"))))
            collections.append(collection)
        landsats = collections[0].merge(collections[1]).sort("CLOUD_COVER").limit(cfg.landsat_candidates)

        def describe(value):
            ls = ee.Image(value)
            date = ls.date()
            s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(point)
            s2 = s2.filterDate(date.advance(-cfg.max_gap_days, "day"), date.advance(cfg.max_gap_days + 0.001, "day"))
            s2 = s2.filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cfg.max_scene_cloud))
            def rank(image):
                gap = image.date().difference(date, "second").abs()
                return image.set("gap_seconds", gap, "pair_rank", gap.multiply(1000).add(ee.Number(image.get("CLOUDY_PIXEL_PERCENTAGE"))))
            s2 = s2.map(rank).filter(ee.Filter.lte("gap_seconds", cfg.max_gap_days * 86400)).sort("pair_rank").limit(cfg.sentinel_candidates)
            def info(image):
                image = ee.Image(image)
                return ee.Dictionary(dict(asset=ee.String("COPERNICUS/S2_SR_HARMONIZED/").cat(image.get("system:index")),
                    date=image.date().format("YYYY-MM-dd"), gap_seconds=image.get("gap_seconds")))
            return ee.Dictionary(dict(landsat=ls.get("asset"), landsat_date=date.format("YYYY-MM-dd"),
                sentinel=s2.toList(cfg.sentinel_candidates).map(info)))
        values = self.retry(lambda: landsats.toList(cfg.landsat_candidates).map(describe).getInfo())
        for row in values:
            # Projection.transform() is WKT, not the six affine coefficients.
            projection = self.retry(lambda: ee.Image(row["landsat"]).select("SR_B4").projection().getInfo())
            row.update(crs=projection["crs"], transform=projection["transform"])
        pairs = [{**{k: v for k, v in row.items() if k != "sentinel"}, "sentinel": s2["asset"],
                  "sentinel_date": s2["date"], "gap_seconds": s2["gap_seconds"]}
                 for row in values for s2 in row["sentinel"]]
        pairs.sort(key=lambda r: identity([cfg.seed, r["landsat"], r["sentinel"]]))
        save_json(path, pairs)
        return pairs

    def image(self, asset, landsat):
        ee = self.ee
        image = ee.Image(asset)
        if landsat:
            rgb = image.select(["SR_B4", "SR_B3", "SR_B2"], ["R", "G", "B"]).multiply(0.0000275).add(-0.2)
            mask = image.select("QA_PIXEL").bitwiseAnd(63).eq(0).And(image.select("QA_RADSAT").eq(0))
        else:
            rgb = image.select(["B4", "B3", "B2"], ["R", "G", "B"]).multiply(0.0001)
            scl = image.select("SCL")
            mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6))
        mask = mask.And(rgb.mask().reduce(ee.Reducer.min())).And(rgb.gte(0).And(rgb.lte(1)).reduce(ee.Reducer.min()))
        mask = mask.focalMin(radius=30, units="meters").clip(self.india)
        rgb = rgb.updateMask(mask)
        if not landsat:
            rgb = rgb.resample("bilinear")
        # RGB uses its declared resampling; QA is a separate nearest-neighbor band.
        return rgb.addBands(mask.rename("valid")).unmask(0, sameFootprint=False).toFloat()

    def download_array(self, image, crs, transform, size):
        import requests
        def fetch():
            url = image.getDownloadURL(dict(crs=crs, crs_transform=transform,
                dimensions=[size, size], format="NPY", bands=["R", "G", "B", "valid"]))
            response = requests.get(url, timeout=(30, 180))
            response.raise_for_status()
            data = np.load(io.BytesIO(response.content), allow_pickle=False)
            if data.shape != (size, size) or not data.dtype.names or not {"R", "G", "B", "valid"}.issubset(data.dtype.names):
                raise ValueError("Unexpected Earth Engine NPY geometry/bands")
            rgb = np.stack([data[name] for name in ("R", "G", "B")]).astype(np.float32)
            valid = (data["valid"][None] >= 0.999).astype(np.float32)
            if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
                raise ValueError("Nonfinite or out-of-range reflectance")
            return rgb, valid
        return self.retry(fetch)

    def accepted(self):
        rows = [json.loads(p.read_text()) for p in sorted((self.root / "accepted").glob("*.json"))]
        for row in rows:
            if not (self.root / row["npz"]).is_file():
                raise FileNotFoundError(f"Accepted pair is missing; restore its NPZ: {row['npz']}")
        return rows

    def collect(self):
        cfg, ee = self.config, self.ee
        quotas = dict(train=(max(cfg.sizes) - cfg.test_count - cfg.val_count) // 5,
                      val=cfg.val_count // 5, test=cfg.test_count // 5)
        records = self.accepted()
        counts = Counter((r["split"], r["scene_class"]) for r in records)
        locations = Counter(r["location_id"] for r in records)
        seen = {r["pair_id"] for r in records}
        consecutive_errors = 0
        for site in self.candidates():
            key = (site["split"], site["scene_class"])
            if counts[key] >= quotas[site["split"]] or locations[site["location_id"]] >= cfg.max_pairs_per_location:
                continue
            try:
                pairs = self.scene_pairs(site)
                consecutive_errors = 0
            except Exception as exc:
                print("Metadata request failed; rerun will retry:", type(exc).__name__, flush=True)
                consecutive_errors += 1
                if consecutive_errors >= 5:
                    raise RuntimeError("Repeated Earth Engine/network failures; completed pairs are preserved. Check quota/auth and rerun.") from exc
                continue
            for pair in pairs:
                if counts[key] >= quotas[site["split"]] or locations[site["location_id"]] >= cfg.max_pairs_per_location:
                    break
                pair_id = identity([site["location_id"], pair["landsat"], pair["sentinel"]])[:24]
                if pair_id in seen or (self.root / "rejected" / f"{pair_id}.json").exists():
                    continue
                try:
                    grid = aligned_grids(site["lon"], site["lat"], pair["crs"], pair["transform"], cfg)
                except ValueError as exc:
                    save_json(self.root / "rejected" / f"{pair_id}.json", dict(pair_id=pair_id, reason=str(exc)))
                    continue
                try:
                    region = ee.Geometry.Rectangle(grid["bounds"], proj=grid["crs"], geodesic=False)
                    fraction = self.retry(lambda: self.cover.eq(site["code"]).reduceRegion(
                        reducer=ee.Reducer.mean(), geometry=region, scale=30, maxPixels=100000,
                        crs=grid["crs"]).get("Map").getInfo())
                    if fraction is None or fraction < cfg.minimum_class_fraction:
                        save_json(self.root / "rejected" / f"{pair_id}.json", dict(pair_id=pair_id, reason="weak_class_fraction", fraction=fraction))
                        continue
                    lr, ml = self.download_array(self.image(pair["landsat"], True), grid["crs"], grid["lr_transform"], cfg.lr_size)
                    hr, mh = self.download_array(self.image(pair["sentinel"], False), grid["crs"], grid["hr_transform"], cfg.lr_size * 3)
                    mh *= ml.repeat(3, 1).repeat(3, 2)
                    ml *= mh.reshape(1, cfg.lr_size, 3, cfg.lr_size, 3).min((2, 4))
                    mh *= ml.repeat(3, 1).repeat(3, 2)
                    if min(float(ml.mean()), float(mh.mean())) < cfg.minimum_valid:
                        save_json(self.root / "rejected" / f"{pair_id}.json", dict(pair_id=pair_id, reason="valid_fraction", lr=float(ml.mean()), hr=float(mh.mean())))
                        continue
                    path = self.root / "master" / f"{pair_id}.npz"
                    path.parent.mkdir(exist_ok=True)
                    temporary = path.with_suffix(".tmp.npz")
                    np.savez_compressed(temporary, lr=lr * ml, hr=hr * mh, valid_mask_lr=ml, valid_mask_hr=mh)
                    temporary.replace(path)
                    row = dict(**site, **pair, pair_id=pair_id, grid=grid, valid_fraction=float(mh.mean()),
                        class_fraction=float(fraction), npz=str(path.relative_to(self.root)))
                    save_json(self.root / "accepted" / f"{pair_id}.json", row)
                    seen.add(pair_id); records.append(row); counts[key] += 1; locations[site["location_id"]] += 1
                    consecutive_errors = 0
                    print(f"Accepted {len(records)}/{max(cfg.sizes)} | {key}: {counts[key]}/{quotas[site['split']]}", flush=True)
                except Exception as exc:
                    consecutive_errors += 1
                    print("Request/download failed, preserved for retry:", pair_id, type(exc).__name__, flush=True)
                    if consecutive_errors >= 5:
                        raise RuntimeError("Repeated failures; check disk, Earth Engine quota/auth or network, then resume.") from exc
            if all(counts[(split, label)] >= count for split, count in quotas.items() for label in CLASSES.values()):
                break
        print("Available:", dict(counts), flush=True)
        selections = make_selection(records, cfg)  # Raises with exact shortfall if incomplete.
        selection_path = self.root / "fixed_selection.json"
        if selection_path.exists() and json.loads(selection_path.read_text()) != selections:
            raise ValueError("Existing fixed selection differs. Preserve the original study; use another root for a new one.")
        save_json(selection_path, selections)
        return selections


def export_datasets(root, config=IndiaPairConfig(), sizes=None):
    """Materialize portable ZIPs without keeping three extra uncompressed data copies."""
    root = Path(root)
    records = [json.loads(p.read_text()) for p in (root / "accepted").glob("*.json")]
    lookup = {r["pair_id"]: r for r in records}
    selections = make_selection(records, config)
    lock = root / "fixed_selection.json"
    if not lock.exists() or json.loads(lock.read_text()) != selections:
        raise ValueError("Finish collection and preserve fixed_selection.json before exporting")
    result = []
    for size in sizes or config.sizes:
        chosen = [lookup[p] for p in selections[str(size)]]
        archive = root / f"india_pairs_{size}.zip"
        if archive.exists():
            with zipfile.ZipFile(archive) as handle:
                if json.loads(handle.read("pair_ids.json")) != selections[str(size)] or handle.testzip():
                    raise ValueError(f"Existing archive failed verification: {archive}")
            result.append(archive); continue
        needed = sum((root / r["npz"]).stat().st_size for r in chosen)
        if shutil.disk_usage(root).free < needed * 1.1 + 500 * 2**20:
            raise OSError(f"Insufficient space to write {archive.name}; need about {needed / 2**30:.2f} GiB plus reserve. No files were deleted.")
        manifest = []
        for r in chosen:
            path = f"{r['split']}/{r['scene_class']}/{r['pair_id']}.npz"
            grid = r["grid"]
            manifest.append(asdict(ManifestRecord(patch=path, tile_id=r["block"], split=r["split"],
                row=grid["row"], col=grid["col"], valid_fraction=r["valid_fraction"],
                source="earth_engine_landsat_sentinel_real_pairs", license_id="USGS-and-Copernicus",
                source_product=r["landsat"] + "|" + r["sentinel"], landsat_product=r["landsat"],
                sentinel_product=r["sentinel"], landsat_acquisition=r["landsat_date"], sentinel_acquisition=r["sentinel_date"],
                day_gap=int(math.ceil(r["gap_seconds"] / 86400)), scale=3,
                target_crs=grid["crs"], target_transform=grid["hr_transform"], scene_class=r["scene_class"])))
        card = dict(size=size, counts=dict(Counter(r["split"] for r in chosen)),
            split_strategy="fixed_6933_spatial_blocks", block_m=config.block_m, guard_m=config.guard_m,
            config=asdict(config), selection_sha256=identity(selections[str(size)]),
            label_note="WorldCover 2021 weak patch labels, not verified contemporary semantic ground truth",
            pixel_protocol="Actual Landsat RGB native 30 m grid; Sentinel RGB bilinearly resampled to aligned 10 m subgrid; raw reflectance [0,1]",
            sources=["LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2", "COPERNICUS/S2_SR_HARMONIZED", "ESA/WorldCover/v200", "FAO/GAUL/2015/level0"])
        temporary = archive.with_suffix(".tmp.zip")
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED, allowZip64=True) as handle:
            handle.writestr("manifest.jsonl", "".join(json.dumps(r) + "\n" for r in manifest))
            handle.writestr("dataset_card.json", json.dumps(card, indent=2))
            handle.writestr("pair_ids.json", json.dumps(selections[str(size)]))
            handle.writestr("pair_metadata.json", json.dumps(chosen, indent=2))
            handle.writestr("README.txt", "Each NPZ contains lr, hr and their validity masks. No LR synthesis.\nKeep val/test fixed across sizes; training sets are nested.\nUse TrustMoE DATASET_PROTOCOL='spatial_blocks', DISPLAY_MAX=0.3.\nAcknowledge USGS Landsat, Copernicus Sentinel, ESA WorldCover and FAO GAUL. Review their source licenses before redistribution.\n")
            for r, record in zip(chosen, manifest):
                handle.write(root / r["npz"], record["patch"])
        temporary.replace(archive)
        result.append(archive)
    return result
