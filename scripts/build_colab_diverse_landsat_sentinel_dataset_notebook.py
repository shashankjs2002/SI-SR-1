"""Build the restart-safe Colab dataset preparation notebook."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "colab" / "Landsat_Sentinel_Diverse_3x_Dataset_Preparation.ipynb"
cells: list[dict[str, object]] = []


def md(value: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": dedent(value).strip().splitlines(True)})


def code(value: str) -> None:
    value = dedent(value).strip() + "\n"
    ast.parse(value)
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": value.splitlines(True)})


md("""
# Prepare a diverse Landsat 30 m / Sentinel-2 10 m dataset

This notebook converts raw paired tiles into one portable dataset for Kaggle. It uses
real Landsat Collection 2 Level-2 observations as LR and Sentinel-2 L2A observations
as HR. It does not synthesize LR from HR.

Expected raw layout (archives or extracted products are accepted):

```text
MyDrive/thesis/landsat_sentinel_diverse_raw/
  urban/          Sentinel SAFE zip(s) and matching Landsat product(s)
  agriculture/    Sentinel SAFE zip(s) and matching Landsat product(s)
  forest/
  water_wetland/
  barren/
  mountain/
  mixed_rural/
```

The exported layout is `split/scene_class/tile/pair/*.npz`. Every NPZ contains `lr`,
`hr`, validity masks, degradation metadata and optionally `lr_ms`. The default split
targets 78/10/12 inside every tile and rejects the export if any class has less than
10% held-out test patches. Spatial guard regions prevent overlapping patches from
appearing in different splits.
""")

md("## 1. Mount Drive and set controls")
code(r'''
from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, tarfile, zipfile

REPOSITORY_URL = "https://github.com/shashankjs2002/SI-SR-1.git"
REPOSITORY_BRANCH = "3x-continued"
REPOSITORY_DIR = Path("/content/geodiff-3x-continued")
SOURCE_ARCHIVE = None  # Optional Drive path to geodiff_diverse_moe_source.zip.

RAW_ROOT = Path("/content/drive/MyDrive/thesis/landsat_sentinel_diverse_raw")
BUILD_ROOT = Path("/content/drive/MyDrive/thesis/geodiff_diverse_3x_build")
EXTRACT_ROOT = BUILD_ROOT / "extracted_products"
PATCH_ROOT = BUILD_ROOT / "working_patches"
RAW_MANIFEST = BUILD_ROOT / "working_manifest.jsonl"
CLASSIFIED_MANIFEST = BUILD_ROOT / "classified_manifest.jsonl"
QUARANTINE = BUILD_ROOT / "quarantine.jsonl"
EXPORT_ROOT = BUILD_ROOT / "geodiff_diverse_landsat_sentinel_3x"

# Directory names are preferred. Use these overrides when a tile is mixed or its
# raw files are not below a recognized class directory.
CATEGORY_BY_TILE = {
    # "44QLL": "urban",
}
CATEGORY_BY_PRODUCT = {
    # "S2A_MSIL2A_202...": "agriculture",
}

PATCH_SIZE = 384              # Sentinel target; Landsat input is 128 x 128.
PATCH_STRIDE = 288            # Overlap is allowed only within a split.
MAX_DAY_GAP = 3
MINIMUM_OVERLAP_FRACTION = 0.10
MINIMUM_VALID_FRACTION = 0.95
TRAIN_FRACTION = 0.78
VALIDATION_FRACTION = 0.10    # Target test fraction is therefore 0.12.
MINIMUM_VALIDATION_FRACTION = 0.08
MINIMUM_TEST_FRACTION = 0.10
INCLUDE_MULTISPECTRAL = True  # Landsat B2..B7; RGB target remains Sentinel B4/B3/B2.
MAX_PAIRS = None              # Set 1 only for a smoke test.
REBUILD = False               # False resumes and includes newly added products.
RANDOM_SEED = 42

for directory in (BUILD_ROOT, EXTRACT_ROOT, PATCH_ROOT, EXPORT_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

def run(command, cwd=None):
    command = list(map(str, command))
    print("+", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_DIR / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTHONUNBUFFERED"] = "1"
    return subprocess.run(command, cwd=cwd, env=environment, check=True)

if TRAIN_FRACTION + VALIDATION_FRACTION >= 1:
    raise ValueError("Train and validation fractions must leave a positive test fraction")
print("Raw:", RAW_ROOT)
print("Persistent build:", BUILD_ROOT)
print("Export:", EXPORT_ROOT)
''')

md("## 2. Install the matching branch")
code(r'''
if SOURCE_ARCHIVE and not REPOSITORY_DIR.exists():
    archive = Path(SOURCE_ARCHIVE)
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (REPOSITORY_DIR / member.filename).resolve()
            if not target.is_relative_to(REPOSITORY_DIR.resolve()):
                raise ValueError(f"Unsafe source archive member: {member.filename}")
        handle.extractall(REPOSITORY_DIR)
elif (REPOSITORY_DIR / ".git").is_dir():
    run(["git", "fetch", "origin", REPOSITORY_BRANCH], REPOSITORY_DIR)
    run(["git", "checkout", REPOSITORY_BRANCH], REPOSITORY_DIR)
    run(["git", "pull", "--ff-only", "origin", REPOSITORY_BRANCH], REPOSITORY_DIR)
elif REPOSITORY_DIR.exists():
    required = REPOSITORY_DIR / "src/geodiff_gan/data/diverse.py"
    if not required.is_file():
        raise RuntimeError(f"{REPOSITORY_DIR} is not a compatible source tree")
else:
    run(["git", "clone", "--depth", "1", "--branch", REPOSITORY_BRANCH, REPOSITORY_URL, REPOSITORY_DIR])

run([sys.executable, "-m", "pip", "install", "-q", "numpy", "rasterio>=1.3", "pandas", "matplotlib", "tqdm", "PyYAML"])
run([sys.executable, "-m", "pip", "install", "-q", "-e", REPOSITORY_DIR, "--no-deps"])
sys.path.insert(0, str(REPOSITORY_DIR / "src"))

from geodiff_gan.data.diverse import infer_scene_class, normalize_scene_class
from geodiff_gan.data.sentinel import discover_safe_products, tile_id_from_product
from geodiff_gan.data.landsat_sentinel import discover_landsat_products, pair_scenes
print("Python:", sys.version)
''')

md("""
## 3. Extract archives incrementally

Every archive gets a persistent completion marker. New archives added later are
discovered on the next run. Existing extracted products and prepared patches are not
deleted. Archives whose path contains no recognizable class can still be labelled by
`CATEGORY_BY_TILE` or `CATEGORY_BY_PRODUCT` after discovery.
""")
code(r'''
def safe_extract(archive, destination):
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    def checked(name):
        target = (destination / name).resolve()
        if not target.is_relative_to(destination):
            raise ValueError(f"Unsafe archive path: {name}")

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                checked(member.filename)
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Archive symlinks are not accepted")
            handle.extractall(destination)
    else:
        with tarfile.open(archive) as handle:
            for member in handle.getmembers():
                checked(member.name)
                if not (member.isfile() or member.isdir()):
                    raise ValueError("Only regular files and directories are accepted")
            handle.extractall(destination, filter="data")

if not RAW_ROOT.is_dir():
    raise FileNotFoundError(RAW_ROOT)

archive_suffixes = (".zip", ".tar", ".tar.gz", ".tgz")
archives = [path for path in RAW_ROOT.rglob("*") if path.is_file() and path.name.casefold().endswith(archive_suffixes)]
print("Archives found:", len(archives))
for number, archive in enumerate(sorted(archives), 1):
    relative_parent = archive.parent.relative_to(RAW_ROOT)
    identity = hashlib.sha256((str(archive.relative_to(RAW_ROOT)) + str(archive.stat().st_size)).encode()).hexdigest()[:12]
    destination = EXTRACT_ROOT / relative_parent / identity
    marker = destination / ".complete.json"
    if marker.exists():
        continue
    print(f"[{number}/{len(archives)}] extracting {archive.name}")
    safe_extract(archive, destination)
    marker.write_text(json.dumps({"archive": str(archive), "size": archive.stat().st_size}, indent=2))
print("Archive extraction complete/resumed")
''')

md("## 4. Discover, pair and label scenes")
code(r'''
search_roots = [RAW_ROOT, EXTRACT_ROOT]
sentinels = {}
landsats = {}
for root in search_roots:
    for product in discover_safe_products(root):
        sentinels.setdefault(product.name.casefold(), product)
    for product in discover_landsat_products(root):
        landsats.setdefault(product.product_id.casefold(), product)
sentinels = list(sentinels.values())
landsats = list(landsats.values())
print("Sentinel products:", len(sentinels), "Landsat products:", len(landsats))
if not sentinels or not landsats:
    raise RuntimeError("No complete Sentinel SAFE and Landsat C2 L2 products were found")

pairs, unmatched = pair_scenes(
    sentinels, landsats,
    max_day_gap=MAX_DAY_GAP,
    minimum_overlap_fraction=MINIMUM_OVERLAP_FRACTION,
)

def category_for_product(product):
    tile = tile_id_from_product(product)
    if tile in CATEGORY_BY_TILE:
        return normalize_scene_class(CATEGORY_BY_TILE[tile])
    matches = [(prefix, value) for prefix, value in CATEGORY_BY_PRODUCT.items()
               if product.name.casefold().startswith(prefix.casefold())]
    if matches:
        return normalize_scene_class(max(matches, key=lambda item: len(item[0]))[1])
    return infer_scene_class(product)

pair_categories = {}
pair_rows = []
for pair in pairs:
    category = category_for_product(pair.sentinel)
    pair_categories[pair.sentinel.name.casefold()] = category
    pair_rows.append({
        "scene_class": category or "UNLABELED",
        "tile_id": tile_id_from_product(pair.sentinel),
        "sentinel": pair.sentinel.name,
        "landsat": pair.landsat.product_id,
        "day_gap": pair.day_gap,
        "overlap": pair.overlap_fraction,
    })

import pandas as pd
pair_table = pd.DataFrame(pair_rows)
display(pair_table)
pair_table.to_csv(BUILD_ROOT / "source_pairs.csv", index=False)
missing = pair_table[pair_table.scene_class.eq("UNLABELED")] if not pair_table.empty else pair_table
if not missing.empty:
    display(missing)
    raise ValueError("Label every pair by class directory, CATEGORY_BY_TILE, or CATEGORY_BY_PRODUCT")
if not pairs:
    raise RuntimeError("No date/overlap-compatible source pairs")
print("Pairs by class:", pair_table.scene_class.value_counts().to_dict())
weak_classes = pair_table.groupby("scene_class").size()
weak_classes = weak_classes[weak_classes < 3]
if not weak_classes.empty:
    print("WARNING: fewer than three independent source pairs in:", weak_classes.to_dict())
    print("All splits will exist spatially, but this cannot establish unseen-tile generalization for those classes.")
print("Unmatched Sentinel products:", len(unmatched))
''')

md("## 5. Prepare or resume aligned 128 to 384 patches")
code(r'''
command = [
    sys.executable, "-m", "geodiff_gan.cli.prepare_landsat_sentinel",
    "--sentinel-input", RAW_ROOT,
    "--landsat-input", RAW_ROOT,
    "--output", PATCH_ROOT,
    "--manifest", RAW_MANIFEST,
    "--state", BUILD_ROOT / "pairing_state.json",
    "--quarantine", QUARANTINE,
    "--patch-size", PATCH_SIZE,
    "--stride", PATCH_STRIDE,
    "--max-day-gap", MAX_DAY_GAP,
    "--minimum-overlap-fraction", MINIMUM_OVERLAP_FRACTION,
    "--minimum-valid-fraction", MINIMUM_VALID_FRACTION,
    "--bandpass-adjustment", "none",
    "--split-strategy", "within-tile-spatial",
    "--train-fraction", TRAIN_FRACTION,
    "--validation-fraction", VALIDATION_FRACTION,
    "--minimum-validation-fraction", MINIMUM_VALIDATION_FRACTION,
    "--minimum-test-fraction", MINIMUM_TEST_FRACTION,
]
# Extracted products may be the only discoverable inputs, so use a combined persistent
# discovery root built with directory links when necessary.
DISCOVERY_ROOT = Path("/content/geodiff_diverse_discovery")
for sensor, products in (("sentinel", sentinels), ("landsat", landsats)):
    root = DISCOVERY_ROOT / sensor
    root.mkdir(parents=True, exist_ok=True)
    for product in products:
        identity = product.name if sensor == "sentinel" else product.product_id
        destination = root / identity
        if sensor == "sentinel":
            if not destination.exists():
                destination.symlink_to(Path(product).resolve(), target_is_directory=True)
        else:
            destination.mkdir(parents=True, exist_ok=True)
            for source in product.files.values():
                link = destination / Path(source).name
                if not link.exists():
                    link.symlink_to(Path(source).resolve())
command[command.index("--sentinel-input") + 1] = DISCOVERY_ROOT / "sentinel"
command[command.index("--landsat-input") + 1] = DISCOVERY_ROOT / "landsat"
if INCLUDE_MULTISPECTRAL:
    command.append("--include-multispectral")
if MAX_PAIRS is not None:
    command += ["--max-pairs", MAX_PAIRS]
if REBUILD:
    command.append("--rebuild")
run(command, REPOSITORY_DIR)
print("Prepared manifest:", RAW_MANIFEST)
print("Rejected/unmatched log:", QUARANTINE)
''')

md("## 6. Add class metadata, validate every split and export portable paths")
code(r'''
from collections import Counter
from geodiff_gan.data.diverse import (
    annotate_scene_classes, export_portable_dataset, validate_category_splits,
    write_dataset_card,
)
from geodiff_gan.data.manifest import load_manifest, write_manifest, validate_within_tile_spatial_isolation

records = load_manifest(RAW_MANIFEST)
product_map = {
    name: category for name, category in pair_categories.items() if category is not None
}
records = annotate_scene_classes(
    records,
    category_by_tile=CATEGORY_BY_TILE,
    category_by_product={**CATEGORY_BY_PRODUCT, **product_map},
    require_all=True,
)
write_manifest(CLASSIFIED_MANIFEST, records)
validate_within_tile_spatial_isolation(records, PATCH_SIZE)
category_rows = validate_category_splits(
    records,
    minimum_test_fraction=MINIMUM_TEST_FRACTION,
    require_each_split=True,
)
summary = pd.DataFrame(category_rows)
display(summary)
summary.to_csv(BUILD_ROOT / "category_split_summary.csv", index=False)
print("All records including spatial guard exclusions:", Counter(record.split for record in records))

portable_manifest = export_portable_dataset(records, EXPORT_ROOT)
preparation = {
    "patch_size": PATCH_SIZE, "stride": PATCH_STRIDE,
    "train_fraction_target": TRAIN_FRACTION,
    "validation_fraction_target": VALIDATION_FRACTION,
    "minimum_test_fraction": MINIMUM_TEST_FRACTION,
    "maximum_day_gap": MAX_DAY_GAP,
    "minimum_valid_fraction": MINIMUM_VALID_FRACTION,
    "include_multispectral": INCLUDE_MULTISPECTRAL,
    "random_seed": RANDOM_SEED,
}
write_dataset_card(EXPORT_ROOT, records, preparation=preparation)
shutil.copy2(QUARANTINE, EXPORT_ROOT / "quarantine.jsonl") if QUARANTINE.exists() else None
(EXPORT_ROOT / "README.txt").write_text(
    "GeoDiff paired NPZ dataset. Layout: split/scene_class/tile/pair/file.npz.\n"
    "Each file stores real Landsat lr, Sentinel-2 hr, masks and metadata.\n"
    "Load manifest.jsonl; relative patch paths are resolved from its directory.\n",
    encoding="utf-8",
)
print("Portable manifest:", portable_manifest)
print("Portable files:", sum(1 for path in EXPORT_ROOT.rglob("*.npz")))
''')

md("## 7. Numerical audit and class-balanced visualization")
code(r'''
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from geodiff_gan.data.manifest import load_manifest

portable_records = load_manifest(EXPORT_ROOT / "manifest.jsonl", resolve_paths=True)
audit_rows = []
for record in portable_records:
    with np.load(record.patch) as data:
        lr, hr = data["lr"], data["hr"]
        valid = data["valid_mask_hr"].astype(bool)
        if lr.shape != (3, PATCH_SIZE // 3, PATCH_SIZE // 3) or hr.shape != (3, PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"Geometry failure: {record.patch} {lr.shape} {hr.shape}")
        if not np.isfinite(lr).all() or not np.isfinite(hr).all():
            raise ValueError(f"Non-finite values: {record.patch}")
        audit_rows.append({"scene_class": record.scene_class, "split": record.split,
                           "tile_id": record.tile_id, "valid": float(valid.mean()),
                           "lr_mean": float(lr.mean()), "hr_mean": float(hr.mean())})
audit = pd.DataFrame(audit_rows)
audit.to_csv(EXPORT_ROOT / "numerical_audit.csv", index=False)
display(audit.groupby(["scene_class", "split"]).agg(count=("valid", "size"),
    valid_mean=("valid", "mean"), lr_mean=("lr_mean", "mean"), hr_mean=("hr_mean", "mean")))

def display_rgb(array, maximum=0.3):
    return np.clip(array.transpose(1, 2, 0) / maximum, 0, 1) ** (1 / 1.4)

chosen = []
for scene_class in sorted({record.scene_class for record in portable_records}):
    candidates = [record for record in portable_records if record.scene_class == scene_class and record.split == "test"]
    if candidates:
        chosen.append(candidates[0])
fig, axes = plt.subplots(len(chosen), 3, figsize=(12, 4 * len(chosen)), squeeze=False)
for row, record in enumerate(chosen):
    with np.load(record.patch) as data:
        lr = torch.from_numpy(data["lr"]).float()
        hr = torch.from_numpy(data["hr"]).float()
        valid = torch.from_numpy(data["valid_mask_hr"][0]).bool()
    bicubic = F.interpolate(lr[None], size=hr.shape[-2:], mode="bicubic", align_corners=False)[0].clamp(0, 1)
    error = (bicubic - hr).abs().mean(0)
    for axis, image, title in zip(axes[row], (lr, hr, error),
            (f"{record.scene_class}: Landsat 30 m\n128 x 128", "Sentinel-2 10 m\n384 x 384", "Bicubic absolute error")):
        if image.ndim == 3:
            axis.imshow(display_rgb(image.numpy()), interpolation="nearest")
        else:
            shown = image.numpy(); shown[~valid.numpy()] = np.nan
            axis.imshow(shown, cmap="turbo", vmin=0, vmax=0.05)
        axis.set_title(title); axis.axis("off")
fig.tight_layout()
figure_path = EXPORT_ROOT / "category_examples.png"
fig.savefig(figure_path, dpi=170)
plt.show()
print("Saved:", figure_path)
''')

md("## 8. Create the Kaggle-upload archive and download link")
code(r'''
manifest_digest = hashlib.sha256((EXPORT_ROOT / "manifest.jsonl").read_bytes()).hexdigest()
ARCHIVE = BUILD_ROOT / f"geodiff_diverse_landsat_sentinel_3x_{manifest_digest[:12]}.zip"
if ARCHIVE.exists():
    print("Reusing archive for this exact manifest:", ARCHIVE)
else:
    temporary = ARCHIVE.with_suffix(".zip.incomplete")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as handle:
        files = [path for path in EXPORT_ROOT.rglob("*") if path.is_file()]
        for number, path in enumerate(files, 1):
            handle.write(path, (Path(EXPORT_ROOT.name) / path.relative_to(EXPORT_ROOT)).as_posix())
            if number % 250 == 0:
                print(f"archived {number}/{len(files)}", flush=True)
    temporary.replace(ARCHIVE)
print("Archive:", ARCHIVE)
print(f"Size: {ARCHIVE.stat().st_size / 2**30:.2f} GiB")
print("SHA256:", hashlib.sha256(ARCHIVE.read_bytes()).hexdigest())

from IPython.display import FileLink, display
display(FileLink(str(ARCHIVE)))
print("For large archives, download from Google Drive or upload this file directly to Kaggle; browser downloads may be unreliable above 2 GB.")
''')


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for index, cell in enumerate(cells):
        cell["id"] = f"diverse-prepare-{index:03d}"
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    build()
