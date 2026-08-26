from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "kaggle" / "GeoDiff_GAN_Kaggle_OLI2MSI_3x.ipynb"


def lines(value: str) -> list[str]:
    return value.strip("\n").splitlines(keepends=True)


def markdown(value: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(value)}


def code(value: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(value),
    }


cells = [
    markdown(
        r"""
# GeoDiff-GAN and pretrained ASDDPM on OLI2MSI 3x and ALSAT 4x

This notebook contains two isolated remote-sensing SR benchmark tracks:

1. Train GeoDiff-GAN 3x on real cross-sensor **OLI2MSI**, then compare it with
   pretrained ASDDPM 3x.
2. Train a separate GeoDiff-GAN 4x model on **ALSAT-2B**, then compare it with
   pretrained ASDDPM 4x.

The OLI2MSI track uses:

```text
Landsat-8 OLI, 30 m, 3 x 128 x 128
                    -> GeoDiff-GAN SR-3x
Sentinel-2 MSI, 10 m, 3 x 384 x 384
```

The official files are `160 x 160 -> 480 x 480`. We make deterministic aligned
`128 x 128 -> 384 x 384` crops because that is the current model-native shape.
For paper-compatible RGB values, the official conversion rule is applied to
both sensors: `clip(surface_reflectance, 0, 0.3) / 0.3`.
No synthetic LR degradation is used. The official 100-pair test split remains
test-only; validation is selected from the official training pairs by a stable
source-ID hash before any crop is written.

**Protocol warning:** these are model-native crop metrics, not full-frame OLI2MSI
benchmark numbers. Report the crop protocol when comparing with a paper.
"""
    ),
    code(
        r"""
from pathlib import Path
import os, sys, subprocess, shutil, json, hashlib, random, time
from collections import Counter

REPOSITORY_URL = "https://github.com/shashankjs2002/SI-SR-1.git"
REPOSITORY_BRANCH = "SR-3x"
REPOSITORY_DIR = Path("/kaggle/working/geodiff-gan-sr3x")
OFFICIAL_REPOSITORY_URL = "https://github.com/wjwjww/OLI2MSI.git"
OFFICIAL_REPOSITORY_DIR = Path("/kaggle/working/OLI2MSI-official")

# Keep the large resumable download cache, but isolate benchmark-normalized
# patches/checkpoints from earlier physical-reflectance and smoke runs.
DATA_CACHE_ROOT = Path("/kaggle/working/geodiff-oli2msi-3x")
DOWNLOAD_ROOT = DATA_CACHE_ROOT / "official_download"
DATA_PROTOCOL_ID = "oli2msi_official_rgb_clip03_center128_v2"
WORK_ROOT = Path("/kaggle/working") / DATA_PROTOCOL_ID
PATCH_ROOT = WORK_ROOT / "patches"
MANIFEST = WORK_ROOT / "manifest.jsonl"
QUARANTINE_LOG = WORK_ROOT / "quarantine.jsonl"

# Recommended: attach OLI2MSI as a private Kaggle dataset and leave this None.
# The notebook recursively discovers train_lr/train_hr/test_lr/test_hr.
OLI2MSI_DATA_ROOT = None
DOWNLOAD_WITH_GDOWN = False
OLI2MSI_DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1kMi-_hSwO3e_Qf68VgLwaaXTNseXxg-d"
)
GDOWN_MAX_CONSECUTIVE_FAILURES = 20
# Emergency pipeline check only. This repartitions downloaded official test pairs
# and therefore cannot produce publishable OLI2MSI test metrics.
ALLOW_TEST_REPARTITION_FOR_PIPELINE_SMOKE = False

FAST_DEV_RUN = True
MAX_TRAIN_SOURCE_PAIRS = 64 if FAST_DEV_RUN else None
MAX_VALIDATION_SOURCE_PAIRS = 16 if FAST_DEV_RUN else None
MAX_TEST_SOURCE_PAIRS = 16 if FAST_DEV_RUN else None
VALIDATION_PERCENT = 10
RANDOM_SEED = 42
MINIMUM_VALID_FRACTION = 0.99
BENCHMARK_REFLECTANCE_MAX = 0.3
LR_CROP_SIZE = 128
SCALE = 3
HR_CROP_SIZE = LR_CROP_SIZE * SCALE

STAGES_TO_RUN = ["base", "vae", "diffusion", "joint"]
EPOCHS = (
    {"base": 1, "vae": 1, "diffusion": 1, "joint": 1}
    if FAST_DEV_RUN
    else {"base": 20, "vae": 20, "diffusion": 60, "joint": 15}
)
STAGE_LEARNING_RATES = {
    "base": 1e-4, "vae": 1e-4, "diffusion": 1e-4, "joint": 1e-5,
}
AUTO_RESUME_TRAINING = True
TRAINING_PROGRESS_MODE = "compact"
VALIDATION_LIMIT = 16 if FAST_DEV_RUN else 100

EVALUATION_SPLIT = "val"  # choose settings on val; use test once at the end
EVALUATION_LIMIT = 16 if FAST_DEV_RUN else 100
EVALUATION_SAMPLES = 2 if FAST_DEV_RUN else 4
EVALUATION_STEPS = 2 if FAST_DEV_RUN else 20
EVALUATION_BACK_PROJECTION_STEPS = 0
EVALUATION_RESIDUAL_SCALE = 1.0
VISUALIZATION_INDEX = 0

CONFIG_ROOT = WORK_ROOT / "configs"
RUN_ROOT = WORK_ROOT / "runs"
EVALUATION_ROOT = WORK_ROOT / "evaluation"
for path in (WORK_ROOT, DOWNLOAD_ROOT, PATCH_ROOT, CONFIG_ROOT, RUN_ROOT, EVALUATION_ROOT):
    path.mkdir(parents=True, exist_ok=True)

def run(command, cwd=None, check=True):
    command = [str(value) for value in command]
    environment = os.environ.copy()
    source = REPOSITORY_DIR / "src"
    if source.exists():
        environment["PYTHONPATH"] = os.pathsep.join(
            value for value in (str(source), environment.get("PYTHONPATH", "")) if value
        )
    environment["PYTHONUNBUFFERED"] = "1"
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=cwd, env=environment, check=check)

print("Work root:", WORK_ROOT)
print("Protocol:", (LR_CROP_SIZE, LR_CROP_SIZE), "->", (HR_CROP_SIZE, HR_CROP_SIZE))
print("Data protocol:", DATA_PROTOCOL_ID)
print("Official RGB stretch: clip(reflectance, 0,", BENCHMARK_REFLECTANCE_MAX, ") /", BENCHMARK_REFLECTANCE_MAX)
"""
    ),
    markdown("## 1. Install GeoDiff-GAN and record the official dataset code revision"),
    code(
        r"""
def clone_or_update(url, destination, branch=None):
    git_dir = destination / ".git"
    if git_dir.is_dir():
        run(["git", "fetch", "--all", "--prune"], cwd=destination)
        if branch:
            run(["git", "checkout", branch], cwd=destination)
            run(["git", "pull", "--ff-only", "origin", branch], cwd=destination)
        else:
            run(["git", "pull", "--ff-only"], cwd=destination)
        return
    if destination.exists():
        raise RuntimeError(
            f"{destination} exists but is not a Git clone. Rename it; this notebook will not delete it."
        )
    command = ["git", "clone", "--depth", "1"]
    if branch:
        command += ["--branch", branch]
    command += [url, destination]
    run(command)

clone_or_update(REPOSITORY_URL, REPOSITORY_DIR, REPOSITORY_BRANCH)
clone_or_update(OFFICIAL_REPOSITORY_URL, OFFICIAL_REPOSITORY_DIR)
run([
    sys.executable, "-m", "pip", "install", "-q",
    "numpy>=1.26", "Pillow>=10", "PyYAML>=6", "rasterio>=1.3", "tqdm>=4.66",
    "pandas>=2", "matplotlib>=3.8", "gdown>=5",
])
run([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"], cwd=REPOSITORY_DIR)

sys.path.insert(0, str(REPOSITORY_DIR / "src"))
import torch, rasterio, yaml, numpy as np, pandas as pd, matplotlib.pyplot as plt
import geodiff_gan

print("GeoDiff repository:", run(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_DIR))
print("OLI2MSI repository:", run(["git", "rev-parse", "HEAD"], cwd=OFFICIAL_REPOSITORY_DIR))
print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
"""
    ),
    markdown(
        r"""
## 2. Locate or download OLI2MSI

The GitHub repository contains code, not the 7.5 GB image archive. Prefer attaching
the official data as a Kaggle dataset. Set `DOWNLOAD_WITH_GDOWN=True` only when the
Google Drive folder is accessible and Kaggle internet is enabled. Downloads resume
into `WORK_ROOT`; existing files are not deleted.
"""
    ),
    code(
        r"""
if DOWNLOAD_WITH_GDOWN:
    import gdown
    from tqdm.auto import tqdm

    print("Reading the official Google Drive folder index...")
    download_plan = gdown.download_folder(
        url=OLI2MSI_DRIVE_FOLDER_URL,
        output=str(DOWNLOAD_ROOT),
        quiet=True,
        use_cookies=False,
        remaining_ok=True,
        skip_download=True,
        resume=True,
    )
    if download_plan is None:
        raise RuntimeError(
            "Google Drive did not return the folder index. Wait for the public quota "
            "to reset or attach OLI2MSI as a Kaggle dataset. Existing files are intact."
        )
    if FAST_DEV_RUN:
        # Google Drive lists train_hr before train_lr. Select matched source IDs
        # from all four folders first so a smoke run does not download thousands
        # of unpaired HR files before seeing its first LR file.
        planned_by_group = {
            name: {} for name in ("train_lr", "train_hr", "test_lr", "test_hr")
        }
        for item in download_plan:
            parts = Path(item.path).parts
            group = next(
                (part.casefold() for part in parts if part.casefold() in planned_by_group),
                None,
            )
            if group is not None:
                planned_by_group[group][Path(item.path).stem.casefold()] = item
        train_keys = sorted(
            planned_by_group["train_lr"].keys() & planned_by_group["train_hr"].keys()
        )
        test_keys = sorted(
            planned_by_group["test_lr"].keys() & planned_by_group["test_hr"].keys()
        )
        dev_train = [
            key for key in train_keys
            if int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 100
            >= VALIDATION_PERCENT
        ][: int(MAX_TRAIN_SOURCE_PAIRS)]
        dev_val = [
            key for key in train_keys
            if int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 100
            < VALIDATION_PERCENT
        ][: int(MAX_VALIDATION_SOURCE_PAIRS)]
        dev_test = test_keys[: int(MAX_TEST_SOURCE_PAIRS)]
        selected_items = []
        for key in dev_train + dev_val:
            selected_items += [
                planned_by_group["train_lr"][key], planned_by_group["train_hr"][key]
            ]
        for key in dev_test:
            selected_items += [
                planned_by_group["test_lr"][key], planned_by_group["test_hr"][key]
            ]
        if not dev_train or not dev_val or not dev_test:
            raise RuntimeError(
                "Could not build matched fast-dev train/val/test pairs from the Drive index."
            )
        download_plan = selected_items
        print(
            "Fast-dev Drive selection:",
            {"train": len(dev_train), "val": len(dev_val), "test": len(dev_test)},
            "source pairs",
        )
    download_state = WORK_ROOT / "gdown_download_state.jsonl"
    download_counts = Counter()
    consecutive_failures = 0
    for item in tqdm(download_plan, desc="OLI2MSI files", unit="file"):
        local_path = Path(item.local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.is_file() and local_path.stat().st_size > 0:
            download_counts["existing"] += 1
            consecutive_failures = 0
            continue
        try:
            result = gdown.download(
                id=item.id,
                output=str(local_path),
                quiet=True,
                use_cookies=False,
                resume=True,
            )
            if result is None or not local_path.is_file() or local_path.stat().st_size == 0:
                raise RuntimeError("gdown returned no completed file")
            download_counts["downloaded"] += 1
            consecutive_failures = 0
        except Exception as error:
            download_counts["failed"] += 1
            consecutive_failures += 1
            with download_state.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "id": item.id,
                    "path": item.path,
                    "local_path": str(local_path),
                    "error": f"{type(error).__name__}: {error}",
                }, ensure_ascii=True) + "\n")
            print(f"\nSkipped inaccessible file: {item.path} ({type(error).__name__})")
            if consecutive_failures >= GDOWN_MAX_CONSECUTIVE_FAILURES:
                print(
                    "Google Drive is repeatedly throttling this Kaggle IP. Stopping "
                    "cleanly. Wait and rerun this cell; completed files will be skipped."
                )
                break
    print("Google Drive state:", dict(download_counts))
    print("Failure log:", download_state)

# Some mirrors package the four directories in one ZIP. Extract only archives
# whose name identifies OLI2MSI, and use a marker so an interrupted session can
# resume without repeatedly expanding completed archives.
import zipfile
archive_roots = [Path(OLI2MSI_DATA_ROOT)] if OLI2MSI_DATA_ROOT is not None else []
archive_roots += [Path("/kaggle/input"), DOWNLOAD_ROOT]
for archive_root in archive_roots:
    if not archive_root.exists():
        continue
    for archive in archive_root.rglob("*.zip"):
        if "oli2msi" not in archive.name.casefold():
            continue
        destination = DOWNLOAD_ROOT / archive.stem
        marker = destination / ".extraction_complete"
        if marker.exists():
            continue
        destination.mkdir(parents=True, exist_ok=True)
        print("Extracting:", archive, "->", destination)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(destination)
        marker.write_text(str(archive), encoding="utf-8")

search_roots = []
if OLI2MSI_DATA_ROOT is not None:
    search_roots.append(Path(OLI2MSI_DATA_ROOT))
search_roots += [Path("/kaggle/input"), DOWNLOAD_ROOT]

def contains_tiffs(path):
    return path.is_dir() and any(
        child.is_file() and child.suffix.lower() in (".tif", ".tiff")
        for child in path.iterdir()
    )

def discover_layout(roots):
    required = ("train_lr", "train_hr", "test_lr", "test_hr")
    for root in roots:
        if not root.exists():
            continue
        candidates = [root] + [path.parent for path in root.rglob("train_lr") if path.is_dir()]
        for candidate in candidates:
            layout = {name: candidate / name for name in required}
            if all(contains_tiffs(path) for path in layout.values()):
                return candidate.resolve(), layout
    return None, None

DATA_ROOT, DATA_DIRS = discover_layout(search_roots)
SMOKE_REPARTITION_ACTIVE = False
if DATA_ROOT is None:
    partial_counts = {}
    for name in ("train_lr", "train_hr", "test_lr", "test_hr"):
        matches = [path for path in DOWNLOAD_ROOT.rglob(name) if path.is_dir()]
        partial_counts[name] = sum(
            1 for directory in matches for item in directory.iterdir()
            if item.is_file() and item.suffix.lower() in (".tif", ".tiff")
        )
    print("Partial download counts:", partial_counts)
    if FAST_DEV_RUN and ALLOW_TEST_REPARTITION_FOR_PIPELINE_SMOKE:
        test_roots = []
        for root in search_roots:
            if root.exists():
                test_roots += [path.parent for path in root.rglob("test_lr") if path.is_dir()]
        smoke_root = next(
            (
                root for root in test_roots
                if contains_tiffs(root / "test_lr") and contains_tiffs(root / "test_hr")
            ),
            None,
        )
        if smoke_root is not None:
            DATA_ROOT = smoke_root.resolve()
            DATA_DIRS = {
                "train_lr": DATA_ROOT / "test_lr",
                "train_hr": DATA_ROOT / "test_hr",
                "test_lr": DATA_ROOT / "test_lr",
                "test_hr": DATA_ROOT / "test_hr",
            }
            SMOKE_REPARTITION_ACTIVE = True
            print(
                "WARNING: PIPELINE-SMOKE REPARTITION ACTIVE. Official test pairs are "
                "being repartitioned into train/val/test. Do not report these metrics."
            )
    if DATA_ROOT is None:
        raise FileNotFoundError(
            "Could not find train_lr/train_hr/test_lr/test_hr. Attach the official OLI2MSI "
            "dataset, set OLI2MSI_DATA_ROOT, or rerun this download cell after Google "
            "Drive's public quota resets. Partial files were preserved. For a non-"
            "publishable pipeline check only, set "
            "ALLOW_TEST_REPARTITION_FOR_PIPELINE_SMOKE=True."
        )
print("OLI2MSI root:", DATA_ROOT)
for name, path in DATA_DIRS.items():
    count = sum(1 for item in path.iterdir() if item.suffix.lower() in (".tif", ".tiff"))
    print(f"{name}: {count:,} TIFF files at {path}")
"""
    ),
    markdown("## 3. Pair audit and one dataset-level reflectance rule"),
    code(
        r"""
def index_tiffs(directory):
    result = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in (".tif", ".tiff"):
            key = path.stem.casefold()
            if key in result:
                raise RuntimeError(f"Duplicate TIFF stem {path.stem!r} in {directory}")
            result[key] = path
    return result

def pair_directories(lr_dir, hr_dir):
    lr_files, hr_files = index_tiffs(lr_dir), index_tiffs(hr_dir)
    common = sorted(lr_files.keys() & hr_files.keys())
    missing_hr = sorted(lr_files.keys() - hr_files.keys())
    missing_lr = sorted(hr_files.keys() - lr_files.keys())
    return [(key, lr_files[key], hr_files[key]) for key in common], missing_hr, missing_lr

official_train_pairs, train_missing_hr, train_missing_lr = pair_directories(
    DATA_DIRS["train_lr"], DATA_DIRS["train_hr"]
)
official_test_pairs, test_missing_hr, test_missing_lr = pair_directories(
    DATA_DIRS["test_lr"], DATA_DIRS["test_hr"]
)
print("Paired official train:", len(official_train_pairs), "official test:", len(official_test_pairs))
print("Unpaired:", {
    "train_missing_hr": len(train_missing_hr), "train_missing_lr": len(train_missing_lr),
    "test_missing_hr": len(test_missing_hr), "test_missing_lr": len(test_missing_lr),
})
if train_missing_hr or train_missing_lr or test_missing_hr or test_missing_lr:
    with QUARANTINE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "stage": "pair_audit",
            "train_missing_hr": train_missing_hr,
            "train_missing_lr": train_missing_lr,
            "test_missing_hr": test_missing_hr,
            "test_missing_lr": test_missing_lr,
        }, ensure_ascii=True) + "\n")
    print(
        "WARNING: unpaired files were excluded and recorded in the quarantine log. "
        "A publication run should use all 5,225 training and 100 test pairs."
    )

expected = {"official_train_pairs": 5225, "official_test_pairs": 100}
observed = {
    "official_train_pairs": len(official_train_pairs),
    "official_test_pairs": len(official_test_pairs),
}
if not FAST_DEV_RUN and observed != expected:
    raise RuntimeError(
        f"Full-run OLI2MSI is incomplete: observed={observed}, expected={expected}. "
        "Rerun the resumable download or attach a complete Kaggle dataset."
    )

def source_split(key):
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "val" if bucket < VALIDATION_PERCENT else "train"

if SMOKE_REPARTITION_ACTIVE:
    smoke_pairs = official_test_pairs
    train_pairs, val_pairs, test_pairs = [], [], []
    for pair in smoke_pairs:
        bucket = int(hashlib.sha256(pair[0].encode("utf-8")).hexdigest()[:8], 16) % 100
        if bucket < 70:
            train_pairs.append(pair)
        elif bucket < 85:
            val_pairs.append(pair)
        else:
            test_pairs.append(pair)
    if not train_pairs or not val_pairs or not test_pairs:
        raise RuntimeError(
            "The smoke-only test repartition could not populate all three splits."
        )
else:
    train_pairs = [pair for pair in official_train_pairs if source_split(pair[0]) == "train"]
    val_pairs = [pair for pair in official_train_pairs if source_split(pair[0]) == "val"]
    test_pairs = official_test_pairs

def truncate(values, maximum):
    return values if maximum is None else values[: int(maximum)]

train_pairs = truncate(train_pairs, MAX_TRAIN_SOURCE_PAIRS)
val_pairs = truncate(val_pairs, MAX_VALIDATION_SOURCE_PAIRS)
test_pairs = truncate(test_pairs, MAX_TEST_SOURCE_PAIRS)
print("Selected source pairs:", {"train": len(train_pairs), "val": len(val_pairs), "test": len(test_pairs)})

split_keys = {
    "train": {pair[0] for pair in train_pairs},
    "val": {pair[0] for pair in val_pairs},
    "test": {pair[0] for pair in test_pairs},
}
overlaps = {
    "train_val": sorted(split_keys["train"] & split_keys["val"]),
    "train_test": sorted(split_keys["train"] & split_keys["test"]),
    "val_test": sorted(split_keys["val"] & split_keys["test"]),
}
if any(overlaps.values()):
    raise RuntimeError(f"OLI2MSI source leakage detected before conversion: {overlaps}")
print("Source split isolation: verified")

def audit_raster(path):
    with rasterio.open(path) as source:
        sample = source.read(out_dtype="float32")
        return {
            "path": str(path), "bands": source.count, "height": source.height,
            "width": source.width, "dtype": source.dtypes[0],
            "minimum": float(np.nanmin(sample)), "p99": float(np.nanpercentile(sample, 99)),
            "maximum": float(np.nanmax(sample)),
        }

audit = []
for _, lr_path, hr_path in (train_pairs + val_pairs + test_pairs)[:8]:
    audit += [audit_raster(lr_path), audit_raster(hr_path)]
audit_table = pd.DataFrame(audit)
display(audit_table)
global_p99 = float(audit_table["p99"].max())
if global_p99 <= 1.5:
    REFLECTANCE_DIVISOR = 1.0
elif global_p99 <= 12000:
    REFLECTANCE_DIVISOR = 10000.0
else:
    raise RuntimeError(
        f"Unsupported OLI2MSI value range (sample p99={global_p99}). "
        "Do not use per-image min/max normalization; inspect the TIFF metadata."
    )
print("One dataset-level reflectance divisor:", REFLECTANCE_DIVISOR)
"""
    ),
    markdown(
        r"""
## 4. Restart-safe conversion to GeoDiff-GAN patches

Each source pair produces one aligned crop. Training offsets are deterministic
pseudo-random values in the valid `0..32` LR range; validation and test use the
center crop. Reflectance is converted using the official fixed `0..0.3` stretch.
Corrupt or insufficiently valid files are logged, not deleted.
"""
    ),
    code(
        r"""
from geodiff_gan.data.manifest import ManifestRecord, load_manifest, write_manifest
from tqdm.auto import tqdm

selected_record_keys = {
    f"{split}:{key}"
    for split, pairs in (("train", train_pairs), ("val", val_pairs), ("test", test_pairs))
    for key, _, _ in pairs
}
loaded_records = load_manifest(MANIFEST) if MANIFEST.exists() else []
stale_records = [
    record for record in loaded_records
    if record.source_product not in selected_record_keys
    or record.source == "oli2msi_test_repartition_pipeline_smoke"
]
if stale_records:
    print(
        "Excluded stale or smoke records from the active manifest:",
        len(stale_records),
    )
existing_records = [
    record for record in loaded_records
    if record.source_product in selected_record_keys
    and record.source != "oli2msi_test_repartition_pipeline_smoke"
]
existing_keys = {record.source_product for record in existing_records}
records_by_key = {record.source_product: record for record in existing_records}

def deterministic_offset(key):
    maximum = 160 - LR_CROP_SIZE
    digest = hashlib.sha256((str(RANDOM_SEED) + ":" + key).encode("utf-8")).digest()
    return digest[0] % (maximum + 1), digest[1] % (maximum + 1)

def read_rgb(path, expected_size):
    with rasterio.open(path) as source:
        if source.count < 3:
            raise ValueError(f"Expected at least 3 bands, found {source.count}")
        if (source.height, source.width) != (expected_size, expected_size):
            raise ValueError(
                f"Expected {expected_size}x{expected_size}, found {source.height}x{source.width}"
            )
        image = source.read([1, 2, 3], out_dtype="float32") / REFLECTANCE_DIVISOR
        masks = source.read_masks([1, 2, 3]) > 0
    valid = masks.all(0) & np.isfinite(image).all(0)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    # Exact fixed stretch from the official OLI2MSI convert2png.py. This is not
    # per-image normalization and therefore preserves cross-image radiometry.
    image = np.clip(image, 0.0, BENCHMARK_REFLECTANCE_MAX)
    image = image / BENCHMARK_REFLECTANCE_MAX
    return image, valid

def append_quarantine(payload):
    with QUARANTINE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

def process_pair(split, pair):
    key, lr_path, hr_path = pair
    record_key = f"{split}:{key}"
    if record_key in existing_keys and Path(records_by_key[record_key].patch).exists():
        return records_by_key[record_key], "skipped"
    try:
        lr, valid_lr = read_rgb(lr_path, 160)
        hr, valid_hr = read_rgb(hr_path, 480)
        if split == "train":
            lr_row, lr_col = deterministic_offset(key)
        else:
            lr_row = lr_col = (160 - LR_CROP_SIZE) // 2
        hr_row, hr_col = lr_row * SCALE, lr_col * SCALE
        lr = lr[:, lr_row:lr_row + LR_CROP_SIZE, lr_col:lr_col + LR_CROP_SIZE]
        valid_lr = valid_lr[lr_row:lr_row + LR_CROP_SIZE, lr_col:lr_col + LR_CROP_SIZE]
        hr = hr[:, hr_row:hr_row + HR_CROP_SIZE, hr_col:hr_col + HR_CROP_SIZE]
        valid_hr = valid_hr[hr_row:hr_row + HR_CROP_SIZE, hr_col:hr_col + HR_CROP_SIZE]
        valid_fraction = min(float(valid_lr.mean()), float(valid_hr.mean()))
        if valid_fraction < MINIMUM_VALID_FRACTION:
            raise ValueError(f"valid_fraction={valid_fraction:.6f}")
        split_dir = PATCH_ROOT / split
        split_dir.mkdir(parents=True, exist_ok=True)
        patch_path = split_dir / f"{key}.npz"
        temporary = patch_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                lr=lr.astype(np.float16),
                clean_lr=lr.astype(np.float16),
                hr=hr.astype(np.float16),
                valid_mask_lr=valid_lr[None].astype(np.uint8),
                valid_mask_hr=valid_hr[None].astype(np.uint8),
                degradation=np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32),
            )
        temporary.replace(patch_path)
        record = ManifestRecord(
            patch=str(patch_path.resolve()), tile_id=key, split=split,
            row=hr_row, col=hr_col, valid_fraction=valid_fraction,
            source=(
                "oli2msi_test_repartition_pipeline_smoke"
                if SMOKE_REPARTITION_ACTIVE
                else DATA_PROTOCOL_ID
            ),
            license_id="see-official-OLI2MSI-distribution",
            source_product=record_key, sentinel_product=hr_path.name,
            landsat_product=lr_path.name, day_gap=0, scale=SCALE,
        )
        return record, "written"
    except Exception as error:
        append_quarantine({
            "split": split, "key": key, "lr": str(lr_path), "hr": str(hr_path),
            "error": f"{type(error).__name__}: {error}",
        })
        return None, "quarantined"

status = Counter()
all_records = list(existing_records)
for split, pairs in (("train", train_pairs), ("val", val_pairs), ("test", test_pairs)):
    for pair in tqdm(pairs, desc=f"prepare {split}", unit="pair"):
        record, state = process_pair(split, pair)
        status[state] += 1
        if record is not None:
            records_by_key[record.source_product] = record
            all_records = list(records_by_key.values())
        if (status["written"] + status["quarantined"]) % 50 == 0:
            write_manifest(MANIFEST, sorted(all_records, key=lambda item: (item.split, item.tile_id)))

write_manifest(MANIFEST, sorted(records_by_key.values(), key=lambda item: (item.split, item.tile_id)))
records = load_manifest(MANIFEST)
split_counts = Counter(record.split for record in records)
print("Preparation:", dict(status))
print("Manifest:", MANIFEST, dict(split_counts))
print("Quarantine log:", QUARANTINE_LOG, "exists=", QUARANTINE_LOG.exists())
if any(split_counts[split] == 0 for split in ("train", "val", "test")):
    raise RuntimeError(f"Incomplete split counts: {split_counts}")
if any(record.source != DATA_PROTOCOL_ID for record in records):
    raise RuntimeError("Active manifest contains records from another data protocol")
manifest_keys = {record.source_product for record in records}
if manifest_keys != selected_record_keys:
    raise RuntimeError(
        "Active manifest does not exactly match the selected source pairs: "
        f"missing={len(selected_record_keys - manifest_keys)} "
        f"unexpected={len(manifest_keys - selected_record_keys)}"
    )
"""
    ),
    markdown("## 5. Verify geometry, pair alignment and reflectance distributions"),
    code(
        r"""
import torch.nn.functional as F

def chw(value):
    value = torch.from_numpy(np.asarray(value)).float()
    if value.ndim == 3 and value.shape[0] not in (1, 3, 6):
        value = value.permute(2, 0, 1)
    return value

def display_rgb(images):
    # Patches already contain the official clip(0, 0.3) / 0.3 conversion.
    # Applying another 0.3 stretch here would double-normalize and saturate them.
    return [image[:3].clamp(0, 1).permute(1, 2, 0) for image in images]

def visualize_pair(split="train", index=0):
    selected = [record for record in records if record.split == split]
    record = selected[int(index) % len(selected)]
    with np.load(record.patch) as data:
        lr, hr = chw(data["lr"]), chw(data["hr"])
        valid_lr = torch.from_numpy(data["valid_mask_lr"][0]).bool()
        valid_hr = torch.from_numpy(data["valid_mask_hr"][0]).bool()
    bicubic = F.interpolate(lr[None], size=hr.shape[-2:], mode="bicubic", align_corners=False)[0]
    hr_at_lr = F.interpolate(hr[None], size=lr.shape[-2:], mode="area")[0]
    shared_valid_lr = valid_lr & (
        F.interpolate(valid_hr[None, None].float(), size=lr.shape[-2:], mode="area")[0, 0] > 0.999
    )
    panels = [(lr, "Landsat OLI benchmark RGB\n30 m | 128 x 128"),
              (bicubic, "Bicubic display\n384 x 384; no new information"),
              (hr, "Sentinel-2 MSI benchmark RGB target\n10 m | 384 x 384")]
    displays = display_rgb([image for image, _ in panels])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, display_image, (_, title) in zip(axes, displays, panels):
        axis.imshow(display_image, interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"{split} source={record.tile_id} | valid={record.valid_fraction:.4f}")
    plt.tight_layout()
    plt.show()
    print({
        "lr_shape": tuple(lr.shape), "hr_shape": tuple(hr.shape),
        "lr_range": (float(lr.min()), float(lr.max())),
        "hr_range": (float(hr.min()), float(hr.max())),
        "valid_lr": float(valid_lr.float().mean()), "valid_hr": float(valid_hr.float().mean()),
        "cross_sensor_lr_scale_l1": float((lr - hr_at_lr).abs().mean(0)[shared_valid_lr].mean()),
        "channel_correlations": [
            float(torch.corrcoef(torch.stack((lr[channel][shared_valid_lr], hr_at_lr[channel][shared_valid_lr])))[0, 1])
            for channel in range(3)
        ],
    })

visualize_pair("train", 0)
"""
    ),
    markdown(
        r"""
## 6. Audit why raw-reflectance PSNR looks unusually high

OLI2MSI's official `convert2png.py` uses
`clip(reflectance, 0, 0.3) / 0.3`. Computing PSNR directly on un-stretched
reflectance while declaring `data_range=1` adds about `10.46 dB` because most
valid values occupy only the lower 30% of that declared range. This cell reports
the same bicubic pairs under physical-reflectance and paper-compatible rules,
for both the complete `160 -> 480` pair and this notebook's center crop.
"""
    ),
    code(
        r"""
def raw_reflectance_rgb(path, expected_size):
    with rasterio.open(path) as source:
        image = source.read([1, 2, 3], out_dtype="float32") / REFLECTANCE_DIVISOR
        mask = (source.read_masks([1, 2, 3]) > 0).all(0)
    if image.shape[-2:] != (expected_size, expected_size):
        raise ValueError(f"Unexpected shape for {path}: {image.shape}")
    valid = mask & np.isfinite(image).all(0)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(image).float(), torch.from_numpy(valid[None]).float()

def masked_psnr_value(prediction, target, mask):
    expanded = mask.expand_as(target)
    mse = ((prediction - target).square() * expanded).sum() / expanded.sum().clamp_min(1)
    return float(-10.0 * torch.log10(mse.clamp_min(1e-12)))

def bicubic_psnr_protocol(lr, hr, lr_mask, hr_mask, benchmark_stretch, quantize):
    if benchmark_stretch:
        lr = lr.clamp(0, BENCHMARK_REFLECTANCE_MAX) / BENCHMARK_REFLECTANCE_MAX
        hr = hr.clamp(0, BENCHMARK_REFLECTANCE_MAX) / BENCHMARK_REFLECTANCE_MAX
    else:
        lr, hr = lr.clamp(0, 1), hr.clamp(0, 1)
    prediction = F.interpolate(
        lr[None], size=hr.shape[-2:], mode="bicubic", align_corners=False
    )[0].clamp(0, 1)
    valid_lr_hr = F.interpolate(
        lr_mask[None], size=hr.shape[-2:], mode="nearest"
    )[0]
    mask = hr_mask * valid_lr_hr
    if quantize:
        prediction = prediction.mul(255).round().div(255)
        hr = hr.mul(255).round().div(255)
    return masked_psnr_value(prediction, hr, mask)

audit_rows = []
for key, lr_path, hr_path in tqdm(test_pairs, desc="bicubic protocol audit", unit="pair"):
    lr_full, lr_mask_full = raw_reflectance_rgb(lr_path, 160)
    hr_full, hr_mask_full = raw_reflectance_rgb(hr_path, 480)
    offset = (160 - LR_CROP_SIZE) // 2
    hr_offset = offset * SCALE
    lr_center = lr_full[:, offset:offset + LR_CROP_SIZE, offset:offset + LR_CROP_SIZE]
    lr_mask_center = lr_mask_full[:, offset:offset + LR_CROP_SIZE, offset:offset + LR_CROP_SIZE]
    hr_center = hr_full[:, hr_offset:hr_offset + HR_CROP_SIZE, hr_offset:hr_offset + HR_CROP_SIZE]
    hr_mask_center = hr_mask_full[:, hr_offset:hr_offset + HR_CROP_SIZE, hr_offset:hr_offset + HR_CROP_SIZE]
    audit_rows.append({
        "source": key,
        "center_raw_reflectance_range1": bicubic_psnr_protocol(
            lr_center, hr_center, lr_mask_center, hr_mask_center, False, False
        ),
        "center_official_clip03_float": bicubic_psnr_protocol(
            lr_center, hr_center, lr_mask_center, hr_mask_center, True, False
        ),
        "center_official_clip03_uint8": bicubic_psnr_protocol(
            lr_center, hr_center, lr_mask_center, hr_mask_center, True, True
        ),
        "full_official_clip03_float": bicubic_psnr_protocol(
            lr_full, hr_full, lr_mask_full, hr_mask_full, True, False
        ),
        "full_official_clip03_uint8": bicubic_psnr_protocol(
            lr_full, hr_full, lr_mask_full, hr_mask_full, True, True
        ),
        "lr_mean_reflectance": float(lr_full.mean()),
        "hr_mean_reflectance": float(hr_full.mean()),
        "hr_p99_reflectance": float(torch.quantile(hr_full.flatten(), 0.99)),
    })

protocol_audit = pd.DataFrame(audit_rows)
display(protocol_audit.describe().T[["mean", "std", "min", "max"]].round(6))
theoretical_shift = 20.0 * np.log10(1.0 / BENCHMARK_REFLECTANCE_MAX)
observed_shift = (
    protocol_audit["center_raw_reflectance_range1"]
    - protocol_audit["center_official_clip03_float"]
).mean()
print(f"Theoretical range-only inflation: {theoretical_shift:.4f} dB")
print(f"Observed center-crop inflation: {observed_shift:.4f} dB")
print(
    "Use center_official_clip03_* for this notebook's internal comparison; "
    "use full_official_clip03_uint8 when reproducing the published full-image protocol."
)
"""
    ),
    markdown("## 7. Build the unchanged SR-3x training configuration"),
    code(
        r"""
import copy

template_path = REPOSITORY_DIR / "configs" / "landsat_sentinel_3x_small.yaml"
runtime_config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
runtime_config["model"].update({
    "scale": SCALE,
    "input_channels": 3,
    "output_channels": 3,
    "base_upsample_mode": "resize_conv",
    "decoder_upsample_mode": "resize_conv",
    "use_text_conditioning": False,
    "use_degradation_conditioning": False,
})
runtime_config["data"].update({
    "manifest": str(MANIFEST),
    "captions": None,
    "input_mode": "paired",
    "target_key": "hr",
    "condition_key": None,
    "train_degradation_sampling": "fixed",
})
runtime_config["training"].update({
    "batch_size": 1 if FAST_DEV_RUN else 2,
    "gradient_accumulation": 8,
    "num_workers": 2,
    "amp": True,
    "progress_mode": TRAINING_PROGRESS_MODE,
    "validation_limit": VALIDATION_LIMIT,
    "keep_best_and_latest": True,
    "auto_resume": AUTO_RESUME_TRAINING,
    "checkpoint_metric": "val_psnr",
    "checkpoint_mode": "max",
    "early_stopping_metric": "val_psnr",
    "early_stopping_mode": "max",
    "early_stopping_patience": 5,
    "early_stopping_min_epochs": 5 if FAST_DEV_RUN else 8,
    "validation_sample_steps": 2 if FAST_DEV_RUN else 10,
    "validation_samples": 1,
    "validation_back_projection_steps": 0,
    "train_back_projection_steps": 0,
})

# OLI2MSI already contains real LR observations. Do not synthesize noise or use
# cross-sensor back-projection as a hard constraint.
runtime_config["training"]["loss_weights"].update({
    "mse": 50.0,
    "multiscale_mse": 25.0,
    "charbonnier": 1.0,
    "ssim": 0.5,
    "consistency": 0.0,
    "perceptual": 0.0,
    "adversarial": 0.0,
    "prompt_alignment": 0.0,
})
display(pd.DataFrame([{
    "scale": runtime_config["model"]["scale"],
    "input_channels": runtime_config["model"]["input_channels"],
    "base_upsampler": runtime_config["model"]["base_upsample_mode"],
    "decoder_upsampler": runtime_config["model"]["decoder_upsample_mode"],
    "input_mode": runtime_config["data"]["input_mode"],
    "text": runtime_config["model"]["use_text_conditioning"],
}]))
"""
    ),
    markdown("## 8. Train or resume base, VAE, diffusion and joint stages"),
    code(
        r"""
def select_checkpoint(output_dir, stage):
    for path in (output_dir / f"{stage}_best.pt", output_dir / f"{stage}_latest.pt"):
        if path.exists():
            return path
    epoch_files = sorted(output_dir.glob(f"{stage}_epoch_*.pt"))
    return epoch_files[-1] if epoch_files else None

def train_stage(stage, initial_checkpoint=None):
    config = copy.deepcopy(runtime_config)
    output_dir = RUN_ROOT / stage
    output_dir.mkdir(parents=True, exist_ok=True)
    config["training"].update({
        "stage": stage,
        "epochs": EPOCHS[stage],
        "learning_rate": STAGE_LEARNING_RATES[stage],
        "output_dir": str(output_dir),
        "init_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
        "resume": None,
        "auto_resume": AUTO_RESUME_TRAINING,
    })
    if stage != "joint":
        config["training"]["module_learning_rate_multipliers"] = {}
    if stage == "diffusion":
        config["training"]["loss_weights"]["diffusion"] = 1.0
        config["training"].update({
            "checkpoint_metric": "val_loss_total", "checkpoint_mode": "min",
            "early_stopping_metric": "val_loss_total", "early_stopping_mode": "min",
            "lr_scheduler_metric": "val_loss_total", "lr_scheduler_mode": "min",
        })
    config_path = CONFIG_ROOT / f"{stage}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"\n===== {stage}: epochs={EPOCHS[stage]} lr={STAGE_LEARNING_RATES[stage]} =====")
    run([sys.executable, "-m", "geodiff_gan.cli.train", "--config", config_path], cwd=REPOSITORY_DIR)
    checkpoint = select_checkpoint(output_dir, stage)
    if checkpoint is None:
        raise RuntimeError(f"No checkpoint produced for {stage}")
    print("Selected:", checkpoint)
    return config_path, checkpoint

CHECKPOINTS, CONFIGS = {}, {}
previous_checkpoint = None
for stage in STAGES_TO_RUN:
    config_path, previous_checkpoint = train_stage(stage, previous_checkpoint)
    CONFIGS[stage], CHECKPOINTS[stage] = config_path, previous_checkpoint
CHECKPOINTS
"""
    ),
    markdown(
        r"""
## 9. Validation metrics and baselines

The table reports PSNR, SSIM, L1, edge F1 and remote-sensing-native ERGAS, SAM,
UIQI and spatial correlation (sCC). Higher is better for PSNR/SSIM/edge F1/UIQI/sCC;
lower is better for L1/ERGAS/SAM. Model selection must use validation only.
"""
    ),
    code(
        r"""
evaluation_config = CONFIGS[STAGES_TO_RUN[-1]]
evaluation_checkpoint = CHECKPOINTS[STAGES_TO_RUN[-1]]
split_output = EVALUATION_ROOT / EVALUATION_SPLIT / "model"
baseline_output = EVALUATION_ROOT / EVALUATION_SPLIT / "baselines.json"

run([
    sys.executable, "-m", "geodiff_gan.cli.evaluate",
    "--config", evaluation_config,
    "--checkpoint", evaluation_checkpoint,
    "--output", split_output,
    "--split", EVALUATION_SPLIT,
    "--samples", EVALUATION_SAMPLES,
    "--steps", EVALUATION_STEPS,
    "--back-projection-steps", EVALUATION_BACK_PROJECTION_STEPS,
    "--residual-scale", EVALUATION_RESIDUAL_SCALE,
    "--mode", "sr",
    "--limit", EVALUATION_LIMIT,
    "--device", "cuda" if torch.cuda.is_available() else "cpu",
    "--progress", "compact",
    "--no-text",
], cwd=REPOSITORY_DIR)

baseline_command = [
    sys.executable, "-m", "geodiff_gan.cli.baselines",
    "--config", evaluation_config,
    "--output", baseline_output,
    "--split", EVALUATION_SPLIT,
    "--limit", EVALUATION_LIMIT,
    "--device", "cuda" if torch.cuda.is_available() else "cpu",
]
if "base" in CHECKPOINTS:
    baseline_command += ["--base-checkpoint", CHECKPOINTS["base"]]
run(baseline_command, cwd=REPOSITORY_DIR)

model_metrics = json.loads((split_output / "metrics.json").read_text(encoding="utf-8"))
baseline_metrics = json.loads(baseline_output.read_text(encoding="utf-8"))
rows = [{"method": "GeoDiff-GAN", **model_metrics}]
for method, values in baseline_metrics.items():
    if isinstance(values, dict):
        rows.append({"method": method, **values})
metric_table = pd.DataFrame(rows)
columns = [
    "method", "count", "l1", "psnr", "ssim", "edge_f1",
    "ergas", "sam_degrees", "uiqi", "scc", "redegradation_l1",
]
display(metric_table[[column for column in columns if column in metric_table.columns]].round(6))

plot_metrics = [metric for metric in ("psnr", "ssim", "ergas", "sam_degrees", "uiqi", "scc")
                if metric in metric_table.columns]
fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
for axis, metric in zip(axes.flat, plot_metrics):
    values = metric_table[["method", metric]].dropna()
    axis.bar(values["method"], values[metric], color=["#1f4e79", "#7f8c8d", "#2a9d8f"][:len(values)])
    axis.set_title(metric)
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
for axis in axes.flat[len(plot_metrics):]:
    axis.axis("off")
plt.tight_layout()
plt.show()
"""
    ),
    markdown("## 10. Inspect any validation or test result by index"),
    code(
        r"""
def visualize_result(index=0, split=EVALUATION_SPLIT, show_base=True):
    selected = [record for record in records if record.split == split]
    index = int(index)
    if not 0 <= index < len(selected):
        raise IndexError(f"Choose {split} index 0..{len(selected)-1}")
    output_dir = EVALUATION_ROOT / "indexed" / split / f"index_{index:05d}"
    run([
        sys.executable, "-m", "geodiff_gan.cli.evaluate",
        "--config", evaluation_config, "--checkpoint", evaluation_checkpoint,
        "--output", output_dir, "--split", split, "--index", index,
        "--samples", EVALUATION_SAMPLES, "--steps", EVALUATION_STEPS,
        "--back-projection-steps", EVALUATION_BACK_PROJECTION_STEPS,
        "--residual-scale", EVALUATION_RESIDUAL_SCALE, "--mode", "sr",
        "--device", "cuda" if torch.cuda.is_available() else "cpu",
        "--progress", "compact", "--no-text",
    ], cwd=REPOSITORY_DIR)
    caches = list(output_dir.glob("*_uncertainty.npz"))
    if len(caches) != 1:
        raise RuntimeError(f"Expected one cache, found {len(caches)} in {output_dir}")
    with np.load(caches[0]) as cache:
        output = chw(cache["mean"])
        base = chw(cache["base"])
    record = selected[index]
    with np.load(record.patch) as patch:
        lr, hr = chw(patch["lr"]), chw(patch["hr"])
        valid = torch.from_numpy(patch["valid_mask_hr"][0]).bool()
    bicubic = F.interpolate(lr[None], size=hr.shape[-2:], mode="bicubic", align_corners=False)[0]
    panels = [(lr, "Landsat OLI original 30 m", False),
              (bicubic, "Bicubic", True)]
    if show_base:
        panels.append((base, "SwinIR base", True))
    panels += [(output, "GeoDiff-GAN", True), (hr, "Sentinel-2 target 10 m", True)]
    displays = display_stretch([image for image, _, _ in panels])
    columns = min(3, len(panels))
    rows_count = (len(panels) + columns - 1) // columns
    fig, axes = plt.subplots(rows_count, columns, figsize=(5 * columns, 5 * rows_count), squeeze=False)
    for axis, display_image, (image, title, comparable) in zip(axes.flat, displays, panels):
        axis.imshow(display_image, interpolation="nearest" if image.shape[-1] == LR_CROP_SIZE else "antialiased")
        if comparable:
            l1 = float((image - hr).abs().mean(0)[valid].mean())
            title += f"\nmasked L1={l1:.5f}"
        axis.set_title(title)
        axis.axis("off")
    for axis in axes.flat[len(panels):]:
        axis.axis("off")
    fig.suptitle(f"{split} index={index} | source={record.tile_id}")
    plt.tight_layout()
    figure_path = output_dir / "comparison.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.show()
    print("Saved:", figure_path)
    print((output_dir / "metrics.json").read_text(encoding="utf-8"))
    return output_dir

VISUAL_RESULT = visualize_result(VISUALIZATION_INDEX, split=EVALUATION_SPLIT, show_base=True)
"""
    ),
    markdown(
        r"""
## 11. Final test protocol

Only after architecture, checkpoint and inference settings are fixed on validation,
set `EVALUATION_SPLIT = "test"` in Cell 1 and rerun Cells 8–9. Do not tune on the
official test split. Save the generated config, checkpoint revision, manifest and
metrics JSON with the paper results.

For a paper-to-paper OLI2MSI comparison, add a separate full-frame `160 -> 480`
tiled-inference evaluation. Do not present the center-crop result as the official
full-frame protocol.
"""
    ),
    markdown(
        r"""
## 12. Install the unchanged pretrained ASDDPM models

This section adds the official pretrained ASDDPM as an external baseline. It
does not edit or retrain ASDDPM:

- OLI2MSI uses the official 3x checkpoint and `rrdb3` encoder.
- ALSAT uses the official 4x checkpoint and `rrdb4` encoder.
- Both use the paper's 100-step diffusion sampler.

The Hugging Face archive is RAR5. Kaggle's old `7z` cannot decode its method, so
the cell installs `unar`. Existing downloads, extracted weights and predictions
are preserved and reused.
"""
    ),
    code(
        r"""
ASDDPM_REPOSITORY_URL = "https://github.com/littlebeen/ASDDPM-Adaptive-Semantic-Enhanced-DDPM.git"
ASDDPM_REPOSITORY_DIR = Path("/kaggle/working/asddpm-official")
ASDDPM_WEIGHT_ROOT = Path("/kaggle/working/asddpm-hf-weights")
ASDDPM_EXTRACT_ROOT = ASDDPM_WEIGHT_ROOT / "extracted"
ASDDPM_HF_REPOSITORY = "littlebeen/ASDDPM"
ASDDPM_SAMPLES = 1
ASDDPM_USE_AMP = False  # official-fidelity default; enable only if memory requires it
ASDDPM_TEST_LIMIT = 16 if FAST_DEV_RUN else 100

if (ASDDPM_REPOSITORY_DIR / ".git").is_dir():
    run(["git", "pull", "--ff-only"], cwd=ASDDPM_REPOSITORY_DIR)
elif ASDDPM_REPOSITORY_DIR.exists():
    raise RuntimeError(
        f"{ASDDPM_REPOSITORY_DIR} exists but is not a Git clone. "
        "Choose another ASDDPM_REPOSITORY_DIR; this notebook will not delete it."
    )
else:
    run(["git", "clone", "--depth", "1", ASDDPM_REPOSITORY_URL, ASDDPM_REPOSITORY_DIR])

run([
    sys.executable, "-m", "pip", "install", "-q",
    "huggingface_hub>=0.25", "einops", "timm", "tensorboardX",
])
from huggingface_hub import hf_hub_download, list_repo_files

ASDDPM_WEIGHT_ROOT.mkdir(parents=True, exist_ok=True)
ASDDPM_EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
hf_files = list_repo_files(ASDDPM_HF_REPOSITORY)
print("ASDDPM Hub files:")
for name in hf_files:
    print(" -", name)

archive_name = next(
    (name for name in hf_files if name.lower().endswith(".rar") and "asddpm" in name.lower()),
    None,
)
if archive_name is None:
    raise FileNotFoundError("No ASDDPM RAR checkpoint archive was found on the official Hub")
ASDDPM_ARCHIVE = Path(hf_hub_download(
    repo_id=ASDDPM_HF_REPOSITORY,
    filename=archive_name,
    local_dir=ASDDPM_WEIGHT_ROOT,
))

rrdb_downloads = []
for scale in (3, 4):
    exact = [
        name for name in hf_files
        if name.lower().endswith((".pt", ".pth"))
        and Path(name).stem.lower() == f"model_best{scale}"
    ]
    if not exact:
        exact = [
            name for name in hf_files
            if name.lower().endswith((".pt", ".pth"))
            and "rrdb" in name.lower() and str(scale) in Path(name).stem
        ]
    if exact:
        rrdb_downloads.append(Path(hf_hub_download(
            repo_id=ASDDPM_HF_REPOSITORY,
            filename=exact[0],
            local_dir=ASDDPM_WEIGHT_ROOT,
        )))

def find_weight(pattern):
    candidates = sorted(
        list(ASDDPM_WEIGHT_ROOT.rglob(pattern)) + list(ASDDPM_EXTRACT_ROOT.rglob(pattern)),
        key=lambda path: (-path.stat().st_size, str(path)),
    )
    return candidates[0] if candidates else None

def find_exact_stem(stem):
    candidates = sorted(
        path for path in ASDDPM_WEIGHT_ROOT.rglob("*")
        if path.is_file() and path.stem.casefold() == stem.casefold()
    )
    return candidates[0] if candidates else None

ASDDPM_OLI_CHECKPOINT = find_weight("*OLI2MSI*ckpt")
ASDDPM_ALSAT_CHECKPOINT = find_weight("*Alsat*ckpt") or find_weight("*ALSAT*ckpt")
if ASDDPM_OLI_CHECKPOINT is None or ASDDPM_ALSAT_CHECKPOINT is None:
    unar = shutil.which("unar")
    if unar is None:
        run(["apt-get", "update", "-qq"])
        run(["apt-get", "install", "-y", "-qq", "unar"])
        unar = shutil.which("unar")
    if unar is None:
        raise RuntimeError("unar installation failed; it is required for the official RAR5 archive")
    run([unar, "-f", "-o", ASDDPM_EXTRACT_ROOT, ASDDPM_ARCHIVE])
    ASDDPM_OLI_CHECKPOINT = find_weight("*OLI2MSI*ckpt")
    ASDDPM_ALSAT_CHECKPOINT = find_weight("*Alsat*ckpt") or find_weight("*ALSAT*ckpt")

ASDDPM_RRDB3 = find_exact_stem("model_best3")
ASDDPM_RRDB4 = find_exact_stem("model_best4")
required_weights = {
    "OLI2MSI ASDDPM": ASDDPM_OLI_CHECKPOINT,
    "ALSAT ASDDPM": ASDDPM_ALSAT_CHECKPOINT,
    "RRDB 3x": ASDDPM_RRDB3,
    "RRDB 4x": ASDDPM_RRDB4,
}
missing = [name for name, path in required_weights.items() if path is None]
if missing:
    raise FileNotFoundError(
        "Official weight files missing: " + ", ".join(missing) +
        ". Review the printed Hub file list; do not substitute a 3x RRDB for the 4x model."
    )
for name, path in required_weights.items():
    print(f"{name}: {path} ({path.stat().st_size / 2**20:.1f} MiB)")

ASDDPM_REVISION = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=ASDDPM_REPOSITORY_DIR, text=True
).strip()
print("Official ASDDPM source revision:", ASDDPM_REVISION)
"""
    ),
    markdown("## 13. Evaluate pretrained ASDDPM 3x on the OLI2MSI test crop protocol"),
    code(
        r"""
OLI_ASDDPM_OUTPUT = EVALUATION_ROOT / "test" / "pretrained_asddpm_3x"
OLI_TEST_MODEL_OUTPUT = EVALUATION_ROOT / "test" / "model"
OLI_TEST_BASELINE_OUTPUT = EVALUATION_ROOT / "test" / "baselines.json"

# Run the saved GeoDiff checkpoint on exactly the same held-out records and
# limit before comparing it with pretrained ASDDPM. Do not tune after this.
run([
    sys.executable, "-m", "geodiff_gan.cli.evaluate",
    "--config", evaluation_config, "--checkpoint", evaluation_checkpoint,
    "--output", OLI_TEST_MODEL_OUTPUT, "--split", "test",
    "--samples", EVALUATION_SAMPLES, "--steps", EVALUATION_STEPS,
    "--back-projection-steps", EVALUATION_BACK_PROJECTION_STEPS,
    "--residual-scale", EVALUATION_RESIDUAL_SCALE, "--mode", "sr",
    "--limit", ASDDPM_TEST_LIMIT, "--device", "cuda",
    "--progress", "compact", "--no-text",
], cwd=REPOSITORY_DIR)
run([
    sys.executable, "-m", "geodiff_gan.cli.baselines",
    "--config", evaluation_config, "--output", OLI_TEST_BASELINE_OUTPUT,
    "--split", "test", "--limit", ASDDPM_TEST_LIMIT, "--device", "cuda",
    "--base-checkpoint", CHECKPOINTS["base"],
], cwd=REPOSITORY_DIR)

oli_asddpm_command = [
    sys.executable, str(REPOSITORY_DIR / "scripts" / "asddpm_pretrained_eval.py"),
    "--official-repo", ASDDPM_REPOSITORY_DIR,
    "--manifest", MANIFEST,
    "--checkpoint", ASDDPM_OLI_CHECKPOINT,
    "--rrdb-checkpoint", ASDDPM_RRDB3,
    "--output", OLI_ASDDPM_OUTPUT,
    "--split", "test",
    "--scale", 3,
    "--limit", ASDDPM_TEST_LIMIT,
    "--samples", ASDDPM_SAMPLES,
    "--seed", RANDOM_SEED,
    "--device", "cuda",
]
if ASDDPM_USE_AMP:
    oli_asddpm_command.append("--amp")
run(oli_asddpm_command, cwd=REPOSITORY_DIR)
OLI_ASDDPM_METRICS = json.loads(
    (OLI_ASDDPM_OUTPUT / "metrics.json").read_text(encoding="utf-8")
)
display(pd.DataFrame([OLI_ASDDPM_METRICS]))
"""
    ),
    markdown(
        r"""
## 14. Prepare ALSAT-2B as a separate 4x experiment

Attach the ALSAT Kaggle dataset shown above. Discovery expects
`Alsat-2B-main/Train/{LR,HR}` and recursively reads categorized test folders such
as `Test/Agriculture/{LR,HR}`, `Test/Special/{LR,HR}` and
`Test/Urban/{LR,HR}`.

Training sources are deterministically divided into train and validation before
cropping. Official test categories remain test-only. Patches follow the paper's
native `32 x 32 -> 128 x 128` 4x protocol. The OLI2MSI workspace is untouched.
"""
    ),
    code(
        r"""
ALSAT_INPUT_ROOT = Path("/kaggle/input")
ALSAT_WORK_ROOT = Path("/kaggle/working/geodiff-alsat-4x")
ALSAT_PATCH_ROOT = ALSAT_WORK_ROOT / "patches"
ALSAT_MANIFEST = ALSAT_WORK_ROOT / "manifest.jsonl"
ALSAT_QUARANTINE = ALSAT_WORK_ROOT / "quarantine.jsonl"
ALSAT_CONFIG_ROOT = ALSAT_WORK_ROOT / "configs"
ALSAT_RUN_ROOT = ALSAT_WORK_ROOT / "runs"
ALSAT_EVALUATION_ROOT = ALSAT_WORK_ROOT / "evaluation"
ALSAT_LR_CROP_SIZE = 32
ALSAT_SCALE = 4
ALSAT_MAX_TRAIN_SOURCES = 16 if FAST_DEV_RUN else None
ALSAT_MAX_VAL_SOURCES = 4 if FAST_DEV_RUN else None
ALSAT_MAX_TEST_SOURCES = 8 if FAST_DEV_RUN else None
ALSAT_TRAIN_PATCHES_PER_SOURCE = 4 if FAST_DEV_RUN else 16
ALSAT_EVAL_PATCHES_PER_SOURCE = 2 if FAST_DEV_RUN else 4
ALSAT_EVALUATION_LIMIT = 16 if FAST_DEV_RUN else 100

for directory in (
    ALSAT_WORK_ROOT, ALSAT_PATCH_ROOT, ALSAT_CONFIG_ROOT,
    ALSAT_RUN_ROOT, ALSAT_EVALUATION_ROOT,
):
    directory.mkdir(parents=True, exist_ok=True)

prepare_alsat_command = [
    sys.executable, str(REPOSITORY_DIR / "scripts" / "prepare_alsat_geodiff.py"),
    "--input", ALSAT_INPUT_ROOT,
    "--output", ALSAT_PATCH_ROOT,
    "--manifest", ALSAT_MANIFEST,
    "--quarantine", ALSAT_QUARANTINE,
    "--lr-crop-size", ALSAT_LR_CROP_SIZE,
    "--scale", ALSAT_SCALE,
    "--validation-percent", 10,
    "--train-patches-per-source", ALSAT_TRAIN_PATCHES_PER_SOURCE,
    "--eval-patches-per-source", ALSAT_EVAL_PATCHES_PER_SOURCE,
]
for flag, value in (
    ("--max-train-sources", ALSAT_MAX_TRAIN_SOURCES),
    ("--max-val-sources", ALSAT_MAX_VAL_SOURCES),
    ("--max-test-sources", ALSAT_MAX_TEST_SOURCES),
):
    if value is not None:
        prepare_alsat_command += [flag, value]
run(prepare_alsat_command, cwd=REPOSITORY_DIR)

ALSAT_RECORDS = load_manifest(ALSAT_MANIFEST)
alsat_patch_counts = Counter(record.split for record in ALSAT_RECORDS)
alsat_source_counts = {
    split: len({record.tile_id for record in ALSAT_RECORDS if record.split == split})
    for split in ("train", "val", "test")
}
print("ALSAT patches:", alsat_patch_counts)
print("ALSAT source pairs:", alsat_source_counts)
if any(alsat_patch_counts[split] == 0 for split in ("train", "val", "test")):
    raise RuntimeError("ALSAT train/val/test must all be non-empty")

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for column, split in enumerate(("train", "val", "test")):
    record = next(record for record in ALSAT_RECORDS if record.split == split)
    with np.load(record.patch) as patch:
        lr = chw(patch["lr"])
        hr = chw(patch["hr"])
    axes[0, column].imshow(lr.permute(1, 2, 0).clamp(0, 1))
    axes[0, column].set_title(f"{split}: ALSAT LR 32x32")
    axes[1, column].imshow(hr.permute(1, 2, 0).clamp(0, 1))
    axes[1, column].set_title(f"{split}: ALSAT HR 128x128")
    axes[0, column].axis("off")
    axes[1, column].axis("off")
plt.tight_layout()
plt.show()
"""
    ),
    markdown("## 15. Train and evaluate a separate GeoDiff-GAN 4x model on ALSAT"),
    code(
        r"""
alsat_runtime_config = copy.deepcopy(runtime_config)
alsat_runtime_config["model"].update({
    "scale": 4,
    "input_channels": 3,
    "output_channels": 3,
    "base_upsample_mode": "resize_conv",
    "decoder_upsample_mode": "resize_conv",
    "use_text_conditioning": False,
    "use_degradation_conditioning": False,
})
alsat_runtime_config["data"].update({
    "manifest": str(ALSAT_MANIFEST),
    "captions": None,
    "input_mode": "paired",
    "target_key": "hr",
    "condition_key": None,
    "train_degradation_sampling": "fixed",
})
alsat_runtime_config["training"].update({
    "batch_size": 2 if FAST_DEV_RUN else 8,
    "gradient_accumulation": 4 if FAST_DEV_RUN else 2,
    "validation_limit": min(ALSAT_EVALUATION_LIMIT, alsat_patch_counts["val"]),
    "keep_best_and_latest": True,
    "auto_resume": AUTO_RESUME_TRAINING,
    "checkpoint_metric": "val_psnr",
    "checkpoint_mode": "max",
    "early_stopping_metric": "val_psnr",
    "early_stopping_mode": "max",
    "train_back_projection_steps": 0,
    "validation_back_projection_steps": 0,
})
alsat_runtime_config["training"]["loss_weights"].update({
    "mse": 50.0,
    "multiscale_mse": 25.0,
    "charbonnier": 1.0,
    "ssim": 0.5,
    "consistency": 0.0,
    "perceptual": 0.0,
    "adversarial": 0.0,
    "prompt_alignment": 0.0,
})

def train_alsat_stage(stage, initial_checkpoint=None):
    config = copy.deepcopy(alsat_runtime_config)
    output_dir = ALSAT_RUN_ROOT / stage
    output_dir.mkdir(parents=True, exist_ok=True)
    config["training"].update({
        "stage": stage,
        "epochs": EPOCHS[stage],
        "learning_rate": STAGE_LEARNING_RATES[stage],
        "output_dir": str(output_dir),
        "init_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
        "resume": None,
        "auto_resume": AUTO_RESUME_TRAINING,
    })
    if stage != "joint":
        config["training"]["module_learning_rate_multipliers"] = {}
    if stage == "diffusion":
        config["training"]["loss_weights"]["diffusion"] = 1.0
        config["training"].update({
            "checkpoint_metric": "val_loss_total", "checkpoint_mode": "min",
            "early_stopping_metric": "val_loss_total", "early_stopping_mode": "min",
            "lr_scheduler_metric": "val_loss_total", "lr_scheduler_mode": "min",
        })
    config_path = ALSAT_CONFIG_ROOT / f"{stage}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"\n===== ALSAT 4x {stage}: epochs={EPOCHS[stage]} =====")
    run([sys.executable, "-m", "geodiff_gan.cli.train", "--config", config_path], cwd=REPOSITORY_DIR)
    checkpoint = select_checkpoint(output_dir, stage)
    if checkpoint is None:
        raise RuntimeError(f"No ALSAT checkpoint produced for {stage}")
    return config_path, checkpoint

ALSAT_CONFIGS, ALSAT_CHECKPOINTS = {}, {}
alsat_previous = None
for stage in STAGES_TO_RUN:
    alsat_config, alsat_previous = train_alsat_stage(stage, alsat_previous)
    ALSAT_CONFIGS[stage] = alsat_config
    ALSAT_CHECKPOINTS[stage] = alsat_previous

ALSAT_MODEL_OUTPUT = ALSAT_EVALUATION_ROOT / "test" / "model"
ALSAT_BASELINE_OUTPUT = ALSAT_EVALUATION_ROOT / "test" / "baselines.json"
run([
    sys.executable, "-m", "geodiff_gan.cli.evaluate",
    "--config", ALSAT_CONFIGS[STAGES_TO_RUN[-1]],
    "--checkpoint", ALSAT_CHECKPOINTS[STAGES_TO_RUN[-1]],
    "--output", ALSAT_MODEL_OUTPUT,
    "--split", "test", "--samples", EVALUATION_SAMPLES,
    "--steps", EVALUATION_STEPS, "--back-projection-steps", 0,
    "--residual-scale", 1.0, "--mode", "sr",
    "--limit", ALSAT_EVALUATION_LIMIT, "--device", "cuda",
    "--progress", "compact", "--no-text",
], cwd=REPOSITORY_DIR)
run([
    sys.executable, "-m", "geodiff_gan.cli.baselines",
    "--config", ALSAT_CONFIGS[STAGES_TO_RUN[-1]],
    "--output", ALSAT_BASELINE_OUTPUT,
    "--split", "test", "--limit", ALSAT_EVALUATION_LIMIT,
    "--device", "cuda", "--base-checkpoint", ALSAT_CHECKPOINTS["base"],
], cwd=REPOSITORY_DIR)
ALSAT_GEODIFF_METRICS = json.loads((ALSAT_MODEL_OUTPUT / "metrics.json").read_text())
ALSAT_BASELINE_METRICS = json.loads(ALSAT_BASELINE_OUTPUT.read_text())
display(pd.DataFrame([
    {"method": "GeoDiff-GAN 4x", **ALSAT_GEODIFF_METRICS},
    *[
        {"method": name, **values}
        for name, values in ALSAT_BASELINE_METRICS.items() if isinstance(values, dict)
    ],
]))
"""
    ),
    markdown("## 16. Evaluate pretrained ASDDPM 4x on the same ALSAT test patches"),
    code(
        r"""
ALSAT_ASDDPM_OUTPUT = ALSAT_EVALUATION_ROOT / "test" / "pretrained_asddpm_4x"
alsat_asddpm_command = [
    sys.executable, str(REPOSITORY_DIR / "scripts" / "asddpm_pretrained_eval.py"),
    "--official-repo", ASDDPM_REPOSITORY_DIR,
    "--manifest", ALSAT_MANIFEST,
    "--checkpoint", ASDDPM_ALSAT_CHECKPOINT,
    "--rrdb-checkpoint", ASDDPM_RRDB4,
    "--output", ALSAT_ASDDPM_OUTPUT,
    "--split", "test", "--scale", 4,
    "--limit", ASDDPM_TEST_LIMIT,
    "--samples", ASDDPM_SAMPLES,
    "--seed", RANDOM_SEED, "--device", "cuda",
]
if ASDDPM_USE_AMP:
    alsat_asddpm_command.append("--amp")
run(alsat_asddpm_command, cwd=REPOSITORY_DIR)
ALSAT_ASDDPM_METRICS = json.loads(
    (ALSAT_ASDDPM_OUTPUT / "metrics.json").read_text(encoding="utf-8")
)
display(pd.DataFrame([ALSAT_ASDDPM_METRICS]))
"""
    ),
    markdown(
        r"""
## 17. Report the two benchmarks separately

Never average OLI2MSI and ALSAT into one score. They have different scale factors,
image statistics and test sets. Each table below compares methods only on the same
manifest, crop geometry, valid mask and normalization. ASDDPM uses its official
100 sampling steps; GeoDiff's step count is reported alongside it.
"""
    ),
    code(
        r"""
def metric_row(dataset, method, values):
    return {
        "dataset": dataset,
        "method": method,
        **{
            key: values.get(key)
            for key in (
                "count", "l1", "psnr", "ssim", "edge_f1", "ergas",
                "sam_degrees", "uiqi", "scc", "diffusion_steps",
                "samples_per_patch",
            )
        },
    }

comparison_rows = [metric_row("OLI2MSI 3x", "ASDDPM pretrained", OLI_ASDDPM_METRICS)]
oli_model_metrics_path = EVALUATION_ROOT / "test" / "model" / "metrics.json"
oli_baseline_path = EVALUATION_ROOT / "test" / "baselines.json"
if oli_model_metrics_path.exists():
    comparison_rows.append(metric_row(
        "OLI2MSI 3x", "GeoDiff-GAN", json.loads(oli_model_metrics_path.read_text())
    ))
if oli_baseline_path.exists():
    oli_baselines = json.loads(oli_baseline_path.read_text())
    for method, values in oli_baselines.items():
        if isinstance(values, dict):
            comparison_rows.append(metric_row("OLI2MSI 3x", method, values))

comparison_rows.extend([
    metric_row("ALSAT 4x", "GeoDiff-GAN", ALSAT_GEODIFF_METRICS),
    metric_row("ALSAT 4x", "ASDDPM pretrained", ALSAT_ASDDPM_METRICS),
])
for method, values in ALSAT_BASELINE_METRICS.items():
    if isinstance(values, dict):
        comparison_rows.append(metric_row("ALSAT 4x", method, values))

BENCHMARK_COMPARISON = pd.DataFrame(comparison_rows)
display(BENCHMARK_COMPARISON.round(6))
comparison_csv = WORK_ROOT / "geodiff_asddpm_3x_4x_comparison.csv"
BENCHMARK_COMPARISON.to_csv(comparison_csv, index=False)

metrics_to_plot = ["psnr", "ssim", "edge_f1", "ergas", "sam_degrees", "scc"]
fig, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False)
for axis, metric in zip(axes.flat, metrics_to_plot):
    pivot = BENCHMARK_COMPARISON.pivot(index="method", columns="dataset", values=metric)
    pivot.plot(kind="bar", ax=axis, rot=20)
    axis.set_title(metric)
    axis.set_xlabel("")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
plt.tight_layout()
comparison_figure = WORK_ROOT / "geodiff_asddpm_3x_4x_comparison.png"
fig.savefig(comparison_figure, dpi=180, bbox_inches="tight")
plt.show()
print("Saved:", comparison_csv)
print("Saved:", comparison_figure)
"""
    ),
    markdown("## 18. Inspect GeoDiff and pretrained ASDDPM outputs by test index"),
    code(
        r"""
def cached_tensor(output_root, patch_path, key="mean"):
    patch_path = str(Path(patch_path).resolve())
    candidates = sorted(Path(output_root).rglob("*.npz"))
    for candidate in candidates:
        with np.load(candidate) as cache:
            if key not in cache.files:
                continue
            source = str(cache["source_patch"].item()) if "source_patch" in cache.files else ""
            if source and str(Path(source).resolve()) == patch_path:
                return chw(cache[key]), candidate
    stem_matches = [candidate for candidate in candidates if Path(patch_path).stem in candidate.stem]
    for candidate in stem_matches:
        with np.load(candidate) as cache:
            if key in cache.files:
                return chw(cache[key]), candidate
    raise FileNotFoundError(f"No {key!r} cache for {patch_path} below {output_root}")

def compare_pretrained_by_index(dataset="OLI2MSI", index=0, show_base=True):
    dataset = dataset.upper()
    if dataset == "OLI2MSI":
        manifest = MANIFEST
        geodiff_root = OLI_TEST_MODEL_OUTPUT
        asddpm_root = OLI_ASDDPM_OUTPUT
        scale = 3
        figure_root = WORK_ROOT / "figures" / "pretrained_comparison"
    elif dataset == "ALSAT":
        manifest = ALSAT_MANIFEST
        geodiff_root = ALSAT_MODEL_OUTPUT
        asddpm_root = ALSAT_ASDDPM_OUTPUT
        scale = 4
        figure_root = ALSAT_WORK_ROOT / "figures" / "pretrained_comparison"
    else:
        raise ValueError("dataset must be 'OLI2MSI' or 'ALSAT'")

    test_records = load_manifest(manifest, split="test")[:ASDDPM_TEST_LIMIT]
    index = int(index)
    if not 0 <= index < len(test_records):
        raise IndexError(f"Choose index 0..{len(test_records)-1}")
    record = test_records[index]
    with np.load(record.patch) as patch:
        lr = chw(patch["lr"])[:3]
        hr = chw(patch["hr"])[:3]
        valid = torch.from_numpy(patch["valid_mask_hr"][0]).bool()
    bicubic = F.interpolate(
        lr[None], size=hr.shape[-2:], mode="bicubic", align_corners=False
    )[0].clamp(0, 1)
    geodiff, geodiff_cache = cached_tensor(geodiff_root, record.patch, "mean")
    asddpm, asddpm_cache = cached_tensor(asddpm_root, record.patch, "mean")
    panels = [(lr, f"Native LR {lr.shape[-1]}x{lr.shape[-2]}", False)]
    panels.append((bicubic, "Bicubic", True))
    if show_base:
        base, _ = cached_tensor(geodiff_root, record.patch, "base")
        panels.append((base, "GeoDiff base", True))
    panels.extend([
        (geodiff, "GeoDiff-GAN", True),
        (asddpm, "ASDDPM pretrained", True),
        (hr, "HR target", True),
    ])

    columns = 3
    row_count = (len(panels) + columns - 1) // columns
    fig, axes = plt.subplots(row_count, columns, figsize=(15, 5 * row_count), squeeze=False)
    for axis, (image, title, compare) in zip(axes.flat, panels):
        axis.imshow(image.clamp(0, 1).permute(1, 2, 0))
        if compare:
            error = (image - hr).abs().mean(0)
            title += f"\nmasked L1={float(error[valid].mean()):.5f}"
        axis.set_title(title)
        axis.axis("off")
    for axis in axes.flat[len(panels):]:
        axis.axis("off")
    fig.suptitle(f"{dataset} {scale}x | test index {index} | {record.tile_id}")
    plt.tight_layout()
    figure_root.mkdir(parents=True, exist_ok=True)
    figure_path = figure_root / f"test_index_{index:04d}.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.show()
    print("GeoDiff cache:", geodiff_cache)
    print("ASDDPM cache:", asddpm_cache)
    print("Saved:", figure_path)
    return figure_path

OLI_COMPARISON_FIGURE = compare_pretrained_by_index("OLI2MSI", 0, show_base=True)
ALSAT_COMPARISON_FIGURE = compare_pretrained_by_index("ALSAT", 0, show_base=True)
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
print(OUTPUT)
