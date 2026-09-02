"""Build the full-frame OLI2MSI strong-base and routed-expert notebook."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
from textwrap import dedent
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "kaggle/GeoDiff_GAN_Kaggle_Tile_Strong_Base_MoE_3x.ipynb"
OUTPUT = ROOT / "kaggle/GeoDiff_GAN_Kaggle_OLI2MSI_Strong_Base_MoE_3x.ipynb"
ARCHIVE = ROOT / "kaggle/geodiff_oli2msi_strong_base_source.zip"


def value(cell):
    return "".join(cell["source"])


def set_markdown(cell, source):
    cell["source"] = dedent(source).strip().splitlines(True)


def set_code(cell, source):
    source = dedent(source).strip() + "\n"
    ast.parse(source)
    cell["source"] = source.splitlines(True)
    cell["execution_count"] = None
    cell["outputs"] = []


def markdown(source):
    cell = {"cell_type": "markdown", "metadata": {}, "source": []}
    set_markdown(cell, source)
    return cell


def code(source):
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [],
    }
    set_code(cell, source)
    return cell


def find(cells, marker, cell_type=None):
    matches = [
        index for index, cell in enumerate(cells)
        if marker in value(cell)
        and (cell_type is None or cell["cell_type"] == cell_type)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one cell containing {marker!r}; found {matches}")
    return matches[0]


def build():
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = copy.deepcopy(notebook["cells"])

    set_markdown(cells[0], """
    # GeoDiff 3x Strong Base and routed experts on official OLI2MSI

    This isolated Kaggle notebook trains the current `3x-continued` architecture on
    real cross-sensor OLI2MSI pairs:

    ```text
    Landsat-8 OLI RGB, 30 m, 3 × 160 × 160
                         → GeoDiff-GAN SR-3x
    Sentinel-2 MSI RGB, 10 m, 3 × 480 × 480
    ```

    Full runs require the official `5,225` training and `100` test pairs. A stable
    source-ID hash reserves 5% of the official training set for validation; the
    official test set is never used for selection. Full frames are stored and used
    for validation/test. Random `32→96`, `64→192`, and `128→384` aligned crops are
    used only during base training.

    The notebook compares the final frozen base with a one-expert control, generic
    MoE, and reliability-supervised MoE. It uses fixed OLI2MSI radiometry,
    resize-convolution, no PixelShuffle, no captions, no synthetic LR, no GAN loss,
    and no natural-image perceptual loss. Training uses compact one-line progress,
    minimum optimizer-step budgets, and full validation at controlled intervals.
    Run the smoke profile first, then
    use a new suite root with `FAST_DEV_RUN=False`.

    Attach the official OLI2MSI data and `geodiff_oli2msi_strong_base_source.zip` as
    private Kaggle datasets. A 35 dB result is a target, not a guarantee.

    **MoE V2 change:** the full run uses two experts and trains both for the first
    five epochs. It then switches to one-expert routing. Each expert receives a
    different detached oracle target based on its relative velocity error, while
    a differentiable balance term discourages collapse. Joint training refines
    only the lightweight router/expert adapters; the shared diffusion trunk and
    deterministic base remain frozen.
    """)

    controls = find(cells, "FAST_DEV_RUN = True", "code")
    set_code(cells[controls], """
    from pathlib import Path
    import os, sys, json, subprocess, time, shutil, hashlib
    from collections import Counter

    FAST_DEV_RUN = True
    REPOSITORY_BRANCH = "3x-continued"
    REPOSITORY_URL = "https://github.com/shashankjs2002/SI-SR-1.git"
    REPOSITORY_DIR = Path("/kaggle/working/geodiff-3x-continued-oli2msi")
    SOURCE_ARCHIVE = None
    SUITE_ROOT = Path(
        "/kaggle/working/geodiff-oli2msi-moe-v2-smoke"
        if FAST_DEV_RUN else "/kaggle/working/geodiff-oli2msi-moe-v2"
    )
    KNOWN_OLI2MSI_ROOT = Path(
        "/kaggle/input/datasets/twilight2002/oli2msi-thesis/"
        "OLI2MSI-dataset/OLI2MSI"
    )
    OLI2MSI_DATA_ROOT = (
        KNOWN_OLI2MSI_ROOT if KNOWN_OLI2MSI_ROOT.exists() else None
    )  # May also be set to a split folder, LR/HR folder, or individual TIFF.
    RESTORE_SUITE_FROM = None

    DATA_PROTOCOL_ID = "oli2msi_official_full160_clip03_uint8grid_v1"
    VALIDATION_PERCENT = 5
    BENCHMARK_REFLECTANCE_MAX = 0.3
    QUANTIZE_TO_UINT8_GRID = True
    MINIMUM_VALID_FRACTION = 0.99
    LR_FRAME_SIZE, SCALE = 160, 3
    HR_FRAME_SIZE = LR_FRAME_SIZE * SCALE
    MAX_TRAIN_SOURCE_PAIRS = 64 if FAST_DEV_RUN else None
    MAX_VALIDATION_SOURCE_PAIRS = 16 if FAST_DEV_RUN else None
    MAX_TEST_SOURCE_PAIRS = 16 if FAST_DEV_RUN else None

    NUM_EXPERTS = 2
    TOP_K = 1
    INCLUDE_MULTISPECTRAL = False
    NEW_MULTISPECTRAL = False
    TRAIN_LR_CROP = 64
    BASE_PHASES = (
        {"name": "crop32", "crop": 32, "epochs": 30, "minutes": 45,
         "learning_rate": 1e-4, "max_batches": 400, "minimum_steps": 3000},
        {"name": "crop64", "crop": 64, "epochs": 24, "minutes": 35,
         "learning_rate": 5e-5, "max_batches": 180, "minimum_steps": 1000},
        {"name": "crop128", "crop": 128, "epochs": 16, "minutes": 20,
         "learning_rate": 1e-5, "max_batches": 60, "minimum_steps": 250},
    )
    EPOCHS = {"base": 12, "vae": 16, "diffusion": 24, "joint": 16}
    SHARED_MINUTES = {"base": 40, "vae": 30}
    PER_EXPERIMENT_MINUTES = {"diffusion": 30, "joint": 25}
    MINIMUM_OPTIMIZER_STEPS = {"vae": 600, "diffusion": 900, "joint": 500}
    VALIDATE_EVERY = {"base": 1, "vae": 2, "diffusion": 4, "joint": 2}
    RUN_EARLIER = {name: False for name in (
        "rgb_standard", "rgb_fidelity", "multispectral_fidelity",
        "rgb_harmonized_fidelity", "multispectral_guided_fidelity")}

    BASE_VALIDATION_LIMIT = 4 if FAST_DEV_RUN else None
    RESIDUAL_VALIDATION_LIMIT = 4 if FAST_DEV_RUN else None
    ALIGNMENT_AUDIT_LIMIT = 4 if FAST_DEV_RUN else 100
    EVAL_LIMIT = 4 if FAST_DEV_RUN else 100
    TEST_EVAL_LIMIT = 4 if FAST_DEV_RUN else 100
    EVAL_SAMPLES = 1 if FAST_DEV_RUN else 4
    EVAL_STEPS = 2 if FAST_DEV_RUN else 20
    RUN_TEST_EVALUATION = False
    RANDOM_SEED = 42

    if not 1 <= TOP_K <= NUM_EXPERTS:
        raise ValueError("Require 1 <= TOP_K <= NUM_EXPERTS")
    SUITE_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT = SUITE_ROOT / "dataset"
    DOWNLOAD_ROOT = SUITE_ROOT / "official_download"
    PATCH_ROOT = DATA_ROOT / "patches"
    MANIFEST = DATA_ROOT / "manifest.jsonl"
    QUARANTINE_LOG = DATA_ROOT / "quarantine.jsonl"
    EXPERIMENT_ROOT = SUITE_ROOT / "experiments"
    FIGURE_ROOT = SUITE_ROOT / "figures"
    for directory in (DATA_ROOT, DOWNLOAD_ROOT, PATCH_ROOT, EXPERIMENT_ROOT, FIGURE_ROOT):
        directory.mkdir(parents=True, exist_ok=True)

    def run(command, cwd=None):
        command = list(map(str, command))
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(REPOSITORY_DIR / "src") + os.pathsep + environment.get("PYTHONPATH", "")
        environment["PYTHONUNBUFFERED"] = "1"
        print("+", " ".join(command), flush=True)
        return subprocess.run(command, cwd=cwd, env=environment, check=True)

    main_minutes = sum(phase["minutes"] for phase in BASE_PHASES)
    main_minutes += SHARED_MINUTES["vae"] + 3 * sum(PER_EXPERIMENT_MINUTES.values())
    print("Suite:", SUITE_ROOT)
    print("Protocol:", LR_FRAME_SIZE, "->", HR_FRAME_SIZE, "at", SCALE, "x")
    print("Nominal main-study training budget:", main_minutes, "minutes")
    """)

    install_md = find(cells, "## 2. Install the matching source", "markdown")
    set_markdown(cells[install_md], "## 1. Install the exact source snapshot and verify the GPU")
    install = find(cells, "geodiff_tile_strong_base_source*.zip", "code")
    install_source = value(cells[install]).replace(
        "geodiff_tile_strong_base_source*.zip",
        "geodiff_oli2msi_strong_base_source*.zip",
    )
    set_code(cells[install], install_source)

    restore_md = find(cells, "## 3. Restore saved session", "markdown")
    set_markdown(cells[restore_md], "## 2. Restore this notebook's saved session artifacts (optional)")

    discover_md = find(cells, "## 4. Discover products", "markdown")
    set_markdown(cells[discover_md], """
    ## 3. Locate the official OLI2MSI folders

    Attach a Kaggle dataset containing `train_lr`, `train_hr`, `test_lr`, and
    `test_hr`. The notebook also accepts an OLI2MSI-named ZIP and extracts it safely
    into this suite. It does not use Google Drive because public quota failures can
    silently leave unmatched partial data. No attached file is modified or deleted.
    """)
    discover = find(cells, "from geodiff_gan.data.sentinel import", "code")
    set_code(cells[discover], """
    search_roots = (
        [Path(OLI2MSI_DATA_ROOT)]
        if OLI2MSI_DATA_ROOT is not None
        else [Path("/kaggle/input"), DOWNLOAD_ROOT]
    )

    for root in list(search_roots):
        if not root.exists():
            continue
        for archive in root.rglob("*.zip"):
            if "oli2msi" not in archive.name.casefold():
                continue
            with zipfile.ZipFile(archive) as handle:
                groups = {
                    part.casefold()
                    for member in handle.namelist()
                    for part in Path(member).parts
                    if part.casefold() in {"train_lr", "train_hr", "test_lr", "test_hr"}
                }
            if groups != {"train_lr", "train_hr", "test_lr", "test_hr"}:
                continue
            destination = DOWNLOAD_ROOT / archive.stem
            marker = destination / ".extraction_complete"
            if not marker.exists():
                print("Extracting", archive)
                extract_safely(archive, destination)
                marker.write_text(str(archive), encoding="utf-8")

    def normalized_name(path):
        return "".join(character for character in path.name.casefold() if character.isalnum())

    def tiff_files(path):
        if not path.is_dir():
            return []
        return [item for item in path.rglob("*")
                if item.is_file() and item.suffix.casefold() in (".tif", ".tiff")]

    def contains_tiffs(path):
        return bool(tiff_files(path))

    def matching_child(parent, aliases):
        if not parent.is_dir():
            return None
        aliases = {"".join(character for character in alias.casefold() if character.isalnum())
                   for alias in aliases}
        matches = [child for child in parent.iterdir()
                   if child.is_dir() and normalized_name(child) in aliases]
        return matches[0] if len(matches) == 1 else None

    def layout_at(candidate):
        flat_aliases = {
            "train_lr": ("train_lr", "train-lr", "trainlr", "lr_train"),
            "train_hr": ("train_hr", "train-hr", "trainhr", "hr_train"),
            "test_lr": ("test_lr", "test-lr", "testlr", "lr_test"),
            "test_hr": ("test_hr", "test-hr", "testhr", "hr_test"),
        }
        flat = {name: matching_child(candidate, aliases) for name, aliases in flat_aliases.items()}
        if all(path is not None and contains_tiffs(path) for path in flat.values()):
            return flat

        train = matching_child(candidate, ("train", "training"))
        test = matching_child(candidate, ("test", "testing"))
        if train is None or test is None:
            return None
        lr_aliases = ("lr", "low_resolution", "low-resolution", "lowres", "oli", "landsat", "l8")
        hr_aliases = ("hr", "high_resolution", "high-resolution", "highres", "msi", "sentinel", "s2")
        nested = {
            "train_lr": matching_child(train, lr_aliases),
            "train_hr": matching_child(train, hr_aliases),
            "test_lr": matching_child(test, lr_aliases),
            "test_hr": matching_child(test, hr_aliases),
        }
        return nested if all(path is not None and contains_tiffs(path) for path in nested.values()) else None

    def discover_layout(roots):
        layouts, signatures = [], set()
        for supplied_root in roots:
            supplied_root = supplied_root.expanduser()
            if not supplied_root.exists():
                continue
            ancestors = [supplied_root]
            current = supplied_root
            while current.parent != current and current != Path("/kaggle/input"):
                current = current.parent
                ancestors.append(current)
                if current == Path("/kaggle/input"):
                    break
            candidates = ancestors + [path for path in supplied_root.rglob("*") if path.is_dir()]
            for candidate in candidates:
                layout = layout_at(candidate)
                if layout is None:
                    continue
                signature = tuple(str(layout[name].resolve()) for name in ("train_lr", "train_hr", "test_lr", "test_hr"))
                if signature not in signatures:
                    signatures.add(signature)
                    layouts.append((candidate.resolve(), layout))
        return layouts

    layouts = discover_layout(search_roots)
    if len(layouts) != 1:
        summary = [(str(root), {name: len(tiff_files(path))
                                for name, path in layout.items()}) for root, layout in layouts]
        nearby = []
        for supplied_root in search_roots:
            if supplied_root.exists():
                for path in [supplied_root] + [item for item in supplied_root.rglob("*") if item.is_dir()]:
                    name = normalized_name(path)
                    if any(token in name for token in ("train", "test", "lr", "hr", "oli", "msi")):
                        count = len(tiff_files(path))
                        if count:
                            nearby.append((str(path), count))
        raise RuntimeError(
            f"Expected exactly one complete OLI2MSI layout, found {len(layouts)}: {summary}. "
            f"Nearby TIFF directories: {nearby[:30]}. Set OLI2MSI_DATA_ROOT to the "
            "dataset root, any split folder, or one LR/HR folder."
        )
    OLI2MSI_ROOT, DATA_DIRS = layouts[0]
    print("OLI2MSI root:", OLI2MSI_ROOT)
    for name, path in DATA_DIRS.items():
        count = len(tiff_files(path))
        print(name, count, path)
    """)

    prepare_md = find(cells, "## 5. Prepare one shared", "markdown")
    set_markdown(cells[prepare_md], """
    ## 4. Audit pairs and prepare restart-safe full-frame NPZ files

    Both sensors receive the same fixed official transformation:
    `clip(reflectance, 0, 0.3) / 0.3`. With `QUANTIZE_TO_UINT8_GRID=True`, values
    are also placed on the 8-bit grid used by the official PNG conversion. There is
    no per-image normalization, target-time calibration, synthetic degradation, or
    cropping during preprocessing. Corrupt and unpaired files are quarantined.
    """)
    prepare = find(cells, "# Reset on cell rerun", "code")
    set_code(cells[prepare], """
    import rasterio
    from tqdm.auto import tqdm
    from geodiff_gan.data.manifest import ManifestRecord, load_manifest, write_manifest
    from geodiff_gan.experiments.tile_moe import balanced_manifest

    lock_json(DATA_ROOT / "protocol_lock.json", {
        "protocol": DATA_PROTOCOL_ID, "validation_percent": VALIDATION_PERCENT,
        "reflectance_max": BENCHMARK_REFLECTANCE_MAX,
        "quantize_uint8_grid": QUANTIZE_TO_UINT8_GRID,
        "lr_frame": LR_FRAME_SIZE, "scale": SCALE, "seed": RANDOM_SEED,
        "fast_dev": FAST_DEV_RUN,
    })

    def index_tiffs(directory):
        indexed = {}
        for path in directory.iterdir():
            if path.is_file() and path.suffix.casefold() in (".tif", ".tiff"):
                key = path.stem.casefold()
                if key in indexed:
                    raise RuntimeError(f"Duplicate TIFF stem {path.stem!r} in {directory}")
                indexed[key] = path
        return indexed

    def pair_directories(lr_directory, hr_directory):
        lr_files, hr_files = index_tiffs(lr_directory), index_tiffs(hr_directory)
        common = sorted(lr_files.keys() & hr_files.keys())
        return ([(key, lr_files[key], hr_files[key]) for key in common],
                sorted(lr_files.keys() - hr_files.keys()),
                sorted(hr_files.keys() - lr_files.keys()))

    official_train_pairs, train_missing_hr, train_missing_lr = pair_directories(DATA_DIRS["train_lr"], DATA_DIRS["train_hr"])
    official_test_pairs, test_missing_hr, test_missing_lr = pair_directories(DATA_DIRS["test_lr"], DATA_DIRS["test_hr"])
    print("Official paired counts:", len(official_train_pairs), len(official_test_pairs))
    missing = {"train_missing_hr": train_missing_hr, "train_missing_lr": train_missing_lr,
               "test_missing_hr": test_missing_hr, "test_missing_lr": test_missing_lr}
    if any(missing.values()):
        with QUARANTINE_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"stage": "pair_audit", **missing}) + "\\n")
        print("Unpaired files quarantined:", {key: len(items) for key, items in missing.items()})
    if not FAST_DEV_RUN and (len(official_train_pairs) != 5225 or len(official_test_pairs) != 100):
        raise RuntimeError("Publication run requires exactly 5,225 official train and 100 official test pairs")

    def source_split(key):
        bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100
        return "val" if bucket < VALIDATION_PERCENT else "train"

    train_pairs = [pair for pair in official_train_pairs if source_split(pair[0]) == "train"]
    val_pairs = [pair for pair in official_train_pairs if source_split(pair[0]) == "val"]
    test_pairs = official_test_pairs
    def truncate(items, maximum):
        return items if maximum is None else items[:int(maximum)]
    train_pairs = truncate(train_pairs, MAX_TRAIN_SOURCE_PAIRS)
    val_pairs = truncate(val_pairs, MAX_VALIDATION_SOURCE_PAIRS)
    test_pairs = truncate(test_pairs, MAX_TEST_SOURCE_PAIRS)
    selected = {"train": train_pairs, "val": val_pairs, "test": test_pairs}
    print("Selected source pairs:", {split: len(items) for split, items in selected.items()})
    if not FAST_DEV_RUN and shutil.disk_usage(SUITE_ROOT).free < 8 * 2**30:
        raise RuntimeError(
            "Less than 8 GiB remains in Kaggle working storage. Attach an already-prepared "
            "suite or free only disposable session artifacts; this notebook deletes nothing."
        )
    split_keys = {split: {item[0] for item in items} for split, items in selected.items()}
    if any(split_keys[left] & split_keys[right] for left, right in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise RuntimeError("Source leakage detected")

    audit_paths = [path for _, lr, hr in (train_pairs + val_pairs + test_pairs)[:8] for path in (lr, hr)]
    p99_values = []
    for path in audit_paths:
        with rasterio.open(path) as source:
            p99_values.append(float(np.nanpercentile(source.read([1, 2, 3], out_dtype="float32"), 99)))
    global_p99 = max(p99_values)
    REFLECTANCE_DIVISOR = 1.0 if global_p99 <= 1.5 else 10000.0 if global_p99 <= 12000 else None
    if REFLECTANCE_DIVISOR is None:
        raise RuntimeError(f"Unsupported OLI2MSI value range: p99={global_p99}")
    print("Dataset-level reflectance divisor:", REFLECTANCE_DIVISOR)

    def read_rgb(path, expected_size):
        with rasterio.open(path) as source:
            if source.count < 3 or (source.height, source.width) != (expected_size, expected_size):
                raise ValueError(f"Unexpected raster geometry: {path} bands={source.count} shape={(source.height, source.width)}")
            image = source.read([1, 2, 3], out_dtype="float32") / REFLECTANCE_DIVISOR
            valid = (source.read_masks([1, 2, 3]) > 0).all(0) & np.isfinite(image).all(0)
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = np.clip(image, 0, BENCHMARK_REFLECTANCE_MAX) / BENCHMARK_REFLECTANCE_MAX
        if QUANTIZE_TO_UINT8_GRID:
            image = np.round(image * 255.0) / 255.0
        return image, valid

    selected_keys = {f"{split}:{key}" for split, items in selected.items() for key, _, _ in items}
    loaded = load_manifest(MANIFEST) if MANIFEST.exists() else []
    records_by_key = {record.source_product: record for record in loaded
                      if record.source_product in selected_keys and record.source == DATA_PROTOCOL_ID
                      and Path(record.patch).exists()}

    def quarantine(payload):
        with QUARANTINE_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\\n")

    status = Counter()
    for split, items in selected.items():
        for key, lr_path, hr_path in tqdm(items, desc=f"prepare {split}", unit="pair"):
            record_key = f"{split}:{key}"
            if record_key in records_by_key:
                status["skipped"] += 1
                continue
            try:
                lr, valid_lr = read_rgb(lr_path, LR_FRAME_SIZE)
                hr, valid_hr = read_rgb(hr_path, HR_FRAME_SIZE)
                valid_fraction = min(float(valid_lr.mean()), float(valid_hr.mean()))
                if valid_fraction < MINIMUM_VALID_FRACTION:
                    raise ValueError(f"valid_fraction={valid_fraction:.6f}")
                directory = PATCH_ROOT / split
                directory.mkdir(parents=True, exist_ok=True)
                patch = directory / f"{key}.npz"
                temporary = patch.with_suffix(".npz.tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(handle, lr=lr.astype(np.float16), clean_lr=lr.astype(np.float16),
                        hr=hr.astype(np.float16), valid_mask_lr=valid_lr[None].astype(np.uint8),
                        valid_mask_hr=valid_hr[None].astype(np.uint8),
                        degradation=np.array([0.5, 0, 0, 0], dtype=np.float32))
                temporary.replace(patch)
                records_by_key[record_key] = ManifestRecord(
                    patch=str(patch.resolve()), tile_id=key, split=split, row=0, col=0,
                    valid_fraction=valid_fraction, source=DATA_PROTOCOL_ID,
                    license_id="see-official-OLI2MSI-distribution", source_product=record_key,
                    landsat_product=lr_path.name, sentinel_product=hr_path.name,
                    day_gap=0, scale=SCALE,
                )
                status["written"] += 1
            except Exception as error:
                status["quarantined"] += 1
                quarantine({"split": split, "key": key, "lr": str(lr_path), "hr": str(hr_path),
                            "error": f"{type(error).__name__}: {error}"})
            if (status["written"] + status["quarantined"]) % 50 == 0:
                write_manifest(MANIFEST, sorted(records_by_key.values(), key=lambda item: (item.split, item.tile_id)))

    write_manifest(MANIFEST, sorted(records_by_key.values(), key=lambda item: (item.split, item.tile_id)))
    records = load_manifest(MANIFEST)
    if {record.source_product for record in records} != selected_keys:
        raise RuntimeError("Prepared manifest does not exactly match selected source pairs; inspect quarantine log")
    split_counts = Counter(record.split for record in records)
    print("Preparation:", dict(status), "manifest:", dict(split_counts))
    lock_json(DATA_ROOT / "manifest_lock.json", {"sha256": sha256(MANIFEST), "records": len(records)})
    MANIFEST = balanced_manifest(MANIFEST, DATA_ROOT / "balanced_manifest.jsonl", RANDOM_SEED)
    records = load_manifest(MANIFEST)
    print("Frozen balanced manifest:", MANIFEST, sha256(MANIFEST))
    """)

    inspect_md = find(cells, "## 6. Inspect true LR", "markdown")
    set_markdown(cells[inspect_md], "## 5. Inspect an actual full-frame OLI2MSI pair")
    inspect = find(cells, "def inspect_pair", "code")
    set_code(cells[inspect], """
    from geodiff_gan.data import SentinelPatchDataset
    from torch.nn import functional as F

    def inspect_pair(index=0, split="train"):
        dataset = SentinelPatchDataset(MANIFEST, split=split, scale=3, augment=False,
            random_degradation=False, input_mode="paired", output_channels=3)
        sample = dataset[int(index) % len(dataset)]
        lr, hr, mask = sample["lr_rgb"], sample["hr"], sample["valid_mask"][0]
        bicubic = F.interpolate(lr[None], size=hr.shape[-2:], mode="bicubic", align_corners=False)[0].clamp(0, 1)
        panels = [(lr, "Landsat OLI original 30 m\\n160 × 160"),
                  (bicubic, "Bicubic display\\n480 × 480; no new information"),
                  (hr, "Sentinel-2 MSI target 10 m\\n480 × 480")]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for axis, (image, title) in zip(axes, panels):
            axis.imshow(image.clamp(0, 1).permute(1, 2, 0), interpolation="nearest")
            axis.set_title(title); axis.axis("off")
        fig.suptitle(f"{split} | {sample['tile_id']} | valid={float(mask.mean()):.4f}")
        fig.tight_layout(); fig.savefig(FIGURE_ROOT / f"pair_{split}_{index}.png", dpi=170); plt.show()
        print({"lr_shape": tuple(lr.shape), "hr_shape": tuple(hr.shape),
               "bicubic_l1": float(((bicubic - hr).abs().mean(0) * mask[0]).sum() / mask[0].sum().clamp_min(1))})
        return sample

    inspect_pair(0, "train")
    """)

    audit_md = find(cells, "## 7. Validation-only reconstruction ceiling audit", "markdown")
    original_ceiling_code = copy.deepcopy(cells[audit_md + 1])
    set_markdown(cells[audit_md], """
    ## 6. Audit the OLI2MSI metric protocol

    Raw reflectance occupies approximately `0..0.3`. Computing PSNR with data range
    one before applying the official stretch inflates PSNR by about 10.46 dB. This
    audit reports raw-range, official float and official 8-bit-grid bicubic PSNR.
    Only the official protocol is used for model training and evaluation.
    """)
    set_code(cells[audit_md + 1], """
    def raw_rgb(path, size):
        with rasterio.open(path) as source:
            image = source.read([1, 2, 3], out_dtype="float32") / REFLECTANCE_DIVISOR
            valid = (source.read_masks([1, 2, 3]) > 0).all(0)
        return torch.from_numpy(np.nan_to_num(image)).float(), torch.from_numpy(valid[None]).float()

    def protocol_psnr(lr, hr, lr_mask, hr_mask, stretch, quantize):
        if stretch:
            lr = lr.clamp(0, BENCHMARK_REFLECTANCE_MAX) / BENCHMARK_REFLECTANCE_MAX
            hr = hr.clamp(0, BENCHMARK_REFLECTANCE_MAX) / BENCHMARK_REFLECTANCE_MAX
        else:
            lr, hr = lr.clamp(0, 1), hr.clamp(0, 1)
        if quantize:
            lr, hr = lr.mul(255).round().div(255), hr.mul(255).round().div(255)
        prediction = F.interpolate(lr[None], size=hr.shape[-2:], mode="bicubic", align_corners=False)[0].clamp(0, 1)
        mask = hr_mask * F.interpolate(lr_mask[None], size=hr.shape[-2:], mode="nearest")[0]
        expanded = mask.expand_as(hr)
        mse = ((prediction - hr).square() * expanded).sum() / expanded.sum().clamp_min(1)
        return float(-10 * torch.log10(mse.clamp_min(1e-12)))

    rows = []
    for key, lr_path, hr_path in tqdm(test_pairs[:100], desc="protocol audit", unit="pair"):
        lr, lr_mask = raw_rgb(lr_path, LR_FRAME_SIZE)
        hr, hr_mask = raw_rgb(hr_path, HR_FRAME_SIZE)
        rows.append({"source": key,
            "raw_reflectance_range1": protocol_psnr(lr, hr, lr_mask, hr_mask, False, False),
            "official_clip03_float": protocol_psnr(lr, hr, lr_mask, hr_mask, True, False),
            "official_clip03_uint8_grid": protocol_psnr(lr, hr, lr_mask, hr_mask, True, True)})
    protocol_audit = pd.DataFrame(rows)
    protocol_audit.to_csv(SUITE_ROOT / "protocol_audit.csv", index=False)
    display(protocol_audit.describe().T[["mean", "std", "min", "max"]])
    print("Expected range-only inflation:", 20 * np.log10(1 / BENCHMARK_REFLECTANCE_MAX), "dB")
    """)
    cells[audit_md + 2:audit_md + 2] = [
        markdown("""
        ## 7. Validation-only alignment and radiometric ceiling audit

        The shift and affine values below are target-assisted diagnostics, not model
        outputs. They identify whether registration or radiometry limits achievable
        PSNR and must never be reported as inference performance.
        """),
        original_ceiling_code,
    ]

    registry = find(cells, "def save_state():", "code")
    registry_source = value(cells[registry])
    registry_source = registry_source.replace(
        'validation_limit=None if not FAST_DEV_RUN else 2,',
        'validation_limit=RESIDUAL_VALIDATION_LIMIT,',
    )
    old_limit = (
        '    configured = config["training"].get("validation_limit")\n'
        '    tile_count = len({r.tile_id for r in records})\n'
        '    if configured is not None:\n'
        '        config["training"]["validation_limit"] = max(int(configured), tile_count)'
    )
    new_limit = (
        '    configured = config["training"].get("validation_limit")\n'
        '    validation_count = sum(record.split == "val" for record in records)\n'
        '    if configured is not None:\n'
        '        config["training"]["validation_limit"] = min(int(configured), validation_count)'
    )
    if old_limit not in registry_source:
        raise RuntimeError("Could not replace tile-specific validation rule")
    registry_source = registry_source.replace(old_limit, new_limit)
    registry_source = registry_source.replace(
        "batch_size=2, gradient_accumulation=8, weight_decay=0.0,",
        "batch_size=2, gradient_accumulation=2, weight_decay=0.0,",
    )
    registry_source = registry_source.replace(
        "early_stopping_patience=6, early_stopping_min_epochs=8,",
        "early_stopping_patience=6, early_stopping_min_epochs=8,\n"
        "            progress_mode=\"compact\", progress_updates_per_epoch=1,\n"
        "            router_warmup_epochs=5,",
    )
    registry_source = registry_source.replace(
        'psnr_first_base=True,',
        'psnr_first_base=True, dataset_protocol=DATA_PROTOCOL_ID, '
        'full_frame_geometry=[LR_FRAME_SIZE, HR_FRAME_SIZE],',
    )
    set_code(cells[registry], registry_source)

    base_train = find(cells, "phase_parent = None", "code")
    set_code(cells[base_train], """
    phase_parent = None
    best_phase = None
    phase_outputs = []
    for phase_index, phase in enumerate(BASE_PHASES):
        phase_root = EXPERIMENT_ROOT / f"{prefix}_strong_base_{phase['name']}"
        phase_config = configure_psnr_base(
            make_config("residual_base", phase_root, crop_size=phase["crop"])
        )
        phase_config["training"]["validation_limit"] = BASE_VALIDATION_LIMIT
        config_path, checkpoint = run_stage(
            REPOSITORY_DIR, phase_config, phase_root, "base", parent=phase_parent,
            epochs=phase["epochs"], minutes=phase["minutes"], fast=FAST_DEV_RUN,
            learning_rate=phase["learning_rate"],
            max_batches_per_epoch=phase["max_batches"],
            minimum_optimizer_steps=phase["minimum_steps"],
            validate_every=VALIDATE_EVERY["base"],
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metrics = payload.get("extra", {}).get("metrics", {})
        if "val_psnr" not in metrics:
            raise RuntimeError(f"Base checkpoint lacks val_psnr: {checkpoint}")
        val_psnr = float(metrics["val_psnr"])
        name = f"{prefix}_{phase['name']}_residual_base"
        register(name, phase_root, config_path, checkpoint)
        row = {"phase": phase["name"], "crop": phase["crop"],
               "validation_psnr": val_psnr, "checkpoint": str(checkpoint),
               "sha256": sha256(checkpoint), "result_name": name}
        phase_outputs.append(row)
        if best_phase is None or val_psnr > best_phase["validation_psnr"]:
            best_phase = row
            phase_parent = checkpoint
            print("New best base phase:", phase["name"], val_psnr)
        else:
            print("Phase did not beat its parent; retaining", best_phase["phase"],
                  best_phase["validation_psnr"], "instead of", phase["name"], val_psnr)
        del payload

    selected = RESULTS.pop(best_phase["result_name"])
    BASE_ROOT = Path(selected["root"])
    BASE_CONFIG_PATH = Path(selected["config"])
    BASE_CHECKPOINT = Path(selected["checkpoint"])
    register(f"{prefix}_residual_base", BASE_ROOT, BASE_CONFIG_PATH, BASE_CHECKPOINT)
    (SUITE_ROOT / "strong_base_phases.json").write_text(json.dumps(phase_outputs, indent=2))
    display(pd.DataFrame(phase_outputs))
    print("Validation-selected shared strong base:", best_phase["phase"], BASE_CHECKPOINT)
    """)

    residual_train = find(cells, "def run_new(profile):", "code")
    residual_source = value(cells[residual_train])
    residual_source = residual_source.replace(
        'parent=BASE_CHECKPOINT, epochs=EPOCHS["vae"], minutes=SHARED_MINUTES["vae"], fast=FAST_DEV_RUN)',
        'parent=BASE_CHECKPOINT, epochs=EPOCHS["vae"], minutes=SHARED_MINUTES["vae"], '
        'fast=FAST_DEV_RUN, minimum_optimizer_steps=MINIMUM_OPTIMIZER_STEPS["vae"], '
        'validate_every=VALIDATE_EVERY["vae"])',
    )
    residual_source = residual_source.replace(
        'epochs=EPOCHS[stage], minutes=PER_EXPERIMENT_MINUTES[stage], fast=FAST_DEV_RUN)',
        'epochs=EPOCHS[stage], minutes=PER_EXPERIMENT_MINUTES[stage], '
        'fast=FAST_DEV_RUN, minimum_optimizer_steps=MINIMUM_OPTIMIZER_STEPS[stage], '
        'validate_every=VALIDATE_EVERY[stage])',
    )
    set_code(cells[residual_train], residual_source)

    evaluate = find(cells, "def evaluate_result", "code")
    evaluate_source = value(cells[evaluate])
    evaluate_source = evaluate_source.replace(
        '''    if split == "val" and limit is not None:
        limit = max(limit, len({r.tile_id for r in records}))
''',
        "",
    )
    evaluate_source = evaluate_source.replace(
        'validation_table = pd.DataFrame([evaluate_result(name, "val") for name in RESULTS])',
        '''def append_bicubic(table, split):
    canonical = RESULTS[f"{prefix}_residual_base"]
    path = Path(canonical["root"]) / "evaluation" / split / "baselines.json"
    payload = json.loads(path.read_text())
    row = {"experiment": "bicubic", "count": payload.get("count"), **payload["bicubic"]}
    return pd.concat([pd.DataFrame([row]), table], ignore_index=True, sort=False)

validation_table = append_bicubic(
    pd.DataFrame([evaluate_result(name, "val") for name in RESULTS]), "val"
)''',
    )
    set_code(cells[evaluate], evaluate_source)

    routing = find(cells, "history_rows = []", "code")
    set_code(cells[routing], """
    history_rows = []
    from geodiff_gan.training.checkpoint import latest_stage_checkpoint
    for name, item in RESULTS.items():
        root = Path(item["root"])
        path = root / "evaluation/val/model/per_patch_metrics.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        routed = [row for row in rows if row.get("expert_weights")]
        if routed:
            weights = np.asarray([row["expert_weights"] for row in routed], dtype=np.float64)
            acceptance = np.asarray([row["router_acceptance"] for row in routed])
            entropy = -(weights * np.log(np.clip(weights, 1e-12, 1))).sum(1)
            entropy /= np.log(weights.shape[1]) if weights.shape[1] > 1 else 1
            fig, axes = plt.subplots(1, 2, figsize=(10, 3))
            axes[0].bar(np.arange(weights.shape[1]), weights.mean(0))
            axes[0].set(xlabel="Expert", ylabel="Mean routing weight",
                        title=name + " (final timestep)", ylim=(0, 1))
            axes[1].hist(acceptance, bins=15, range=(0, 1))
            axes[1].set(xlabel="Residual acceptance", ylabel="Images")
            fig.tight_layout(); fig.savefig(FIGURE_ROOT / f"{name}_routing.png", dpi=150); plt.show()
            effective = (weights >= 0.05).mean(0)
            print("Effective usage (weight >= 0.05):", effective.round(3).tolist())
            print("Mean normalized routing entropy:", round(float(entropy.mean()), 4))
        for done in root.glob("runs/*/completed.json"):
            stage = done.parent.name
            saved = json.loads(done.read_text())
            payload = torch.load(saved["checkpoint"], map_location="cpu", weights_only=False)
            last = torch.load(latest_stage_checkpoint(done.parent, stage), map_location="cpu", weights_only=False)
            extra = last.get("extra", {})
            history_rows.append({
                "experiment": name,
                "stage": stage,
                "best_epoch": payload["epoch"] + 1,
                "completed_epochs": last["epoch"] + 1,
                "batches": extra.get("total_train_batches"),
                "optimizer_step_attempts": extra.get("total_optimizer_step_attempts"),
                "minimum_optimizer_steps": extra.get("minimum_optimizer_steps", 0),
                "minimum_steps_met": extra.get("minimum_optimizer_steps_met", True),
                "stage_minutes": extra.get("stage_elapsed_seconds", 0) / 60,
                "best_metrics": payload["extra"]["metrics"],
            })
            del payload, last
    (SUITE_ROOT / "training_selection.json").write_text(json.dumps(history_rows, indent=2))
    display(pd.DataFrame(history_rows).drop(columns=["best_metrics"], errors="ignore"))
    print("Training loss/usage histories:", list(EXPERIMENT_ROOT.glob("*/runs/*/training_history.jsonl")))
    """)

    routing = find(cells, "history_rows = []", "code")
    set_code(cells[routing], """
    history_rows = []
    from geodiff_gan.training.checkpoint import latest_stage_checkpoint
    for name, item in RESULTS.items():
        root = Path(item["root"])
        path = root / "evaluation/val/model/per_patch_metrics.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        routed = [row for row in rows if row.get("expert_weights")]
        if routed:
            weights = np.asarray([row["expert_weights"] for row in routed], dtype=np.float64)
            acceptance = np.asarray([row["router_acceptance"] for row in routed])
            entropy = -(weights * np.log(np.clip(weights, 1e-12, 1))).sum(1)
            entropy /= np.log(weights.shape[1]) if weights.shape[1] > 1 else 1
            fig, axes = plt.subplots(1, 2, figsize=(10, 3))
            axes[0].bar(np.arange(weights.shape[1]), weights.mean(0))
            axes[0].set(xlabel="Expert", ylabel="Mean routing weight",
                        title=name + " (final timestep)", ylim=(0, 1))
            axes[1].hist(acceptance, bins=15, range=(0, 1))
            axes[1].set(xlabel="Residual acceptance", ylabel="Images")
            fig.tight_layout(); fig.savefig(FIGURE_ROOT / f"{name}_routing.png", dpi=150); plt.show()
            effective = (weights >= 0.05).mean(0)
            print("Effective usage (weight >= 0.05):", effective.round(3).tolist())
            print("Mean normalized routing entropy:", round(float(entropy.mean()), 4))
        for done in root.glob("runs/*/completed.json"):
            stage = done.parent.name
            saved = json.loads(done.read_text())
            payload = torch.load(saved["checkpoint"], map_location="cpu", weights_only=False)
            last = torch.load(latest_stage_checkpoint(done.parent, stage), map_location="cpu", weights_only=False)
            extra = last.get("extra", {})
            history_rows.append({
                "experiment": name,
                "stage": stage,
                "best_epoch": payload["epoch"] + 1,
                "completed_epochs": last["epoch"] + 1,
                "batches": extra.get("total_train_batches"),
                "optimizer_step_attempts": extra.get("total_optimizer_step_attempts"),
                "minimum_optimizer_steps": extra.get("minimum_optimizer_steps", 0),
                "minimum_steps_met": extra.get("minimum_optimizer_steps_met", True),
                "stage_minutes": extra.get("stage_elapsed_seconds", 0) / 60,
                "best_metrics": payload["extra"]["metrics"],
            })
            del payload, last
    (SUITE_ROOT / "training_selection.json").write_text(json.dumps(history_rows, indent=2))
    display(pd.DataFrame(history_rows).drop(columns=["best_metrics"], errors="ignore"))
    print("Training loss/usage histories:", list(EXPERIMENT_ROOT.glob("*/runs/*/training_history.jsonl")))
    """)

    test_cell = find(cells, "test_table = pd.DataFrame", "code")
    test_source = value(cells[test_cell]).replace(
        'test_table = pd.DataFrame([evaluate_result(name, "test") for name in RESULTS])',
        'test_table = append_bicubic(\n'
        '        pd.DataFrame([evaluate_result(name, "test") for name in RESULTS]), "test"\n'
        '    )',
    )
    set_code(cells[test_cell], test_source)

    for cell in cells:
        if cell["cell_type"] == "markdown":
            source = value(cell)
            source = source.replace("real Landsat/Sentinel", "official OLI2MSI")
            source = source.replace("full 128->384 patches", "full 160→480 frames")
            source = source.replace("full 128->384", "full 160→480")
            source = source.replace("## 1. Controls (set before training)", "## Before you run")
            source = source.replace(
                "For the time-limited screening run, validation is capped at 64 tile-interleaved patches.",
                "For the time-limited screening run, validation uses a fixed source-ID subset."
            )
            source = source.replace("tile/split filters", "source/split filters")
            cell["source"] = source.splitlines(True)
        elif "Patch bootstrap is descriptive" in value(cell):
            source = value(cell).replace(
                "Patch bootstrap is descriptive: adjacent/overlapping patches are correlated. Use tile-level resampling for geographic claims.",
                "Pair bootstrap is descriptive. Report the fixed official test set and add scene/source-grouped uncertainty when source dependencies are known.",
            )
            set_code(cell, source)
        elif "equal geographic footprint" in value(cell):
            source = value(cell).replace(
                "| equal geographic footprint",
                "| full official frame",
            )
            set_code(cell, source)
        elif "VIEW_REPOSITORY =" in value(cell):
            source = value(cell).replace(
                'Path("/kaggle/working/geodiff-3x-continued-strong-base")',
                'Path("/kaggle/working/geodiff-3x-continued-oli2msi")',
            ).replace(
                'Path("/kaggle/working/geodiff-tiles-strong-base-v2")',
                'Path("/kaggle/working/geodiff-oli2msi-moe-v2")',
            )
            set_code(cell, source)

    # The OLI2MSI study is intentionally limited to the shared strong base and
    # three routed-residual controls. Tile-specific historical profiles would add
    # cost and the multispectral variants have no corresponding OLI2MSI inputs.
    cells = [
        cell for cell in cells
        if not any(
            marker in value(cell)
            for marker in (
                "Optional earlier profile:",
                'run_earlier("rgb_standard")',
                'run_earlier("rgb_fidelity")',
                'run_earlier("multispectral_fidelity")',
                'run_earlier("rgb_harmonized_fidelity")',
                'run_earlier("multispectral_guided_fidelity")',
            )
        )
    ]
    heading_replacements = {
        "## Experiment B: one-expert control": "## 11A. One-expert residual control",
        "## Experiment C: generic MoE": "## 11B. Generic routed experts",
        "## Experiment D: reliability-supervised MoE": "## 11C. Reliability-supervised experts",
        "same full patches": "same full frames",
    }
    for cell in cells:
        if cell["cell_type"] == "markdown":
            source = value(cell)
            for old, new in heading_replacements.items():
                source = source.replace(old, new)
            cell["source"] = source.splitlines(True)

    # A full publication run must never inherit tile-preparation variables.
    forbidden = ("SENTINEL_INPUT", "LANDSAT_INPUT", "MAX_DAY_GAP", "prepare_landsat_sentinel")
    joined = "\n".join(value(cell) for cell in cells)
    if any(token in joined for token in forbidden):
        raise RuntimeError("Tile-specific preparation leaked into the OLI2MSI notebook")

    for index, cell in enumerate(cells):
        cell["id"] = f"oli2msi-strong-{index:03d}"
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
            ast.parse(value(cell))
    notebook["cells"] = cells
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")

    files = list((ROOT / "src").rglob("*.py")) + list((ROOT / "configs").glob("*.yaml"))
    files += [ROOT / "pyproject.toml", ROOT / "README.md"]
    with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(OUTPUT)
    print(ARCHIVE, "sha256=" + hashlib.sha256(ARCHIVE.read_bytes()).hexdigest())


if __name__ == "__main__":
    build()
