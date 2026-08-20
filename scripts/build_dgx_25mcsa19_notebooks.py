from __future__ import annotations

import json
import ast
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
DGX = ROOT / "dgx"
ASSIGNED_ROOT = "/workspace/temp/25mcsa19/working/temp/25mcsa19"
KERNEL_NAME = "geodiff-py311-25mcsa19"
KERNEL_DISPLAY = "GeoDiff-GAN 25MCSA19 Python 3.11"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def write_notebook(path: Path, cells: list[dict]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": KERNEL_DISPLAY,
                "language": "python",
                "name": KERNEL_NAME,
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    print(path)


def normalize_reused_cell(cell: dict) -> dict:
    if cell.get("cell_type") != "code":
        return cell
    source = "".join(cell.get("source", ""))
    try:
        ast.parse(source)
        return {
            **cell,
            "execution_count": None,
            "outputs": [],
        }
    except SyntaxError:
        pass
    lines = source.splitlines(keepends=True)
    if len(lines) <= 1:
        return cell
    normalized = [lines[0]]
    for line in lines[1:]:
        normalized.append(line[8:] if line.startswith(" " * 8) else line)
    return {
        **cell,
        "execution_count": None,
        "outputs": [],
        "source": normalized,
    }


KERNEL_SETUP = rf"""
from pathlib import Path
import json, os, shutil, subprocess, sys

ASSIGNED_ROOT = Path("{ASSIGNED_ROOT}").resolve()
ASSIGNED_ROOT.mkdir(parents=True, exist_ok=True)
VENV = ASSIGNED_ROOT / ".venvs" / "geodiff-py311"
KERNEL_DIR = ASSIGNED_ROOT / "jupyter_kernels" / "{KERNEL_NAME}"

def run(command, check=True, cwd=None):
    command = [str(value) for value in command]
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, check=check, cwd=cwd)

uv = shutil.which("uv")
if uv is None:
    BOOTSTRAP = ASSIGNED_ROOT / ".bootstrap"
    run([sys.executable, "-m", "pip", "install", "--prefix", BOOTSTRAP, "uv"])
    uv = str(BOOTSTRAP / "bin" / "uv")
if not Path(uv).exists() and shutil.which(uv) is None:
    raise RuntimeError(f"uv was not found at {{uv}}. Ask the administrator to provide uv or Python 3.11.")

run([uv, "python", "install", "3.11"])
run([uv, "venv", "--python", "3.11", "--seed", VENV])
PYTHON311 = VENV / "bin" / "python"
run([PYTHON311, "-m", "ensurepip", "--upgrade"], check=False)
run([PYTHON311, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel", "ipykernel"])

KERNEL_DIR.mkdir(parents=True, exist_ok=True)
(KERNEL_DIR / "kernel.json").write_text(json.dumps({{
    "argv": [str(PYTHON311), "-m", "ipykernel_launcher", "-f", "{{connection_file}}"],
    "display_name": "{KERNEL_DISPLAY}",
    "language": "python",
    "env": {{
        "PYTHONUNBUFFERED": "1",
        "XDG_CACHE_HOME": str(ASSIGNED_ROOT / ".cache"),
        "HF_HOME": str(ASSIGNED_ROOT / ".cache" / "huggingface"),
        "TORCH_HOME": str(ASSIGNED_ROOT / ".cache" / "torch"),
        "KAGGLE_CONFIG_DIR": str(ASSIGNED_ROOT / "secrets" / "kaggle"),
        "KAGGLEHUB_CACHE": str(ASSIGNED_ROOT / "downloads" / "kagglehub_cache")
    }}
}}, indent=2), encoding="utf-8")

print("Created Python:", PYTHON311)
print("Created local kernelspec:", KERNEL_DIR)
print("No files outside the assigned root were modified by default.")
print()
print("If this kernel is not visible in Jupyter, run this optional command manually:")
print(f"{{PYTHON311}} -m ipykernel install --user --name {KERNEL_NAME} --display-name '{KERNEL_DISPLAY}'")
print("Then switch Kernel ->", "{KERNEL_DISPLAY}", "and rerun from the runtime cell.")
"""


RUNTIME = rf"""
from pathlib import Path
import json, os, shutil, subprocess, sys, time

ASSIGNED_ROOT = Path("{ASSIGNED_ROOT}").resolve()
if not str(ASSIGNED_ROOT).startswith("{ASSIGNED_ROOT}"):
    raise RuntimeError(f"Unsafe root: {{ASSIGNED_ROOT}}")
ASSIGNED_ROOT.mkdir(parents=True, exist_ok=True)

THESIS_ROOT = ASSIGNED_ROOT
REPOSITORY_DIR = THESIS_ROOT / "geodiff-gan"
DOWNLOAD_ROOT = THESIS_ROOT / "downloads"
DATASET_ROOT = THESIS_ROOT / "datasets" / "sentinel2-bharat"
WORK_ROOT = THESIS_ROOT / "geodiff-output"
SOURCE_ROOT = THESIS_ROOT / "sota_sources"
BACKUP_ROOT = THESIS_ROOT / "backups"
REPOSITORY_URL = "https://github.com/shashankjs2002/SI-SR-1.git"
KAGGLE_DATASET = "twilight2002/sentinel2-bharat"

for path in (DOWNLOAD_ROOT, DATASET_ROOT, WORK_ROOT, SOURCE_ROOT, BACKUP_ROOT):
    path.mkdir(parents=True, exist_ok=True)

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["XDG_CACHE_HOME"] = str(THESIS_ROOT / ".cache")
os.environ["HF_HOME"] = str(THESIS_ROOT / ".cache" / "huggingface")
os.environ["TRANSFORMERS_CACHE"] = str(THESIS_ROOT / ".cache" / "huggingface")
os.environ["TORCH_HOME"] = str(THESIS_ROOT / ".cache" / "torch")
os.environ["KAGGLE_CONFIG_DIR"] = str(THESIS_ROOT / "secrets" / "kaggle")
os.environ["KAGGLEHUB_CACHE"] = str(DOWNLOAD_ROOT / "kagglehub_cache")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def assert_inside(path):
    path = Path(path).resolve()
    if not str(path).startswith(str(THESIS_ROOT)):
        raise RuntimeError(f"Refusing to touch path outside assigned root: {{path}}")
    return path

def run(command, cwd=None, env=None, check=True):
    command = [str(value) for value in command]
    environment = os.environ.copy()
    if env:
        environment.update(env)
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=cwd, env=environment, check=check)

PYTHON = Path(sys.executable)
PIP = [PYTHON, "-m", "pip"]
print("Python:", sys.version)
print("Executable:", PYTHON)
print("Assigned root:", THESIS_ROOT)
if sys.version_info < (3, 10):
    raise RuntimeError("Switch to the GeoDiff-GAN 25MCSA19 Python 3.11 kernel before continuing.")
"""


CLONE_INSTALL = """
import os, sys
from pathlib import Path

if REPOSITORY_DIR.exists():
    if (REPOSITORY_DIR / ".git").exists():
        print("Updating existing Git clone:", REPOSITORY_DIR)
        run(["git", "pull", "--ff-only"], cwd=REPOSITORY_DIR, check=False)
    elif (REPOSITORY_DIR / "pyproject.toml").exists() and (REPOSITORY_DIR / "src").exists():
        print("Using existing uploaded source tree:", REPOSITORY_DIR)
    else:
        raise RuntimeError(f"{REPOSITORY_DIR} exists but is not a GeoDiff-GAN source tree.")
else:
    run(["git", "clone", "--depth", "1", REPOSITORY_URL, REPOSITORY_DIR])

def patch_geodiff_source_compatibility():
    # Patch older cloned source trees inside the assigned DGX folder only.
    system_path = REPOSITORY_DIR / "src" / "geodiff_gan" / "models" / "system.py"
    parameters_path = REPOSITORY_DIR / "src" / "geodiff_gan" / "parameters.py"
    for path in (system_path, parameters_path):
        assert_inside(path)
        if not path.exists():
            raise FileNotFoundError(path)

    system_text = system_path.read_text(encoding="utf-8")
    changed = False
    if "self.output_channels = output_channels" not in system_text:
        if "self.input_channels = input_channels\\n" in system_text:
            system_text = system_text.replace(
                "        self.input_channels = input_channels\\n",
                "        self.input_channels = input_channels\\n"
                "        self.output_channels = output_channels\\n",
                1,
            )
        else:
            system_text = system_text.replace(
                "        self.scale = scale\\n",
                "        self.scale = scale\\n"
                "        self.input_channels = input_channels\\n"
                "        self.output_channels = output_channels\\n",
                1,
            )
        changed = True
    if changed:
        system_path.write_text(system_text, encoding="utf-8")
        print("Patched GeoDiffGAN input/output channel attributes:", system_path)

    parameters_text = parameters_path.read_text(encoding="utf-8")
    if 'model_channels = config.get("model", config)' not in parameters_text:
        needle = '    model = GeoDiffGAN.from_config(config)\\n'
        replacement = (
            '    model = GeoDiffGAN.from_config(config)\\n'
            '    model_channels = config.get("model", config)\\n'
            '    output_channels = int(\\n'
            '        getattr(model, "output_channels", model_channels.get("output_channels", 3))\\n'
            '    )\\n'
        )
        parameters_text = parameters_text.replace(needle, replacement, 1)
        parameters_text = parameters_text.replace(
            '        output_channels=model.output_channels,\\n',
            '        output_channels=output_channels,\\n',
        )
        parameters_text = parameters_text.replace(
            '        condition_channels=model.output_channels,\\n',
            '        condition_channels=output_channels,\\n',
        )
        parameters_path.write_text(parameters_text, encoding="utf-8")
        print("Patched parameter-report output channel fallback:", parameters_path)

patch_geodiff_source_compatibility()

run([*PIP, "install", "--upgrade", "pip", "setuptools", "wheel"])
run([*PIP, "install", "numpy>=1.26", "Pillow>=10", "PyYAML>=6", "tqdm>=4.66", "rasterio>=1.3", "pandas>=2", "matplotlib>=3.8", "kaggle", "kagglehub", "ipykernel"])
run([*PIP, "install", "-e", ".", "--no-deps"], cwd=REPOSITORY_DIR)
sys.path.insert(0, str(REPOSITORY_DIR / "src"))
os.chdir(REPOSITORY_DIR)
run(["git", "log", "-1", "--oneline"], cwd=REPOSITORY_DIR, check=False)
"""


GPU_CHECK = """
import torch

print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available(), "visible GPUs:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable. Ask the administrator to check the DGX Jupyter container.")
if torch.cuda.device_count() != 1:
    raise RuntimeError("Exactly one GPU must be visible. Coordinate with other users and set CUDA_VISIBLE_DEVICES before starting.")
props = torch.cuda.get_device_properties(0)
print("GPU:", props.name, "VRAM GiB:", round(props.total_memory / 1024**3, 2))
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
"""


KAGGLE_DOWNLOAD = """
from getpass import getpass
import json, os, zipfile

KAGGLE_DIR = THESIS_ROOT / "secrets" / "kaggle"
KAGGLE_JSON = KAGGLE_DIR / "kaggle.json"
KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["KAGGLE_CONFIG_DIR"] = str(KAGGLE_DIR)

if not KAGGLE_JSON.exists():
    username = input("Kaggle username: ").strip()
    key = getpass("Kaggle API key: ").strip()
    if not username or not key:
        raise RuntimeError("Kaggle credentials not provided")
    KAGGLE_JSON.write_text(json.dumps({"username": username, "key": key}), encoding="utf-8")
    KAGGLE_JSON.chmod(0o600)

DOWNLOAD_DATASET = True
DOWNLOAD_METHOD = "kaggle_cli"  # kaggle_cli | kagglehub

if DOWNLOAD_DATASET and not list(DATASET_ROOT.rglob("*.SAFE")):
    if DOWNLOAD_METHOD == "kaggle_cli":
        run([PYTHON, "-m", "kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", DOWNLOAD_ROOT])
        archives = sorted(DOWNLOAD_ROOT.glob("*.zip"))
        if not archives:
            raise RuntimeError("Kaggle CLI did not produce a zip archive")
        archive = archives[-1]
        with zipfile.ZipFile(archive) as handle:
            bad = handle.testzip()
            if bad is not None:
                raise RuntimeError(f"Corrupt zip member: {bad}")
            handle.extractall(DATASET_ROOT)
        print("Extracted:", archive, "->", DATASET_ROOT)
    elif DOWNLOAD_METHOD == "kagglehub":
        import kagglehub
        path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
        print("kagglehub path:", path)
        DATASET_ROOT = path
    else:
        raise ValueError(DOWNLOAD_METHOD)
else:
    print("Reusing dataset:", DATASET_ROOT)

safe_count = len(list(DATASET_ROOT.rglob("*.SAFE")))
print("SAFE directories:", safe_count)
if safe_count == 0:
    raise RuntimeError(f"No SAFE products found below {DATASET_ROOT}")
"""


MULTISPECTRAL_PREP = """
from collections import Counter
import json
from geodiff_gan.data import sentinel as sentinel_module

PATCH_SIZE = 512
PATCH_STRIDE = 384
MINIMUM_VALID_FRACTION = 0.95
MAX_PRODUCTS = None       # set small number for smoke test
REBUILD_PATCHES = False
PATCH_ROOT = WORK_ROOT / "patches_ms"
RAW_MANIFEST = WORK_ROOT / "manifest_ms_raw.jsonl"
PREPARATION_STATE = WORK_ROOT / "preparation-ms-state.json"

safe_candidates = sorted(DATASET_ROOT.rglob("*.SAFE"))
safe_products = sentinel_module.discover_safe_products(DATASET_ROOT)
print(f"Canonical={len(safe_products)}, scanned={len(safe_candidates)}, wrappers ignored={len(safe_candidates)-len(safe_products)}")
for product in safe_products[:20]:
    print(" -", sentinel_module.source_product_name(product), "->", product)
if not safe_products:
    raise RuntimeError(f"No canonical SAFE product below {DATASET_ROOT}")

command = [
    PYTHON, "-m", "geodiff_gan.cli.prepare_sentinel",
    "--input", DATASET_ROOT,
    "--output", PATCH_ROOT,
    "--manifest", RAW_MANIFEST,
    "--state", PREPARATION_STATE,
    "--patch-size", PATCH_SIZE,
    "--stride", PATCH_STRIDE,
    "--minimum-valid-fraction", MINIMUM_VALID_FRACTION,
    "--multispectral-preset", "rgb-nir-swir",
    "--unmatched-split", "train",
]
if MAX_PRODUCTS is not None:
    command.extend(["--max-products", MAX_PRODUCTS])
if REBUILD_PATCHES:
    command.append("--rebuild")
for historical in (WORK_ROOT / "manifest_ms.before-edge-filter.jsonl", WORK_ROOT / "rejected-ms-edge-patches.jsonl"):
    if historical.exists():
        command.extend(["--completed-manifest", historical])
run(command, cwd=REPOSITORY_DIR)

records = [json.loads(line) for line in RAW_MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
print("Patches:", len(records), "products:", len({r["source_product"] for r in records}), "tiles:", Counter(r["tile_id"] for r in records))
"""


QUARANTINE = """
import json, shutil
import numpy as np
from collections import Counter

MANIFEST = RAW_MANIFEST
BACKUP_MANIFEST = WORK_ROOT / "manifest_ms.before-edge-filter.jsonl"
REJECTED_MANIFEST = WORK_ROOT / "rejected-ms-edge-patches.jsonl"
QUARANTINE_ROOT = WORK_ROOT / "quarantine_ms_edge_patches"
BLACK_THRESHOLD = 1e-6
EDGE_WIDTH = 32
CORNER_SIZE = 64
MAX_EDGE_BLACK_FRACTION = 0.002
MAX_CORNER_BLACK_FRACTION = 0.002
MAX_OVERALL_BLACK_FRACTION = 0.005

active_records = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
BACKUP_MANIFEST.write_text("".join(json.dumps(record) + "\\n" for record in active_records), encoding="utf-8")

def inspect_patch(path):
    path = Path(path)
    if not path.exists():
        return True, {"reason": "missing_patch"}
    try:
        with np.load(path) as data:
            hr = np.asarray(data["hr"], dtype=np.float32)
            if "ms_hr" not in data:
                return True, {"reason": "missing_ms_hr"}
            ms_hr = np.asarray(data["ms_hr"], dtype=np.float32)
    except Exception as error:
        return True, {"reason": f"unreadable:{type(error).__name__}"}
    if hr.shape[0] != 3 or ms_hr.shape[0] != 6:
        return True, {"reason": f"unexpected_shape:hr={hr.shape},ms={ms_hr.shape}"}
    if not np.isfinite(hr).all() or not np.isfinite(ms_hr).all():
        return True, {"reason": "nonfinite"}
    black = np.all(hr <= BLACK_THRESHOLD, axis=0)
    h, w = black.shape
    edge = min(EDGE_WIDTH, h // 2, w // 2)
    corner = min(CORNER_SIZE, h // 2, w // 2)
    edge_pixels = np.concatenate([black[:edge].ravel(), black[-edge:].ravel(), black[edge:-edge, :edge].ravel(), black[edge:-edge, -edge:].ravel()])
    corner_pixels = np.concatenate([black[:corner, :corner].ravel(), black[:corner, -corner:].ravel(), black[-corner:, :corner].ravel(), black[-corner:, -corner:].ravel()])
    stats = {
        "overall_black_fraction": float(black.mean()),
        "edge_black_fraction": float(edge_pixels.mean()),
        "corner_black_fraction": float(corner_pixels.mean()),
    }
    reasons = []
    if stats["overall_black_fraction"] > MAX_OVERALL_BLACK_FRACTION:
        reasons.append("overall_black")
    if stats["edge_black_fraction"] > MAX_EDGE_BLACK_FRACTION:
        reasons.append("edge_black")
    if stats["corner_black_fraction"] > MAX_CORNER_BLACK_FRACTION:
        reasons.append("corner_black")
    stats["reason"] = ",".join(reasons) if reasons else "accepted"
    return bool(reasons), stats

kept, rejected = [], []
for index, record in enumerate(active_records, start=1):
    reject, stats = inspect_patch(record["patch"])
    if reject:
        rejected_record = dict(record)
        rejected_record["original_patch"] = record["patch"]
        rejected_record["filter"] = stats
        source = Path(record["patch"])
        if source.exists():
            destination = QUARANTINE_ROOT / record["tile_id"] / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != destination.resolve():
                shutil.move(str(source), str(destination))
            rejected_record["patch"] = str(destination)
        rejected.append(rejected_record)
    else:
        kept.append(record)
    if index % 200 == 0 or index == len(active_records):
        print(f"Scanned {index}/{len(active_records)}")

MANIFEST.write_text("".join(json.dumps(record) + "\\n" for record in kept), encoding="utf-8")
REJECTED_MANIFEST.write_text("".join(json.dumps(record) + "\\n" for record in rejected), encoding="utf-8")
ACCEPTED_MANIFEST = MANIFEST
records = kept
print("Accepted:", len(kept), "Rejected:", len(rejected), "splits:", Counter(r["split"] for r in kept))
if not kept:
    raise RuntimeError("All patches were rejected.")
"""


SPLIT = """
from collections import Counter, defaultdict
import hashlib, json, random

SPLIT_SEED = 20260708
SPLIT_MODE = "spatial_blocks"  # spatial_blocks | patch_random
SPATIAL_BLOCK_PIXELS = 2048
FINAL_MANIFEST = WORK_ROOT / "manifest_ms_80_10_10.jsonl"

def seed_for(tile):
    return int.from_bytes(hashlib.sha256(f"{SPLIT_SEED}:{tile}".encode()).digest()[:8], "little")

def assign(group):
    tile = group[0]["tile_id"]
    if SPLIT_MODE == "patch_random":
        random.Random(seed_for(tile)).shuffle(group)
        n = len(group)
        n_test = max(1, round(0.1 * n))
        n_val = max(1, round(0.1 * n))
        for i, record in enumerate(group):
            record["split"] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
        return group
    blocks = defaultdict(list)
    for record in group:
        blocks[(record["row"] // SPATIAL_BLOCK_PIXELS, record["col"] // SPATIAL_BLOCK_PIXELS)].append(record)
    keys = list(blocks)
    random.Random(seed_for(tile)).shuffle(keys)
    targets = {"test": 0.1 * len(group), "val": 0.1 * len(group)}
    chosen = {}
    for split in ("test", "val"):
        count = 0
        while keys and (count < targets[split] or count == 0):
            key = keys.pop()
            chosen[key] = split
            count += len(blocks[key])
    for key in keys:
        chosen[key] = "train"
    output = []
    for key, values in blocks.items():
        for record in values:
            record["split"] = chosen[key]
            output.append(record)
    return output

accepted = [json.loads(line) for line in ACCEPTED_MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
by_tile = defaultdict(list)
for record in accepted:
    by_tile[record["tile_id"]].append(dict(record))
records = []
for tile, group in sorted(by_tile.items()):
    records.extend(assign(group))
FINAL_MANIFEST.write_text("".join(json.dumps(record) + "\\n" for record in records), encoding="utf-8")
MANIFEST = FINAL_MANIFEST
counts = Counter(record["split"] for record in records)
print("Manifest:", FINAL_MANIFEST, "overall:", counts)
for tile in sorted(by_tile):
    values = {split: sum(r["tile_id"] == tile and r["split"] == split for r in records) for split in ("train", "val", "test")}
    total = sum(values.values())
    print(tile, values, {key: round(value / total, 3) for key, value in values.items()})
if any(counts[split] == 0 for split in ("train", "val", "test")):
    raise RuntimeError(f"Missing split: {counts}")
"""


VISUALIZE = """
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from geodiff_gan.models.degradation import random_degradation

DEGRADATION_SEVERITY = "mild"
DEGRADATION_SEED = 42
VISUALIZE_ACCEPTED = 5

accepted_records = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
selected = random.Random(42).sample(accepted_records, k=min(VISUALIZE_ACCEPTED, len(accepted_records)))

def chw_to_rgb(tensor):
    return tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()

figure, axes = plt.subplots(len(selected), 5, figsize=(20, 4 * len(selected)), squeeze=False)
for row, record in enumerate(selected):
    with np.load(record["patch"]) as data:
        hr = torch.from_numpy(data["hr"]).float()
        ms_hr = torch.from_numpy(data["ms_hr"]).float()
    generator = torch.Generator().manual_seed(DEGRADATION_SEED + row)
    observed_lr, parameters, clean_lr = random_degradation(hr.unsqueeze(0), scale=4, generator=generator, return_clean=True, severity=DEGRADATION_SEVERITY)
    ms_lr, _, _ = random_degradation(ms_hr.unsqueeze(0), scale=4, generator=torch.Generator().manual_seed(DEGRADATION_SEED + row), return_clean=True, severity=DEGRADATION_SEVERITY)
    bicubic = F.interpolate(observed_lr, size=hr.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)[0]
    nir_lr = ms_lr[0, 3:4].repeat(3, 1, 1)
    panels = [
        (hr, f"HR RGB\\n{record['tile_id']} {record['split']}"),
        (clean_lr[0], "Clean RGB LR"),
        (observed_lr[0], "Observed RGB LR"),
        (nir_lr, "Observed NIR LR"),
        (bicubic, "Bicubic RGB x4"),
    ]
    for column, (image, title) in enumerate(panels):
        axes[row, column].imshow(chw_to_rgb(image))
        axes[row, column].set_title(title, fontsize=10)
        axes[row, column].axis("off")
plt.tight_layout()
plt.show()
"""


BUILD_VARIANT_CONFIGS = """
import copy, yaml
from geodiff_gan.config import load_config
from geodiff_gan.parameters import build_parameter_report

VARIANTS_TO_RUN = ["small_improved", "medium", "small_improved_ms", "medium_ms"]
STAGES_TO_RUN = ["base", "vae", "diffusion", "joint"]
USE_CAPTIONS_FOR_GEODIFF = False
CAPTION_JSONL = WORK_ROOT / "captions_grounded" / "captions_grounded_qwen3vl.jsonl"
AUTO_RESUME_TRAINING = True
VALIDATION_LIMIT = 128

EPOCHS_BY_VARIANT = {
    "small_improved": {"base": 8, "vae": 8, "diffusion": 20, "joint": 8},
    "medium": {"base": 8, "vae": 8, "diffusion": 20, "joint": 8},
    "small_improved_ms": {"base": 8, "vae": 8, "diffusion": 20, "joint": 8},
    "medium_ms": {"base": 8, "vae": 8, "diffusion": 20, "joint": 8},
}
BATCH_BY_VARIANT = {
    "small_improved": {"batch": 4, "accumulation": 4},
    "medium": {"batch": 2, "accumulation": 8},
    "small_improved_ms": {"batch": 4, "accumulation": 4},
    "medium_ms": {"batch": 2, "accumulation": 8},
}

CONFIG_FILES = {
    "small_improved": REPOSITORY_DIR / "configs" / "small_12tile_improved.yaml",
    "medium": REPOSITORY_DIR / "configs" / "medium.yaml",
    "small_improved_ms": REPOSITORY_DIR / "configs" / "small_12tile_improved_multispectral.yaml",
    "medium_ms": REPOSITORY_DIR / "configs" / "medium_multispectral.yaml",
}
DEFAULTS = REPOSITORY_DIR / "configs" / "default.yaml"
CONFIG_ROOT = WORK_ROOT / "configs"
RUN_ROOT = WORK_ROOT / "runs"

def build_config(variant):
    config = load_config(CONFIG_FILES[variant], DEFAULTS)
    config["data"].update({
        "manifest": str(FINAL_MANIFEST),
        "captions": str(CAPTION_JSONL) if USE_CAPTIONS_FOR_GEODIFF and CAPTION_JSONL.exists() else None,
        "caption_field": "caption",
        "caption_sampling": "random" if USE_CAPTIONS_FOR_GEODIFF and CAPTION_JSONL.exists() else "fixed",
        "random_caption_fields": ["brief", "descriptive", "analytical", "positional"],
        "target_key": "hr",
        "train_degradation_sampling": "random",
        "degradation_seed": 42,
        "degradation_severity": "mild",
    })
    if variant.endswith("_ms"):
        config["data"]["condition_key"] = "ms_hr"
        config["model"]["input_channels"] = 6
        config["model"]["output_channels"] = 3
    else:
        config["data"]["condition_key"] = None
        config["model"]["input_channels"] = 3
        config["model"]["output_channels"] = 3
    config["model"]["decoder_upsample_mode"] = "resize_conv"
    config["model"]["use_text_conditioning"] = bool(config["data"]["captions"])
    if not config["model"]["use_text_conditioning"]:
        config["text_encoder"] = {"kind": "hash", "context_dim": config["model"]["context_dim"], "max_tokens": 1}
        config["prompts"] = {"null_probability": 1.0, "paraphrase_probability": 0.0, "mismatch_probability": 0.0}
    config["training"].update({
        "batch_size": BATCH_BY_VARIANT[variant]["batch"],
        "gradient_accumulation": BATCH_BY_VARIANT[variant]["accumulation"],
        "num_workers": 8,
        "amp": True,
        "gradient_checkpointing": True,
        "auto_resume": True,
        "progress_mode": "compact",
        "progress_updates_per_epoch": 4,
        "validate_every": 1,
        "validation_limit": VALIDATION_LIMIT,
        "keep_best_and_latest": True,
        "checkpoint_metric": "val_l1",
        "checkpoint_mode": "min",
        "early_stopping_patience": 4,
        "early_stopping_min_epochs": 4,
        "early_stopping_min_delta": 0.00005,
    })
    config.setdefault("debug", {})
    config["debug"].update({"enabled": False, "print_tensor_stats": False})
    return config

CONFIGS = {name: build_config(name) for name in VARIANTS_TO_RUN}
train_count = sum(record["split"] == "train" for record in records)
for variant, config in CONFIGS.items():
    directory = CONFIG_ROOT / variant
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "template.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    report = build_parameter_report(config, patches=train_count, world_size=1)
    print(variant, "input_channels", config["model"]["input_channels"], "condition_key", config["data"]["condition_key"], "core", f"{report['core_model']['scalar_parameters']:,}", "joint", f"{report['training_stages']['joint']['total_optimized_parameters']:,}")
"""


TRAIN_VARIANTS = """
import copy, yaml

def select_checkpoint(directory, stage):
    best = directory / f"{stage}_best.pt"
    if best.exists():
        return best
    latest = directory / f"{stage}_latest.pt"
    if latest.exists():
        return latest
    candidates = sorted(directory.glob(f"{stage}_epoch_*.pt"))
    if not candidates:
        raise RuntimeError(f"No checkpoint for {stage} in {directory}")
    return candidates[-1]

def train_variant(variant):
    previous = None
    checkpoints = {}
    for stage in STAGES_TO_RUN:
        config = copy.deepcopy(CONFIGS[variant])
        output = RUN_ROOT / variant / stage
        config["training"].update({
            "stage": stage,
            "epochs": EPOCHS_BY_VARIANT[variant][stage],
            "output_dir": str(output),
            "init_checkpoint": str(previous) if previous else None,
            "resume": None,
            "auto_resume": AUTO_RESUME_TRAINING,
        })
        if stage == "joint":
            config["training"]["learning_rate"] = 1e-5
            config["training"]["loss_weights"]["adversarial"] = 0.003
        config_path = CONFIG_ROOT / variant / f"{stage}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(f"\\n===== {variant}: {stage} =====")
        run([PYTHON, "-m", "geodiff_gan.cli.train", "--config", config_path], cwd=REPOSITORY_DIR)
        previous = select_checkpoint(output, stage)
        checkpoints[stage] = previous
        print("Selected:", previous)
    return checkpoints

CHECKPOINTS_BY_VARIANT = {}
for variant in VARIANTS_TO_RUN:
    CHECKPOINTS_BY_VARIANT[variant] = train_variant(variant)
CHECKPOINTS_BY_VARIANT
"""


EVALUATE_VARIANTS = """
import json, pandas as pd

EVALUATION_LIMIT = 40
EVALUATION_SAMPLES = 2
EVALUATION_STEPS = 20
EVAL_ROOT = WORK_ROOT / "evaluation"
EVALUATION_RESULTS = {}
for variant in VARIANTS_TO_RUN:
    config_path = CONFIG_ROOT / variant / "joint.yaml"
    checkpoint = CHECKPOINTS_BY_VARIANT[variant]["joint"]
    output = EVAL_ROOT / variant / "test"
    run([PYTHON, "-m", "geodiff_gan.cli.evaluate", "--config", config_path, "--checkpoint", checkpoint, "--output", output, "--split", "test", "--samples", EVALUATION_SAMPLES, "--steps", EVALUATION_STEPS, "--back-projection-steps", 3, "--mode", "sr", "--limit", EVALUATION_LIMIT, "--device", "cuda", "--progress", "compact", "--no-text"], cwd=REPOSITORY_DIR)
    baseline = EVAL_ROOT / variant / "test_baselines.json"
    run([PYTHON, "-m", "geodiff_gan.cli.baselines", "--config", config_path, "--base-checkpoint", CHECKPOINTS_BY_VARIANT[variant]["base"], "--output", baseline, "--split", "test", "--limit", EVALUATION_LIMIT, "--device", "cuda", "--progress", "compact"], cwd=REPOSITORY_DIR)
    EVALUATION_RESULTS[variant] = {"model": json.loads((output / "metrics.json").read_text()), "baselines": json.loads(baseline.read_text())}

rows = []
for variant, payload in EVALUATION_RESULTS.items():
    rows.append({"variant": variant, "method": "GeoDiff-GAN", **payload["model"]})
    for method, values in payload["baselines"].items():
        if isinstance(values, dict):
            rows.append({"variant": variant, "method": method, **values})
metric_table = pd.DataFrame(rows)
columns = [name for name in ["variant", "method", "count", "l1", "psnr", "ssim", "edge_f1", "redegradation_l1", "observed_lr_noise_to_signal"] if name in metric_table]
display(metric_table[columns].round(6))
EVAL_ROOT.mkdir(parents=True, exist_ok=True)
metric_table.to_csv(EVAL_ROOT / "rgb_vs_multispectral_variant_comparison.csv", index=False)
"""


SIDE_BY_SIDE = """
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from geodiff_gan.data import SentinelPatchDataset

dataset = SentinelPatchDataset(FINAL_MANIFEST, split="test", scale=4, caption_file=None, augment=False, random_degradation=False, degradation_seed=42, degradation_severity="mild")
for index in range(min(4, len(dataset))):
    sample = dataset[index]
    stem = Path(sample["patch"]).stem
    hr = sample["hr"]
    lr = sample["lr_rgb"] if "lr_rgb" in sample else sample["lr"]
    bicubic = F.interpolate(lr[None], size=hr.shape[-2:], mode="bicubic", align_corners=False)[0].clamp(0, 1)
    outputs = {}
    for variant in VARIANTS_TO_RUN:
        result = EVAL_ROOT / variant / "test" / f"{stem}_uncertainty.npz"
        if result.exists():
            with np.load(result) as data:
                outputs[variant] = torch.from_numpy(data["mean"]).float()
    panels = [(F.interpolate(lr[None], size=hr.shape[-2:], mode="nearest")[0], "Input LR"), (bicubic, "Bicubic")] + [(outputs[v], v) for v in VARIANTS_TO_RUN if v in outputs] + [(hr, "Target HR")]
    fig, axes = plt.subplots(2, len(panels), figsize=(4 * len(panels), 8), squeeze=False)
    for column, (image, title) in enumerate(panels):
        axes[0, column].imshow(image.clamp(0, 1).permute(1, 2, 0))
        axes[0, column].set_title(title)
        axes[0, column].axis("off")
        error = (image - hr).abs().mean(0)
        axes[1, column].imshow(error, cmap="turbo", vmin=0, vmax=max(0.05, float(error.quantile(0.99))))
        axes[1, column].set_title(f"error mean={float(error.mean()):.4f}")
        axes[1, column].axis("off")
    fig.suptitle(stem)
    plt.tight_layout()
    plt.show()
"""


BACKUP = """
archive_base = BACKUP_ROOT / f"geodiff_multispectral_variants_{time.strftime('%Y%m%d_%H%M%S')}"
archive_path = shutil.make_archive(str(archive_base), "gztar", root_dir=WORK_ROOT)
print("Backup archive:", archive_path)
print("No dataset or checkpoint files were deleted.")
"""


def build_variants_notebook() -> None:
    cells = [
        md(
            """
            # GeoDiff-GAN DGX 25MCSA19: RGB and Multispectral Variants

            Runs inside `/workspace/temp/25mcsa19/working/temp/25mcsa19` only.
            It prepares Sentinel-2 RGB+NIR+SWIR patches, trains RGB `small_improved` and
            `medium`, and trains multispectral `small_improved_ms` and `medium_ms`.

            Prediction target remains RGB 10 m. Multispectral bands are conditioning only.
            """
        ),
        md("## 0. Create the local Python 3.11 environment"),
        code(KERNEL_SETUP),
        md("## 1. Runtime root guard\n\nSwitch to the Python 3.11 kernel created above, then run from here."),
        code(RUNTIME),
        md("## 2. Clone/update repository and install dependencies"),
        code(CLONE_INSTALL),
        md("## 3. Verify the allocated A100 GPU"),
        code(GPU_CHECK),
        md("## 4. Kaggle credentials and dataset download inside the assigned folder"),
        code(KAGGLE_DOWNLOAD),
        md("## 5. Incremental multispectral Sentinel-2 preprocessing"),
        code(MULTISPECTRAL_PREP),
        md("## 6. Quarantine corrupt/black edge patches"),
        code(QUARANTINE),
        md("## 7. Deterministic 80/10/10 spatial-block split within every tile"),
        code(SPLIT),
        md("## 8. Visualize multispectral preparation samples"),
        code(VISUALIZE),
        md("## 9. Build RGB and multispectral GeoDiff-GAN configs"),
        code(BUILD_VARIANT_CONFIGS),
        md("## 10. Train variants sequentially with automatic resume"),
        code(TRAIN_VARIANTS),
        md("## 11. Evaluate all variants on the same test split"),
        code(EVALUATE_VARIANTS),
        md("## 12. Side-by-side qualitative comparison"),
        code(SIDE_BY_SIDE),
        md("## 13. Backup outputs without deleting anything"),
        code(BACKUP),
    ]
    write_notebook(DGX / "GeoDiff_GAN_DGX_25MCSA19_Multispectral_Variants.ipynb", cells)


def build_caption_notebook() -> None:
    source = DGX / "GeoDiff_GAN_DGX_Grounded_Multispectral_Captioning.ipynb"
    notebook = json.loads(source.read_text(encoding="utf-8"))
    cells = [
        md(
            """
            # GeoDiff-GAN DGX 25MCSA19: Grounded Multispectral Captioning

            Runs only under `/workspace/temp/25mcsa19/working/temp/25mcsa19`.
            Captions are grounded with RGB, NDVI, MNDWI, NDBI and SCL evidence,
            checkpointed to SQLite, and exported as GeoDiff-GAN-compatible JSONL.
            """
        ),
        md("## 0. Create the local Python 3.11 environment"),
        code(KERNEL_SETUP),
        md("## 1. Runtime root guard\n\nSwitch to the Python 3.11 kernel created above, then run from here."),
        code(
            RUNTIME
            + """

DATASET_ROOT = THESIS_ROOT / "datasets" / "sentinel2-bharat"
WORK_ROOT = THESIS_ROOT / "geodiff-output"
manifest_candidates = [
    WORK_ROOT / "manifest_ms_80_10_10.jsonl",
    WORK_ROOT / "manifest_ms_raw.jsonl",
    WORK_ROOT / "manifest_dgx_80_10_10.jsonl",
    WORK_ROOT / "manifest_raw.jsonl",
    WORK_ROOT / "manifest.jsonl",
]
MANIFEST = next((path for path in manifest_candidates if path.exists()), manifest_candidates[0])
CAPTION_ROOT = WORK_ROOT / "captions_grounded"
CAPTION_ROOT.mkdir(parents=True, exist_ok=True)
CAPTION_DB = CAPTION_ROOT / "captions_grounded.sqlite"
CAPTION_JSONL = CAPTION_ROOT / "captions_grounded_qwen3vl.jsonl"
EVIDENCE_JSONL = CAPTION_ROOT / "spectral_evidence.jsonl"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
PREFERRED_CAPTION = "descriptive"
DISPLAY_GAIN = 2.5
MAX_NEW_TOKENS = 320
CHECKPOINT_EVERY = 1
EXPORT_EVERY = 25
CAPTION_SPLIT = "all"
START_INDEX = 0
END_INDEX = None
RETRY_GENERATION = 2
FORCE_RECATION_INDICES = []
USE_4BIT = False
RUN_TEST_CAPTIONS = True
TEST_CAPTION_COUNT = 10
TEST_CAPTION_JSONL = CAPTION_ROOT / "test_captions_10.jsonl"

print("Dataset:", DATASET_ROOT)
print("Manifest:", MANIFEST)
print("Caption DB:", CAPTION_DB)
if not MANIFEST.exists():
    raise FileNotFoundError(f"No manifest found: {manifest_candidates}")
if not DATASET_ROOT.exists():
    raise FileNotFoundError(f"Original SAFE dataset required for grounded evidence: {DATASET_ROOT}")
"""
        ),
    ]
    # Reuse the proven long captioning workflow after its original path cell.
    for cell in notebook["cells"][3:]:
        normalized_cell = normalize_reused_cell(cell)
        cells.append(normalized_cell)
        if normalized_cell["cell_type"] == "code" and "def export_jsonl()" in "".join(normalized_cell.get("source", "")):
            cells.append(md("## 7b. Generate 10 test captions before full captioning"))
            cells.append(
                code(
                    """
                    if RUN_TEST_CAPTIONS:
                        test_records = records[:TEST_CAPTION_COUNT]
                        test_rows = []
                        for record in tqdm(test_records, desc="test grounded captions"):
                            evidence, arrays = extract_evidence(record)
                            panel = build_panel(arrays)
                            raw_response = generate_response(panel, caption_prompt(evidence))
                            parsed = parse_json_object(raw_response)
                            payload, issues = validate_payload(parsed, evidence)
                            if issues:
                                payload = safe_fallback(payload, evidence)
                                payload, issues = validate_payload(payload, evidence)
                            test_rows.append({
                                "patch": record["patch"],
                                "manifest_index": record["_manifest_index"],
                                "tile_id": record.get("tile_id", ""),
                                "split": record.get("split", ""),
                                "caption": top_level_caption(payload),
                                "captions": payload,
                                "issues": issues,
                                "spectral_evidence": evidence,
                                "source_model": MODEL_ID,
                            })
                        TEST_CAPTION_JSONL.write_text(
                            "".join(json.dumps(row, ensure_ascii=True) + "\\n" for row in test_rows),
                            encoding="utf-8",
                        )
                        print("Wrote:", TEST_CAPTION_JSONL)
                        for row in test_rows[:3]:
                            print("\\nindex", row["manifest_index"], row["patch"])
                            print("brief:", row["captions"].get("brief", ""))
                            print("descriptive:", row["captions"].get("descriptive", ""))
                            print("issues:", row["issues"])
                    else:
                        print("RUN_TEST_CAPTIONS=False")
                    """
                )
            )
    write_notebook(DGX / "GeoDiff_GAN_DGX_25MCSA19_Grounded_Multispectral_Captioning.ipynb", cells)


SOTA_NOTEBOOK = rf"""
from pathlib import Path
import json, os, subprocess, sys, time, shutil

{RUNTIME}
"""


SOTA_INSTALL_CLONE = """
from geodiff_gan.benchmark.models import MODEL_SPECS

run([*PIP, "install", "timm>=1.0.15", "einops>=0.8", "pandas>=2", "matplotlib>=3.8"])

SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
REPOSITORIES = {key: spec.repository for key, spec in MODEL_SPECS.items()}
DIRECTORIES = {key: spec.directory for key, spec in MODEL_SPECS.items()}
for key, url in REPOSITORIES.items():
    destination = SOURCE_ROOT / DIRECTORIES[key]
    if destination.exists():
        print("Keeping existing source:", key, destination)
    else:
        run(["git", "clone", "--depth", "1", url, destination])
"""


SOTA_DATASET = """
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from geodiff_gan.data import SentinelPatchDataset

candidate_manifests = [
    WORK_ROOT / "manifest_ms_80_10_10.jsonl",
    WORK_ROOT / "manifest_dgx_80_10_10.jsonl",
    WORK_ROOT / "manifest_ms_raw.jsonl",
    WORK_ROOT / "manifest_raw.jsonl",
    WORK_ROOT / "manifest.jsonl",
]
MANIFEST = next((path for path in candidate_manifests if path.exists()), None)
if MANIFEST is None:
    raise FileNotFoundError("Run the multispectral variant notebook preprocessing first.")
records = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
split_counts = Counter(record["split"] for record in records)
print("Manifest:", MANIFEST)
print("Split counts:", split_counts)
if any(split_counts[name] == 0 for name in ("train", "val", "test")):
    raise RuntimeError(f"Manifest must contain train/val/test records: {split_counts}")

dataset = SentinelPatchDataset(MANIFEST, split="train", scale=4, caption_file=None, augment=False, random_degradation=False, degradation_seed=42, degradation_severity="mild")
sample = dataset[0]
lr = sample["lr_rgb"] if "lr_rgb" in sample else sample["lr"]
hr = sample["hr"]
bicubic = F.interpolate(lr[None], size=hr.shape[-2:], mode="bicubic", align_corners=False)[0].clamp(0, 1)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for axis, image, title in zip(axes, [lr, bicubic, hr], ["LR 128x128", "Bicubic 512x512", "Target 512x512"]):
    axis.imshow(image.clamp(0, 1).permute(1, 2, 0))
    axis.set_title(title)
    axis.axis("off")
plt.tight_layout()
plt.show()
"""


SOTA_PROFILE = """
PROFILE = "screening"  # smoke | screening | paper
ARCHITECTURE_MODE = "official"
LR_CROP = 128          # keeps the task as 128x128 -> 512x512
INCLUDE_MAMBA_MODELS = False
MODELS_TO_RUN = [
    "swinir", "hat", "srformer", "dat", "omnisr",
    "ttst", "mfghmoe", "swin2mose", "atd",
]
if INCLUDE_MAMBA_MODELS:
    MODELS_TO_RUN += ["mambair", "mambairv2", "fremamba"]

PROFILES = {
    "smoke": {"max_updates": 20, "validate_every": 10, "validation_limit": 4, "test_limit": 4, "early_stopping_patience": 2},
    "screening": {"max_updates": 5000, "validate_every": 500, "validation_limit": 32, "test_limit": 40, "early_stopping_patience": 4},
    "paper": {"max_updates": 50000, "validate_every": 2000, "validation_limit": 128, "test_limit": 80, "early_stopping_patience": 6},
}
SETTINGS = PROFILES[PROFILE]
BATCH_SETTINGS = {name: (1, 8) for name in MODELS_TO_RUN}
BATCH_SETTINGS["omnisr"] = (2, 4)
BENCHMARK_ROOT = WORK_ROOT / "sota_benchmark"
print("Models:", MODELS_TO_RUN)
print("Official architecture mode:", ARCHITECTURE_MODE)
print("LR crop:", LR_CROP, "=> output", LR_CROP * 4)
"""


SOTA_OPTIONAL_MAMBA = """
if INCLUDE_MAMBA_MODELS:
    result = run([PYTHON, "-m", "pip", "install", "causal-conv1d>=1.4.0", "mamba-ssm>=2.2.0", "thop"], check=False)
    if result.returncode != 0:
        raise RuntimeError("Mamba dependencies failed. Set INCLUDE_MAMBA_MODELS=False.")
else:
    print("Mamba models disabled; no CUDA extension installation needed.")
"""


SOTA_PROBE = """
PROBE_RESULTS = {}
for model_name in MODELS_TO_RUN:
    command = [PYTHON, "-m", "geodiff_gan.cli.benchmark_sota", "--model", model_name, "--source-root", SOURCE_ROOT, "--manifest", MANIFEST, "--output", BENCHMARK_ROOT / ARCHITECTURE_MODE / model_name, "--architecture-mode", ARCHITECTURE_MODE, "--probe-only"]
    completed = subprocess.run([str(value) for value in command], cwd=REPOSITORY_DIR, text=True, capture_output=True)
    print(completed.stdout)
    if completed.returncode != 0:
        print(completed.stderr)
        raise RuntimeError(f"Probe failed for {model_name}")
    PROBE_RESULTS[model_name] = json.loads(completed.stdout[completed.stdout.index("{"):])
print("All probes passed.")
"""


SOTA_TRAIN = """
TRAINING_RESULTS = {}
for model_name in MODELS_TO_RUN:
    batch_size, accumulation = BATCH_SETTINGS[model_name]
    output = BENCHMARK_ROOT / ARCHITECTURE_MODE / PROFILE / model_name
    command = [
        PYTHON, "-m", "geodiff_gan.cli.benchmark_sota",
        "--model", model_name,
        "--source-root", SOURCE_ROOT,
        "--manifest", MANIFEST,
        "--output", output,
        "--architecture-mode", ARCHITECTURE_MODE,
        "--max-updates", SETTINGS["max_updates"],
        "--batch-size", batch_size,
        "--accumulation", accumulation,
        "--learning-rate", 2e-4,
        "--weight-decay", 1e-4,
        "--lr-crop", LR_CROP,
        "--num-workers", 6,
        "--validation-limit", SETTINGS["validation_limit"],
        "--test-limit", SETTINGS["test_limit"],
        "--validate-every", SETTINGS["validate_every"],
        "--early-stopping-patience", SETTINGS["early_stopping_patience"],
        "--degradation-seed", 42,
        "--degradation-severity", "mild",
        "--seed", 42,
    ]
    print("\\n" + "=" * 80)
    print("Training", model_name, "->", output)
    print("=" * 80)
    run(command, cwd=REPOSITORY_DIR)
    TRAINING_RESULTS[model_name] = json.loads((output / "test_metrics.json").read_text(encoding="utf-8"))
"""


SOTA_COMPARE = """
import pandas as pd

rows = []
for model_name in MODELS_TO_RUN:
    output = BENCHMARK_ROOT / ARCHITECTURE_MODE / PROFILE / model_name
    metrics = json.loads((output / "test_metrics.json").read_text(encoding="utf-8"))
    run_info = json.loads((output / "run.json").read_text(encoding="utf-8"))
    rows.append({"family": "SOTA official", "method": model_name, "architecture_mode": run_info["architecture_mode"], "parameters": run_info["parameters"], **metrics})

for variant in ("small_improved", "medium", "small_improved_ms", "medium_ms"):
    metrics_path = WORK_ROOT / "evaluation" / variant / "test" / "metrics.json"
    if metrics_path.exists():
        rows.append({"family": "GeoDiff-GAN", "method": f"geodiff_{variant}", **json.loads(metrics_path.read_text(encoding="utf-8"))})

comparison = pd.DataFrame(rows)
preferred = ["family", "method", "architecture_mode", "parameters", "count", "l1", "psnr", "ssim", "edge_f1", "redegradation_l1", "ergas", "sam_degrees", "uiqi", "scc"]
comparison = comparison[[name for name in preferred if name in comparison.columns]].sort_values("psnr", ascending=False)
display(comparison.round(6))
out = BENCHMARK_ROOT / ARCHITECTURE_MODE / PROFILE / "all_model_comparison.csv"
out.parent.mkdir(parents=True, exist_ok=True)
comparison.to_csv(out, index=False)
print("Wrote:", out)
"""


SOTA_SIDE_BY_SIDE = """
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

image_roots = {name: BENCHMARK_ROOT / ARCHITECTURE_MODE / PROFILE / name / "images" / "test" for name in MODELS_TO_RUN}
available = [name for name, root in image_roots.items() if list(root.glob("*_hr.png"))]
if not available:
    print("No saved qualitative outputs yet.")
else:
    reference_model = available[0]
    reference_files = sorted(image_roots[reference_model].glob("*_hr.png"))
    for reference_hr in reference_files[:5]:
        prefix = reference_hr.name.removesuffix("_hr.png")
        lr_path = image_roots[reference_model] / f"{prefix}_lr.png"
        panels = [("Observed LR", Image.open(lr_path)), ("Target HR", Image.open(reference_hr))]
        for model_name, root in image_roots.items():
            path = root / f"{prefix}_sr.png"
            if path.exists():
                panels.append((model_name, Image.open(path)))
        columns = 4
        rows_count = int(np.ceil(len(panels) / columns))
        figure, axes = plt.subplots(rows_count, columns, figsize=(16, 4 * rows_count))
        axes = np.asarray(axes).reshape(-1)
        for axis, (title, image) in zip(axes, panels):
            axis.imshow(image)
            axis.set_title(title)
            axis.axis("off")
        for axis in axes[len(panels):]:
            axis.axis("off")
        figure.suptitle(prefix)
        plt.tight_layout()
        plt.show()
"""


def build_sota_notebook() -> None:
    cells = [
        md(
            """
            # GeoDiff-GAN DGX 25MCSA19: Recent SOTA Benchmarks

            Trains official x4 SISR/remote-sensing SR repositories on the same
            Sentinel-2 128x128 -> 512x512 task. Competitor source code, published
            upsampling heads, input size and output size are not changed in
            `ARCHITECTURE_MODE='official'`.
            """
        ),
        md("## 0. Create the local Python 3.11 environment"),
        code(KERNEL_SETUP),
        md("## 1. Runtime root guard and repository install"),
        code(RUNTIME),
        code(CLONE_INSTALL),
        code(GPU_CHECK),
        md("## 2. Clone official SOTA repositories inside the assigned folder"),
        code(SOTA_INSTALL_CLONE),
        md("## 3. Load the shared Sentinel-2 split"),
        code(SOTA_DATASET),
        md("## 4. Select official benchmark profile"),
        code(SOTA_PROFILE),
        md("## 5. Optional Mamba dependencies"),
        code(SOTA_OPTIONAL_MAMBA),
        md("## 6. Probe input/output and architecture before training"),
        code(SOTA_PROBE),
        md("## 7. Train models sequentially with automatic resume"),
        code(SOTA_TRAIN),
        md("## 8. Compare SOTA and GeoDiff-GAN metrics"),
        code(SOTA_COMPARE),
        md("## 9. Side-by-side qualitative outputs"),
        code(SOTA_SIDE_BY_SIDE),
        md("## 10. Backup benchmark outputs"),
        code(BACKUP),
    ]
    write_notebook(DGX / "GeoDiff_GAN_DGX_25MCSA19_SOTA_Benchmark.ipynb", cells)


def main() -> None:
    build_variants_notebook()
    build_caption_notebook()
    build_sota_notebook()


if __name__ == "__main__":
    main()
