from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "kaggle" / "GeoDiff_GAN_OLI2MSI_FidelityTrust_3x.ipynb"


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
# GeoDiff-Fidelity 3x on OLI2MSI

This notebook trains the fidelity-first GeoDiff-GAN extension on the official
OLI2MSI paired-sensor task:

```text
Landsat-8 OLI RGB, 30 m, 3 x 160 x 160
                         -> GeoDiff-Fidelity SR-3x
Sentinel-2 MSI RGB, 10 m, 3 x 480 x 480
```

The experiment adds a stronger resize-convolution Swin anchor and a learned
**base-referenced trust projection**. The stochastic branch predicts detail,
but a spatial controller decides how much of that detail can improve on the
deterministic base. Its supervision is the local least-squares residual gain.
All learned upsampling in this variant uses resize-convolution; it contains no
PixelShuffle layer.

The full official test set is never used for checkpoint or residual-scale
selection. A `35 dB` test PSNR is an experimental target, not a guaranteed
outcome. Report the achieved result and the complete protocol.
"""
    ),
    code(
        r"""
from pathlib import Path
from collections import Counter
import copy, hashlib, json, os, random, shutil, subprocess, sys, time

REPOSITORY_URL = "https://github.com/shashankjs2002/SI-SR-1.git"
REPOSITORY_BRANCH = "SR-3x"
REPOSITORY_DIR = Path("/kaggle/working/geodiff-gan-sr3x-fidelity")

PROTOCOL_ID = "oli2msi_fullframe_clip03_fidelity_trust_v1"
WORK_ROOT = Path("/kaggle/working") / PROTOCOL_ID
PATCH_ROOT = WORK_ROOT / "patches"
MANIFEST = WORK_ROOT / "manifest.jsonl"
QUARANTINE = WORK_ROOT / "quarantine.jsonl"
CONFIG_ROOT = WORK_ROOT / "configs"
RUN_ROOT = WORK_ROOT / "runs"
EVAL_ROOT = WORK_ROOT / "evaluation"
ARTIFACT_ROOT = WORK_ROOT / "paper_artifacts"

# Set this only if auto-discovery under /kaggle/input is ambiguous.
OLI2MSI_DATA_ROOT = None
FAST_DEV_RUN = False
REQUIRE_COMPLETE_OFFICIAL_DATA = not FAST_DEV_RUN
EXPECTED_TRAIN_PAIRS = 5225
EXPECTED_TEST_PAIRS = 100
VALIDATION_PERCENT = 10
RANDOM_SEED = 42
MINIMUM_VALID_FRACTION = 0.99
BENCHMARK_REFLECTANCE_MAX = 0.3
LR_SIZE = 160
SCALE = 3
HR_SIZE = LR_SIZE * SCALE

MAX_TRAIN_PAIRS = 64 if FAST_DEV_RUN else None
MAX_VAL_PAIRS = 16 if FAST_DEV_RUN else None
MAX_TEST_PAIRS = 16 if FAST_DEV_RUN else None

STAGES = ["base", "vae", "diffusion", "joint"]
EPOCHS = (
    {"base": 1, "vae": 1, "diffusion": 1, "joint": 1}
    if FAST_DEV_RUN
    else {"base": 40, "vae": 20, "diffusion": 60, "joint": 15}
)
LEARNING_RATES = {
    "base": 2e-4,
    "vae": 1e-4,
    "diffusion": 1e-4,
    "joint": 2e-5,
}
BATCH_SIZES = {"base": 1, "vae": 1, "diffusion": 1, "joint": 1}
ACCUMULATION = {"base": 4, "vae": 8, "diffusion": 4, "joint": 8}
AUTO_RESUME = True
VALIDATION_LIMIT = 16 if FAST_DEV_RUN else 100

VAL_SAMPLES = 2 if FAST_DEV_RUN else 4
VAL_STEPS = 2 if FAST_DEV_RUN else 20
VAL_LIMIT = 16 if FAST_DEV_RUN else 100
RESIDUAL_SCALE_CANDIDATES = (
    [0.0, 1.0]
    if FAST_DEV_RUN
    else [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
)
RUN_FINAL_TEST = True
TEST_SAMPLES = 2 if FAST_DEV_RUN else 4
TEST_STEPS = 2 if FAST_DEV_RUN else 20
TEST_LIMIT = 16 if FAST_DEV_RUN else 100
VISUALIZATION_INDEX = 0

for directory in (WORK_ROOT, PATCH_ROOT, CONFIG_ROOT, RUN_ROOT, EVAL_ROOT, ARTIFACT_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

def run(command, cwd=None, check=True):
    command = [str(value) for value in command]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=cwd, env=environment, check=check)

print("Protocol:", PROTOCOL_ID)
print("Work root:", WORK_ROOT)
print("Full publication run:", not FAST_DEV_RUN)
"""
    ),
    markdown("## 1. Install the exact SR-3x branch"),
    code(
        r"""
if REPOSITORY_DIR.exists() and not (REPOSITORY_DIR / ".git").is_dir():
    # Preserve the existing directory; clone beside it instead of deleting it.
    REPOSITORY_DIR = REPOSITORY_DIR.with_name(REPOSITORY_DIR.name + "-git")

if (REPOSITORY_DIR / ".git").is_dir():
    run(["git", "fetch", "origin", REPOSITORY_BRANCH], cwd=REPOSITORY_DIR)
    run(["git", "checkout", REPOSITORY_BRANCH], cwd=REPOSITORY_DIR)
    run(["git", "pull", "--ff-only", "origin", REPOSITORY_BRANCH], cwd=REPOSITORY_DIR)
else:
    run([
        "git", "clone", "--depth", "1", "--branch", REPOSITORY_BRANCH,
        REPOSITORY_URL, REPOSITORY_DIR,
    ])

requirements = REPOSITORY_DIR / "requirements-kaggle.txt"
if requirements.exists():
    run([sys.executable, "-m", "pip", "install", "-q", "-r", requirements], cwd=REPOSITORY_DIR)
run([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"], cwd=REPOSITORY_DIR)
sys.path.insert(0, str(REPOSITORY_DIR / "src"))
os.chdir(REPOSITORY_DIR)

import torch, numpy as np, pandas as pd, matplotlib.pyplot as plt, yaml, rasterio
import geodiff_gan

commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_DIR, text=True
).strip()
branch = subprocess.check_output(
    ["git", "branch", "--show-current"], cwd=REPOSITORY_DIR, text=True
).strip()
print("Repository:", REPOSITORY_DIR)
print("Branch:", branch, "commit:", commit)
print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())
if branch != REPOSITORY_BRANCH:
    raise RuntimeError(f"Expected branch {REPOSITORY_BRANCH}, found {branch}")
if not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle GPU accelerator before training.")
"""
    ),
    markdown("## 2. Discover and audit the complete official OLI2MSI pairs"),
    code(
        r"""
search_roots = [Path(OLI2MSI_DATA_ROOT)] if OLI2MSI_DATA_ROOT else []
search_roots.append(Path("/kaggle/input"))

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
        candidates = [root]
        candidates += [path.parent for path in root.rglob("train_lr") if path.is_dir()]
        for candidate in candidates:
            layout = {name: candidate / name for name in required}
            if all(contains_tiffs(path) for path in layout.values()):
                return candidate.resolve(), layout
    return None, None

DATA_ROOT, DATA_DIRS = discover_layout(search_roots)
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find train_lr/train_hr/test_lr/test_hr under /kaggle/input. "
        "Attach the official OLI2MSI dataset or set OLI2MSI_DATA_ROOT."
    )

def index_tiffs(directory):
    indexed = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in (".tif", ".tiff"):
            key = path.stem.casefold()
            if key in indexed:
                raise RuntimeError(f"Duplicate TIFF stem {path.stem!r} in {directory}")
            indexed[key] = path
    return indexed

def pair_directories(lr_dir, hr_dir):
    lr_files, hr_files = index_tiffs(lr_dir), index_tiffs(hr_dir)
    common = sorted(lr_files.keys() & hr_files.keys())
    return (
        [(key, lr_files[key], hr_files[key]) for key in common],
        sorted(lr_files.keys() - hr_files.keys()),
        sorted(hr_files.keys() - lr_files.keys()),
    )

official_train, train_missing_hr, train_missing_lr = pair_directories(
    DATA_DIRS["train_lr"], DATA_DIRS["train_hr"]
)
official_test, test_missing_hr, test_missing_lr = pair_directories(
    DATA_DIRS["test_lr"], DATA_DIRS["test_hr"]
)
observed = {"train": len(official_train), "test": len(official_test)}
print("Data root:", DATA_ROOT)
print("Official paired files:", observed)
print("Unpaired counts:", {
    "train_missing_hr": len(train_missing_hr),
    "train_missing_lr": len(train_missing_lr),
    "test_missing_hr": len(test_missing_hr),
    "test_missing_lr": len(test_missing_lr),
})
if REQUIRE_COMPLETE_OFFICIAL_DATA and observed != {
    "train": EXPECTED_TRAIN_PAIRS, "test": EXPECTED_TEST_PAIRS
}:
    raise RuntimeError(
        "Publication run requires the complete official OLI2MSI distribution: "
        f"expected {EXPECTED_TRAIN_PAIRS}/100 train/test pairs, observed {observed}."
    )

def split_for_source(key):
    bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100
    return "val" if bucket < VALIDATION_PERCENT else "train"

train_pairs = [pair for pair in official_train if split_for_source(pair[0]) == "train"]
val_pairs = [pair for pair in official_train if split_for_source(pair[0]) == "val"]
test_pairs = list(official_test)
if MAX_TRAIN_PAIRS is not None:
    train_pairs = train_pairs[:MAX_TRAIN_PAIRS]
if MAX_VAL_PAIRS is not None:
    val_pairs = val_pairs[:MAX_VAL_PAIRS]
if MAX_TEST_PAIRS is not None:
    test_pairs = test_pairs[:MAX_TEST_PAIRS]
print("Selected source pairs:", {
    "train": len(train_pairs), "val": len(val_pairs), "test": len(test_pairs)
})

sets = {name: {item[0] for item in pairs} for name, pairs in (
    ("train", train_pairs), ("val", val_pairs), ("test", test_pairs)
)}
if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
    raise RuntimeError("Source-level train/validation/test leakage detected")
print("Source isolation: verified")
"""
    ),
    markdown("## 3. Restart-safe full-frame conversion"),
    code(
        r"""
from tqdm.auto import tqdm
from geodiff_gan.data.manifest import ManifestRecord, load_manifest, write_manifest

def raster_audit(path):
    with rasterio.open(path) as source:
        values = source.read([1, 2, 3], out_dtype="float32")
    return {
        "path": str(path), "bands": values.shape[0],
        "height": values.shape[1], "width": values.shape[2],
        "p99": float(np.nanpercentile(values, 99)),
        "max": float(np.nanmax(values)),
    }

audit_rows = []
for _, lr_path, hr_path in (train_pairs + val_pairs + test_pairs)[:8]:
    audit_rows += [raster_audit(lr_path), raster_audit(hr_path)]
audit = pd.DataFrame(audit_rows)
display(audit)
global_p99 = float(audit["p99"].max())
if global_p99 <= 1.5:
    REFLECTANCE_DIVISOR = 1.0
elif global_p99 <= 12000:
    REFLECTANCE_DIVISOR = 10000.0
else:
    raise RuntimeError(f"Unsupported TIFF range; sampled p99={global_p99}")
print("Dataset-level reflectance divisor:", REFLECTANCE_DIVISOR)

def read_rgb(path, expected_size):
    with rasterio.open(path) as source:
        if source.count < 3 or (source.height, source.width) != (expected_size, expected_size):
            raise ValueError(
                f"{path.name}: expected >=3 bands at {expected_size}x{expected_size}; "
                f"found {source.count} at {source.height}x{source.width}"
            )
        image = source.read([1, 2, 3], out_dtype="float32") / REFLECTANCE_DIVISOR
        masks = source.read_masks([1, 2, 3]) > 0
    valid = masks.all(0) & np.isfinite(image).all(0)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    image = np.clip(image, 0.0, BENCHMARK_REFLECTANCE_MAX) / BENCHMARK_REFLECTANCE_MAX
    return image, valid

selected = {
    f"{split}:{key}": (split, key, lr_path, hr_path)
    for split, pairs in (("train", train_pairs), ("val", val_pairs), ("test", test_pairs))
    for key, lr_path, hr_path in pairs
}
existing = load_manifest(MANIFEST) if MANIFEST.exists() else []
records_by_key = {
    record.source_product: record
    for record in existing
    if record.source == PROTOCOL_ID
    and record.source_product in selected
    and Path(record.patch).exists()
}

def quarantine(payload):
    with QUARANTINE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

status = Counter()
for record_key, (split, key, lr_path, hr_path) in tqdm(
    selected.items(), desc="OLI2MSI full frames", unit="pair"
):
    if record_key in records_by_key:
        status["existing"] += 1
        continue
    try:
        lr, valid_lr = read_rgb(lr_path, LR_SIZE)
        hr, valid_hr = read_rgb(hr_path, HR_SIZE)
        valid_fraction = min(float(valid_lr.mean()), float(valid_hr.mean()))
        if valid_fraction < MINIMUM_VALID_FRACTION:
            raise ValueError(f"valid_fraction={valid_fraction:.6f}")
        destination = PATCH_ROOT / split / f"{key}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".npz.tmp")
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
        temporary.replace(destination)
        records_by_key[record_key] = ManifestRecord(
            patch=str(destination.resolve()), tile_id=key, split=split,
            row=0, col=0, valid_fraction=valid_fraction,
            source=PROTOCOL_ID, license_id="see-official-OLI2MSI-distribution",
            source_product=record_key, landsat_product=lr_path.name,
            sentinel_product=hr_path.name, day_gap=0, scale=SCALE,
        )
        status["written"] += 1
    except Exception as error:
        status["quarantined"] += 1
        quarantine({
            "key": key, "split": split, "lr": str(lr_path), "hr": str(hr_path),
            "error": f"{type(error).__name__}: {error}",
        })
    if (status["written"] + status["quarantined"]) % 50 == 0:
        write_manifest(MANIFEST, sorted(records_by_key.values(), key=lambda x: (x.split, x.tile_id)))

write_manifest(MANIFEST, sorted(records_by_key.values(), key=lambda x: (x.split, x.tile_id)))
records = load_manifest(MANIFEST)
split_counts = Counter(record.split for record in records)
print("Conversion:", dict(status))
print("Manifest counts:", dict(split_counts))
if set(records_by_key) != set(selected):
    raise RuntimeError(
        f"Manifest is incomplete: missing={len(set(selected) - set(records_by_key))}. "
        f"Inspect {QUARANTINE}."
    )
"""
    ),
    markdown("## 4. Visual and numerical pair sanity check"),
    code(
        r"""
import torch.nn.functional as F
from geodiff_gan.metrics import basic_metrics

def chw(value):
    return torch.from_numpy(np.asarray(value)).float()

def show_pair(split="train", index=0):
    candidates = [record for record in records if record.split == split]
    record = candidates[int(index) % len(candidates)]
    with np.load(record.patch) as data:
        lr, hr = chw(data["lr"]), chw(data["hr"])
        valid_hr = chw(data["valid_mask_hr"])
    bicubic = F.interpolate(
        lr[None], size=hr.shape[-2:], mode="bicubic", align_corners=False
    )[0].clamp(0, 1)
    downsampled_hr = F.interpolate(
        hr[None], size=lr.shape[-2:], mode="area"
    )[0]
    cross_sensor_l1 = float(((lr - downsampled_hr).abs() * F.interpolate(
        valid_hr[None], size=lr.shape[-2:], mode="nearest"
    )[0]).mean())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis, image, title in zip(
        axes,
        (lr, bicubic, hr),
        (
            "Landsat OLI input | 30 m | 160 x 160",
            "Bicubic | 480 x 480 | no learned detail",
            "Sentinel-2 MSI target | 10 m | 480 x 480",
        ),
    ):
        axis.imshow(image[:3].clamp(0, 1).permute(1, 2, 0), interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"{record.tile_id} | downsampled cross-sensor L1={cross_sensor_l1:.5f}")
    plt.tight_layout()
    plt.show()
    return record

_ = show_pair("train", 0)
"""
    ),
    markdown("## 5. Build and audit the fidelity-trust model"),
    code(
        r"""
from geodiff_gan.parameters import build_parameter_report

template_path = REPOSITORY_DIR / "configs" / "oli2msi_fidelity_trust_3x.yaml"
template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
template["data"]["manifest"] = str(MANIFEST)
template["data"]["captions"] = None
template["debug"]["output_dir"] = str(WORK_ROOT / "debug")

assert template["model"]["base_architecture"] == "fidelity_swinir"
assert template["model"]["base_upsample_mode"] == "resize_conv"
assert template["model"]["decoder_upsample_mode"] == "resize_conv"
assert template["model"]["use_base_referenced_trust"] is True
report = build_parameter_report(template, patches=split_counts["train"], world_size=1)
print("Core parameters:", f"{report['core_model']['scalar_parameters']:,}")
display(pd.DataFrame([
    {"module": name, "parameters": values["scalar_parameters"]}
    for name, values in report["core_modules"].items()
]).sort_values("parameters", ascending=False))
"""
    ),
    markdown("## 6. Train with immutable checkpoint lineage"),
    code(
        r"""
from geodiff_gan.training.checkpoint import (
    best_stage_checkpoint, checkpoint_sha256, latest_stage_checkpoint
)

def choose_checkpoint(directory, stage):
    checkpoint = best_stage_checkpoint(directory, stage)
    if checkpoint is None:
        checkpoint = latest_stage_checkpoint(directory, stage)
    if checkpoint is None:
        raise FileNotFoundError(f"No {stage} checkpoint in {directory}")
    return checkpoint

def lineage_directory(stage, parent):
    lineage = "root" if parent is None else checkpoint_sha256(parent)[:12]
    return RUN_ROOT / stage / lineage

def train_stage(stage, parent):
    config = copy.deepcopy(template)
    output_dir = lineage_directory(stage, parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    training = config["training"]
    training.update({
        "stage": stage,
        "epochs": EPOCHS[stage],
        "batch_size": BATCH_SIZES[stage],
        "gradient_accumulation": ACCUMULATION[stage],
        "learning_rate": LEARNING_RATES[stage],
        "output_dir": str(output_dir),
        "init_checkpoint": str(parent) if parent is not None else None,
        "auto_resume": AUTO_RESUME,
        "resume": None,
        "validation_limit": VALIDATION_LIMIT,
        "enforce_init_checkpoint_lineage": True,
    })
    if stage == "diffusion":
        training.update({
            "checkpoint_metric": "val_loss_diffusion",
            "checkpoint_mode": "min",
            "early_stopping_metric": "val_loss_diffusion",
            "early_stopping_mode": "min",
            "early_stopping_min_epochs": 15 if not FAST_DEV_RUN else 1,
            "lr_scheduler_metric": "val_loss_diffusion",
            "lr_scheduler_mode": "min",
        })
    else:
        training.update({
            "checkpoint_metric": "val_psnr",
            "checkpoint_mode": "max",
            "early_stopping_metric": "val_psnr",
            "early_stopping_mode": "max",
        })
    if stage == "joint":
        # Preserve the already selected diffusion model while the trust/decoder
        # learns to minimize sampled reconstruction risk.
        training["trainable_modules"] = [
            "mapper", "decoder", "trust_controller"
        ]
        training["loss_weights"]["diffusion"] = 0.0
        training["validation_sample_steps"] = VAL_STEPS
        training["validation_samples"] = 1
        training["early_stopping_min_epochs"] = 5 if not FAST_DEV_RUN else 1
        training["module_learning_rate_multipliers"] = {
            "mapper": 0.5,
            "decoder": 0.5,
            "trust_controller": 1.0,
        }
    config_dir = CONFIG_ROOT / stage
    config_dir.mkdir(parents=True, exist_ok=True)
    lineage = "root" if parent is None else checkpoint_sha256(parent)[:12]
    config_path = config_dir / f"{lineage}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"\n===== {stage}: epochs={EPOCHS[stage]} lineage={lineage} =====")
    run([sys.executable, "-m", "geodiff_gan.cli.train", "--config", config_path], cwd=REPOSITORY_DIR)
    return config_path, choose_checkpoint(output_dir, stage)

CHECKPOINTS, CONFIGS = {}, {}
parent = None
for stage in STAGES:
    config_path, checkpoint = train_stage(stage, parent)
    CONFIGS[stage], CHECKPOINTS[stage] = config_path, checkpoint
    parent = checkpoint
    print("Selected:", checkpoint)

CHECKPOINTS
"""
    ),
    markdown("## 7. Verify parent fingerprints and plot training curves"),
    code(
        r"""
lineage_rows = []
parent = None
for stage in STAGES:
    checkpoint = CHECKPOINTS[stage]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    stored = payload.get("extra", {}).get("parent_checkpoint_sha256")
    expected = checkpoint_sha256(parent) if parent is not None else None
    lineage_rows.append({
        "stage": stage,
        "checkpoint": str(checkpoint),
        "parent_matches": stored == expected,
        "selection_metric": payload.get("extra", {}).get("checkpoint_selection", {}).get("metric"),
        "selection_mode": payload.get("extra", {}).get("checkpoint_selection", {}).get("mode"),
        "selection_value": payload.get("extra", {}).get("checkpoint_selection", {}).get("value"),
    })
    parent = checkpoint
lineage_table = pd.DataFrame(lineage_rows)
display(lineage_table)
if not lineage_table["parent_matches"].all():
    raise RuntimeError("Checkpoint lineage audit failed")

fig, axes = plt.subplots(1, len(STAGES), figsize=(18, 4))
for axis, stage in zip(axes, STAGES):
    history_path = Path(yaml.safe_load(CONFIGS[stage].read_text())["training"]["output_dir"]) / "training_history.jsonl"
    rows = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    frame = pd.DataFrame([{"epoch": row["epoch"] + 1, **row["metrics"]} for row in rows])
    metric = "val_psnr" if "val_psnr" in frame else "val_loss_diffusion"
    axis.plot(frame["epoch"], frame[metric], marker="o", linewidth=1.5)
    axis.set_title(f"{stage}: {metric}")
    axis.set_xlabel("epoch")
    axis.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(ARTIFACT_ROOT / "training_curves.png", dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        "## 8. Validation-only residual calibration\n\n"
        "The test set remains unopened. External scale `1.0` means the learned "
        "trust map is used exactly as trained; other values are declared inference ablations."
    ),
    code(
        r"""
JOINT_CONFIG = CONFIGS["joint"]
JOINT_CHECKPOINT = CHECKPOINTS["joint"]
validation_rows = []
for residual_scale in RESIDUAL_SCALE_CANDIDATES:
    destination = EVAL_ROOT / "val" / f"scale_{residual_scale:.2f}"
    run([
        sys.executable, "-m", "geodiff_gan.cli.evaluate",
        "--config", JOINT_CONFIG,
        "--checkpoint", JOINT_CHECKPOINT,
        "--output", destination,
        "--split", "val",
        "--samples", VAL_SAMPLES,
        "--steps", VAL_STEPS,
        "--back-projection-steps", 0,
        "--residual-scale", residual_scale,
        "--mode", "sr",
        "--limit", VAL_LIMIT,
        "--device", "cuda",
        "--progress", "compact",
        "--no-text",
    ], cwd=REPOSITORY_DIR)
    metrics = json.loads((destination / "metrics.json").read_text())
    validation_rows.append({"residual_scale": residual_scale, **metrics})

validation_table = pd.DataFrame(validation_rows).sort_values("psnr", ascending=False)
display(validation_table[[
    "residual_scale", "psnr", "base_psnr", "psnr_delta_vs_base", "ssim", "l1",
    "edge_f1", "ergas", "sam_degrees", "fraction_beating_base_psnr",
]])
BEST_RESIDUAL_SCALE = float(validation_table.iloc[0]["residual_scale"])
print("Validation-selected residual scale:", BEST_RESIDUAL_SCALE)
(ARTIFACT_ROOT / "validation_scale_selection.json").write_text(
    json.dumps({
        "selected_scale": BEST_RESIDUAL_SCALE,
        "selection_split": "val",
        "rows": validation_rows,
    }, indent=2), encoding="utf-8"
)
"""
    ),
    markdown("## 9. Final test evaluation (run once after all choices are fixed)"),
    code(
        r"""
FINAL_RESULTS = None
if RUN_FINAL_TEST:
    model_output = EVAL_ROOT / "test" / "model"
    baseline_output = EVAL_ROOT / "test" / "baselines.json"
    run([
        sys.executable, "-m", "geodiff_gan.cli.evaluate",
        "--config", JOINT_CONFIG,
        "--checkpoint", JOINT_CHECKPOINT,
        "--output", model_output,
        "--split", "test",
        "--samples", TEST_SAMPLES,
        "--steps", TEST_STEPS,
        "--back-projection-steps", 0,
        "--residual-scale", BEST_RESIDUAL_SCALE,
        "--mode", "sr",
        "--limit", TEST_LIMIT,
        "--device", "cuda",
        "--progress", "compact",
        "--no-text",
    ], cwd=REPOSITORY_DIR)
    run([
        sys.executable, "-m", "geodiff_gan.cli.baselines",
        "--config", JOINT_CONFIG,
        "--base-checkpoint", CHECKPOINTS["base"],
        "--output", baseline_output,
        "--split", "test",
        "--limit", TEST_LIMIT,
        "--device", "cuda",
    ], cwd=REPOSITORY_DIR)
    model_metrics = json.loads((model_output / "metrics.json").read_text())
    baseline_metrics = json.loads(baseline_output.read_text())
    rows = [{"method": "GeoDiff-Fidelity", **model_metrics}]
    rows += [
        {"method": name, **values}
        for name, values in baseline_metrics.items() if isinstance(values, dict)
    ]
    FINAL_RESULTS = pd.DataFrame(rows)
    display(FINAL_RESULTS[[
        "method", "l1", "psnr", "ssim", "edge_f1", "ergas",
        "sam_degrees", "uiqi", "scc", "redegradation_l1",
    ]])
    achieved = float(model_metrics["psnr"])
    print(f"Measured test PSNR: {achieved:.4f} dB")
    print("35 dB target reached:", achieved >= 35.0)
    (ARTIFACT_ROOT / "final_test_results.json").write_text(json.dumps({
        "protocol": PROTOCOL_ID,
        "repository_commit": commit,
        "checkpoint": str(JOINT_CHECKPOINT),
        "checkpoint_sha256": checkpoint_sha256(JOINT_CHECKPOINT),
        "validation_selected_residual_scale": BEST_RESIDUAL_SCALE,
        "model": model_metrics,
        "baselines": baseline_metrics,
    }, indent=2), encoding="utf-8")
else:
    print("RUN_FINAL_TEST=False; validation is complete and the official test remains unopened.")
"""
    ),
    markdown("## 10. Inspect any held-out test pair and the learned trust map"),
    code(
        r"""
from geodiff_gan.data import SentinelPatchDataset
from geodiff_gan.models.system import GeoDiffGAN
from geodiff_gan.training.checkpoint import load_checkpoint
from geodiff_gan.metrics import basic_metrics

def inspect_test(index=0, samples=TEST_SAMPLES, steps=TEST_STEPS):
    config = yaml.safe_load(Path(JOINT_CONFIG).read_text())
    dataset = SentinelPatchDataset(
        MANIFEST, split="test", scale=3, augment=False,
        random_degradation=False, input_mode="paired",
    )
    item = dataset[int(index) % len(dataset)]
    device = torch.device("cuda")
    model = GeoDiffGAN.from_config(config).to(device).eval()
    load_checkpoint(JOINT_CHECKPOINT, model, strict=False)
    lr = item["lr"][None].to(device)
    hr = item["hr"][None].to(device)
    degradation = item["degradation"][None].to(device)
    context = torch.zeros(1, 1, model.context_dim, device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        base = model.predict_base(lr)
        features = model.lr_encoder(lr)
        generated = [
            model.sample(
                lr, context, degradation=degradation, sample_steps=steps,
                back_projection_steps=0,
                generator=torch.Generator(device=device).manual_seed(seed),
                base=base, lr_features=features,
            )
            for seed in range(samples)
        ]
        stack = torch.stack([value.image for value in generated]).float()
        raw_mean = stack.mean(0)
        uncertainty = stack.var(0, unbiased=False).mean(1)
        evidence = torch.stack([value.evidence_confidence for value in generated]).mean(0)
        trust = torch.stack([value.trust_map for value in generated]).mean(0)
        output, confidence, abstention = model.apply_uncertainty_abstention(
            raw_mean, base.float(), evidence.float(), uncertainty, trust_map=trust.float()
        )
        output = (base.float() + BEST_RESIDUAL_SCALE * (output - base.float())).clamp(0, 1)
    bicubic = F.interpolate(lr[:, :3], size=hr.shape[-2:], mode="bicubic", align_corners=False)
    panels = [
        (F.interpolate(lr[:, :3], size=hr.shape[-2:], mode="nearest")[0], "Landsat input (nearest display)"),
        (bicubic[0], "Bicubic"),
        (base[0].float(), "Fidelity Swin base"),
        (output[0], "GeoDiff-Fidelity"),
        (hr[0], "Sentinel target"),
        ((output - hr).abs().mean(1, keepdim=True)[0].repeat(3, 1, 1), "Output absolute error"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for axis, (image, title) in zip(axes.flat, panels):
        axis.imshow(image.clamp(0, 1).permute(1, 2, 0).cpu())
        axis.set_title(title)
        axis.axis("off")
    plt.tight_layout()
    figure_path = ARTIFACT_ROOT / f"test_{int(index):04d}_comparison.png"
    plt.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, value, title in zip(
        axes,
        (trust[0, 0], confidence[0, 0], uncertainty[0]),
        ("Learned residual trust", "Combined confidence", "Sample uncertainty"),
    ):
        artist = axis.imshow(value.float().cpu(), cmap="viridis")
        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(artist, ax=axis, fraction=0.046)
    plt.tight_layout()
    policy_path = ARTIFACT_ROOT / f"test_{int(index):04d}_trust.png"
    plt.savefig(policy_path, dpi=180, bbox_inches="tight")
    plt.show()
    print("Saved:", figure_path, policy_path)
    return {"item": item, "base": base.cpu(), "output": output.cpu(), "target": hr.cpu()}

INSPECTION = inspect_test(VISUALIZATION_INDEX)
"""
    ),
    markdown("## 11. Optional intermediate diagnostic export"),
    code(
        r"""
RUN_DETAILED_DEBUG = False
if RUN_DETAILED_DEBUG:
    debug_root = ARTIFACT_ROOT / f"debug_test_{VISUALIZATION_INDEX}"
    run([
        sys.executable, "-m", "geodiff_gan.cli.debug",
        "--config", JOINT_CONFIG,
        "--checkpoint", JOINT_CHECKPOINT,
        "--output", debug_root,
        "--split", "test",
        "--index", VISUALIZATION_INDEX,
        "--mode", "sr",
        "--steps", TEST_STEPS,
        "--diffusion-every", max(1, TEST_STEPS // 4),
        "--save-tensors",
    ], cwd=REPOSITORY_DIR)
    print("Diagnostic directory:", debug_root)
else:
    print("Set RUN_DETAILED_DEBUG=True to export every internal module view.")
"""
    ),
    markdown(
        r"""
## 12. Paper ablations that must be run before claiming novelty

Use the same split, seed and update count for every row:

1. Legacy compact base + old evidence gate.
2. Fidelity Swin base only.
3. Fidelity base + fixed global residual gain `0.25`.
4. Fidelity base + learned trust, but no oracle trust loss.
5. Fidelity base + learned trust, but no spatial base guard.
6. Full proposed model.

The claim is supported only if the learned trust model improves held-out PSNR,
SSIM and the fraction of patches beating its own base. A parameter-matched
baseline is required to show that gains are not merely from added capacity.
"""
    ),
    markdown("## 13. Package metrics, configs and figures (not training data or checkpoints)"),
    code(
        r"""
bundle_stamp = time.strftime("%Y%m%d_%H%M%S")
bundle_root = WORK_ROOT / f"result_bundle_{bundle_stamp}"
bundle_root.mkdir(parents=True, exist_ok=True)
for source in (CONFIG_ROOT, EVAL_ROOT, ARTIFACT_ROOT):
    destination = bundle_root / source.name
    shutil.copytree(source, destination)
(bundle_root / "run_metadata.json").write_text(json.dumps({
    "protocol": PROTOCOL_ID,
    "repository_branch": REPOSITORY_BRANCH,
    "repository_commit": commit,
    "manifest": str(MANIFEST),
    "split_counts": dict(split_counts),
    "checkpoints": {name: str(path) for name, path in CHECKPOINTS.items()},
    "checkpoint_hashes": {name: checkpoint_sha256(path) for name, path in CHECKPOINTS.items()},
}, indent=2), encoding="utf-8")
archive = shutil.make_archive(
    str(WORK_ROOT / f"geodiff_fidelity_oli2msi_results_{bundle_stamp}"),
    "zip",
    bundle_root,
)
print("Result archive:", archive)
print("Checkpoints remain in:", RUN_ROOT)
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUTPUT)
