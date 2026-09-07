"""Build the diverse-scene routed GeoDiff Kaggle experiment notebook."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from textwrap import dedent
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "kaggle" / "GeoDiff_GAN_Kaggle_Diverse_Tile_MoE_3x.ipynb"
cells: list[dict[str, object]] = []


def md(value: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": dedent(value).strip().splitlines(True)})


def code(value: str) -> None:
    value = dedent(value).strip() + "\n"
    ast.parse(value)
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": value.splitlines(True)})


md("""
# GeoDiff 3x on a diverse Landsat/Sentinel tile dataset

This notebook consumes the output of `Landsat_Sentinel_Diverse_3x_Dataset_Preparation.ipynb`.
It does not prepare raw scenes and does not synthesize LR. Every experiment uses the
same real Landsat 30 m LR, Sentinel-2 10 m HR, masks and fixed train/validation/test split.

Controlled experiments:

1. previous standard GeoDiff control on the same data;
2. residual reconstruction base;
3. one-expert residual diffusion control;
4. generic sparse MoE;
5. reliability-supervised sparse MoE;
6. reliability MoE with balanced reconstruction, structure and adversarial losses.

Bicubic is also evaluated. Results include overall and per-class PSNR, SSIM, L1,
Edge F1, ERGAS, SAM, UIQI and sCC; paired confidence intervals; category/expert routing;
parameters; latency; and indexed visual comparisons. Test is evaluated only after the
predeclared experiments finish. A diverse dataset is not evidence of superiority by
itself: the saved audits must show specialization and improvement over the one-expert
control on held-out data.
""")

md("## 1. Controls")
code(r'''
from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, time, zipfile

FAST_DEV_RUN = False
REPOSITORY_URL = "https://github.com/shashankjs2002/SI-SR-1.git"
REPOSITORY_BRANCH = "3x-continued"
REPOSITORY_DIR = Path("/kaggle/working/geodiff-diverse-moe-3x")
SOURCE_ARCHIVE = None  # Auto-discovers geodiff_diverse_moe_source*.zip when omitted.
PREPARED_DATASET_ROOT = None  # Set only if auto-discovery reports multiple datasets.
SUITE_ROOT = Path("/kaggle/working/geodiff-diverse-moe-smoke" if FAST_DEV_RUN else "/kaggle/working/geodiff-diverse-moe-v1")
RESTORE_SUITE_FROM = None

NUM_EXPERTS = 5
TOP_K = 2
TRAIN_LR_CROP = 64
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 2
NUM_WORKERS = 2
EPOCHS = (
    {"base": 1, "vae": 1, "diffusion": 1, "joint": 1}
    if FAST_DEV_RUN else
    {"base": 30, "vae": 12, "diffusion": 30, "joint": 10}
)
LEARNING_RATES = {"base": 1e-4, "vae": 1e-4, "diffusion": 1e-4, "joint": 1e-5}

RUN_STANDARD_GEODIFF_CONTROL = True
PROFILES = ("residual_base", "single_expert", "generic_moe", "reliability_moe", "balanced_adversarial_moe")
EVAL_SAMPLES = 1 if FAST_DEV_RUN else 4
EVAL_STEPS = 2 if FAST_DEV_RUN else 20
EVAL_LIMIT = 4 if FAST_DEV_RUN else None
RUN_TEST_EVALUATION = True
RANDOM_SEED = 42

if not 1 <= TOP_K <= NUM_EXPERTS:
    raise ValueError("Require 1 <= TOP_K <= NUM_EXPERTS")
SUITE_ROOT.mkdir(parents=True, exist_ok=True)
DATA_ROOT = SUITE_ROOT / "dataset"
EXPERIMENT_ROOT = SUITE_ROOT / "experiments"
FIGURE_ROOT = SUITE_ROOT / "figures"
AUDIT_ROOT = SUITE_ROOT / "research_audit"
for directory in (DATA_ROOT, EXPERIMENT_ROOT, FIGURE_ROOT, AUDIT_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

def run(command, cwd=None):
    command = list(map(str, command))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_DIR / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTHONUNBUFFERED"] = "1"
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=cwd, env=environment, check=True)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

print("Epochs:", EPOCHS)
print("Batch:", BATCH_SIZE, "accumulation:", GRADIENT_ACCUMULATION)
print("Experts:", NUM_EXPERTS, "top-k:", TOP_K)
''')

md("## 2. Install the exact source and check GPU")
code(r'''
archives = ([Path(SOURCE_ARCHIVE)] if SOURCE_ARCHIVE else
            list(Path("/kaggle/input").rglob("geodiff_diverse_moe_source*.zip")))
if len(archives) > 1:
    raise RuntimeError("Set SOURCE_ARCHIVE to one exact source snapshot")
if not REPOSITORY_DIR.exists():
    if archives:
        with zipfile.ZipFile(archives[0]) as handle:
            for member in handle.infolist():
                target = (REPOSITORY_DIR / member.filename).resolve()
                if not target.is_relative_to(REPOSITORY_DIR.resolve()):
                    raise ValueError(f"Unsafe source archive member: {member.filename}")
            handle.extractall(REPOSITORY_DIR)
    else:
        run(["git", "clone", "--depth", "1", "--branch", REPOSITORY_BRANCH, REPOSITORY_URL, REPOSITORY_DIR])
elif (REPOSITORY_DIR / ".git").is_dir():
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPOSITORY_DIR, text=True).strip()
    if branch != REPOSITORY_BRANCH:
        raise RuntimeError(f"Expected branch {REPOSITORY_BRANCH}, found {branch}")

run([sys.executable, "-m", "pip", "install", "-q", "numpy", "Pillow", "PyYAML", "tqdm", "pandas", "matplotlib"])
run([sys.executable, "-m", "pip", "install", "-q", "-e", REPOSITORY_DIR, "--no-deps"])
sys.path.insert(0, str(REPOSITORY_DIR / "src"))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from IPython.display import display
print("Python:", sys.version)
print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle GPU; do not reinstall PyTorch")
print("GPU:", torch.cuda.get_device_name(0))

source_files = sorted((REPOSITORY_DIR / "src/geodiff_gan").rglob("*.py"))
source_digest = hashlib.sha256("".join(
    str(path.relative_to(REPOSITORY_DIR)) + sha256(path) for path in source_files
).encode()).hexdigest()
source_lock = SUITE_ROOT / "source_lock.json"
source_receipt = {"branch": REPOSITORY_BRANCH, "source_sha256": source_digest}
if source_lock.exists() and json.loads(source_lock.read_text()) != source_receipt:
    raise ValueError("Source changed inside this experiment suite; use a new SUITE_ROOT")
source_lock.write_text(json.dumps(source_receipt, indent=2))
print("Source SHA256:", source_digest)
''')

md("## 3. Restore an earlier Kaggle session (optional)")
code(r'''
if RESTORE_SUITE_FROM:
    source = Path(RESTORE_SUITE_FROM)
    if source.name != SUITE_ROOT.name:
        raise ValueError("Restore source must have the same suite directory name")
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        destination = SUITE_ROOT / path.relative_to(source)
        if destination.exists() and sha256(destination) != sha256(path):
            raise RuntimeError(f"Restore conflict, nothing overwritten: {destination}")
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    print("Restored missing files from", source)
else:
    print("Restore skipped")
''')

md("## 4. Locate and validate the uploaded prepared dataset")
code(r'''
from collections import Counter
from geodiff_gan.data.diverse import validate_category_splits
from geodiff_gan.data.manifest import load_manifest, validate_within_tile_spatial_isolation

if PREPARED_DATASET_ROOT:
    candidates = [Path(PREPARED_DATASET_ROOT)]
else:
    candidates = sorted({path.parent for path in Path("/kaggle/input").rglob("dataset_card.json")
                         if (path.parent / "manifest.jsonl").is_file()})
if len(candidates) != 1:
    raise RuntimeError(f"Expected one prepared diverse dataset, found {candidates}. Set PREPARED_DATASET_ROOT.")
SOURCE_DATASET_ROOT = candidates[0]
SOURCE_MANIFEST = SOURCE_DATASET_ROOT / "manifest.jsonl"
SOURCE_CARD = json.loads((SOURCE_DATASET_ROOT / "dataset_card.json").read_text())
if SOURCE_CARD.get("format") != "geodiff-paired-npz-v1" or SOURCE_CARD.get("scale") != 3:
    raise ValueError(f"Unsupported dataset card: {SOURCE_CARD}")

# Experiment snapshots require absolute immutable paths; /kaggle/input is read-only.
source_rows = [json.loads(line) for line in SOURCE_MANIFEST.read_text().splitlines() if line.strip()]
for row in source_rows:
    path = Path(row["patch"])
    row["patch"] = str(path if path.is_absolute() else (SOURCE_DATASET_ROOT / path).resolve())
RUNTIME_MANIFEST = DATA_ROOT / "manifest_absolute.jsonl"
runtime_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in source_rows)
if RUNTIME_MANIFEST.exists() and RUNTIME_MANIFEST.read_text() != runtime_text:
    raise ValueError("Runtime manifest changed; use a new SUITE_ROOT")
RUNTIME_MANIFEST.write_text(runtime_text)

records = load_manifest(RUNTIME_MANIFEST)
validate_within_tile_spatial_isolation(records, patch_size=384)
category_rows = validate_category_splits(records, minimum_test_fraction=0.10)
category_table = pd.DataFrame(category_rows)
display(category_table)
category_table.to_csv(AUDIT_ROOT / "dataset_category_splits.csv", index=False)
print("Active split totals:", Counter(record.split for record in records))
print("Tiles:", Counter(record.tile_id for record in records))
print("Manifest SHA256:", sha256(RUNTIME_MANIFEST))
''')

md("## 5. Inspect one sample from every class")
code(r'''
from torch.nn import functional as F

def display_rgb(array, maximum=0.3):
    return np.clip(array.transpose(1, 2, 0) / maximum, 0, 1) ** (1 / 1.4)

classes = sorted({record.scene_class for record in records})
figure, axes = plt.subplots(len(classes), 3, figsize=(12, 4 * len(classes)), squeeze=False)
for row, scene_class in enumerate(classes):
    record = next(record for record in records if record.scene_class == scene_class and record.split == "test")
    with np.load(record.patch) as data:
        lr = torch.from_numpy(data["lr"]).float()
        hr = torch.from_numpy(data["hr"]).float()
        valid = torch.from_numpy(data["valid_mask_hr"][0]).bool()
    bicubic = F.interpolate(lr[None], size=hr.shape[-2:], mode="bicubic", align_corners=False)[0].clamp(0, 1)
    error = (bicubic - hr).abs().mean(0); error[~valid] = float("nan")
    axes[row, 0].imshow(display_rgb(lr.numpy()), interpolation="nearest")
    axes[row, 1].imshow(display_rgb(hr.numpy()), interpolation="nearest")
    axes[row, 2].imshow(error, cmap="turbo", vmin=0, vmax=0.05)
    axes[row, 0].set_title(f"{scene_class}: Landsat 30 m | {lr.shape[-1]} px")
    axes[row, 1].set_title(f"Sentinel-2 10 m | {hr.shape[-1]} px")
    axes[row, 2].set_title("Bicubic absolute error")
    for axis in axes[row]: axis.axis("off")
figure.tight_layout()
figure.savefig(FIGURE_ROOT / "dataset_class_examples.png", dpi=170)
plt.show()
''')

md("## 6. Configuration, registry and restart-safe stage helper")
code(r'''
import copy, yaml
from geodiff_gan.experiments.tile_moe import balanced_manifest, experiment_config, run_stage
from geodiff_gan.models.system import GeoDiffGAN

MANIFEST = balanced_manifest(RUNTIME_MANIFEST, DATA_ROOT / "balanced_manifest.jsonl", RANDOM_SEED)
records = load_manifest(MANIFEST)
RESULTS = {}
STATE_PATH = SUITE_ROOT / "suite_state.json"
if STATE_PATH.exists():
    saved = json.loads(STATE_PATH.read_text())
    if saved["manifest_sha256"] != sha256(MANIFEST):
        raise ValueError("Saved experiment uses another dataset")
    RESULTS.update(saved.get("results", {}))

def save_state():
    value = {
        "manifest": str(MANIFEST), "manifest_sha256": sha256(MANIFEST),
        "dataset_card": SOURCE_CARD, "results": RESULTS,
        "display_max": 0.3, "display_gamma": 1 / 1.4,
    }
    temporary = STATE_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2)); temporary.replace(STATE_PATH)

def make_config(profile, root):
    config = experiment_config(
        REPOSITORY_DIR, profile, MANIFEST, root,
        experts=NUM_EXPERTS, top_k=TOP_K, crop_size=TRAIN_LR_CROP,
        multispectral=False, fast=FAST_DEV_RUN, full_dataset_epochs=True,
    )
    config["data"]["balance_scene_classes"] = True
    config["training"].update(
        batch_size=BATCH_SIZE,
        gradient_accumulation=GRADIENT_ACCUMULATION,
        validation_batch_size=1,
        num_workers=NUM_WORKERS,
        persistent_workers=False,
        progress_mode="compact",
        progress_updates_per_epoch=2,
        validation_limit=len([record for record in records if record.split == "val"]),
        router_warmup_epochs=2,
    )
    config["training"].pop("max_batches_per_epoch", None)
    return config

def register(name, root, config, checkpoint):
    RESULTS[name] = {"root": str(root), "config": str(config), "checkpoint": str(checkpoint),
                     "checkpoint_sha256": sha256(checkpoint)}
    save_state()

prefix = "rgb"
parameter_profiles = (*(("rgb_standard",) if RUN_STANDARD_GEODIFF_CONTROL else ()), *PROFILES)
for profile in parameter_profiles:
    model = GeoDiffGAN.from_config(make_config(profile, EXPERIMENT_ROOT / f"{prefix}_{profile}"))
    if any(isinstance(module, torch.nn.PixelShuffle) for module in model.modules()):
        raise RuntimeError(f"PixelShuffle found in {profile}")
    print(profile, f"{sum(parameter.numel() for parameter in model.parameters()) / 1e6:.3f} M parameters")
    del model
save_state()
''')

md("## 7. Experiment 0: previous standard GeoDiff control")
code(r'''
if RUN_STANDARD_GEODIFF_CONTROL:
    STANDARD_ROOT = EXPERIMENT_ROOT / "rgb_standard_control"
    STANDARD_CONFIG = make_config("rgb_standard", STANDARD_ROOT)
    parent = None
    for stage in ("base", "vae", "diffusion", "joint"):
        config_path, parent = run_stage(
            REPOSITORY_DIR, STANDARD_CONFIG, STANDARD_ROOT, stage, parent=parent,
            epochs=EPOCHS[stage], minutes=None,
            learning_rate=LEARNING_RATES[stage], fast=FAST_DEV_RUN,
        )
    register("rgb_standard_control", STANDARD_ROOT, config_path, parent)
    display(RESULTS["rgb_standard_control"])
else:
    print("Standard GeoDiff control disabled")
''')

md("## 8. Experiment A: residual base")
code(r'''
BASE_ROOT = EXPERIMENT_ROOT / "rgb_residual_base"
BASE_CONFIG = make_config("residual_base", BASE_ROOT)
BASE_CONFIG_PATH, BASE_CHECKPOINT = run_stage(
    REPOSITORY_DIR, BASE_CONFIG, BASE_ROOT, "base",
    epochs=EPOCHS["base"], minutes=None, learning_rate=LEARNING_RATES["base"], fast=FAST_DEV_RUN,
)
register("rgb_residual_base", BASE_ROOT, BASE_CONFIG_PATH, BASE_CHECKPOINT)
print("Base:", BASE_CHECKPOINT)
''')

md("## 9. Shared VAE/LR encoder warm-up")
code(r'''
VAE_ROOT = EXPERIMENT_ROOT / "rgb_shared_vae"
VAE_CONFIG = make_config("residual_base", VAE_ROOT)
VAE_CONFIG_PATH, VAE_CHECKPOINT = run_stage(
    REPOSITORY_DIR, VAE_CONFIG, VAE_ROOT, "vae", parent=BASE_CHECKPOINT,
    epochs=EPOCHS["vae"], minutes=None, learning_rate=LEARNING_RATES["vae"], fast=FAST_DEV_RUN,
)
print("Shared VAE:", VAE_CHECKPOINT)

def train_residual_experiment(profile):
    name = f"rgb_{profile}"
    root = EXPERIMENT_ROOT / name
    config = make_config(profile, root)
    parent = VAE_CHECKPOINT
    for stage in ("diffusion", "joint"):
        config_path, parent = run_stage(
            REPOSITORY_DIR, config, root, stage, parent=parent,
            epochs=EPOCHS[stage], minutes=None,
            learning_rate=LEARNING_RATES[stage], fast=FAST_DEV_RUN,
        )
    register(name, root, config_path, parent)
    return RESULTS[name]
''')

md("## 10. Experiment B: one expert")
code('display(train_residual_experiment("single_expert"))')
md("## 11. Experiment C: generic sparse MoE")
code('display(train_residual_experiment("generic_moe"))')
md("## 12. Experiment D: reliability-supervised MoE")
code('display(train_residual_experiment("reliability_moe"))')
md("## 13. Experiment E: balanced adversarial MoE")
code('display(train_residual_experiment("balanced_adversarial_moe"))')

md("## 14. Validation evaluation and model selection")
code(r'''
def evaluate_result(name, split="val"):
    item = RESULTS[name]
    output_root = Path(item["root"]) / "evaluation" / split
    base_only = name == "rgb_residual_base"
    metrics_path = output_root / "model" / "metrics.json"
    if not metrics_path.exists():
        command = [
            sys.executable, "-m", "geodiff_gan.cli.evaluate",
            "--config", item["config"], "--checkpoint", item["checkpoint"],
            "--output", output_root / "model", "--split", split,
            "--samples", 1 if base_only else EVAL_SAMPLES,
            "--steps", 1 if base_only else EVAL_STEPS,
            "--back-projection-steps", 0,
            "--residual-scale", 0.0 if base_only else 1.0,
            "--device", "cuda", "--progress", "compact", "--no-text",
        ]
        if EVAL_LIMIT is not None: command += ["--limit", EVAL_LIMIT]
        run(command, REPOSITORY_DIR)
    baseline_path = output_root / "baselines.json"
    if not baseline_path.exists():
        command = [
            sys.executable, "-m", "geodiff_gan.cli.baselines",
            "--config", item["config"], "--base-checkpoint", item["checkpoint"],
            "--output", baseline_path, "--split", split,
            "--device", "cuda", "--progress", "compact",
        ]
        if EVAL_LIMIT is not None: command += ["--limit", EVAL_LIMIT]
        run(command, REPOSITORY_DIR)
    return {"experiment": name, **json.loads(metrics_path.read_text())}

validation_table = pd.DataFrame([evaluate_result(name, "val") for name in RESULTS])
validation_table.to_csv(AUDIT_ROOT / "validation_overall.csv", index=False)
display(validation_table[["experiment", "count", "psnr", "base_psnr", "psnr_delta_vs_base",
    "ssim", "l1", "edge_f1", "ergas", "sam_degrees", "uiqi", "scc", "fraction_beating_base_psnr"]])

candidate_rows = validation_table[validation_table.experiment.isin(
    ("rgb_generic_moe", "rgb_reliability_moe", "rgb_balanced_adversarial_moe")
)].copy()
candidate_rows["eligible"] = ((candidate_rows.psnr_delta_vs_base >= -0.05) &
                              (candidate_rows.ssim >= candidate_rows.base_ssim - 0.001))
eligible = candidate_rows[candidate_rows.eligible]
if eligible.empty:
    SELECTED_MODEL = candidate_rows.sort_values(["psnr", "ssim"], ascending=False).iloc[0].experiment
    print("WARNING: no refinement met validation non-inferiority; diagnostic selection:", SELECTED_MODEL)
else:
    SELECTED_MODEL = eligible.sort_values(["edge_f1", "psnr", "ssim"], ascending=False).iloc[0].experiment
print("Validation-selected model:", SELECTED_MODEL)
(AUDIT_ROOT / "selected_model.json").write_text(json.dumps({"model": SELECTED_MODEL, "rule":
    "PSNR delta >= -0.05 dB and SSIM delta >= -0.001; then highest Edge F1"}, indent=2))
''')

md("## 15. Overall and per-class validation comparison")
code(r'''
METRICS = ("psnr", "ssim", "l1", "edge_f1", "ergas", "sam_degrees", "uiqi", "scc")
DIRECTIONS = {metric: ("lower" if metric in ("l1", "ergas", "sam_degrees") else "higher") for metric in METRICS}

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def class_metric_table(split):
    rows = []
    for name, item in RESULTS.items():
        path = Path(item["root"]) / "evaluation" / split / "model" / "per_patch_metrics.jsonl"
        for row in read_jsonl(path):
            rows.append({"experiment": name, "scene_class": row["scene_class"],
                         **{metric: row[f"output_{metric}"] for metric in METRICS}})
    base_item = RESULTS["rgb_residual_base"]
    baseline_path = Path(base_item["root"]) / "evaluation" / split / "baselines_per_patch.jsonl"
    for row in read_jsonl(baseline_path):
        if row["method"] == "bicubic":
            rows.append({"experiment": "bicubic", "scene_class": row["scene_class"],
                         **{metric: row[metric] for metric in METRICS}})
    frame = pd.DataFrame(rows)
    return frame.groupby(["scene_class", "experiment"], as_index=False)[list(METRICS)].mean()

validation_by_class = class_metric_table("val")
validation_by_class.to_csv(AUDIT_ROOT / "validation_by_class.csv", index=False)
display(validation_by_class)

fig, axes = plt.subplots(2, 3, figsize=(17, 9))
for axis, metric in zip(axes.flat, ("psnr", "ssim", "edge_f1", "ergas", "sam_degrees", "scc")):
    pivot = validation_by_class.pivot(index="scene_class", columns="experiment", values=metric)
    pivot.plot.bar(ax=axis, width=0.8)
    axis.set_title(f"{metric} ({DIRECTIONS[metric]} is better)")
    axis.set_xlabel(""); axis.tick_params(axis="x", rotation=25)
    axis.legend(fontsize=6)
fig.tight_layout(); fig.savefig(FIGURE_ROOT / "validation_by_class.png", dpi=180); plt.show()
''')

md("## 16. Locked test evaluation")
code(r'''
if not RUN_TEST_EVALUATION:
    print("Test remains locked. Set RUN_TEST_EVALUATION=True only after fixing every design choice.")
else:
    test_table = pd.DataFrame([evaluate_result(name, "test") for name in RESULTS])
    base_baselines = json.loads((Path(RESULTS["rgb_residual_base"]["root"]) / "evaluation/test/baselines.json").read_text())
    bicubic = {"experiment": "bicubic", "count": base_baselines["count"], **base_baselines["bicubic"]}
    test_table = pd.concat([pd.DataFrame([bicubic]), test_table], ignore_index=True)
    expected_test = len([record for record in records if record.split == "test"])
    if not FAST_DEV_RUN and not all(test_table["count"].astype(int).eq(expected_test)):
        raise RuntimeError(
            f"Incomplete test evaluation: expected {expected_test}, got "
            f"{test_table[['experiment', 'count']].to_dict('records')}"
        )
    test_table.to_csv(AUDIT_ROOT / "test_overall.csv", index=False)
    display(test_table[[column for column in ("experiment", "count", "psnr", "base_psnr", "psnr_delta_vs_base",
        "ssim", "l1", "edge_f1", "ergas", "sam_degrees", "uiqi", "scc", "fraction_beating_base_psnr")
        if column in test_table]])
    test_by_class = class_metric_table("test")
    test_by_class.to_csv(AUDIT_ROOT / "test_by_class.csv", index=False)
    display(test_by_class)
''')

md("## 17. Paired confidence intervals and source-pair robustness")
code(r'''
def bootstrap_mean(values, groups=None, repeats=5000, seed=42):
    frame = pd.DataFrame({"value": np.asarray(values, dtype=float),
                          "group": np.arange(len(values)) if groups is None else np.asarray(groups)})
    unique = frame.group.unique()
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        estimates.append(np.mean([frame.loc[frame.group.eq(group), "value"].mean() for group in sampled]))
    return float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975))

comparison_rows = []
split = "test" if RUN_TEST_EVALUATION else "val"
metric_rows = {
    name: {row["patch"]: row for row in read_jsonl(
        Path(item["root"]) / "evaluation" / split / "model/per_patch_metrics.jsonl"
    )}
    for name, item in RESULTS.items()
}
for name in RESULTS:
    if name == "rgb_residual_base":
        continue
    references = ["rgb_residual_base"]
    if name not in ("rgb_single_expert",) and "rgb_single_expert" in RESULTS:
        references.append("rgb_single_expert")
    for reference in references:
        common = sorted(set(metric_rows[name]) & set(metric_rows[reference]))
        if not common:
            raise RuntimeError(f"No paired evaluation rows for {name} and {reference}")
        for metric in ("psnr", "ssim", "l1", "edge_f1", "ergas", "sam_degrees", "uiqi", "scc"):
            sign = -1 if metric in ("l1", "ergas", "sam_degrees") else 1
            deltas = np.asarray([
                sign * (
                    metric_rows[name][patch][f"output_{metric}"]
                    - metric_rows[reference][patch][f"output_{metric}"]
                )
                for patch in common
            ])
            groups = [
                metric_rows[name][patch].get("source_pair")
                or metric_rows[name][patch]["tile_id"]
                for patch in common
            ]
            patch_ci = bootstrap_mean(deltas, repeats=3000, seed=RANDOM_SEED)
            group_ci = bootstrap_mean(deltas, groups=groups, repeats=3000, seed=RANDOM_SEED) if len(set(groups)) > 1 else (float("nan"), float("nan"))
            comparison_rows.append({"experiment": name, "reference": reference, "metric": metric,
                "direction_adjusted_improvement": deltas.mean(),
                "patch_ci_low": patch_ci[0], "patch_ci_high": patch_ci[1],
                "source_pair_ci_low": group_ci[0], "source_pair_ci_high": group_ci[1],
                "fraction_better": float((deltas > 0).mean()), "source_pairs": len(set(groups))})
confidence_table = pd.DataFrame(comparison_rows)
confidence_table.to_csv(AUDIT_ROOT / f"{split}_paired_confidence_intervals.csv", index=False)
display(confidence_table)
print("Source-pair confidence intervals are primary; patch intervals are descriptive because nearby patches are correlated.")
''')

md("## 18. Expert specialization, collapse and category routing")
code(r'''
def normalized_mutual_information(labels, experts):
    labels, experts = np.asarray(labels), np.asarray(experts)
    total = len(labels); mi = 0.0
    for label in np.unique(labels):
        for expert in np.unique(experts):
            joint = np.mean((labels == label) & (experts == expert))
            if joint > 0:
                mi += joint * np.log(joint / (np.mean(labels == label) * np.mean(experts == expert)))
    def entropy(values):
        probabilities = np.asarray([np.mean(values == value) for value in np.unique(values)])
        return -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
    denominator = np.sqrt(entropy(labels) * entropy(experts))
    return float(mi / denominator) if denominator > 0 else 0.0

routing_summary = []
for name, item in RESULTS.items():
    if name in ("rgb_residual_base", "rgb_single_expert"): continue
    path = Path(item["root"]) / "evaluation" / split / "model/per_patch_metrics.jsonl"
    rows = read_jsonl(path)
    weights = np.asarray([row["expert_weights"] for row in rows], dtype=float)
    if weights.ndim != 2 or weights.shape[1] < 2:
        print(name, "has no multi-expert routing to audit")
        continue
    labels = np.asarray([row["scene_class"] for row in rows])
    dominant = weights.argmax(1)
    for scene_class in sorted(set(labels)):
        selected = weights[labels == scene_class]
        for expert in range(weights.shape[1]):
            routing_summary.append({"experiment": name, "scene_class": scene_class, "expert": expert,
                "mean_weight": selected[:, expert].mean(),
                "selection_fraction": (selected[:, expert] > 0).mean(),
                "dominant_fraction": (selected.argmax(1) == expert).mean()})
    print(name, "NMI(class, dominant expert)=", normalized_mutual_information(labels, dominant),
          "dead experts=", np.where((weights > 0).sum(0) == 0)[0].tolist())
routing_table = pd.DataFrame(routing_summary)
routing_table.to_csv(AUDIT_ROOT / f"{split}_routing_by_class.csv", index=False)
display(routing_table)
for name in routing_table.experiment.unique():
    pivot = routing_table[routing_table.experiment.eq(name)].pivot(index="scene_class", columns="expert", values="selection_fraction")
    fig, axis = plt.subplots(figsize=(8, max(3, .6 * len(pivot))))
    image = axis.imshow(pivot, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(pivot.columns)), [f"E{x}" for x in pivot.columns])
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    axis.set_title(name + ": expert selection fraction by class")
    fig.colorbar(image, ax=axis); fig.tight_layout()
    fig.savefig(FIGURE_ROOT / f"{name}_{split}_routing_by_class.png", dpi=180); plt.show()
''')

md("## 19. Parameters and inference latency")
code(r'''
from geodiff_gan.config import load_config
from geodiff_gan.training.checkpoint import load_checkpoint
from geodiff_gan.data import SentinelPatchDataset

parameter_rows, latency_rows = [], []
benchmark_record = next(record for record in records if record.split == split)
for name, item in RESULTS.items():
    config = load_config(item["config"])
    model = GeoDiffGAN.from_config(config).cuda().eval()
    load_checkpoint(item["checkpoint"], model, strict=False)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    parameter_rows.append({"experiment": name, "parameters": parameters, "million": parameters / 1e6,
                           "checkpoint_mb": Path(item["checkpoint"]).stat().st_size / 2**20})
    dataset = SentinelPatchDataset(MANIFEST, split=split, scale=3, input_mode="paired", augment=False)
    sample = dataset[[record.patch for record in dataset.records].index(benchmark_record.patch)]
    lr = sample["lr"][None].cuda(); degradation = sample["degradation"][None].cuda()
    context = torch.zeros(1, 1, config["model"].get("context_dim", 64), device="cuda")
    repeats = 3
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        base = model.predict_base(lr)
        features = None if name == "rgb_residual_base" else model.lr_encoder(lr)
        if name != "rgb_residual_base":
            model.sample(lr, context, degradation=degradation, base=base, lr_features=features,
                         mode="sr", sample_steps=EVAL_STEPS, back_projection_steps=0)
        torch.cuda.synchronize(); timings = []
        for repeat in range(repeats):
            start = time.perf_counter()
            if name == "rgb_residual_base":
                model.predict_base(lr)
            else:
                model.sample(lr, context, degradation=degradation, base=base, lr_features=features,
                             mode="sr", sample_steps=EVAL_STEPS, back_projection_steps=0,
                             generator=torch.Generator(device="cuda").manual_seed(repeat))
            torch.cuda.synchronize(); timings.append((time.perf_counter() - start) * 1000)
    latency_rows.append({"experiment": name, "steps": EVAL_STEPS, "mean_ms": np.mean(timings),
                         "std_ms": np.std(timings), "images_per_second": 1000 / np.mean(timings)})
    del model; torch.cuda.empty_cache()
parameter_table = pd.DataFrame(parameter_rows); latency_table = pd.DataFrame(latency_rows)
parameter_table.to_csv(AUDIT_ROOT / "parameters.csv", index=False)
latency_table.to_csv(AUDIT_ROOT / "latency.csv", index=False)
display(parameter_table); display(latency_table)
print("Hardware:", torch.cuda.get_device_name(0), "PyTorch:", torch.__version__, "FP16, batch=1")
''')

md("## 20. Indexed visual comparison")
code(r'''
from geodiff_gan.experiments.tile_viewer import SavedTileResults

def show_results(index=0, split_name="test", scene_class="All", models=None, show_base=False, show_errors=False):
    viewer = SavedTileResults(SUITE_ROOT)
    selected = [record for record in viewer.records if record["split"] == split_name and
                (scene_class == "All" or record.get("scene_class") == scene_class)]
    if not selected:
        raise ValueError(f"No records for {split_name}/{scene_class}")
    record = selected[int(index)]
    tile_records = viewer.select(split_name, record["tile_id"])
    tile_index = next(i for i, value in enumerate(tile_records) if value["patch"] == record["patch"])
    panels, details = viewer.panels(tile_index, split_name, names=models, show_base=show_base,
                                    tile=record["tile_id"], errors=show_errors)
    columns = min(3, len(panels)); rows_count = (len(panels) + columns - 1) // columns
    figure, axes = plt.subplots(rows_count, columns, figsize=(5 * columns, 5 * rows_count), squeeze=False)
    for axis in axes.flat: axis.axis("off")
    for axis, (image, title) in zip(axes.flat, panels):
        if image.ndim == 2: axis.imshow(image, cmap="turbo", vmin=0, vmax=0.05)
        else: axis.imshow(viewer.display(image), interpolation="nearest")
        axis.set_title(title, fontsize=9)
    figure.suptitle(f"{split_name}/{scene_class}/{index} | {record['tile_id']}")
    figure.tight_layout()
    destination = FIGURE_ROOT / f"comparison_{split_name}_{scene_class}_{index}.png"
    figure.savefig(destination, dpi=180); plt.show(); display(details)
    print("Saved:", destination)
    return details

show_results(0, split_name="test" if RUN_TEST_EVALUATION else "val", models=[SELECTED_MODEL], show_base=True)
# Example: show_results(2, "test", "urban", models=["rgb_single_expert", "rgb_generic_moe", "rgb_reliability_moe"])
''')

md("## 21. Optional saved-result Gradio explorer")
code(r'''
START_SERVER = False
PUBLIC_SHARE = False
if START_SERVER:
    run([sys.executable, "-m", "pip", "install", "-q", "gradio>=4.44,<6"])
    from geodiff_gan.experiments.tile_viewer import build_app
    app = build_app(SUITE_ROOT)
    app.launch(share=PUBLIC_SHARE, server_name="0.0.0.0")
else:
    print("Set START_SERVER=True. After a kernel restart, restore/install source and run only this cell with the correct SUITE_ROOT.")
''')

md("## 22. Build and download the research-result bundle")
code(r'''
context = {
    "research_question": "Does sparse routed residual diffusion improve heterogeneous real cross-sensor SR?",
    "dataset_manifest_sha256": sha256(MANIFEST),
    "dataset_card": SOURCE_CARD,
    "experiments": list(RESULTS),
    "selected_model": SELECTED_MODEL,
    "selection_used_validation_only": True,
    "test_was_enabled": RUN_TEST_EVALUATION,
    "experts": NUM_EXPERTS, "top_k": TOP_K,
    "epochs": EPOCHS, "batch_size": BATCH_SIZE,
    "gradient_accumulation": GRADIENT_ACCUMULATION,
    "evaluation_samples": EVAL_SAMPLES, "evaluation_steps": EVAL_STEPS,
    "limitations": [
        "Within-tile spatial holdout does not establish unseen-region transfer.",
        "Nearby patches remain correlated; source-pair bootstrap is primary.",
        "A diverse dataset alone does not prove MoE specialization or superiority.",
    ],
}
(AUDIT_ROOT / "research_context.json").write_text(json.dumps(context, indent=2))

bundle = SUITE_ROOT.parent / f"{SUITE_ROOT.name}_research_results.zip"
selected_files = set()
for pattern in (
    "research_audit/**/*", "figures/*.png", "suite_state.json", "source_lock.json",
    "experiments/*/configs/*.yaml", "experiments/*/configs/*.json",
    "experiments/*/runs/*/completed.json", "experiments/*/runs/*/training_history.jsonl",
    "experiments/*/evaluation/*/model/metrics.json",
    "experiments/*/evaluation/*/model/per_patch_metrics.jsonl",
    "experiments/*/evaluation/*/baselines.json",
    "experiments/*/evaluation/*/baselines_per_patch.jsonl",
):
    selected_files.update(path for path in SUITE_ROOT.glob(pattern) if path.is_file())
# Include five cached outputs per model/split for paper figures; full predictions remain in Kaggle output.
for directory in SUITE_ROOT.glob("experiments/*/evaluation/*/model"):
    selected_files.update(sorted(directory.glob("*_uncertainty.npz"))[:5])
temporary = bundle.with_suffix(".zip.incomplete")
with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as handle:
    for path in sorted(selected_files):
        handle.write(path, path.relative_to(SUITE_ROOT))
temporary.replace(bundle)
print("Bundle:", bundle)
print(f"Size: {bundle.stat().st_size / 2**20:.1f} MiB")
print("SHA256:", sha256(bundle))
from IPython.display import FileLink
display(FileLink(str(bundle)))
print("Share this ZIP with the final tables. It excludes raw data, Kaggle credentials and large checkpoints.")
''')


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for index, cell in enumerate(cells):
        cell["id"] = f"diverse-moe-{index:03d}"
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    files = list((ROOT / "src").rglob("*.py")) + list((ROOT / "configs").glob("*.yaml"))
    files += [ROOT / "pyproject.toml", ROOT / "README.md"]
    archive = ROOT / "kaggle" / "geodiff_diverse_moe_source.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(files):
            handle.write(path, path.relative_to(ROOT).as_posix())
    print(OUTPUT)
    print(archive, "sha256=" + hashlib.sha256(archive.read_bytes()).hexdigest())


if __name__ == "__main__":
    build()
