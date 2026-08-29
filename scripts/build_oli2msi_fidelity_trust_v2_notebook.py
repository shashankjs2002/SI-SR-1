from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "kaggle" / "GeoDiff_GAN_OLI2MSI_FidelityTrust_V2_3x.ipynb"


def source(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def markdown(text: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": source(text)}


def code(text: str) -> dict[str, object]:
    ast.parse(text.strip("\n"))
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source(text),
    }


cells: list[dict[str, object]] = []
cells += [
    markdown(
        r"""
# GeoDiff FidelityTrust V2 on official OLI2MSI

The locked task is actual Landsat-8 OLI RGB at 30 m (`160 x 160`) to
Sentinel-2 MSI RGB at 10 m (`480 x 480`). V2 races two deterministic anchors,
then calibrates per-band trust over sampled diffusion residuals. It contains no
PixelShuffle, GAN loss, natural-image perceptual loss, or test-time target
calibration. The official 100-pair test is opened only after the validation
gates pass. A 35 dB result is a target, not a guaranteed outcome.

The default `kaggle_5h` execution profile is a bounded experiment, not the
full publication schedule. It uses RDN directly, stops every stage at a hard
wall-clock budget, and does not duplicate the complete pipeline in an all-data
refit. Switch to `publication` only when resumable multi-session storage is
available.
"""
    ),
    markdown("## 0. Protocol and controls"),
    code(
        r"""
from pathlib import Path
from collections import Counter
import copy, hashlib, inspect, json, math, os, shutil, stat, subprocess, sys, time

REPOSITORY_URL = "https://github.com/shashankjs2002/SI-SR-1.git"
REPOSITORY_BRANCH = "SR-3x"
REPOSITORY_DIR = Path("/kaggle/working/geodiff-gan-sr3x-fidelity-v2")
EXECUTION_PROFILE = "kaggle_5h"  # kaggle_5h | publication | smoke
PROFILE_SUFFIX = "" if EXECUTION_PROFILE == "publication" else f"_{EXECUTION_PROFILE}"
PROTOCOL_ID = "oli2msi_fullframe_clip03_fidelity_trust_v2" + PROFILE_SUFFIX
WORK_ROOT = Path("/kaggle/working") / PROTOCOL_ID
PATCH_ROOT, CONFIG_ROOT = WORK_ROOT / "patches", WORK_ROOT / "configs"
RUN_ROOT, EVAL_ROOT = WORK_ROOT / "runs", WORK_ROOT / "evaluation"
ARTIFACT_ROOT = WORK_ROOT / "paper_artifacts"
MANIFEST, REFIT_MANIFEST = WORK_ROOT / "manifest.jsonl", WORK_ROOT / "manifest_refit.jsonl"
QUARANTINE, STATE_PATH = WORK_ROOT / "quarantine.jsonl", WORK_ROOT / "state.json"
TEST_LOCK = WORK_ROOT / "official_test_lock.json"
BASELINE_BUNDLE = WORK_ROOT / "baseline_v1_read_only"

OLI2MSI_DATA_ROOT = None
V1_BASE_CHECKPOINT_OVERRIDE = None
V1_CONFIG_OVERRIDE = None
ATTACHED_RDN_CHECKPOINT = None  # Optional .pt under /kaggle/input.
AUTO_DISCOVER_ATTACHED_RDN = True
REPORTED_V1_VALIDATION_BASE_PSNR = 35.819096
FORCE_REEVALUATE = False

EXPECTED_TRAIN, EXPECTED_TEST = 5225, 100
VALIDATION_PERCENT, RANDOM_SEED = 10, 42
LR_SIZE, SCALE, HR_SIZE = 160, 3, 480
REFLECTANCE_MAX, MINIMUM_VALID_FRACTION = 0.3, 0.99
PROFILES = {
    "kaggle_5h": {
        "updates": {"race": 0, "base32": 18_000, "base64": 1_500, "spatial": 0, "vae": 600, "diffusion": 1_200, "proposal": 1_200, "trust": 1_200},
        "minutes": {"race": 0, "base32": 105, "base64": 35, "spatial": 0, "vae": 30, "diffusion": 60, "proposal": 40, "trust": 45},
        "architecture_race": False, "spatial_training": False,
        "full_frame_epochs": 0, "train_validation_limit": 64,
        "final_refit": False, "final_test": True,
        "final_samples": 4, "final_steps": 20,
    },
    "publication": {
        "updates": {"race": 25_000, "base32": 150_000, "base64": 40_000, "spatial": 10_000, "vae": 60_000, "diffusion": 150_000, "proposal": 30_000, "trust": 10_000},
        "minutes": {name: 0 for name in ("race", "base32", "base64", "spatial", "vae", "diffusion", "proposal", "trust")},
        "architecture_race": True, "spatial_training": True,
        "full_frame_epochs": 8, "train_validation_limit": None,
        "final_refit": True, "final_test": True,
        "final_samples": 4, "final_steps": 20,
    },
    "smoke": {
        "updates": {"race": 20, "base32": 30, "base64": 20, "spatial": 10, "vae": 20, "diffusion": 30, "proposal": 20, "trust": 10},
        "minutes": {name: 0 for name in ("race", "base32", "base64", "spatial", "vae", "diffusion", "proposal", "trust")},
        "architecture_race": True, "spatial_training": True,
        "full_frame_epochs": 1, "train_validation_limit": 8,
        "final_refit": False, "final_test": True,
        "final_samples": 2, "final_steps": 2,
    },
}
if EXECUTION_PROFILE not in PROFILES:
    raise ValueError(f"Unknown execution profile: {EXECUTION_PROFILE}")
PROFILE = PROFILES[EXECUTION_PROFILE]
FAST_DEV_RUN = EXECUTION_PROFILE == "smoke"
UPDATES, STAGE_MINUTES = PROFILE["updates"], PROFILE["minutes"]
RUN_ARCHITECTURE_RACE = PROFILE["architecture_race"]
RUN_SPATIAL_TRAINING = PROFILE["spatial_training"]
FULL_FRAME_EPOCHS = PROFILE["full_frame_epochs"]
TRAIN_VALIDATION_LIMIT = PROFILE["train_validation_limit"]
RUN_FINAL_REFIT, RUN_FINAL_TEST = PROFILE["final_refit"], PROFILE["final_test"]
VAL_LIMIT = 8 if FAST_DEV_RUN else None
FINAL_SAMPLES, FINAL_STEPS = PROFILE["final_samples"], PROFILE["final_steps"]

for directory in (WORK_ROOT, PATCH_ROOT, CONFIG_ROOT, RUN_ROOT, EVAL_ROOT, ARTIFACT_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

def run(command, cwd=None, check=True):
    command = [str(value) for value in command]
    environment = os.environ.copy(); environment["PYTHONUNBUFFERED"] = "1"
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=cwd, env=environment, check=check)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def state():
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}

def save_state(**values):
    current = state(); current.update(values)
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(current, indent=2)); temporary.replace(STATE_PATH)
    return current

print(PROTOCOL_ID, WORK_ROOT, "profile=", EXECUTION_PROFILE)
print("Training budget (minutes):", STAGE_MINUTES, "total=", sum(STAGE_MINUTES.values()))
if EXECUTION_PROFILE == "kaggle_5h":
    print("Fresh /kaggle/working storage detected by design. Attach an earlier notebook output as a Kaggle input to reuse its RDN checkpoint.")
save_state(execution_profile=EXECUTION_PROFILE, stage_minutes=STAGE_MINUTES, update_caps=UPDATES)
"""
    ),
    markdown("## 1. Install and verify SR-3x"),
    code(
        r"""
if REPOSITORY_DIR.exists() and not (REPOSITORY_DIR / ".git").is_dir():
    REPOSITORY_DIR = REPOSITORY_DIR.with_name(REPOSITORY_DIR.name + "-git")
if (REPOSITORY_DIR / ".git").is_dir():
    run(["git", "fetch", "origin", REPOSITORY_BRANCH], cwd=REPOSITORY_DIR)
    run(["git", "checkout", REPOSITORY_BRANCH], cwd=REPOSITORY_DIR)
    run(["git", "pull", "--ff-only", "origin", REPOSITORY_BRANCH], cwd=REPOSITORY_DIR)
else:
    run(["git", "clone", "--depth", "1", "--branch", REPOSITORY_BRANCH, REPOSITORY_URL, REPOSITORY_DIR])
requirements = REPOSITORY_DIR / "requirements-kaggle.txt"
if requirements.exists():
    run([sys.executable, "-m", "pip", "install", "-q", "-r", requirements])
run([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"], cwd=REPOSITORY_DIR)
sys.path.insert(0, str(REPOSITORY_DIR / "src")); os.chdir(REPOSITORY_DIR)

import numpy as np, pandas as pd, matplotlib.pyplot as plt, rasterio, torch, yaml
from torch import nn
from geodiff_gan.models.system import GeoDiffGAN
from geodiff_gan.training.trainer import Trainer

branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPOSITORY_DIR, text=True).strip()
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_DIR, text=True).strip()
if branch != REPOSITORY_BRANCH or not torch.cuda.is_available():
    raise RuntimeError(f"Expected branch {REPOSITORY_BRANCH} with a GPU; got {branch}, CUDA={torch.cuda.is_available()}")
template_path = REPOSITORY_DIR / "configs" / "oli2msi_fidelity_trust_v2_3x.yaml"
if not template_path.exists():
    raise RuntimeError("The checked-out branch does not contain FidelityTrust V2")
probe = GeoDiffGAN.from_config(yaml.safe_load(template_path.read_text()))
offenders = [name for name, module in probe.named_modules() if isinstance(module, nn.PixelShuffle)]
if offenders:
    raise RuntimeError(f"PixelShuffle found: {offenders}")
del probe
print(branch, commit, torch.__version__, torch.cuda.get_device_name(0))
wall_time_supported = "max_wall_time_minutes" in inspect.getsource(Trainer.train)
if EXECUTION_PROFILE == "kaggle_5h" and not wall_time_supported:
    print("WARNING: this remote branch predates hard wall-time stopping. Optimizer-step caps still bound the run, but commit/push the current SR-3x changes for exact time limits.")
save_state(repository_commit=commit, repository_branch=branch, wall_time_supported=wall_time_supported)
"""
    ),
    markdown("## 2. Preserve V1 artifacts"),
    code(
        r"""
if not (BASELINE_BUNDLE / "inventory.json").exists():
    BASELINE_BUNDLE.mkdir(parents=True, exist_ok=True)
    roots = [
        Path("/kaggle/working/oli2msi_official_rgb_clip03_center128_v2"),
        Path("/kaggle/working/oli2msi_fullframe_clip03_fidelity_trust_v1"),
        Path("/kaggle/working/geodiff-oli2msi-3x"),
    ]
    old_root = next((item for item in roots if item.exists()), None)
    if old_root:
        for child in old_root.iterdir():
            if child.name in {"patches", "official_download", "dataset"}:
                continue
            destination = BASELINE_BUNDLE / child.name
            shutil.copytree(child, destination, dirs_exist_ok=True) if child.is_dir() else shutil.copy2(child, destination)
    (BASELINE_BUNDLE / "reported_metrics.json").write_text(json.dumps({
        "source": str(old_root) if old_root else None,
        "validation_base_psnr": REPORTED_V1_VALIDATION_BASE_PSNR,
        "reported_test_psnr": 33.922831, "reported_test_ssim": 0.916337,
    }, indent=2))
    inventory = []
    for path in sorted(BASELINE_BUNDLE.rglob("*")):
        if path.is_file() and path.name != "inventory.json":
            inventory.append({"path": str(path.relative_to(BASELINE_BUNDLE)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    (BASELINE_BUNDLE / "inventory.json").write_text(json.dumps(inventory, indent=2))
    for path in BASELINE_BUNDLE.rglob("*"):
        if path.is_file(): path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
print("V1 inventory files:", len(json.loads((BASELINE_BUNDLE / "inventory.json").read_text())))
"""
    ),
    markdown("## 3. Discover, split, and lock official pairs"),
    code(
        r"""
def tiff_index(directory):
    return {path.stem.casefold(): path for path in Path(directory).iterdir() if path.suffix.lower() in (".tif", ".tiff")}

def discover_data():
    roots = [Path(OLI2MSI_DATA_ROOT)] if OLI2MSI_DATA_ROOT else []
    roots.append(Path("/kaggle/input"))
    for root in roots:
        candidates = [root] + ([path.parent for path in root.rglob("train_lr")] if root.exists() else [])
        for candidate in candidates:
            layout = {name: candidate / name for name in ("train_lr", "train_hr", "test_lr", "test_hr")}
            if all(path.is_dir() and any(path.iterdir()) for path in layout.values()): return candidate, layout
    raise FileNotFoundError("Attach official OLI2MSI train_lr/train_hr/test_lr/test_hr")

DATA_ROOT, DATA_DIRS = discover_data()
def pairs(lr_dir, hr_dir):
    lr, hr = tiff_index(lr_dir), tiff_index(hr_dir)
    return [(key, lr[key], hr[key]) for key in sorted(lr.keys() & hr.keys())]
official_train, official_test = pairs(DATA_DIRS["train_lr"], DATA_DIRS["train_hr"]), pairs(DATA_DIRS["test_lr"], DATA_DIRS["test_hr"])
if not FAST_DEV_RUN and (len(official_train), len(official_test)) != (EXPECTED_TRAIN, EXPECTED_TEST):
    raise RuntimeError(f"Expected 5225/100, found {len(official_train)}/{len(official_test)}")
def assigned_split(key):
    return "val" if int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100 < VALIDATION_PERCENT else "train"
train_pairs = [pair for pair in official_train if assigned_split(pair[0]) == "train"]
val_pairs = [pair for pair in official_train if assigned_split(pair[0]) == "val"]
test_pairs = list(official_test)
if FAST_DEV_RUN: train_pairs, val_pairs, test_pairs = train_pairs[:16], val_pairs[:8], test_pairs[:8]
sets = {name: {pair[0] for pair in value} for name, value in (("train", train_pairs), ("val", val_pairs), ("test", test_pairs))}
if any(sets[a] & sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
    raise RuntimeError("Source leakage")
lock = {"count": len(test_pairs), "keys_sha256": hashlib.sha256("\n".join(sorted(sets["test"])).encode()).hexdigest(), "normalization": "clip(0,0.3)/0.3", "geometry": "160->480"}
if TEST_LOCK.exists() and json.loads(TEST_LOCK.read_text()) != lock: raise RuntimeError("Test protocol changed")
TEST_LOCK.write_text(json.dumps(lock, indent=2))
print(DATA_ROOT, {name: len(value) for name, value in sets.items()})
"""
    ),
    markdown("## 4. Restart-safe full-frame conversion"),
    code(
        r"""
from tqdm.auto import tqdm
from geodiff_gan.data.manifest import ManifestRecord, load_manifest, write_manifest

sample_paths = [path for _, lr, hr in (train_pairs + val_pairs + test_pairs)[:8] for path in (lr, hr)]
def p99(path):
    with rasterio.open(path) as source: return float(np.nanpercentile(source.read([1, 2, 3], out_dtype="float32"), 99))
sample_p99 = max(p99(path) for path in sample_paths)
DIVISOR = 1.0 if sample_p99 <= 1.5 else 10_000.0
if sample_p99 > 12_000: raise RuntimeError(f"Unsupported TIFF range: {sample_p99}")

def read_rgb(path, size):
    with rasterio.open(path) as source:
        if source.count < 3 or (source.height, source.width) != (size, size): raise ValueError(f"Bad geometry: {path}")
        image = source.read([1, 2, 3], out_dtype="float32") / DIVISOR
        valid = (source.read_masks([1, 2, 3]) > 0).all(0)
    valid &= np.isfinite(image).all(0)
    return np.clip(np.nan_to_num(image), 0, REFLECTANCE_MAX) / REFLECTANCE_MAX, valid

selected = {f"{split}:{key}": (split, key, lr, hr) for split, rows in (("train", train_pairs), ("val", val_pairs), ("test", test_pairs)) for key, lr, hr in rows}
old = load_manifest(MANIFEST) if MANIFEST.exists() else []
records_by_key = {record.source_product: record for record in old if record.source_product in selected and Path(record.patch).exists()}
status = Counter()
for record_key, (split, key, lr_path, hr_path) in tqdm(selected.items(), desc="OLI2MSI"):
    if record_key in records_by_key: status["existing"] += 1; continue
    try:
        lr, mlr = read_rgb(lr_path, LR_SIZE); hr, mhr = read_rgb(hr_path, HR_SIZE)
        valid_fraction = min(float(mlr.mean()), float(mhr.mean()))
        if valid_fraction < MINIMUM_VALID_FRACTION: raise ValueError(f"valid={valid_fraction}")
        destination = PATCH_ROOT / split / f"{key}.npz"; destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, lr=lr.astype(np.float16), clean_lr=lr.astype(np.float16), hr=hr.astype(np.float16), valid_mask_lr=mlr[None].astype(np.uint8), valid_mask_hr=mhr[None].astype(np.uint8), degradation=np.array([0.5, 0, 0, 0], np.float32))
        temporary.replace(destination)
        records_by_key[record_key] = ManifestRecord(patch=str(destination.resolve()), tile_id=key, split=split, row=0, col=0, valid_fraction=valid_fraction, source=PROTOCOL_ID, source_product=record_key, landsat_product=lr_path.name, sentinel_product=hr_path.name, scale=3)
        status["written"] += 1
    except Exception as error:
        status["quarantined"] += 1
        with QUARANTINE.open("a") as handle: handle.write(json.dumps({"key": key, "error": str(error)}) + "\n")
    if (status["written"] + status["quarantined"]) % 50 == 0: write_manifest(MANIFEST, list(records_by_key.values()))
write_manifest(MANIFEST, sorted(records_by_key.values(), key=lambda item: (item.split, item.tile_id)))
records = load_manifest(MANIFEST); split_counts = Counter(record.split for record in records)
if set(records_by_key) != set(selected): raise RuntimeError(f"Incomplete conversion; inspect {QUARANTINE}")
print(status, split_counts, sha256(MANIFEST)); save_state(manifest_sha256=sha256(MANIFEST), split_counts=dict(split_counts))
"""
    ),
    markdown("## 5. Geometry and visual audit"),
    code(
        r"""
import torch.nn.functional as F
from geodiff_gan.data import SentinelPatchDataset
record = next(record for record in records if record.split == "train")
with np.load(record.patch) as data: lr, hr = torch.from_numpy(data["lr"]).float(), torch.from_numpy(data["hr"]).float()
bicubic = F.interpolate(lr[None], size=hr.shape[-2:], mode="bicubic", align_corners=False)[0].clamp(0, 1)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for axis, image, title in zip(axes, (lr, bicubic, hr), ("Landsat 30 m, 160 native pixels", "Bicubic display, 480 pixels", "Sentinel 10 m, 480 pixels")):
    axis.imshow(image.permute(1, 2, 0).clamp(0, 1)); axis.set_title(title); axis.axis("off")
plt.tight_layout(); plt.show()
for crop in (32, 64, None):
    sample = SentinelPatchDataset(MANIFEST, split="train", scale=3, input_mode="paired", paired_lr_crop_size=crop, augment=False, random_degradation=False)[0]
    size = 160 if crop is None else crop
    assert sample["lr"].shape[-2:] == (size, size) and sample["hr"].shape[-2:] == (3 * size, 3 * size)
print("Aligned 32->96, 64->192, 160->480 verified; bicubic PSNR=", float(-10 * torch.log10((bicubic - hr).square().mean())))
"""
    ),
]

cells += [
    markdown("## 21. Full-frame validation at mandatory residual scale 1.0"),
    code(
        r"""
from geodiff_gan.fidelity import fidelity_acceptance_report
s = state(); validation = evaluate("dev_scale1", s["dev_trust_config"], s["dev_trust"], split="val", samples=FINAL_SAMPLES, steps=FINAL_STEPS, scale=1.0, limit=VAL_LIMIT)
rows = [json.loads(line) for line in (EVAL_ROOT / "dev_scale1" / "val" / "per_patch_metrics.jsonl").read_text().splitlines() if line]
acceptance = fidelity_acceptance_report(previous_base_psnr=s["previous_base_validation_psnr"], new_base_psnr=validation["base_psnr"], final_psnr=validation["psnr"], base_ssim=validation["base_ssim"], final_ssim=validation["ssim"], per_image_psnr_delta=[row["psnr_delta_vs_base"] for row in rows], bootstrap_samples=500 if FAST_DEV_RUN else 10_000)
(ARTIFACT_ROOT / "validation_metrics.json").write_text(json.dumps(validation, indent=2)); (ARTIFACT_ROOT / "pretest_acceptance.json").write_text(json.dumps(acceptance, indent=2))
display(pd.DataFrame([{**validation, **{f"gate_{key}": value for key, value in acceptance["checks"].items()}}])); print("PASS" if acceptance["passed"] else "FAIL"); save_state(validation=validation, acceptance=acceptance)
"""
    ),
    markdown("## 22. Validation-only global, local-scalar, and local-per-band oracle ceilings"),
    code(
        r"""
from geodiff_gan.fidelity import per_image_psnr, residual_oracle_variants
evaluation_dir = EVAL_ROOT / "dev_scale1" / "val"; oracle_rows = []
for result_path in tqdm(sorted(evaluation_dir.glob("*_uncertainty.npz")), desc="residual oracle"):
    with np.load(result_path) as result:
        patch = Path(str(result["source_patch"])); base = torch.from_numpy(result["base"])[None].float(); candidate = torch.from_numpy(result["candidate_residual"])[None].float()
    with np.load(patch) as data: target = torch.from_numpy(data["hr"])[None].float(); mask = torch.from_numpy(data["valid_mask_hr"])[None].float()
    variants = residual_oracle_variants(base, candidate, target, mask=mask, smoothing_window=9, ridge=1e-6)
    row = {"patch": str(patch), "base": float(per_image_psnr(base, target, mask)[0])}
    for name, payload in variants.items(): row[name] = float(per_image_psnr(payload["image"], target, mask)[0]); row[f"{name}_trust"] = float(payload["trust"].mean())
    oracle_rows.append(row)
oracle = pd.DataFrame(oracle_rows); oracle.to_csv(ARTIFACT_ROOT / "residual_oracle.csv", index=False)
summary = {"count": len(oracle), **{name: float((oracle[name] - oracle.base).mean()) for name in ("global_scalar", "local_scalar", "local_per_band")}}
(ARTIFACT_ROOT / "residual_oracle.json").write_text(json.dumps(summary, indent=2)); display(pd.DataFrame([summary]))
"""
    ),
    markdown("## 23. Validation ablations; never select residual scale 0"),
    code(
        r"""
s = state(); specifications = [("base_exact", 1, 1, 0.0), ("proposal_8step", 1, 8, 1.0), ("trust_2sample", 2, 20, 1.0), ("final_4sample", FINAL_SAMPLES, FINAL_STEPS, 1.0)]
ablation = []
for label, samples, steps, scale in specifications:
    metric = evaluate(f"ablation_{label}", s["dev_trust_config"], s["dev_trust"], "val", samples, steps, scale, VAL_LIMIT); ablation.append({"label": label, **metric})
ablation = pd.DataFrame(ablation); ablation.to_csv(ARTIFACT_ROOT / "ablations.csv", index=False)
display(ablation[["label", "psnr", "base_psnr", "psnr_delta_vs_base", "ssim", "fraction_beating_base_psnr", "edge_f1", "ergas", "sam_degrees"]])
"""
    ),
    markdown("## 24. Select the final checkpoint; optionally refit on all training pairs"),
    code(
        r"""
from dataclasses import replace as dataclass_replace
s = state()
if not s.get("acceptance", {}).get("passed"):
    final_config, final_checkpoint = Path(s["dev_trust_config"]), Path(s["dev_trust"])
    save_state(
        test_locked=True, refit_skipped=True,
        refit_reason="validation acceptance gates failed",
        final_config=str(final_config), final_checkpoint=str(final_checkpoint),
        final_checkpoint_sha256=checkpoint_sha256(final_checkpoint),
    )
    print("Validation gates failed. Test remains locked, but validation artifacts will still be plotted and bundled.")
elif not RUN_FINAL_REFIT:
    final_config, final_checkpoint = Path(s["dev_trust_config"]), Path(s["dev_trust"])
    save_state(
        refit_skipped=True,
        refit_reason=f"disabled by {EXECUTION_PROFILE} time budget",
        final_config=str(final_config), final_checkpoint=str(final_checkpoint),
        final_checkpoint_sha256=checkpoint_sha256(final_checkpoint),
    )
    print("All-data refit skipped; using the validation-selected development checkpoint:", final_checkpoint)
else:
    refit_records = [dataclass_replace(record, split="train") if record.split == "val" else record for record in records]; write_manifest(REFIT_MANIFEST, refit_records)
    counts = Counter(record.split for record in refit_records)
    if counts["train"] != len(train_pairs) + len(val_pairs) or counts["test"] != len(test_pairs): raise RuntimeError(counts)

    def refit(label, stage, parent, crop, updates, batch, accumulation, lr, objective, modules=None, epochs=1, sample_steps=None, trust_samples=None):
        overrides = None
        if sample_steps:
            overrides = {"joint_latent_source": "sampled", "joint_sample_steps": sample_steps, "trust_samples": trust_samples}
        path = locked_config(
            label, stage, parent=parent, manifest=REFIT_MANIFEST,
            model=s["winner_model"], crop=crop, max_steps=updates, epochs=epochs,
            batch=batch, accumulation=accumulation, lr=lr, objective=objective,
            modules=modules, validate_every=1, validation_limit=None,
            max_wall_minutes=0,
            training_overrides=overrides,
        )
        return path, train(label, path, stage, best=False)

    _, b32 = refit("refit_base32", "base", None, 32, UPDATES["base32"], 4, 4, 1e-4, BASE_WARM)
    _, b64 = refit("refit_base64", "base", b32, 64, UPDATES["base64"], 1, 16, 5e-5, BASE_WARM)
    _, base = refit("refit_base_full", "base", b64, None, 0, 1, 8, 1e-5, BASE_FULL, epochs=FULL_FRAME_EPOCHS)
    if s["spatial_head_enabled"]: _, base = refit("refit_base_spatial", "base", base, 64, UPDATES["spatial"], 1, 16, 1e-5, BASE_FULL)
    _, vae = refit("refit_vae", "vae", base, 64, UPDATES["vae"], 2, 8, 1e-4, VAE_LOSS, ["vae", "lr_encoder", "mapper", "decoder"])
    _, diffusion = refit("refit_diffusion", "diffusion", vae, 64, UPDATES["diffusion"], 2, 8, 1e-4, DIFFUSION_LOSS, ["diffusion"])
    _, proposal = refit("refit_proposal", "joint", diffusion, 64, UPDATES["proposal"], 1, 8, 2e-5, PROPOSAL_LOSS, ["lr_encoder", "mapper", "decoder"], sample_steps=8, trust_samples=1)
    final_config, final_checkpoint = refit("refit_trust", "joint", proposal, 64, UPDATES["trust"], 1, 8, 1e-5, TRUST_LOSS, ["trust_controller"], sample_steps=20, trust_samples=2)
    refit_checkpoints = {"base32": str(b32), "base64": str(b64), "base": str(base), "vae": str(vae), "diffusion": str(diffusion), "proposal": str(proposal), "trust": str(final_checkpoint)}
    save_state(
        refit_manifest_sha256=sha256(REFIT_MANIFEST), refit_checkpoints=refit_checkpoints,
        refit_checkpoint_sha256={key: checkpoint_sha256(path) for key, path in refit_checkpoints.items()},
        final_config=str(final_config), final_checkpoint=str(final_checkpoint),
        final_checkpoint_sha256=checkpoint_sha256(final_checkpoint),
    )
    print(final_checkpoint)
"""
    ),
    markdown("## 25. One-time official 100-pair test"),
    code(
        r"""
s = state()
if not RUN_FINAL_TEST or not s.get("acceptance", {}).get("passed"):
    test = None
    print("Official test skipped because the validation gates did not pass or RUN_FINAL_TEST=False.")
else:
    if "final_checkpoint" not in s: raise RuntimeError("No final checkpoint was selected")
    if not FAST_DEV_RUN and split_counts["test"] != 100: raise RuntimeError("All 100 test pairs are required")
    test = evaluate("official_test_locked", s["final_config"], s["final_checkpoint"], "test", 4, 20, 1.0, 8 if FAST_DEV_RUN else 100)
    criteria = {"psnr_35": test["psnr"] >= 35, "gain_005": test["psnr_delta_vs_base"] >= 0.05, "ssim_non_degradation": test["ssim"] >= test["base_ssim"], "count": test["count"] == (8 if FAST_DEV_RUN else 100)}
    report = {"metrics": test, "criteria": criteria, "passed": all(criteria.values())}; (ARTIFACT_ROOT / "official_test.json").write_text(json.dumps(report, indent=2)); display(pd.DataFrame([{**test, **criteria}])); save_state(final_test=report)
"""
    ),
    markdown("## 26. PSNR, SSIM, and paired-delta plots"),
    code(
        r"""
s = state(); validation, test = s["validation"], s.get("final_test", {}).get("metrics"); rows = [{"split": "validation", "base_psnr": validation["base_psnr"], "psnr": validation["psnr"], "base_ssim": validation["base_ssim"], "ssim": validation["ssim"]}]
if test: rows.append({"split": "test", "base_psnr": test["base_psnr"], "psnr": test["psnr"], "base_ssim": test["base_ssim"], "ssim": test["ssim"]})
table = pd.DataFrame(rows); x = np.arange(len(table)); fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for axis, left, right, ylabel in ((axes[0], "base_psnr", "psnr", "PSNR (dB)"), (axes[1], "base_ssim", "ssim", "SSIM")):
    axis.bar(x-0.18, table[left], 0.36, label="base"); axis.bar(x+0.18, table[right], 0.36, label="V2"); axis.set_xticks(x, table.split); axis.set_ylabel(ylabel); axis.legend()
fig.tight_layout(); fig.savefig(ARTIFACT_ROOT / "psnr_ssim.png", dpi=180); plt.show()
per_image = [json.loads(line) for line in (EVAL_ROOT / "dev_scale1" / "val" / "per_patch_metrics.jsonl").read_text().splitlines() if line]; delta = np.array([row["psnr_delta_vs_base"] for row in per_image])
plt.figure(figsize=(8, 4)); plt.hist(delta, bins=30); plt.axvline(0, color="red"); plt.title(f"{np.mean(delta>0):.1%} beat base"); plt.xlabel("PSNR delta (dB)"); plt.tight_layout(); plt.savefig(ARTIFACT_ROOT / "paired_delta.png", dpi=180); plt.show()
"""
    ),
    markdown("## 27. Indexed LR, base, V2, target, residual, and RGB trust"),
    code(
        r"""
def show_result(index=0, split="test"):
    s = state(); label = "official_test_locked" if split == "test" else "dev_scale1"
    if split == "test" and "final_test" not in s: raise RuntimeError("Test is still locked")
    root = EVAL_ROOT / label / split; rows = [json.loads(line) for line in (root / "per_patch_metrics.jsonl").read_text().splitlines() if line]; row = rows[index % len(rows)]; patch = Path(row["patch"])
    result_path = None
    for candidate in root.glob("*_uncertainty.npz"):
        with np.load(candidate) as item:
            if str(item["source_patch"]) == str(patch): result_path = candidate; break
    if result_path is None: raise FileNotFoundError(patch)
    with np.load(patch) as data: lr, hr = torch.from_numpy(data["lr"]).float(), torch.from_numpy(data["hr"]).float()
    with np.load(result_path) as data: base, output, residual, trust = (torch.from_numpy(data[key]).float() for key in ("base", "mean", "net_addition", "trust_map"))
    lr_view = F.interpolate(lr[None], size=hr.shape[-2:], mode="nearest")[0]; fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    titles = ("Landsat 30 m", f"Base {row['base_psnr']:.2f} dB", f"V2 {row['output_psnr']:.2f} dB", "Sentinel 10 m")
    for axis, image, title in zip(axes, (lr_view, base, output, hr), titles): axis.imshow(image.permute(1,2,0).clamp(0,1)); axis.set_title(title); axis.axis("off")
    plt.tight_layout(); plt.show(); fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    magnitude = max(0.01, float(residual.abs().quantile(0.995))); axes[0].imshow((residual.permute(1,2,0)/(2*magnitude)+0.5).clamp(0,1)); axes[0].set_title("accepted residual")
    for channel, name in enumerate(("red trust", "green trust", "blue trust")): axes[channel+1].imshow(trust[channel], cmap="viridis", vmin=0, vmax=1); axes[channel+1].set_title(name)
    for axis in axes: axis.axis("off")
    plt.tight_layout(); plt.show(); return row
show_result(0, "test" if state().get("final_test") else "val")
"""
    ),
    markdown("## 28. Bundle immutable evidence"),
    code(
        r"""
s = state(); stamp = time.strftime("%Y%m%d_%H%M%S"); bundle = WORK_ROOT / f"fidelity_trust_v2_bundle_{stamp}"; bundle.mkdir(parents=True)
for directory in (CONFIG_ROOT, ARTIFACT_ROOT, EVAL_ROOT): shutil.copytree(directory, bundle / directory.name, dirs_exist_ok=True)
for item in (STATE_PATH, MANIFEST, REFIT_MANIFEST, TEST_LOCK):
    if item.exists(): shutil.copy2(item, bundle / item.name)
checkpoint_paths = {
    key: Path(s[key]) for key in (
        "winner_race_checkpoint", "dev_base32", "dev_base64", "dev_base_final",
        "dev_vae", "dev_diffusion", "dev_proposal", "dev_trust",
        "final_checkpoint",
    ) if s.get(key)
}
checkpoint_paths.update({
    f"refit_{key}": Path(path)
    for key, path in s.get("refit_checkpoints", {}).items()
})
for key, path in checkpoint_paths.items():
    if path.exists():
        destination = bundle / "checkpoints" / f"{key}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
inventory = [{"path": str(path.relative_to(bundle)), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(bundle.rglob("*")) if path.is_file()]
(bundle / "inventory.json").write_text(json.dumps(inventory, indent=2)); archive = shutil.make_archive(str(bundle), "zip", root_dir=bundle); print(bundle, archive, len(inventory))
"""
    ),
    markdown(
        r"""
## Reporting rule

Always report the achieved number, the full-frame `160 -> 480` protocol, all
100 official test pairs, four samples, 20 diffusion steps, residual scale 1.0,
the execution profile, whether all-data refit was skipped, and every failed
gate. Do not present cropped `32 -> 96` paper results as if they used this
stricter full-frame protocol. A validation-only result is not a test result.
"""
    ),
]

cells += [
    markdown("## 12. Train the selected base on aligned 32 -> 96 crops"),
    code(
        r"""
s = state(); model = s["winner_model"]; parent = s.get("winner_race_checkpoint")
BASE32_CONFIG = locked_config("dev_base32", "base", parent=parent, model=model, crop=32, max_steps=UPDATES["base32"], batch=4, accumulation=4, lr=1e-4, objective=BASE_WARM, max_wall_minutes=STAGE_MINUTES["base32"])
if parent and int(s.get("attached_rdn_maturity", 0)) >= 2:
    DEV_BASE32 = Path(parent)
    print("Reusing attached crop-32-or-later RDN checkpoint:", DEV_BASE32)
else:
    DEV_BASE32 = train("dev_base32", BASE32_CONFIG, "base")
save_state(dev_base32=str(DEV_BASE32)); display(checkpoint_metrics(DEV_BASE32))
"""
    ),
    markdown("## 13. Refine the base on aligned 64 -> 192 crops"),
    code(
        r"""
s = state(); BASE64_CONFIG = locked_config("dev_base64", "base", parent=s["dev_base32"], model=s["winner_model"], crop=64, max_steps=UPDATES["base64"], batch=1, accumulation=16, lr=5e-5, objective=BASE_WARM, max_wall_minutes=STAGE_MINUTES["base64"])
if s.get("winner_race_checkpoint") and int(s.get("attached_rdn_maturity", 0)) >= 3:
    DEV_BASE64 = Path(s["winner_race_checkpoint"])
    print("Reusing attached crop-64-or-later RDN checkpoint:", DEV_BASE64)
else:
    DEV_BASE64 = train("dev_base64", BASE64_CONFIG, "base")
save_state(dev_base64=str(DEV_BASE64)); display(checkpoint_metrics(DEV_BASE64))
"""
    ),
    markdown("## 14. Optional full-frame pure-MSE fine-tuning"),
    code(
        r"""
s = state()
if FULL_FRAME_EPOCHS > 0:
    BASE_FULL_CONFIG = locked_config("dev_base_full", "base", parent=s["dev_base64"], model=s["winner_model"], crop=None, max_steps=0, epochs=FULL_FRAME_EPOCHS, batch=1, accumulation=8, lr=1e-5, objective=BASE_FULL, validate_every=1)
    DEV_BASE_FULL = train("dev_base_full", BASE_FULL_CONFIG, "base"); display(checkpoint_metrics(DEV_BASE_FULL))
else:
    BASE_FULL_CONFIG, DEV_BASE_FULL = BASE64_CONFIG, DEV_BASE64
    print("Full-frame training skipped by the time budget; full-frame validation is still mandatory.")
save_state(dev_base_full=str(DEV_BASE_FULL), dev_base_config=str(BASE_FULL_CONFIG))
"""
    ),
    markdown("## 15. Validation-only radiometric ceilings"),
    code(
        r"""
from torch.utils.data import DataLoader
from geodiff_gan.fidelity import radiometric_ceiling_report
def radiometric_scan(config_path, model_checkpoint):
    config = yaml.safe_load(Path(config_path).read_text()); dataset = SentinelPatchDataset(MANIFEST, split="val", scale=3, input_mode="paired", augment=False, random_degradation=False)
    model = GeoDiffGAN.from_config(config).cuda().eval(); load_checkpoint(model_checkpoint, model, strict=False, prefer_ema=True); rows = []
    for index, batch in enumerate(tqdm(DataLoader(dataset, batch_size=1, num_workers=2), desc="radiometric oracle")):
        if VAL_LIMIT is not None and index >= VAL_LIMIT: break
        lr, hr, mask = batch["lr"].cuda(), batch["hr"].cuda(), batch["valid_mask"].cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16): base = model.predict_base(lr)
        report = radiometric_ceiling_report(base.float(), hr, mask, local_window=31)
        rows.append({"base": float(report["base_psnr"].mean()), "affine": float(report["affine_psnr"].mean()), "local": float(report["local_low_frequency_psnr"].mean())})
    del model; table = pd.DataFrame(rows)
    summary = {"count": len(table), "base_psnr": float(table.base.mean()), "affine_gain_db": float((table.affine-table.base).mean()), "spatial_gain_beyond_affine_db": float((table.local-table.affine).mean())}
    summary["enable_spatial_head"] = summary["spatial_gain_beyond_affine_db"] >= 0.20
    table.to_csv(ARTIFACT_ROOT / "radiometric_oracle.csv", index=False); (ARTIFACT_ROOT / "radiometric_oracle.json").write_text(json.dumps(summary, indent=2)); return summary
s = state(); radiometric = radiometric_scan(s["dev_base_config"], s["dev_base_full"]); display(pd.DataFrame([radiometric])); save_state(radiometric_oracle=radiometric, spatial_head_enabled=radiometric["enable_spatial_head"])
"""
    ),
    markdown("## 16. Conditionally enable the spatial radiometric head"),
    code(
        r"""
s = state(); base_path, base_config, model = s["dev_base_full"], s["dev_base_config"], dict(s["winner_model"])
if s["spatial_head_enabled"] and RUN_SPATIAL_TRAINING:
    model["base_spatial_radiometric_calibration"] = True
    path = locked_config("dev_base_spatial", "base", parent=base_path, model=model, crop=64, max_steps=UPDATES["spatial"], batch=1, accumulation=16, lr=1e-5, objective=BASE_FULL, validate_every=2, max_wall_minutes=STAGE_MINUTES["spatial"])
    base_path, base_config = str(train("dev_base_spatial", path, "base")), str(path)
elif s["spatial_head_enabled"]:
    print("Spatial head has oracle support but is deferred by the 5-hour profile.")
else:
    print("Spatial head disabled: validation oracle gain < 0.20 dB")
save_state(winner_model=model, dev_base_final=base_path, dev_base_final_config=base_config)
"""
    ),
    markdown("## 17. Retrain residual VAE and deterministic proposal heads"),
    code(
        r"""
s = state(); VAE_CONFIG = locked_config("dev_vae", "vae", parent=s["dev_base_final"], model=s["winner_model"], crop=64, max_steps=UPDATES["vae"], batch=2, accumulation=8, lr=1e-4, objective=VAE_LOSS, modules=["vae", "lr_encoder", "mapper", "decoder"], max_wall_minutes=STAGE_MINUTES["vae"])
DEV_VAE = train("dev_vae", VAE_CONFIG, "vae"); save_state(dev_vae=str(DEV_VAE)); display(checkpoint_metrics(DEV_VAE))
"""
    ),
    markdown("## 18. Retrain diffusion for the new base residual"),
    code(
        r"""
s = state(); DIFFUSION_CONFIG = locked_config("dev_diffusion", "diffusion", parent=s["dev_vae"], model=s["winner_model"], crop=64, max_steps=UPDATES["diffusion"], batch=2, accumulation=8, lr=1e-4, objective=DIFFUSION_LOSS, modules=["diffusion"], max_wall_minutes=STAGE_MINUTES["diffusion"])
DEV_DIFFUSION = train("dev_diffusion", DIFFUSION_CONFIG, "diffusion"); save_state(dev_diffusion=str(DEV_DIFFUSION)); display(checkpoint_metrics(DEV_DIFFUSION))
"""
    ),
    markdown("## 19. Refine proposals with actual frozen-diffusion 8-step samples"),
    code(
        r"""
s = state(); PROPOSAL_CONFIG = locked_config(
    "dev_proposal", "joint", parent=s["dev_diffusion"], model=s["winner_model"],
    crop=64, max_steps=UPDATES["proposal"], batch=1, accumulation=8, lr=2e-5,
    objective=PROPOSAL_LOSS, modules=["lr_encoder", "mapper", "decoder"],
    validate_every=2, max_wall_minutes=STAGE_MINUTES["proposal"], training_overrides={
        "joint_latent_source": "sampled", "joint_sample_steps": 8,
        "trust_samples": 1, "validation_sample_steps": 20,
        "validation_samples": 2,
    },
)
DEV_PROPOSAL = train("dev_proposal", PROPOSAL_CONFIG, "joint"); save_state(dev_proposal=str(DEV_PROPOSAL)); display(checkpoint_metrics(DEV_PROPOSAL))
"""
    ),
    markdown("## 20. Calibrate per-band trust from two 20-step candidates"),
    code(
        r"""
s = state(); TRUST_CONFIG = locked_config(
    "dev_trust", "joint", parent=s["dev_proposal"], model=s["winner_model"],
    crop=64, max_steps=UPDATES["trust"], batch=1, accumulation=8, lr=1e-5,
    objective=TRUST_LOSS, modules=["trust_controller"], validate_every=1,
    max_wall_minutes=STAGE_MINUTES["trust"],
    training_overrides={
        "joint_latent_source": "sampled", "joint_sample_steps": 20,
        "trust_samples": 2, "validation_sample_steps": 20,
        "validation_samples": 4, "checkpoint_metric": "val_psnr_gain_vs_base",
        "checkpoint_mode": "max",
    },
)
DEV_TRUST = train("dev_trust", TRUST_CONFIG, "joint"); save_state(dev_trust=str(DEV_TRUST), dev_trust_config=str(TRUST_CONFIG)); display(checkpoint_metrics(DEV_TRUST))
"""
    ),
]

cells += [
    markdown("## 6. Locked config, checkpoint, and evaluation helpers"),
    code(
        r"""
from geodiff_gan.training.checkpoint import best_stage_checkpoint, checkpoint_sha256, latest_stage_checkpoint, load_checkpoint
TEMPLATE = yaml.safe_load((REPOSITORY_DIR / "configs" / "oli2msi_fidelity_trust_v2_3x.yaml").read_text())
LOSS_NAMES = list(TEMPLATE["training"]["loss_weights"])
def losses(**values):
    result = {name: 0.0 for name in LOSS_NAMES}; result.update({name: float(value) for name, value in values.items()}); return result
BASE_WARM = losses(mse=1, charbonnier=0.05, ssim=0.02)
BASE_FULL = losses(mse=1)
VAE_LOSS = losses(vae_reconstruction=1, kl=0.0001, mse=1, charbonnier=0.05, residual_supervision=0.25)
DIFFUSION_LOSS = losses(diffusion=1)
PROPOSAL_LOSS = losses(mse=1, charbonnier=0.05, ssim=0.02, residual_supervision=1, base_guard=10, local_excess_mse=5, evidence_improvement=0.1)
TRUST_LOSS = losses(mse=1, base_guard=20, local_excess_mse=20, trust_projection=1)
RDN = {"base_architecture": "fidelity_rdn", "base_embed_dim": 64, "base_rdn_blocks": 20, "base_rdn_layers": 6, "base_rdn_growth": 32}
SWIN = {"base_architecture": "fidelity_swinir_v2", "base_embed_dim": 120, "base_heads": 6, "base_swin_groups": 6, "base_swin_blocks_per_group": 6, "window_size": 8}

def locked_config(
    label, stage, parent=None, manifest=MANIFEST, model=None, crop=32,
    max_steps=0, epochs=1, batch=4, accumulation=4, lr=1e-4,
    objective=None, modules=None, validate_every=5,
    validation_limit=TRAIN_VALIDATION_LIMIT, max_wall_minutes=0,
    training_overrides=None,
):
    config = copy.deepcopy(TEMPLATE); config["data"].update({"manifest": str(manifest), "paired_lr_crop_size": crop})
    if model: config["model"].update(model)
    training = config["training"]
    training.update({
        "stage": stage, "output_dir": str(RUN_ROOT / label), "init_checkpoint": str(parent) if parent else None,
        "resume": None, "auto_resume": True, "epochs": epochs, "max_optimizer_steps": max_steps,
        "max_wall_time_minutes": max_wall_minutes,
        "batch_size": batch, "gradient_accumulation": accumulation, "learning_rate": lr, "weight_decay": 0,
        "loss_weights": objective or BASE_WARM, "validate_every": validate_every,
        "validation_limit": validation_limit,
        "lr_scheduler_type": "cosine_warmup" if max_steps else "none", "warmup_steps": min(1000, max_steps // 10),
        "ema_decay": 0.9999, "use_ema_for_evaluation": True, "init_use_ema": True,
    })
    if modules is None: training.pop("trainable_modules", None)
    else: training["trainable_modules"] = list(modules)
    if training_overrides:
        training.update(training_overrides)
    path = CONFIG_ROOT / f"{label}.yaml"; serialized = yaml.safe_dump(config, sort_keys=False)
    if path.exists() and path.read_text() != serialized and any((RUN_ROOT / label).glob("*.pt")): raise RuntimeError(f"Config changed after training: {path}")
    path.write_text(serialized); return path

def checkpoint(label, stage, best=True):
    directory = RUN_ROOT / label
    result = best_stage_checkpoint(directory, stage) if best else None
    result = result or latest_stage_checkpoint(directory, stage)
    if result is None: raise FileNotFoundError(directory)
    return result

def train(label, config, stage, best=True):
    run([sys.executable, "-m", "geodiff_gan.cli.train", "--config", config], cwd=REPOSITORY_DIR)
    result = checkpoint(label, stage, best); print(result, checkpoint_sha256(result)); return result

def checkpoint_metrics(path):
    return torch.load(path, map_location="cpu", weights_only=False).get("extra", {}).get("metrics", {})

def evaluate(label, config, model_checkpoint, split="val", samples=4, steps=20, scale=1.0, limit=None):
    output = EVAL_ROOT / label / split; output.mkdir(parents=True, exist_ok=True)
    config_data = yaml.safe_load(Path(config).read_text())
    config_manifest = Path(config_data["data"]["manifest"])
    protocol = {"checkpoint_sha256": checkpoint_sha256(model_checkpoint), "config_sha256": sha256(config), "manifest_sha256": sha256(config_manifest), "split": split, "samples": samples, "steps": steps, "scale": scale, "limit": limit}
    marker, metrics = output / "protocol.json", output / "metrics.json"
    if metrics.exists() and marker.exists() and not FORCE_REEVALUATE:
        if json.loads(marker.read_text()) != protocol: raise RuntimeError(f"Protocol mismatch: {output}")
        return json.loads(metrics.read_text())
    command = [sys.executable, "-m", "geodiff_gan.cli.evaluate", "--config", config, "--checkpoint", model_checkpoint, "--output", output, "--split", split, "--samples", samples, "--steps", steps, "--back-projection-steps", 0, "--residual-scale", scale, "--mode", "sr", "--device", "cuda", "--progress", "compact", "--no-text"]
    if limit is not None: command += ["--limit", limit]
    run(command, cwd=REPOSITORY_DIR); marker.write_text(json.dumps(protocol, indent=2)); return json.loads(metrics.read_text())
print("Helpers ready")
"""
    ),
    markdown("## 7. Remeasure the V1 base on this validation split when available"),
    code(
        r"""
old_checkpoint = Path(V1_BASE_CHECKPOINT_OVERRIDE) if V1_BASE_CHECKPOINT_OVERRIDE else None
old_config = Path(V1_CONFIG_OVERRIDE) if V1_CONFIG_OVERRIDE else None
if old_checkpoint is None:
    options = sorted(BASELINE_BUNDLE.rglob("base_best.pt")); old_checkpoint = options[-1] if options else None
if old_config is None and old_checkpoint:
    option = old_checkpoint.parent / "resolved_config.json"; old_config = option if option.exists() else None
previous_base_psnr, provenance = REPORTED_V1_VALIDATION_BASE_PSNR, "reported prior validation"
if old_checkpoint and old_checkpoint.exists() and old_config and old_config.exists():
    config = json.loads(old_config.read_text()) if old_config.suffix == ".json" else yaml.safe_load(old_config.read_text())
    config["data"].update({"manifest": str(MANIFEST), "input_mode": "paired"}); config["training"]["use_ema_for_evaluation"] = True
    path = CONFIG_ROOT / "v1_same_validation.yaml"; path.write_text(yaml.safe_dump(config, sort_keys=False))
    output = EVAL_ROOT / "v1_base_same_validation.json"
    run([sys.executable, "-m", "geodiff_gan.cli.baselines", "--config", path, "--output", output, "--base-checkpoint", old_checkpoint, "--split", "val", "--device", "cuda", "--progress", "compact"], cwd=REPOSITORY_DIR)
    previous_base_psnr = float(json.loads(output.read_text())["base"]["psnr"]); provenance = str(old_checkpoint)
print(previous_base_psnr, provenance); save_state(previous_base_validation_psnr=previous_base_psnr, baseline_provenance=provenance)
"""
    ),
    markdown("## 8. Configure architecture selection"),
    code(
        r"""
if RUN_ARCHITECTURE_RACE:
    RDN_RACE_CONFIG = locked_config("race_rdn", "base", model=RDN, crop=32, max_steps=UPDATES["race"], batch=4, accumulation=4, objective=BASE_WARM, max_wall_minutes=STAGE_MINUTES["race"])
    SWIN_RACE_CONFIG = locked_config("race_swin", "base", model=SWIN, crop=32, max_steps=UPDATES["race"], batch=4, accumulation=4, objective=BASE_WARM, max_wall_minutes=STAGE_MINUTES["race"])
    print(RDN_RACE_CONFIG, SWIN_RACE_CONFIG)
else:
    RDN_RACE_CONFIG = SWIN_RACE_CONFIG = None
    print("Architecture race skipped by", EXECUTION_PROFILE, "; using FidelityRDN. SwinIR remains available in the publication profile.")
"""
    ),
    markdown("## 9. Run or resume RDN"),
    code("""if RUN_ARCHITECTURE_RACE:\n    RDN_RACE = train('race_rdn', RDN_RACE_CONFIG, 'base')\n    display(checkpoint_metrics(RDN_RACE))\nelse:\n    RDN_RACE = None\n    print('RDN will be trained directly in the bounded base stage.')"""),
    markdown("## 10. Run or resume SwinIR-v2"),
    code("""if RUN_ARCHITECTURE_RACE:\n    SWIN_RACE = train('race_swin', SWIN_RACE_CONFIG, 'base')\n    display(checkpoint_metrics(SWIN_RACE))\nelse:\n    SWIN_RACE = None\n    print('SwinIR-v2 skipped: its measured runtime does not fit the 5-6 hour session.')"""),
    markdown("## 11. Select by PSNR, SSIM, then measured memory"),
    code(
        r"""
def peak_memory(config_path, path):
    config = yaml.safe_load(Path(config_path).read_text()); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = GeoDiffGAN.from_config(config).cuda().eval(); load_checkpoint(path, model, strict=False, prefer_ema=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16): model.predict_base(torch.zeros(1, 3, 160, 160, device="cuda"))
    torch.cuda.synchronize(); value = torch.cuda.max_memory_allocated() / 1024**2; del model; torch.cuda.empty_cache(); return value
def compatible_attached_rdn(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = payload.get("config", {}).get("model", {})
        return payload.get("stage") == "base" and model.get("base_architecture") == "fidelity_rdn" and int(model.get("scale", 0)) == 3
    except Exception:
        return False

def rdn_checkpoint_maturity(path):
    label = str(path).lower()
    return next((rank for token, rank in (("dev_base_final", 5), ("dev_base_spatial", 5), ("dev_base_full", 4), ("dev_base64", 3), ("dev_base32", 2), ("race_rdn", 1)) if token in label), 0)

def discover_attached_rdn():
    if ATTACHED_RDN_CHECKPOINT:
        path = Path(ATTACHED_RDN_CHECKPOINT)
        if not path.exists() or not compatible_attached_rdn(path):
            raise RuntimeError(f"ATTACHED_RDN_CHECKPOINT is missing or incompatible: {path}")
        return path
    if not AUTO_DISCOVER_ATTACHED_RDN:
        return None
    extracted_root = WORK_ROOT / "attached_checkpoint_bundles"
    for archive in sorted(Path("/kaggle/input").glob("**/fidelity_trust_v2_bundle_*.zip")):
        destination = extracted_root / archive.stem
        if not destination.exists():
            destination.mkdir(parents=True)
            shutil.unpack_archive(str(archive), str(destination))
            print("Extracted attached V2 bundle:", archive)
    candidates = []
    for root in (Path("/kaggle/input"), extracted_root):
        for pattern in ("**/dev_base*.pt", "**/race_rdn*.pt", "**/base_best.pt"):
            candidates.extend(root.glob(pattern))
    compatible = []
    for path in sorted(set(candidates)):
        if compatible_attached_rdn(path):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            maturity = rdn_checkpoint_maturity(path)
            compatible.append((maturity, int(payload.get("extra", {}).get("optimizer_step", 0)), int(payload.get("epoch", -1)), path))
    return max(compatible, default=(0, 0, -1, None), key=lambda item: (item[0], item[1], item[2]))[3]

if RUN_ARCHITECTURE_RACE:
    rows = []
    for name, config, path, definition in (("fidelity_rdn", RDN_RACE_CONFIG, RDN_RACE, RDN), ("fidelity_swinir_v2", SWIN_RACE_CONFIG, SWIN_RACE, SWIN)):
        metric = checkpoint_metrics(path); rows.append({"architecture": name, "val_psnr": metric["val_psnr"], "val_ssim": metric["val_ssim"], "memory_mib": peak_memory(config, path), "checkpoint": str(path), "definition": definition})
    race = pd.DataFrame(rows).sort_values("val_psnr", ascending=False).reset_index(drop=True)
    if abs(race.loc[0, "val_psnr"] - race.loc[1, "val_psnr"]) <= 0.03: race = race.sort_values(["val_ssim", "memory_mib"], ascending=[False, True]).reset_index(drop=True)
    display(race.drop(columns="definition")); winner = race.iloc[0]
    race_records = json.loads(race.drop(columns="definition").to_json(orient="records"))
    winner_architecture, winner_model, winner_checkpoint = str(winner["architecture"]), dict(winner["definition"]), str(winner["checkpoint"])
    winner_maturity = 1
else:
    attached = discover_attached_rdn()
    winner_architecture = "fidelity_rdn"
    winner_model = dict(torch.load(attached, map_location="cpu", weights_only=False)["config"]["model"]) if attached else dict(RDN)
    winner_checkpoint = str(attached) if attached else None
    winner_maturity = rdn_checkpoint_maturity(attached) if attached else 0
    race_records = [{"architecture": winner_architecture, "selection": "fixed by 5-hour budget", "checkpoint": winner_checkpoint}]
    print("Attached RDN warm start:" if attached else "No attached RDN found; starting RDN from bicubic identity.", attached or "")
save_state(
    winner_architecture=winner_architecture,
    winner_model=winner_model,
    winner_race_checkpoint=winner_checkpoint,
    attached_rdn_maturity=winner_maturity,
    race=race_records,
)
"""
    ),
]

# Feature blocks are assembled independently above. Sort numbered sections into
# their intended execution order, then validate every generated code cell.
import re

title_cells: list[dict[str, object]] = []
groups: dict[int, list[dict[str, object]]] = {}
active: int | None = None
for cell in cells:
    text = "".join(cell["source"])
    match = re.search(r"^##\s+(\d+)\.", text, flags=re.MULTILINE)
    if match:
        active = int(match.group(1))
        groups.setdefault(active, []).append(cell)
    elif re.search(r"^##\s+Reporting rule", text, flags=re.MULTILINE):
        active = 10_000
        groups.setdefault(active, []).append(cell)
    elif active is None:
        title_cells.append(cell)
    else:
        groups[active].append(cell)
missing = set(range(29)) - set(groups)
if missing:
    raise RuntimeError(f"Missing sections: {sorted(missing)}")
cells = title_cells + [cell for number in sorted(groups) for cell in groups[number]]
for cell in cells:
    if cell["cell_type"] == "code":
        ast.parse("".join(cell["source"]))

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
OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {OUTPUT} with {len(cells)} clean cells")
