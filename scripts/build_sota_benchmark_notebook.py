from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "dgx" / "GeoDiff_GAN_DGX_A100_SOTA_Benchmark.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown(
        """# Sentinel-2 4x SR: GeoDiff-GAN vs Recent Open-Source Models

This notebook trains recent super-resolution backbones on the **same Sentinel-2
manifest used by the GeoDiff-GAN three-variant notebook** and compares all methods
with identical RGB metrics.

## Important correction

There is no single "2026 satellite SR SOTA" leaderboard. Classical bicubic SR,
blind real-world SR, continuous-scale remote-sensing SR, and synthetic
Sentinel-2 40 m to 10 m SR are different tasks. This notebook therefore performs
a controlled architecture comparison on our task instead of copying scores from
unrelated datasets.

Default papers with official training repositories:

1. SwinIR, ICCV Workshops 2021.
2. HAT, CVPR 2023.
3. SRFormer, ICCV 2023.
4. DAT, ICCV 2023.
5. OmniSR, CVPR 2023.
6. TTST, IEEE TIP 2024, remote-sensing specific.
7. MFG-HMoE, IEEE GRSL 2025, remote-sensing specific.

FreMamba/FMSR, IEEE TMM 2024, is an optional eighth model because its compiled
Mamba CUDA extension is more environment-sensitive.

The default `official` mode retains each repository's published x4
reconstruction head, including PixelShuffle or MoE upsampling where used.
This is required for a claimable architecture comparison.

An optional `resize_conv_ablation` mode studies upsampling artifacts. It changes
the competitor architectures and therefore cannot be reported as a faithful
baseline against the named published methods.
"""
    ),
    markdown(
        """## 1. Runtime and paths

This notebook assumes the earlier DGX notebook has already:

- cloned this GeoDiff-GAN repository;
- prepared/quarantined patches;
- written the final train/val/test manifest;
- trained any GeoDiff variants you want to compare.

It does not recreate SAFE patches and does not delete prior outputs.
"""
    ),
    code(
        """from pathlib import Path
import json, os, shutil, subprocess, sys, time

THESIS_ROOT = Path.home() / "geodiff_dgx"
REPOSITORY_DIR = THESIS_ROOT / "geodiff-gan"
WORK_ROOT = THESIS_ROOT / "geodiff-output"
SOURCE_ROOT = THESIS_ROOT / "sota_sources"
BENCHMARK_ROOT = WORK_ROOT / "sota_benchmark"

candidate_manifests = [
    WORK_ROOT / "manifest_dgx_80_10_10.jsonl",
    WORK_ROOT / "manifest.jsonl",
    WORK_ROOT / "manifest_raw.jsonl",
]
MANIFEST = next((path for path in candidate_manifests if path.exists()), None)
if MANIFEST is None:
    raise FileNotFoundError(
        "No prepared manifest found. Run the data-preparation cells in the "
        "three-model DGX notebook first."
    )
if not REPOSITORY_DIR.exists():
    raise FileNotFoundError(f"GeoDiff-GAN repository missing: {REPOSITORY_DIR}")

PYTHON = sys.executable
SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)
print("Python:", PYTHON)
print("Repository:", REPOSITORY_DIR)
print("Manifest:", MANIFEST)
print("External sources:", SOURCE_ROOT)
print("Outputs:", BENCHMARK_ROOT)
"""
    ),
    markdown("## 2. Install shared dependencies and clone official repositories"),
    code(
        """def run(command, cwd=None, check=True):
    command = [str(value) for value in command]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=cwd, env=environment, check=check)

run([PYTHON, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"], cwd=REPOSITORY_DIR)
run([
    PYTHON, "-m", "pip", "install", "-q",
    "timm>=1.0.15", "einops>=0.8", "pandas>=2", "matplotlib>=3.8"
])

REPOSITORIES = {
    "SwinIR": "https://github.com/JingyunLiang/SwinIR.git",
    "HAT": "https://github.com/XPixelGroup/HAT.git",
    "SRFormer": "https://github.com/HVision-NKU/SRFormer.git",
    "DAT": "https://github.com/zhengchen1999/DAT.git",
    "OmniSR": "https://github.com/Francis0625/Omni-SR.git",
    "MFG-HMoE": "https://github.com/Mr-Bamboo/MFG-HMoE.git",
    "TTST": "https://github.com/XY-boy/TTST.git",
    "FreMamba": "https://github.com/XY-boy/FreMamba.git",
}
for directory, url in REPOSITORIES.items():
    destination = SOURCE_ROOT / directory
    if destination.exists():
        print("Keeping existing source:", destination)
    else:
        run(["git", "clone", "--depth", "1", url, destination])
"""
    ),
    markdown("## 3. Verify dataset splits and visualize the shared degradation"),
    code(
        """from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from geodiff_gan.data import SentinelPatchDataset

records = [
    json.loads(line)
    for line in MANIFEST.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
split_counts = Counter(record["split"] for record in records)
tile_counts = Counter((record["tile_id"], record["split"]) for record in records)
print("Split counts:", split_counts)
print("Unique tiles:", len({record["tile_id"] for record in records}))
if any(split_counts[name] == 0 for name in ("train", "val", "test")):
    raise RuntimeError(f"Manifest must contain train/val/test records: {split_counts}")

dataset = SentinelPatchDataset(
    MANIFEST,
    split="train",
    scale=4,
    caption_file=None,
    augment=False,
    random_degradation=False,
    degradation_seed=42,
    degradation_severity="mild",
)
figure, axes = plt.subplots(3, 4, figsize=(16, 12))
for row, index in enumerate(np.linspace(0, len(dataset) - 1, 3, dtype=int)):
    sample = dataset[int(index)]
    lr = sample["lr"]
    hr = sample["hr"]
    clean_lr = sample["clean_lr"]
    bicubic = F.interpolate(
        lr[None], size=hr.shape[-2:], mode="bicubic", align_corners=False
    )[0]
    panels = [
        (lr, "Observed LR"),
        (clean_lr, "Clean sensor LR"),
        (bicubic, "Bicubic x4"),
        (hr, "Target HR"),
    ]
    for column, (image, title) in enumerate(panels):
        axes[row, column].imshow(image.clamp(0, 1).permute(1, 2, 0))
        axes[row, column].set_title(title)
        axes[row, column].axis("off")
plt.tight_layout()
plt.show()
"""
    ),
    markdown(
        """## 4. Select benchmark profile

Use `screening` first. It can reject broken or clearly weak candidates without
spending the full three-day allocation. Use `paper` only for the shortlisted
models. Fair comparison is by optimizer updates, not epochs, because model
throughput differs substantially.
"""
    ),
    code(
        """PROFILE = "screening"  # smoke | screening | paper
ARCHITECTURE_MODE = "official"  # official | resize_conv_ablation
DEFAULT_MODELS = [
    "swinir", "hat", "srformer", "dat", "omnisr", "ttst", "mfghmoe"
]
INCLUDE_FREMAMBA = False
MODELS_TO_RUN = DEFAULT_MODELS + (["fremamba"] if INCLUDE_FREMAMBA else [])

PROFILES = {
    "smoke": {
        "max_updates": 20,
        "validate_every": 10,
        "validation_limit": 4,
        "test_limit": 4,
        "early_stopping_patience": 2,
    },
    "screening": {
        "max_updates": 5000,
        "validate_every": 500,
        "validation_limit": 32,
        "test_limit": 40,
        "early_stopping_patience": 4,
    },
    "paper": {
        "max_updates": 50000,
        "validate_every": 2000,
        "validation_limit": 128,
        "test_limit": 40,
        "early_stopping_patience": 6,
    },
}
if PROFILE not in PROFILES:
    raise ValueError(PROFILE)
SETTINGS = PROFILES[PROFILE]

BATCH_SETTINGS = {
    "swinir": (1, 8),
    "hat": (1, 8),
    "srformer": (1, 8),
    "dat": (1, 8),
    "omnisr": (4, 2),
    "ttst": (1, 8),
    "mfghmoe": (1, 8),
    "fremamba": (1, 8),
}
print("Profile:", PROFILE, SETTINGS)
print("Architecture mode:", ARCHITECTURE_MODE)
print("Models:", MODELS_TO_RUN)
"""
    ),
    markdown(
        """## 5. Optional FreMamba CUDA dependency

Run this cell only when `INCLUDE_FREMAMBA=True`. If installation fails, keep the
six-model default benchmark. Do not change the PyTorch build merely to force
Mamba installation during a limited DGX allocation.
"""
    ),
    code(
        """if INCLUDE_FREMAMBA:
    result = run(
        [
            PYTHON, "-m", "pip", "install",
            "causal-conv1d>=1.4.0", "mamba-ssm>=2.2.0", "thop"
        ],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "FreMamba dependencies failed. Set INCLUDE_FREMAMBA=False and "
            "continue with the six validated default models."
        )
else:
    print("FreMamba disabled; no CUDA extension installation needed.")
"""
    ),
    markdown(
        """## 6. Probe every architecture before training

Each probe builds the selected architecture and runs a real 32x32 to 128x128
forward pass. In `official` mode, the reported PixelShuffle modules are expected
for papers that use sub-pixel reconstruction.
"""
    ),
    code(
        """PROBE_RESULTS = {}
for model_name in MODELS_TO_RUN:
    command = [
        PYTHON, "-m", "geodiff_gan.cli.benchmark_sota",
        "--model", model_name,
        "--source-root", SOURCE_ROOT,
        "--manifest", MANIFEST,
        "--output", BENCHMARK_ROOT / ARCHITECTURE_MODE / model_name,
        "--architecture-mode", ARCHITECTURE_MODE,
        "--probe-only",
    ]
    completed = subprocess.run(
        [str(value) for value in command],
        cwd=REPOSITORY_DIR,
        text=True,
        capture_output=True,
    )
    print(completed.stdout)
    if completed.returncode != 0:
        print(completed.stderr)
        raise RuntimeError(f"Probe failed for {model_name}")
    PROBE_RESULTS[model_name] = json.loads(completed.stdout[completed.stdout.index("{"):])
print("All probes passed.")
"""
    ),
    markdown(
        """## 7. Train sequentially with automatic resume

- Training LR is regenerated randomly every sample access.
- Validation/test LR is deterministic with seed 42.
- Every model receives the same Charbonnier, SSIM, gradient, and wavelet losses.
- Only `best.pt` and `latest.pt` are retained.
- Rerunning this cell resumes from `latest.pt`.
- Models train one at a time so one user occupies only the allocated GPU.
"""
    ),
    code(
        """TRAINING_RESULTS = {}
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
        "--lr-crop", 64,
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
    TRAINING_RESULTS[model_name] = json.loads(
        (output / "test_metrics.json").read_text(encoding="utf-8")
    )
"""
    ),
    markdown("## 8. Training curves and early-stopping behavior"),
    code(
        """import pandas as pd

figure, axes = plt.subplots(
    len(MODELS_TO_RUN), 2, figsize=(14, 4 * len(MODELS_TO_RUN)), squeeze=False
)
for row, model_name in enumerate(MODELS_TO_RUN):
    history_path = (
        BENCHMARK_ROOT / ARCHITECTURE_MODE / PROFILE / model_name / "history.jsonl"
    )
    history = pd.DataFrame(
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    axes[row, 0].plot(history["update"], history["loss"], alpha=0.8)
    axes[row, 0].set_title(f"{model_name}: train loss")
    axes[row, 0].set_xlabel("optimizer update")
    validation = history.dropna(subset=["val_psnr"]) if "val_psnr" in history else history.iloc[:0]
    if len(validation):
        axes[row, 1].plot(validation["update"], validation["val_psnr"], marker="o", label="PSNR")
        second = axes[row, 1].twinx()
        second.plot(
            validation["update"], validation["val_ssim"],
            marker="s", color="tab:orange", label="SSIM"
        )
        second.set_ylabel("SSIM")
    axes[row, 1].set_title(f"{model_name}: validation")
    axes[row, 1].set_xlabel("optimizer update")
    axes[row, 1].set_ylabel("PSNR (dB)")
plt.tight_layout()
plt.show()
"""
    ),
    markdown("## 9. Merge benchmark metrics with prior GeoDiff-GAN outputs"),
    code(
        """rows = []
for model_name in MODELS_TO_RUN:
    output = BENCHMARK_ROOT / ARCHITECTURE_MODE / PROFILE / model_name
    metrics = json.loads((output / "test_metrics.json").read_text(encoding="utf-8"))
    run_info = json.loads((output / "run.json").read_text(encoding="utf-8"))
    rows.append({
        "family": (
            "faithful official architecture"
            if ARCHITECTURE_MODE == "official"
            else "resize-conv architecture ablation"
        ),
        "method": model_name,
        "architecture_mode": run_info["architecture_mode"],
        "source_revision": run_info.get("source_revision"),
        "parameters": run_info["parameters"],
        **metrics,
    })

for variant in ("small_improved", "medium", "large"):
    metrics_path = WORK_ROOT / "evaluation" / variant / "test" / "metrics.json"
    if metrics_path.exists():
        rows.append({
            "family": "GeoDiff-GAN",
            "method": f"geodiff_{variant}",
            "parameters": None,
            **json.loads(metrics_path.read_text(encoding="utf-8")),
        })

comparison = pd.DataFrame(rows)
preferred = [
    "family", "method", "architecture_mode", "source_revision",
    "parameters", "count", "l1", "psnr", "ssim",
    "edge_f1", "redegradation_l1", "lpips", "dists",
]
comparison = comparison[[name for name in preferred if name in comparison.columns]]
comparison = comparison.sort_values("psnr", ascending=False)
display(comparison.round(6))
comparison.to_csv(
    BENCHMARK_ROOT / ARCHITECTURE_MODE / PROFILE / "all_model_comparison.csv",
    index=False,
)
"""
    ),
    markdown("## 10. Metric comparison plots"),
    code(
        """plot_data = comparison.copy()
figure, axes = plt.subplots(2, 2, figsize=(16, 11))
metrics = [
    ("psnr", True, "PSNR (higher is better)"),
    ("ssim", True, "SSIM (higher is better)"),
    ("edge_f1", True, "Edge F1 (higher is better)"),
    ("redegradation_l1", False, "LR re-degradation L1 (lower is better)"),
]
colors = ["#2563eb" if family == "GeoDiff-GAN" else "#64748b" for family in plot_data["family"]]
for axis, (metric, higher, title) in zip(axes.flat, metrics):
    if metric not in plot_data:
        axis.axis("off")
        continue
    ordered = plot_data.sort_values(metric, ascending=not higher)
    ordered_colors = [
        "#2563eb" if family == "GeoDiff-GAN" else "#64748b"
        for family in ordered["family"]
    ]
    axis.barh(ordered["method"], ordered[metric], color=ordered_colors)
    axis.invert_yaxis()
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.show()
"""
    ),
    markdown(
        """## 11. Side-by-side qualitative comparison

The external harness saves the first five deterministic test predictions. The
cell aligns those images by filename and also loads matching GeoDiff stochastic
means when available.
"""
    ),
    code(
        """from PIL import Image

image_roots = {
    name: BENCHMARK_ROOT / ARCHITECTURE_MODE / PROFILE / name / "images" / "test"
    for name in MODELS_TO_RUN
}
reference_model = next(name for name, root in image_roots.items() if list(root.glob("*_hr.png")))
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
    ),
    markdown(
        """## 12. Interpretation rules

1. Do not call the highest PSNR model universally SOTA. It is best only under
   this dataset, split, degradation, update budget, and metric implementation.
2. Compare `edge_f1` and visual texture with PSNR/SSIM. A smooth model can score
   well while missing roads, roofs, and field boundaries.
3. Compare `redegradation_l1`. A sharp image that cannot reproduce the observed
   LR is less evidence-constrained.
4. The generic models test backbone strength. TTST, MFG-HMoE, and optional
   FreMamba test remote-sensing-specific inductive biases.
5. GeoDiff-GAN uses a more complex multi-stage and stochastic training pipeline.
   For a strict architecture-only claim, rerun GeoDiff under a matched update or
   wall-clock budget and report both comparisons.
6. `official` mode is architecturally faithful, but the models are retrained
   under our Sentinel-2 data and loss protocol. Do not present these numbers as
   the papers' original DIV2K/UCMerced/AID scores.
7. `resize_conv_ablation` changes competitor architectures. It can answer an
   upsampling-ablation question but cannot serve as a published-method baseline.
"""
    ),
    markdown("## 13. Backup manifest, metrics, histories, and checkpoints"),
    code(
        """BACKUP_ROOT = THESIS_ROOT / "backups"
BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
timestamp = time.strftime("%Y%m%d_%H%M%S")
archive_base = BACKUP_ROOT / f"sota_benchmark_{ARCHITECTURE_MODE}_{PROFILE}_{timestamp}"
archive_path = shutil.make_archive(
    str(archive_base),
    "gztar",
    root_dir=BENCHMARK_ROOT / ARCHITECTURE_MODE / PROFILE,
)
print("Backup:", archive_path)
print(
    "Copy this archive and the GeoDiff evaluation directory off the shared DGX "
    "before the allocation ends."
)
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
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUTPUT)
