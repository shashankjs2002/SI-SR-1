from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "dgx" / "GeoDiff_GAN_Saved_Models_Evaluation.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip() + "\n"}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip() + "\n",
    }


cells = [
    md(
        """
        # Evaluate Saved GeoDiff-GAN and SOTA SR Models

        This notebook uses the downloaded `geodiff_dgx` backup and **does not train**.

        It:

        - repairs old absolute manifest paths into a separate runtime manifest;
        - discovers saved GeoDiff-GAN and SOTA checkpoints;
        - evaluates all available models on identical deterministic 4x LR inputs;
        - reports L1, PSNR, SSIM, edge F1, LR re-degradation error, LPIPS, and DISTS;
        - caches per-image outputs so interrupted evaluation resumes;
        - shows LR, bicubic, target, every external model, and every GeoDiff variant side by side
          for a user-selected test index.

        LPIPS/DISTS are perceptual metrics where **lower is better**. PSNR, SSIM, and edge F1 are
        **higher is better**. This benchmark remains specific to this dataset, split, degradation
        seed, and inference configuration.
        """
    ),
    md("## 1. Locate the extracted backup and repository"),
    code(
        r"""
        from pathlib import Path
        import os, sys, json, shutil, subprocess

        # Set this manually only if automatic discovery selects the wrong folder.
        BACKUP_PARENT = Path.cwd()
        GEODIFF_ROOT_OVERRIDE = None

        def valid_root(path):
            path = Path(path)
            return (
                (path / "geodiff-output").is_dir()
                and (path / "geodiff-gan").is_dir()
                and (path / "sota_sources").is_dir()
            )

        candidates = [BACKUP_PARENT, *BACKUP_PARENT.parents]
        # Search only below the current working directory. A recursive home-directory
        # scan can be extremely slow on machines with large mounted drives.
        candidates.extend(BACKUP_PARENT.glob("*/geodiff_dgx"))
        candidates.extend(BACKUP_PARENT.glob("*/*/geodiff_dgx"))
        if GEODIFF_ROOT_OVERRIDE is not None:
            candidates.insert(0, Path(GEODIFF_ROOT_OVERRIDE))
        THESIS_ROOT = next((path.resolve() for path in candidates if valid_root(path)), None)
        if THESIS_ROOT is None:
            raise FileNotFoundError(
                "Could not locate geodiff_dgx. Set GEODIFF_ROOT_OVERRIDE to the folder "
                "containing geodiff-output, geodiff-gan, and sota_sources."
            )

        REPOSITORY_DIR = THESIS_ROOT / "geodiff-gan"
        WORK_ROOT = THESIS_ROOT / "geodiff-output"
        SOURCE_ROOT = THESIS_ROOT / "sota_sources"
        RESULT_ROOT = WORK_ROOT / "saved_model_comparison"
        RESULT_ROOT.mkdir(parents=True, exist_ok=True)

        sys.path.insert(0, str(REPOSITORY_DIR / "src"))
        os.chdir(REPOSITORY_DIR)
        print("THESIS_ROOT:", THESIS_ROOT)
        print("Repository:", REPOSITORY_DIR)
        print("Output:", WORK_ROOT)
        print("SOTA sources:", SOURCE_ROOT)
        print("Comparison output:", RESULT_ROOT)
        """
    ),
    md("## 2. Install only evaluation dependencies"),
    code(
        r"""
        def run(command, cwd=None):
            command = [str(value) for value in command]
            print("+", " ".join(command), flush=True)
            subprocess.run(command, cwd=cwd, check=True)

        run([
            sys.executable, "-m", "pip", "install", "-q",
            "lpips>=0.1.4", "DISTS-pytorch>=0.1",
            "pandas>=2", "matplotlib>=3.7", "tqdm>=4.66",
            "einops>=0.8", "timm>=1.0.15",
        ])
        # FreMamba additionally needs mamba-ssm compiled for this Torch/CUDA installation.
        """
    ),
    md("## 3. Repair manifest paths without changing the downloaded manifest"),
    code(
        r"""
        import json
        from collections import Counter

        manifest_candidates = [
            WORK_ROOT / "manifest_dgx_80_10_10.jsonl",
            WORK_ROOT / "manifest_raw.jsonl",
            WORK_ROOT / "manifest.jsonl",
        ]
        SOURCE_MANIFEST = next((path for path in manifest_candidates if path.exists()), None)
        if SOURCE_MANIFEST is None:
            raise FileNotFoundError(manifest_candidates)
        RUNTIME_MANIFEST = RESULT_ROOT / "manifest_runtime_paths.jsonl"

        records = [
            json.loads(line)
            for line in SOURCE_MANIFEST.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        patch_files = {path.name: path for path in (WORK_ROOT / "patches").rglob("*.npz")}
        unresolved = []
        for record in records:
            original = Path(record["patch"])
            if original.exists():
                resolved = original
            else:
                resolved = patch_files.get(original.name)
            if resolved is None or not resolved.exists():
                unresolved.append(record["patch"])
            else:
                record["patch"] = str(resolved.resolve())

        if unresolved:
            raise FileNotFoundError(
                f"{len(unresolved)} manifest patches cannot be resolved. First examples: "
                f"{unresolved[:5]}"
            )
        RUNTIME_MANIFEST.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        print("Source manifest:", SOURCE_MANIFEST)
        print("Runtime manifest:", RUNTIME_MANIFEST)
        print("Records:", len(records), "splits:", Counter(r["split"] for r in records))
        """
    ),
    md("## 4. Evaluation controls"),
    code(
        r"""
        import torch

        SPLIT = "test"
        EVALUATION_LIMIT = None  # None evaluates every patch in SPLIT.
        DEGRADATION_SEED = 42
        DEGRADATION_SEVERITY = "mild"
        OPTIONAL_METRICS = True

        # GeoDiff is stochastic and much slower than deterministic competitors.
        GEODIFF_SAMPLES = 2
        GEODIFF_STEPS = 20
        GEODIFF_BACK_PROJECTION_STEPS = 3

        # Use the same model architecture mode as the stored SOTA checkpoint.
        # Usually "official"; it is read from each checkpoint when available.
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        AMP = DEVICE.type == "cuda"
        if DEVICE.type == "cuda":
            torch.backends.cudnn.benchmark = True
        print("Device:", DEVICE)
        if DEVICE.type == "cuda":
            print("GPU:", torch.cuda.get_device_name(0))
        """
    ),
    md("## 5. Discover all saved checkpoints"),
    code(
        r"""
        import re
        import pandas as pd

        from geodiff_gan.benchmark.models import MODEL_SPECS

        def newest(paths):
            paths = [path for path in paths if path.exists()]
            return max(paths, key=lambda path: path.stat().st_mtime) if paths else None

        SOTA_CHECKPOINTS = {}
        benchmark_root = WORK_ROOT / "sota_benchmark"
        for model_name in MODEL_SPECS:
            best = list(benchmark_root.rglob(f"{model_name}/best.pt"))
            latest = list(benchmark_root.rglob(f"{model_name}/latest.pt"))
            selected = newest(best) or newest(latest)
            if selected is not None:
                SOTA_CHECKPOINTS[model_name] = selected

        GEODIFF_CHECKPOINTS = {}
        GEODIFF_CONFIGS = {}
        runs_root = WORK_ROOT / "runs"
        configs_root = WORK_ROOT / "configs"
        for variant in ("small_improved", "medium", "large"):
            directory = runs_root / variant / "joint"
            candidates = [directory / "joint_best.pt"]
            candidates.extend(directory.glob("joint_epoch_*.pt"))
            selected = newest(candidates)
            config_candidates = [
                configs_root / variant / "joint.yaml",
                configs_root / variant / "template.yaml",
            ]
            config = next((path for path in config_candidates if path.exists()), None)
            if selected is not None and config is not None:
                GEODIFF_CHECKPOINTS[variant] = selected
                GEODIFF_CONFIGS[variant] = config

        discovery = []
        for name, path in SOTA_CHECKPOINTS.items():
            discovery.append({"family": "SOTA", "model": name, "checkpoint": str(path)})
        for name, path in GEODIFF_CHECKPOINTS.items():
            discovery.append({
                "family": "GeoDiff-GAN", "model": name, "checkpoint": str(path),
                "config": str(GEODIFF_CONFIGS[name]),
            })
        display(pd.DataFrame(discovery))
        if not SOTA_CHECKPOINTS:
            print("WARNING: no SOTA checkpoints found.")
        if not GEODIFF_CHECKPOINTS:
            print("WARNING: no GeoDiff joint checkpoints found.")
        """
    ),
    md("## 6. Shared deterministic dataset and metric helpers"),
    code(
        r"""
        import gc
        from collections import defaultdict

        import numpy as np
        import torch
        from torch.nn import functional as F
        from tqdm.auto import tqdm

        from geodiff_gan.data import SentinelPatchDataset
        from geodiff_gan.metrics import OptionalMetricSuite, basic_metrics

        dataset = SentinelPatchDataset(
            RUNTIME_MANIFEST,
            split=SPLIT,
            scale=4,
            caption_file=None,
            augment=False,
            random_degradation=False,
            degradation_seed=DEGRADATION_SEED,
            degradation_severity=DEGRADATION_SEVERITY,
        )
        TOTAL = len(dataset) if EVALUATION_LIMIT is None else min(len(dataset), EVALUATION_LIMIT)
        if TOTAL == 0:
            raise RuntimeError(f"No {SPLIT} patches")
        optional_suite = OptionalMetricSuite(DEVICE, enabled=OPTIONAL_METRICS)
        print("Evaluation patches:", TOTAL)

        def to_device(sample):
            return {
                key: value.unsqueeze(0).to(DEVICE)
                for key, value in sample.items()
                if torch.is_tensor(value)
            }

        def release_model(model):
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        """
    ),
    md("## 7. Evaluate saved SOTA checkpoints, with per-patch resume cache"),
    code(
        r"""
        from geodiff_gan.benchmark.models import build_benchmark_model

        def evaluate_sota(model_name, checkpoint):
            output_dir = RESULT_ROOT / "predictions" / f"sota_{model_name}"
            output_dir.mkdir(parents=True, exist_ok=True)
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            architecture_mode = state.get("architecture_mode", "official")
            model = build_benchmark_model(
                model_name, SOURCE_ROOT, architecture_mode=architecture_mode
            ).to(DEVICE).eval()
            model.load_state_dict(state["model"], strict=True)
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            totals = defaultdict(float)

            for index in tqdm(range(TOTAL), desc=f"SOTA {model_name}", unit="patch"):
                sample = dataset[index]
                cache = output_dir / f"{index:05d}.npz"
                tensors = to_device(sample)
                target = tensors["hr"]
                if cache.exists():
                    with np.load(cache) as values:
                        prediction = torch.from_numpy(values["prediction"]).unsqueeze(0).to(DEVICE)
                else:
                    with torch.inference_mode(), torch.autocast(
                        device_type=DEVICE.type, dtype=torch.float16, enabled=AMP
                    ):
                        prediction = model(tensors["lr"]).clamp(0, 1)
                    np.savez_compressed(
                        cache, prediction=prediction[0].float().cpu().numpy()
                    )
                metrics = basic_metrics(
                    prediction.float(), target.float(), tensors["clean_lr"].float(),
                    tensors["degradation"].float(), scale=4,
                    severity=DEGRADATION_SEVERITY,
                )
                metrics.update(optional_suite(prediction.float(), target.float()))
                for key, value in metrics.items():
                    totals[key] += value
            result = {key: value / TOTAL for key, value in totals.items()}
            result.update({
                "count": TOTAL,
                "parameters": parameter_count,
                "architecture_mode": architecture_mode,
                "checkpoint": str(checkpoint),
            })
            release_model(model)
            return result

        SOTA_RESULTS = {}
        for model_name, checkpoint in SOTA_CHECKPOINTS.items():
            try:
                SOTA_RESULTS[model_name] = evaluate_sota(model_name, checkpoint)
                (RESULT_ROOT / f"sota_{model_name}_metrics.json").write_text(
                    json.dumps(SOTA_RESULTS[model_name], indent=2), encoding="utf-8"
                )
            except Exception as error:
                print(f"FAILED {model_name}: {type(error).__name__}: {error}")
        """
    ),
    md("## 8. Evaluate saved GeoDiff-GAN variants"),
    code(
        r"""
        from geodiff_gan.config import load_config
        from geodiff_gan.models.system import GeoDiffGAN
        from geodiff_gan.training.checkpoint import load_checkpoint

        def evaluate_geodiff(variant, checkpoint, config_path):
            config = load_config(config_path)
            # Always use the repaired manifest and caption-free context for this comparison.
            config["data"]["manifest"] = str(RUNTIME_MANIFEST)
            model = GeoDiffGAN.from_config(config).to(DEVICE).eval()
            load_checkpoint(checkpoint, model, strict=False)
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            output_dir = RESULT_ROOT / "predictions" / f"geodiff_{variant}"
            output_dir.mkdir(parents=True, exist_ok=True)
            totals = defaultdict(float)

            for index in tqdm(range(TOTAL), desc=f"GeoDiff {variant}", unit="patch"):
                sample = dataset[index]
                cache = output_dir / f"{index:05d}.npz"
                tensors = to_device(sample)
                target = tensors["hr"]
                if cache.exists():
                    with np.load(cache) as values:
                        prediction = torch.from_numpy(values["prediction"]).unsqueeze(0).to(DEVICE)
                        uncertainty = torch.from_numpy(values["uncertainty"]).to(DEVICE)
                else:
                    lr = tensors["lr"]
                    context = torch.zeros(
                        1, 1, int(config["model"].get("context_dim", 768)),
                        device=DEVICE, dtype=lr.dtype,
                    )
                    with torch.inference_mode(), torch.autocast(
                        device_type=DEVICE.type, dtype=torch.float16, enabled=AMP
                    ):
                        base = model.base(lr)
                        lr_features = model.lr_encoder(lr)
                        outputs = []
                        for seed in range(GEODIFF_SAMPLES):
                            generator = torch.Generator(device=DEVICE).manual_seed(seed)
                            outputs.append(model.sample(
                                lr, context, degradation=tensors["degradation"],
                                projection_lr=tensors["clean_lr"], mode="sr",
                                sample_steps=GEODIFF_STEPS,
                                back_projection_steps=GEODIFF_BACK_PROJECTION_STEPS,
                                generator=generator, base=base, lr_features=lr_features,
                            ))
                        stack = torch.stack([output.image for output in outputs]).float()
                        raw_mean = stack.mean(0)
                        uncertainty = stack.var(0, unbiased=False).mean(1)
                        evidence = torch.stack(
                            [output.evidence_confidence for output in outputs]
                        ).float().mean(0)
                        prediction, _, _ = model.apply_uncertainty_abstention(
                            raw_mean, outputs[0].base, evidence, uncertainty
                        )
                    np.savez_compressed(
                        cache,
                        prediction=prediction[0].float().cpu().numpy(),
                        uncertainty=uncertainty[0].float().cpu().numpy(),
                    )
                metrics = basic_metrics(
                    prediction.float(), target.float(), tensors["clean_lr"].float(),
                    tensors["degradation"].float(), scale=4,
                    severity=DEGRADATION_SEVERITY,
                )
                metrics.update(optional_suite(prediction.float(), target.float()))
                for key, value in metrics.items():
                    totals[key] += value
            result = {key: value / TOTAL for key, value in totals.items()}
            result.update({
                "count": TOTAL,
                "parameters": parameter_count,
                "samples": GEODIFF_SAMPLES,
                "diffusion_steps": GEODIFF_STEPS,
                "checkpoint": str(checkpoint),
            })
            release_model(model)
            return result

        GEODIFF_RESULTS = {}
        for variant, checkpoint in GEODIFF_CHECKPOINTS.items():
            try:
                GEODIFF_RESULTS[variant] = evaluate_geodiff(
                    variant, checkpoint, GEODIFF_CONFIGS[variant]
                )
                (RESULT_ROOT / f"geodiff_{variant}_metrics.json").write_text(
                    json.dumps(GEODIFF_RESULTS[variant], indent=2), encoding="utf-8"
                )
            except Exception as error:
                print(f"FAILED GeoDiff {variant}: {type(error).__name__}: {error}")
        """
    ),
    md("## 9. Metric table and plots"),
    code(
        r"""
        rows = []
        for name, values in SOTA_RESULTS.items():
            rows.append({"family": "SOTA", "method": name, **values})
        for name, values in GEODIFF_RESULTS.items():
            rows.append({"family": "GeoDiff-GAN", "method": f"geodiff_{name}", **values})

        comparison = pd.DataFrame(rows)
        preferred = [
            "family", "method", "count", "parameters", "l1", "psnr", "ssim",
            "edge_f1", "redegradation_l1", "lpips", "dists",
            "architecture_mode", "samples", "diffusion_steps",
        ]
        comparison = comparison[[column for column in preferred if column in comparison]]
        comparison = comparison.sort_values(["lpips", "psnr"], ascending=[True, False])
        display(comparison.round(6))
        comparison.to_csv(RESULT_ROOT / "saved_models_comparison.csv", index=False)

        import matplotlib.pyplot as plt
        metrics_to_plot = [
            metric for metric in ("psnr", "ssim", "edge_f1", "lpips", "dists")
            if metric in comparison.columns
        ]
        figure, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 5))
        axes = np.atleast_1d(axes)
        for axis, metric in zip(axes, metrics_to_plot):
            ordered = comparison.sort_values(metric, ascending=metric in ("lpips", "dists", "l1"))
            axis.barh(ordered["method"], ordered[metric])
            axis.set_title(f"{metric} ({'lower' if metric in ('lpips','dists','l1') else 'higher'} is better)")
            axis.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        plt.show()
        """
    ),
    md("## 10. Side-by-side output for any test index"),
    code(
        r"""
        # Change only this value and rerun this cell.
        IMAGE_INDEX = 0
        if not 0 <= IMAGE_INDEX < TOTAL:
            raise IndexError(f"IMAGE_INDEX must be in [0, {TOTAL - 1}]")

        sample = dataset[IMAGE_INDEX]
        lr = sample["lr"]
        target = sample["hr"]
        bicubic = F.interpolate(
            lr.unsqueeze(0), size=target.shape[-2:],
            mode="bicubic", align_corners=False,
        )[0].clamp(0, 1)
        nearest_lr = F.interpolate(
            lr.unsqueeze(0), size=target.shape[-2:], mode="nearest"
        )[0]

        panels = [
            ("Input LR (nearest 4x)", nearest_lr),
            ("Bicubic", bicubic),
        ]
        for name in SOTA_RESULTS:
            path = RESULT_ROOT / "predictions" / f"sota_{name}" / f"{IMAGE_INDEX:05d}.npz"
            if path.exists():
                with np.load(path) as values:
                    panels.append((name, torch.from_numpy(values["prediction"])))
        for name in GEODIFF_RESULTS:
            path = RESULT_ROOT / "predictions" / f"geodiff_{name}" / f"{IMAGE_INDEX:05d}.npz"
            if path.exists():
                with np.load(path) as values:
                    panels.append((f"GeoDiff {name}", torch.from_numpy(values["prediction"])))
        panels.append(("Target HR", target))

        columns = 4
        rows_count = int(np.ceil(len(panels) / columns))
        figure, axes = plt.subplots(rows_count * 2, columns, figsize=(20, 8 * rows_count))
        axes = np.asarray(axes).reshape(rows_count * 2, columns)
        vmax = 0.08
        for position, (title, image) in enumerate(panels):
            block = position // columns
            column = position % columns
            image = image.float().clamp(0, 1)
            error = (image - target).abs().mean(0)
            axes[block * 2, column].imshow(image.permute(1, 2, 0))
            axes[block * 2, column].set_title(title)
            axes[block * 2, column].axis("off")
            axes[block * 2 + 1, column].imshow(error, cmap="turbo", vmin=0, vmax=vmax)
            axes[block * 2 + 1, column].set_title(f"absolute error: {float(error.mean()):.5f}")
            axes[block * 2 + 1, column].axis("off")
        for position in range(len(panels), rows_count * columns):
            block, column = position // columns, position % columns
            axes[block * 2, column].axis("off")
            axes[block * 2 + 1, column].axis("off")
        figure.suptitle(
            f"Test index {IMAGE_INDEX}: {Path(sample['patch']).name}",
            fontsize=16,
        )
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## Interpretation

        - Use the same `EVALUATION_LIMIT`, manifest, seed, and degradation severity for every row.
        - Do not compare old `test_metrics.json` files if they used different limits or optional metrics.
        - LPIPS and DISTS may download pretrained feature weights on first use.
        - A model can have higher PSNR but worse LPIPS or edge F1 because it produces smoother output.
        - Low re-degradation error means the SR output remains consistent with the observed LR evidence.
        - GeoDiff uses stochastic samples; record `GEODIFF_SAMPLES` and `GEODIFF_STEPS` in the paper.
        """
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
