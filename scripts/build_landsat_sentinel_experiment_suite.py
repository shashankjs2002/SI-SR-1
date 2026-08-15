from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "kaggle" / "GeoDiff_GAN_Kaggle_Landsat_Sentinel_3x_Experiment_Suite.ipynb"


def source(value: str) -> list[str]:
    value = textwrap.dedent(value).strip("\n") + "\n"
    return value.splitlines(keepends=True)


def markdown(value: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": source(value)}


def code(value: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source(value),
    }


cells: list[dict[str, object]] = [
    markdown(
        """
        # GeoDiff-GAN 3x Controlled Experiment Suite

        This notebook compares real Landsat 30 m to Sentinel-2 10 m super-resolution under
        three isolated experiment tracks:

        1. **RGB standard:** no PSNR-specific fidelity objective.
        2. **RGB fidelity:** PSNR/SSIM-first losses and validation-only residual selection.
        3. **Multispectral fidelity:** six real Landsat bands condition an RGB 10 m target.
        4. **Spatial K-fold cross-validation:** all three tracks, executed only after the
           single-split experiments are complete.

        Prepared data, checkpoints, evaluation caches, tables, figures, and folds use separate
        directories. Output panels are saved as PNG files and displayed inline. The last section
        starts a Gradio explorer for dataset inspection, per-model analysis, and same-image
        visual comparison.

        The two RGB tracks deliberately use the same observations for a controlled loss
        ablation. Each run receives its own immutable manifest snapshot, while the large patch
        arrays remain deduplicated to avoid wasting Kaggle storage.

        **Scientific scope:** per-tile spatial folds measure within-tile spatial generalization.
        They do not replace a separate unseen-tile geographic-generalization experiment.
        """
    ),
    markdown("## 1. Controls and isolated storage roots"),
    code(
        """
        from pathlib import Path
        import os, sys, subprocess, json, shutil, time

        REPOSITORY_URL = "https://github.com/shashankjs2002/SI-SR-1.git"
        REPOSITORY_BRANCH = "SR-3x"
        REPOSITORY_DIR = Path("/kaggle/working/geodiff-gan-sr3x-suite")
        SENTINEL_INPUT = Path("/kaggle/input")
        LANDSAT_INPUT = Path("/kaggle/input")
        SUITE_ROOT = Path("/kaggle/working/geodiff-ls-s2-3x-suite")

        FAST_DEV_RUN = True
        RUN_DATA_PREPARATION = True
        REBUILD_DATA = False
        MAX_PAIRS = 1 if FAST_DEV_RUN else None
        PATCH_SIZE = 384
        PATCH_STRIDE = 288
        MAX_DAY_GAP = 3
        MINIMUM_OVERLAP_FRACTION = 0.10
        MINIMUM_VALID_FRACTION = 0.95
        SPLIT_STRATEGY = "within-tile-spatial"
        TRAIN_FRACTION = 0.80
        VALIDATION_FRACTION = 0.10
        BANDPASS_ADJUSTMENT = "none"

        RUN_RGB_STANDARD = True
        RUN_RGB_FIDELITY = True
        RUN_MULTISPECTRAL_FIDELITY = True
        # Keep this enabled for progress-report metrics. Set False only when the
        # session has no Internet and LPIPS/DISTS weights are not cached.
        RUN_OPTIONAL_METRICS = True
        RUN_GRADIO = False

        K_FOLDS = 5
        RUN_CV_RGB_STANDARD = False
        RUN_CV_RGB_FIDELITY = False
        RUN_CV_MULTISPECTRAL_FIDELITY = False
        CV_FOLDS_TO_RUN = list(range(K_FOLDS))

        EPOCHS = (
            {"base": 1, "vae": 1, "diffusion": 1, "joint": 1}
            if FAST_DEV_RUN
            else {"base": 20, "vae": 20, "diffusion": 60, "joint": 15}
        )
        LEARNING_RATES = {
            "base": 1e-4,
            "vae": 1e-4,
            "diffusion": 1e-4,
            "joint": 1e-5,
        }
        EVALUATION_SAMPLES = 2 if FAST_DEV_RUN else 4
        EVALUATION_STEPS = 2 if FAST_DEV_RUN else 20
        EVALUATION_LIMIT = 4 if FAST_DEV_RUN else 100
        VALIDATION_LIMIT = 4 if FAST_DEV_RUN else 64
        FIDELITY_RESIDUAL_SCALES = [0.0, 0.25, 0.5, 0.75, 1.0]
        FIDELITY_PROJECTION_STEPS = [0]
        SAVED_EXAMPLES_PER_EXPERIMENT = 3 if FAST_DEV_RUN else 8
        COMPARISON_COMMON_INDEX = 0
        RANDOM_SEED = 42

        DATA_ROOTS = {
            "rgb": SUITE_ROOT / "datasets" / "rgb",
            "multispectral": SUITE_ROOT / "datasets" / "multispectral",
        }
        EXPERIMENT_ROOT = SUITE_ROOT / "experiments"
        CV_ROOT = SUITE_ROOT / "cross_validation"
        SUITE_TABLE_ROOT = SUITE_ROOT / "tables"
        SUITE_FIGURE_ROOT = SUITE_ROOT / "figures"
        GRADIO_ROOT = SUITE_ROOT / "gradio"
        for path in (
            *DATA_ROOTS.values(), EXPERIMENT_ROOT, CV_ROOT,
            SUITE_TABLE_ROOT, SUITE_FIGURE_ROOT, GRADIO_ROOT,
        ):
            path.mkdir(parents=True, exist_ok=True)

        suite_settings = {
            "fast_dev_run": FAST_DEV_RUN,
            "patch_size": PATCH_SIZE,
            "patch_stride": PATCH_STRIDE,
            "max_day_gap": MAX_DAY_GAP,
            "split_strategy": SPLIT_STRATEGY,
            "k_folds": K_FOLDS,
            "epochs": EPOCHS,
            "evaluation_samples": EVALUATION_SAMPLES,
            "evaluation_steps": EVALUATION_STEPS,
            "optional_metrics": RUN_OPTIONAL_METRICS,
        }
        (SUITE_ROOT / "suite_settings.json").write_text(
            json.dumps(suite_settings, indent=2), encoding="utf-8"
        )

        def run(command, cwd=None, check=True):
            command = [str(value) for value in command]
            environment = os.environ.copy()
            source_root = REPOSITORY_DIR / "src"
            if source_root.exists():
                environment["PYTHONPATH"] = os.pathsep.join(
                    value for value in (
                        str(source_root), environment.get("PYTHONPATH", "")
                    ) if value
                )
            environment["PYTHONUNBUFFERED"] = "1"
            print("+", " ".join(command), flush=True)
            return subprocess.run(command, cwd=cwd, env=environment, check=check)

        print("Suite root:", SUITE_ROOT)
        print("Fast development run:", FAST_DEV_RUN)
        print("Cross-validation enabled:", {
            "rgb_standard": RUN_CV_RGB_STANDARD,
            "rgb_fidelity": RUN_CV_RGB_FIDELITY,
            "multispectral_fidelity": RUN_CV_MULTISPECTRAL_FIDELITY,
        })
        """
    ),
    markdown("## 2. Clone, install, and verify the runtime"),
    code(
        """
        if REPOSITORY_DIR.exists() and not (REPOSITORY_DIR / ".git").is_dir():
            raise RuntimeError(
                f"{REPOSITORY_DIR} exists but is not a Git clone. Choose another "
                "REPOSITORY_DIR; this notebook will not delete it."
            )
        if (REPOSITORY_DIR / ".git").is_dir():
            run(["git", "fetch", "origin", REPOSITORY_BRANCH], cwd=REPOSITORY_DIR)
            run(["git", "checkout", REPOSITORY_BRANCH], cwd=REPOSITORY_DIR)
            run(["git", "pull", "--ff-only"], cwd=REPOSITORY_DIR)
        else:
            run([
                "git", "clone", "--depth", "1", "--branch", REPOSITORY_BRANCH,
                REPOSITORY_URL, REPOSITORY_DIR,
            ])

        run([
            sys.executable, "-m", "pip", "install", "-q", "-r",
            REPOSITORY_DIR / "requirements-kaggle.txt",
        ])
        run([
            sys.executable, "-m", "pip", "install", "-q",
            "pandas", "matplotlib", "gradio>=4.44,<6", "plotly>=5.20",
        ])
        if RUN_OPTIONAL_METRICS:
            run([
                sys.executable, "-m", "pip", "install", "-q",
                "lpips>=0.1.4", "DISTS-pytorch>=0.1",
            ])
        run([
            sys.executable, "-m", "pip", "install", "-q", "-e",
            REPOSITORY_DIR, "--no-deps",
        ])
        source_root = str(REPOSITORY_DIR / "src")
        if source_root not in sys.path:
            sys.path.insert(0, source_root)

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import torch
        from torch.nn import functional as F
        from IPython.display import Image as DisplayImage, display

        print("Python:", sys.version)
        print("Torch:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
        """
    ),
    markdown("## 3. Discover paired Sentinel-2 and Landsat products"),
    code(
        """
        from geodiff_gan.data.sentinel import discover_safe_products
        from geodiff_gan.data.landsat_sentinel import (
            discover_landsat_products,
            pair_scenes,
        )

        sentinel_products = discover_safe_products(SENTINEL_INPUT)
        landsat_products = discover_landsat_products(LANDSAT_INPUT)
        print("Canonical Sentinel SAFE products:", len(sentinel_products))
        print("Complete Landsat products:", len(landsat_products))
        if not sentinel_products:
            raise FileNotFoundError("No canonical Sentinel-2 L2A SAFE products were found.")
        if not landsat_products:
            raise FileNotFoundError(
                "No complete Landsat C2 L2 products were found. Multispectral mode "
                "requires SR_B2 through SR_B7 plus QA_PIXEL and QA_RADSAT."
            )
        pairs, unmatched = pair_scenes(
            sentinel_products,
            landsat_products,
            max_day_gap=MAX_DAY_GAP,
            minimum_overlap_fraction=MINIMUM_OVERLAP_FRACTION,
        )
        pair_table = pd.DataFrame([{
            "sentinel": pair.sentinel.name,
            "landsat": pair.landsat.product_id,
            "day_gap": pair.day_gap,
            "overlap_fraction": pair.overlap_fraction,
            "has_B5_B6_B7": all(
                layer in pair.landsat.files for layer in ("SR_B5", "SR_B6", "SR_B7")
            ),
        } for pair in pairs])
        display(pair_table)
        if not pairs:
            raise RuntimeError("No date-and-footprint-compatible scene pairs were found.")
        if not pair_table["has_B5_B6_B7"].all():
            print("WARNING: RGB can run, but multispectral preparation needs SR_B5-B7.")
        """
    ),
    markdown("## 4. Prepare the isolated RGB paired dataset"),
    code(
        """
        def prepare_dataset(dataset_name, include_multispectral):
            root = DATA_ROOTS[dataset_name]
            patch_root = root / "patches"
            manifest = root / "manifest.jsonl"
            state = root / "pairing_state.json"
            table_root = root / "tables"
            figure_root = root / "figures"
            for path in (patch_root, table_root, figure_root):
                path.mkdir(parents=True, exist_ok=True)
            if RUN_DATA_PREPARATION:
                command = [
                    sys.executable, "-m", "geodiff_gan.cli.prepare_landsat_sentinel",
                    "--sentinel-input", SENTINEL_INPUT,
                    "--landsat-input", LANDSAT_INPUT,
                    "--output", patch_root,
                    "--manifest", manifest,
                    "--state", state,
                    "--patch-size", PATCH_SIZE,
                    "--stride", PATCH_STRIDE,
                    "--max-day-gap", MAX_DAY_GAP,
                    "--minimum-overlap-fraction", MINIMUM_OVERLAP_FRACTION,
                    "--minimum-valid-fraction", MINIMUM_VALID_FRACTION,
                    "--bandpass-adjustment", BANDPASS_ADJUSTMENT,
                    "--split-strategy", SPLIT_STRATEGY,
                    "--train-fraction", TRAIN_FRACTION,
                    "--validation-fraction", VALIDATION_FRACTION,
                    "--unmatched-split", "train",
                ]
                if include_multispectral:
                    command.append("--include-multispectral")
                if MAX_PAIRS is not None:
                    command.extend(["--max-pairs", MAX_PAIRS])
                if REBUILD_DATA:
                    command.append("--rebuild")
                run(command, cwd=REPOSITORY_DIR)
            if not manifest.exists():
                raise FileNotFoundError(f"Missing prepared manifest: {manifest}")
            return {
                "name": dataset_name,
                "root": root,
                "patch_root": patch_root,
                "manifest": manifest,
                "state": state,
                "table_root": table_root,
                "figure_root": figure_root,
                "multispectral": include_multispectral,
            }

        RGB_DATA = prepare_dataset("rgb", include_multispectral=False)
        print(RGB_DATA)
        """
    ),
    markdown("## 5. Prepare the isolated six-band multispectral paired dataset"),
    code(
        """
        MULTISPECTRAL_DATA = prepare_dataset(
            "multispectral",
            include_multispectral=True,
        )
        print(MULTISPECTRAL_DATA)
        print(
            "Multispectral channel order: R, G, B, NIR, SWIR1, SWIR2. "
            "All six are real Landsat OLI 30 m measurements."
        )
        """
    ),
    markdown("## 6. Audit and save dataset split statistics"),
    code(
        """
        from collections import Counter, defaultdict

        def read_jsonl(path):
            return [
                json.loads(line)
                for line in Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        def audit_dataset(dataset):
            records = read_jsonl(dataset["manifest"])
            rows = []
            for tile_id in sorted({record["tile_id"] for record in records}):
                tile_records = [r for r in records if r["tile_id"] == tile_id]
                counts = Counter(r["split"] for r in tile_records)
                rows.append({"tile_id": tile_id, **counts})
                for split in ("train", "val", "test"):
                    if not counts[split]:
                        raise RuntimeError(
                            f"{dataset['name']} tile {tile_id} is missing {split}."
                        )
            table = pd.DataFrame(rows).fillna(0)
            table.to_csv(dataset["table_root"] / "split_counts_by_tile.csv", index=False)
            active = [r for r in records if r["split"] in ("train", "val", "test")]
            sample_path = Path(active[0]["patch"])
            with np.load(sample_path) as sample:
                keys = sorted(sample.files)
                shapes = {key: list(sample[key].shape) for key in sample.files}
            summary = {
                "dataset": dataset["name"],
                "records_total": len(records),
                "records_active": len(active),
                "split_counts": dict(Counter(r["split"] for r in records)),
                "tiles": sorted({r["tile_id"] for r in records}),
                "sample_keys": keys,
                "sample_shapes": shapes,
            }
            (dataset["table_root"] / "dataset_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            display(table)
            print(json.dumps(summary, indent=2))
            return records, active, table

        RGB_RECORDS, RGB_ACTIVE, RGB_SPLIT_TABLE = audit_dataset(RGB_DATA)
        MS_RECORDS, MS_ACTIVE, MS_SPLIT_TABLE = audit_dataset(MULTISPECTRAL_DATA)
        """
    ),
    markdown("## 7. Save and display RGB and multispectral data diagnostics"),
    code(
        """
        def chw(array):
            tensor = torch.from_numpy(array).float()
            return tensor if tensor.shape[0] <= 16 else tensor.permute(2, 0, 1)

        def target_stretch(images, target, valid):
            values = target[:, valid]
            low = torch.quantile(values, 0.02)
            high = torch.quantile(values, 0.98).clamp_min(low + 1e-6)
            return [((image - low) / (high - low)).clamp(0, 1) for image in images]

        def unit_stretch(image, low_quantile=0.02, high_quantile=0.98):
            low = torch.quantile(image.float(), low_quantile)
            high = torch.quantile(image.float(), high_quantile).clamp_min(low + 1e-6)
            return ((image - low) / (high - low)).clamp(0, 1)

        def save_dataset_diagnostic(dataset, records, index=0):
            active = [r for r in records if r["split"] in ("train", "val", "test")]
            record = active[index % len(active)]
            with np.load(record["patch"]) as data:
                lr = chw(data["lr"])
                hr = chw(data["hr"])
                valid = torch.from_numpy(data["valid_mask_hr"][0]).bool()
                lr_ms = chw(data["lr_ms"]) if "lr_ms" in data.files else None
            bicubic = F.interpolate(
                lr[None], size=hr.shape[-2:], mode="bicubic", align_corners=False
            )[0].clamp(0, 1)
            panels = [(bicubic, "Landsat RGB bicubic"), (hr, "Sentinel-2 RGB target")]
            if lr_ms is not None:
                false_color = unit_stretch(lr_ms[[3, 0, 1]])
                red, nir, swir1 = lr_ms[0], lr_ms[3], lr_ms[4]
                ndvi = (nir - red) / (nir + red).clamp_min(1e-6)
                ndbi = (swir1 - nir) / (swir1 + nir).clamp_min(1e-6)
            displays = target_stretch([image for image, _ in panels], hr, valid)
            columns = 4 if lr_ms is not None else 2
            fig, axes = plt.subplots(1, columns, figsize=(5 * columns, 5), squeeze=False)
            for axis, image, (_, title) in zip(axes[0], displays, panels):
                axis.imshow(image.permute(1, 2, 0))
                axis.set_title(title)
                axis.axis("off")
            if lr_ms is not None:
                axes[0, 2].imshow(false_color.permute(1, 2, 0).clamp(0, 1))
                axes[0, 2].set_title("Landsat NIR/R/G false color")
                axes[0, 3].imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
                axes[0, 3].contour(ndbi.numpy(), levels=[0], colors="cyan", linewidths=0.5)
                axes[0, 3].set_title("NDVI; cyan=NDBI zero contour")
                axes[0, 2].axis("off")
                axes[0, 3].axis("off")
            fig.suptitle(
                f"{dataset['name']} | tile={record['tile_id']} | split={record['split']} | "
                f"gap={record.get('day_gap')} day(s)"
            )
            output = dataset["figure_root"] / f"dataset_index_{index:05d}.png"
            fig.tight_layout()
            fig.savefig(output, dpi=180, bbox_inches="tight")
            plt.show()
            display(DisplayImage(filename=str(output)))
            print("Saved:", output)
            return output

        RGB_DATA_FIGURE = save_dataset_diagnostic(RGB_DATA, RGB_RECORDS, 0)
        MS_DATA_FIGURE = save_dataset_diagnostic(MULTISPECTRAL_DATA, MS_RECORDS, 0)
        """
    ),
    markdown("## 8. Shared experiment, training, evaluation, and figure helpers"),
    code(
        """
        import copy
        import yaml
        from geodiff_gan.config import load_config

        STANDARD_LOSSES = {
            "charbonnier": 1.0,
            "mse": 0.0,
            "multiscale_mse": 0.0,
            "consistency": 0.0,
            "ssim": 0.2,
            "gradient": 0.1,
            "perceptual": 0.1,
            "wavelet": 0.05,
            "radiometric": 0.0,
            "residual_supervision": 0.0,
            "base_guard": 0.0,
            "kl": 0.0001,
            "vae_reconstruction": 1.0,
            "diffusion": 1.0,
            "evidence_calibration": 0.1,
            "evidence_improvement": 0.0,
            "adversarial": 0.01,
        }
        FIDELITY_LOSSES = {
            "charbonnier": 1.0,
            "mse": 50.0,
            "multiscale_mse": 25.0,
            "consistency": 0.0,
            "ssim": 0.5,
            "gradient": 0.05,
            "perceptual": 0.0,
            "wavelet": 0.01,
            "radiometric": 0.1,
            "residual_supervision": 0.5,
            "base_guard": 100.0,
            "kl": 0.0001,
            "vae_reconstruction": 1.0,
            "diffusion": 0.25,
            "evidence_calibration": 0.02,
            "evidence_improvement": 0.1,
            "adversarial": 0.0,
        }

        EXPERIMENT_SPECS = {
            "rgb_standard": {
                "dataset": RGB_DATA,
                "multispectral": False,
                "fidelity": False,
                "description": "RGB without fidelity-specific losses",
            },
            "rgb_fidelity": {
                "dataset": RGB_DATA,
                "multispectral": False,
                "fidelity": True,
                "description": "RGB PSNR/SSIM fidelity-first",
            },
            "multispectral_fidelity": {
                "dataset": MULTISPECTRAL_DATA,
                "multispectral": True,
                "fidelity": True,
                "description": "Six-band Landsat conditioning with RGB fidelity target",
            },
        }

        def experiment_paths(name, root=None):
            root = Path(root) if root is not None else EXPERIMENT_ROOT / name
            paths = {
                "root": root,
                "dataset": root / "dataset",
                "configs": root / "configs",
                "runs": root / "runs",
                "evaluation": root / "evaluation",
                "tables": root / "tables",
                "figures": root / "figures",
            }
            for path in paths.values():
                path.mkdir(parents=True, exist_ok=True)
            return paths

        def base_runtime_config(spec, manifest, paths):
            config = load_config(
                REPOSITORY_DIR / "configs" / "landsat_sentinel_3x_small.yaml",
                REPOSITORY_DIR / "configs" / "default.yaml",
            )
            config["data"].update({
                "manifest": str(manifest),
                "captions": None,
                "input_mode": "paired",
                "target_key": "hr",
                "condition_key": "lr_ms" if spec["multispectral"] else None,
                "train_degradation_sampling": "fixed",
            })
            config["model"].update({
                "scale": 3,
                "input_channels": 6 if spec["multispectral"] else 3,
                "output_channels": 3,
                "base_upsample_mode": "resize_conv",
                "decoder_upsample_mode": "resize_conv",
                "use_text_conditioning": False,
                "use_degradation_conditioning": False,
                "use_back_projection": True,
            })
            config["training"].update({
                "batch_size": 1,
                "gradient_accumulation": 1 if FAST_DEV_RUN else 8,
                "num_workers": 2,
                "amp": True,
                "gradient_checkpointing": True,
                "auto_resume": True,
                "progress_mode": "compact",
                "progress_updates_per_epoch": 2,
                "validate_every": 1,
                "validation_limit": VALIDATION_LIMIT,
                "keep_best_and_latest": True,
                "checkpoint_metric": "val_psnr",
                "checkpoint_mode": "max",
                "early_stopping_metric": "val_psnr",
                "early_stopping_mode": "max",
                "early_stopping_patience": 5,
                "early_stopping_min_epochs": 1 if FAST_DEV_RUN else 5,
                "train_back_projection_steps": 0,
                "validation_sample_steps": 2 if FAST_DEV_RUN else 10,
                "validation_samples": 1,
                "validation_back_projection_steps": 0,
                "base_guard_margin": 0.0,
                "evidence_improvement_temperature": 0.0001,
                "module_learning_rate_multipliers": {
                    "diffusion": 0.25,
                    "lr_encoder": 0.5,
                    "mapper": 1.0,
                    "decoder": 1.0,
                },
                "lr_scheduler_metric": "val_psnr",
                "lr_scheduler_mode": "max",
                "lr_scheduler_factor": 0.5,
                "lr_scheduler_patience": 2,
                "lr_scheduler_threshold": 1e-4,
                "lr_scheduler_min_lr": 1e-7,
            })
            config["training"]["loss_weights"].update(
                FIDELITY_LOSSES if spec["fidelity"] else STANDARD_LOSSES
            )
            config.setdefault("debug", {})
            config["debug"].update({
                "enabled": False,
                "output_dir": str(paths["root"] / "training_debug"),
                "max_exports_per_epoch": 0,
                "fail_on_nonfinite": True,
            })
            return config

        def select_checkpoint(output_dir, stage):
            output_dir = Path(output_dir)
            for name in (f"{stage}_best.pt", f"{stage}_latest.pt"):
                candidate = output_dir / name
                if candidate.exists():
                    return candidate
            candidates = sorted(output_dir.glob(f"{stage}_epoch_*.pt"))
            return candidates[-1] if candidates else None

        def train_experiment(name, spec, manifest=None, root=None, enabled=True):
            source_manifest = Path(manifest or spec["dataset"]["manifest"])
            paths = experiment_paths(name, root=root)
            manifest = paths["dataset"] / "manifest.jsonl"
            source_bytes = source_manifest.read_bytes()
            if manifest.exists() and manifest.read_bytes() != source_bytes:
                raise RuntimeError(
                    f"Dataset snapshot changed for {name}: {manifest}. Use a new "
                    "experiment root rather than mixing checkpoints with another split."
                )
            if not manifest.exists():
                manifest.write_bytes(source_bytes)
            config = base_runtime_config(spec, manifest, paths)
            checkpoints = {}
            configs = {}
            previous = None
            for stage in ("base", "vae", "diffusion", "joint"):
                stage_config = copy.deepcopy(config)
                stage_output = paths["runs"] / stage
                stage_output.mkdir(parents=True, exist_ok=True)
                stage_config["training"].update({
                    "stage": stage,
                    "epochs": EPOCHS[stage],
                    "learning_rate": LEARNING_RATES[stage],
                    "output_dir": str(stage_output),
                    "init_checkpoint": str(previous) if previous else None,
                    "resume": None,
                    "auto_resume": True,
                })
                if stage != "joint":
                    stage_config["training"]["module_learning_rate_multipliers"] = {}
                if stage == "diffusion":
                    stage_config["training"]["loss_weights"]["diffusion"] = 1.0
                    stage_config["training"].update({
                        "checkpoint_metric": "val_loss_total",
                        "checkpoint_mode": "min",
                        "early_stopping_metric": "val_loss_total",
                        "early_stopping_mode": "min",
                        "lr_scheduler_metric": "val_loss_total",
                        "lr_scheduler_mode": "min",
                    })
                config_path = paths["configs"] / f"{stage}.yaml"
                config_path.write_text(
                    yaml.safe_dump(stage_config, sort_keys=False), encoding="utf-8"
                )
                configs[stage] = config_path
                if enabled:
                    print(f"\\n===== {name}: {stage} =====")
                    run([
                        sys.executable, "-m", "geodiff_gan.cli.train",
                        "--config", config_path,
                    ], cwd=REPOSITORY_DIR)
                checkpoint = select_checkpoint(stage_output, stage)
                if checkpoint is None:
                    raise FileNotFoundError(
                        f"No {stage} checkpoint for {name}. Enable its RUN flag first."
                    )
                checkpoints[stage] = checkpoint
                previous = checkpoint
                curve = stage_output / "training_curves.png"
                if curve.exists():
                    destination = paths["figures"] / f"training_{stage}.png"
                    shutil.copy2(curve, destination)
                    display(DisplayImage(filename=str(destination)))
            descriptor = {
                "name": name,
                "description": spec["description"],
                "manifest": str(manifest),
                "multispectral": spec["multispectral"],
                "fidelity": spec["fidelity"],
                "paths": {key: str(value) for key, value in paths.items()},
                "configs": {key: str(value) for key, value in configs.items()},
                "checkpoints": {key: str(value) for key, value in checkpoints.items()},
            }
            (paths["root"] / "experiment.json").write_text(
                json.dumps(descriptor, indent=2), encoding="utf-8"
            )
            return {**descriptor, "paths": paths, "configs": configs, "checkpoints": checkpoints}

        def evaluation_command(result, output, split, residual_scale, projection_steps, limit):
            command = [
                sys.executable, "-m", "geodiff_gan.cli.evaluate",
                "--config", result["configs"]["joint"],
                "--checkpoint", result["checkpoints"]["joint"],
                "--output", output,
                "--split", split,
                "--samples", EVALUATION_SAMPLES,
                "--steps", EVALUATION_STEPS,
                "--back-projection-steps", projection_steps,
                "--residual-scale", residual_scale,
                "--mode", "sr",
                "--limit", limit,
                "--device", "cuda" if torch.cuda.is_available() else "cpu",
                "--progress", "compact",
                "--no-text",
            ]
            if RUN_OPTIONAL_METRICS:
                command.append("--optional-metrics")
            return command

        def render_saved_examples(name, result, output_dir, maximum):
            figure_dir = result["paths"]["figures"] / "test_examples"
            figure_dir.mkdir(parents=True, exist_ok=True)
            caches = sorted(Path(output_dir).glob("*_uncertainty.npz"))[:maximum]
            saved = []
            for example_index, cache_path in enumerate(caches):
                with np.load(cache_path) as cache:
                    prediction = chw(cache["mean"])
                    base = chw(cache["base"])
                    residual = chw(cache["net_addition"])
                    source_patch = Path(str(cache["source_patch"].item()))
                with np.load(source_patch) as patch:
                    lr = chw(patch["lr"])
                    hr = chw(patch["hr"])
                    valid = torch.from_numpy(patch["valid_mask_hr"][0]).bool()
                bicubic = F.interpolate(
                    lr[None], size=hr.shape[-2:], mode="bicubic", align_corners=False
                )[0].clamp(0, 1)
                displays = target_stretch([bicubic, base, prediction, hr], hr, valid)
                error = (prediction - hr).abs().mean(0) * valid
                residual_limit = float(torch.quantile(residual.abs(), 0.99).clamp_min(1e-6))
                residual_display = (0.5 + residual * (0.45 / residual_limit)).clamp(0, 1)
                fig, axes = plt.subplots(2, 3, figsize=(15, 10), squeeze=False)
                panels = [
                    (displays[0], "Landsat bicubic", "rgb"),
                    (displays[1], "SwinIR base", "rgb"),
                    (displays[2], f"{name} output", "rgb"),
                    (displays[3], "Sentinel target", "rgb"),
                    (residual_display, "Final correction (signed)", "rgb"),
                    (error, f"Absolute error; L1={float(error[valid].mean()):.5f}", "heat"),
                ]
                for axis, (image, title, kind) in zip(axes.flat, panels):
                    if kind == "rgb":
                        axis.imshow(image.permute(1, 2, 0))
                    else:
                        axis.imshow(image, cmap="turbo", vmin=0, vmax=max(0.03, float(torch.quantile(error[valid], 0.99))))
                    axis.set_title(title)
                    axis.axis("off")
                fig.suptitle(source_patch.stem)
                destination = figure_dir / f"test_example_{example_index:05d}.png"
                fig.tight_layout()
                fig.savefig(destination, dpi=180, bbox_inches="tight")
                plt.show()
                display(DisplayImage(filename=str(destination)))
                saved.append(destination)
            return saved

        def evaluate_experiment(name, result, select_fidelity):
            paths = result["paths"]
            if select_fidelity:
                sweep_rows = []
                for projection_steps in FIDELITY_PROJECTION_STEPS:
                    for residual_scale in FIDELITY_RESIDUAL_SCALES:
                        label = f"bp{projection_steps}_alpha{residual_scale:.2f}".replace(".", "p")
                        output = paths["evaluation"] / "validation_sweep" / label
                        output.mkdir(parents=True, exist_ok=True)
                        run(evaluation_command(
                            result, output, "val", residual_scale,
                            projection_steps, EVALUATION_LIMIT,
                        ), cwd=REPOSITORY_DIR)
                        metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
                        sweep_rows.append({
                            "residual_scale": residual_scale,
                            "projection_steps": projection_steps,
                            **metrics,
                        })
                sweep = pd.DataFrame(sweep_rows).sort_values(
                    ["psnr", "ssim"], ascending=False
                ).reset_index(drop=True)
                sweep.to_csv(paths["tables"] / "validation_fidelity_sweep.csv", index=False)
                display(sweep)
                selected = sweep.iloc[0].to_dict()
            else:
                selected = {"residual_scale": 1.0, "projection_steps": 0}
            selection_path = paths["tables"] / "selected_inference.json"
            selection_path.write_text(json.dumps(selected, indent=2), encoding="utf-8")

            test_output = paths["evaluation"] / "test"
            test_output.mkdir(parents=True, exist_ok=True)
            run(evaluation_command(
                result,
                test_output,
                "test",
                float(selected["residual_scale"]),
                int(selected["projection_steps"]),
                EVALUATION_LIMIT,
            ), cwd=REPOSITORY_DIR)
            baseline_path = paths["tables"] / "test_baselines.json"
            baseline_command = [
                sys.executable, "-m", "geodiff_gan.cli.baselines",
                "--config", result["configs"]["joint"],
                "--base-checkpoint", result["checkpoints"]["base"],
                "--output", baseline_path,
                "--split", "test",
                "--limit", EVALUATION_LIMIT,
                "--device", "cuda" if torch.cuda.is_available() else "cpu",
                "--progress", "compact",
            ]
            if RUN_OPTIONAL_METRICS:
                baseline_command.append("--optional-metrics")
            run(baseline_command, cwd=REPOSITORY_DIR)
            model_metrics = json.loads((test_output / "metrics.json").read_text(encoding="utf-8"))
            baselines = json.loads(baseline_path.read_text(encoding="utf-8"))
            rows = [{"experiment": name, "method": "GeoDiff-GAN", **model_metrics}]
            for method, values in baselines.items():
                if isinstance(values, dict):
                    rows.append({"experiment": name, "method": method, **values})
            table = pd.DataFrame(rows)
            table.to_csv(paths["tables"] / "test_metrics.csv", index=False)
            display(table)
            saved_examples = render_saved_examples(
                name, result, test_output, SAVED_EXAMPLES_PER_EXPERIMENT
            )
            return {
                "name": name,
                "selection": selected,
                "metrics": model_metrics,
                "baselines": baselines,
                "table": table,
                "test_output": test_output,
                "saved_examples": saved_examples,
            }
        """
    ),
    markdown(
        """
        ## 9. Experiment A: RGB without fidelity-specific losses

        This is the control experiment. MSE, multiscale MSE, radiometric matching,
        residual supervision, base guard, and evidence-improvement losses are disabled.
        Perceptual, wavelet, gradient, and low-weight adversarial objectives remain active.
        """
    ),
    code(
        """
        RGB_STANDARD_RESULT = train_experiment(
            "rgb_standard",
            EXPERIMENT_SPECS["rgb_standard"],
            enabled=RUN_RGB_STANDARD,
        )
        """
    ),
    markdown("### 9.1 Evaluate, save, and display RGB-standard outputs"),
    code(
        """
        RGB_STANDARD_EVALUATION = evaluate_experiment(
            "rgb_standard",
            RGB_STANDARD_RESULT,
            select_fidelity=False,
        )
        """
    ),
    markdown(
        """
        ## 10. Experiment B: RGB fidelity-first

        This experiment uses the same RGB data and architecture, but activates the
        PSNR/SSIM-focused objective. Residual scale is selected only on validation data.
        """
    ),
    code(
        """
        RGB_FIDELITY_RESULT = train_experiment(
            "rgb_fidelity",
            EXPERIMENT_SPECS["rgb_fidelity"],
            enabled=RUN_RGB_FIDELITY,
        )
        """
    ),
    markdown("### 10.1 Validation selection, test evaluation, and saved outputs"),
    code(
        """
        RGB_FIDELITY_EVALUATION = evaluate_experiment(
            "rgb_fidelity",
            RGB_FIDELITY_RESULT,
            select_fidelity=True,
        )
        """
    ),
    markdown(
        """
        ## 11. Experiment C: multispectral fidelity-first

        Input is six-channel Landsat OLI `[R,G,B,NIR,SWIR1,SWIR2]` at 30 m. Output and
        target remain Sentinel-2 RGB at 10 m. This isolates the value of additional
        spectral evidence without changing output dimensionality.
        """
    ),
    code(
        """
        MULTISPECTRAL_RESULT = train_experiment(
            "multispectral_fidelity",
            EXPERIMENT_SPECS["multispectral_fidelity"],
            enabled=RUN_MULTISPECTRAL_FIDELITY,
        )
        """
    ),
    markdown("### 11.1 Validation selection, test evaluation, and saved outputs"),
    code(
        """
        MULTISPECTRAL_EVALUATION = evaluate_experiment(
            "multispectral_fidelity",
            MULTISPECTRAL_RESULT,
            select_fidelity=True,
        )
        """
    ),
    markdown("## 12. Compare the three single-split experiments"),
    code(
        """
        EXPERIMENT_RESULTS = {
            "rgb_standard": RGB_STANDARD_RESULT,
            "rgb_fidelity": RGB_FIDELITY_RESULT,
            "multispectral_fidelity": MULTISPECTRAL_RESULT,
        }
        EXPERIMENT_EVALUATIONS = {
            "rgb_standard": RGB_STANDARD_EVALUATION,
            "rgb_fidelity": RGB_FIDELITY_EVALUATION,
            "multispectral_fidelity": MULTISPECTRAL_EVALUATION,
        }
        comparison_rows = [
            {"experiment": name, **evaluation["metrics"]}
            for name, evaluation in EXPERIMENT_EVALUATIONS.items()
        ]
        EXPERIMENT_COMPARISON = pd.DataFrame(comparison_rows)
        EXPERIMENT_COMPARISON.to_csv(
            SUITE_TABLE_ROOT / "single_split_model_comparison.csv", index=False
        )
        display(EXPERIMENT_COMPARISON)

        metrics_to_plot = [
            metric for metric in ("psnr", "ssim", "l1", "edge_f1", "lpips", "dists")
            if metric in EXPERIMENT_COMPARISON.columns
            and EXPERIMENT_COMPARISON[metric].notna().any()
        ]
        columns = min(3, max(1, len(metrics_to_plot)))
        rows = (len(metrics_to_plot) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(5.5 * columns, 4.5 * rows), squeeze=False)
        for axis, metric in zip(axes.flat, metrics_to_plot):
            axis.bar(EXPERIMENT_COMPARISON["experiment"], EXPERIMENT_COMPARISON[metric])
            better = "higher" if metric in ("psnr", "ssim", "edge_f1") else "lower"
            axis.set_title(f"{metric}: {better} is better")
            axis.tick_params(axis="x", rotation=20)
            axis.grid(axis="y", alpha=0.25)
        for axis in axes.flat[len(metrics_to_plot):]:
            axis.axis("off")
        comparison_figure = SUITE_FIGURE_ROOT / "single_split_metric_comparison.png"
        fig.tight_layout()
        fig.savefig(comparison_figure, dpi=180, bbox_inches="tight")
        plt.show()
        display(DisplayImage(filename=str(comparison_figure)))
        print("Saved:", comparison_figure)
        """
    ),
    markdown("## 13. Save a same-ground-patch visual comparison across all models"),
    code(
        """
        def record_identity(record):
            return (
                record.get("tile_id"), record.get("sentinel_product"),
                record.get("landsat_product"), int(record["row"]), int(record["col"]),
            )

        def test_record_maps():
            return {
                "rgb": {
                    record_identity(record): record
                    for record in RGB_RECORDS if record["split"] == "test"
                },
                "multispectral": {
                    record_identity(record): record
                    for record in MS_RECORDS if record["split"] == "test"
                },
            }

        TEST_MAPS = test_record_maps()
        COMMON_TEST_KEYS = sorted(set(TEST_MAPS["rgb"]) & set(TEST_MAPS["multispectral"]))
        if not COMMON_TEST_KEYS:
            raise RuntimeError("RGB and multispectral test manifests have no common patch.")

        def evaluate_common_patch(experiment_name, common_key):
            result = EXPERIMENT_RESULTS[experiment_name]
            evaluation = EXPERIMENT_EVALUATIONS[experiment_name]
            dataset_key = "multispectral" if result["multispectral"] else "rgb"
            manifest_records = [
                record for record in read_jsonl(result["manifest"])
                if record["split"] == "test"
            ]
            index_by_key = {
                record_identity(record): index for index, record in enumerate(manifest_records)
            }
            dataset_index = index_by_key[common_key]
            output = result["paths"]["evaluation"] / "indexed_common" / str(dataset_index)
            output.mkdir(parents=True, exist_ok=True)
            selection = evaluation["selection"]
            command = evaluation_command(
                result, output, "test",
                float(selection["residual_scale"]),
                int(selection["projection_steps"]),
                1,
            )
            limit_position = command.index("--limit")
            command[limit_position:limit_position + 2] = ["--index", dataset_index]
            cached = sorted(output.glob("*_uncertainty.npz"))
            if cached:
                cache_path = cached[0]
            else:
                run(command, cwd=REPOSITORY_DIR)
                cache_path = next(output.glob("*_uncertainty.npz"))
            with np.load(cache_path) as cache:
                return chw(cache["mean"]), chw(cache["base"]), cache_path

        common_key = COMMON_TEST_KEYS[COMPARISON_COMMON_INDEX % len(COMMON_TEST_KEYS)]
        rgb_record = TEST_MAPS["rgb"][common_key]
        with np.load(rgb_record["patch"]) as patch:
            comparison_lr = chw(patch["lr"])
            comparison_hr = chw(patch["hr"])
            comparison_valid = torch.from_numpy(patch["valid_mask_hr"][0]).bool()
        comparison_bicubic = F.interpolate(
            comparison_lr[None], size=comparison_hr.shape[-2:],
            mode="bicubic", align_corners=False,
        )[0].clamp(0, 1)
        common_outputs = {}
        for experiment_name in EXPERIMENT_RESULTS:
            common_outputs[experiment_name] = evaluate_common_patch(experiment_name, common_key)[0]
        images = [comparison_bicubic, *common_outputs.values(), comparison_hr]
        displays = target_stretch(images, comparison_hr, comparison_valid)
        titles = ["Landsat bicubic", *common_outputs.keys(), "Sentinel target"]
        fig, axes = plt.subplots(2, len(images), figsize=(4.5 * len(images), 9), squeeze=False)
        for column, (image, display_image, title) in enumerate(zip(images, displays, titles)):
            axes[0, column].imshow(display_image.permute(1, 2, 0))
            axes[0, column].set_title(title)
            axes[0, column].axis("off")
            error = (image - comparison_hr).abs().mean(0) * comparison_valid
            axes[1, column].imshow(error, cmap="turbo", vmin=0, vmax=max(0.03, float(torch.quantile(error[comparison_valid], 0.99))))
            axes[1, column].set_title(f"masked L1={float(error[comparison_valid].mean()):.5f}")
            axes[1, column].axis("off")
        same_image_figure = SUITE_FIGURE_ROOT / "same_patch_all_models.png"
        fig.suptitle(str(common_key))
        fig.tight_layout()
        fig.savefig(same_image_figure, dpi=180, bbox_inches="tight")
        plt.show()
        display(DisplayImage(filename=str(same_image_figure)))
        print("Saved:", same_image_figure)
        """
    ),
    markdown(
        """
        # Spatial K-fold cross-validation

        The following cells are intentionally last. Each profile receives independent fold
        manifests, configs, checkpoints, test caches, tables, and figures. Five folds across
        three four-stage models means fifteen complete training pipelines; enable only the
        profiles that fit the available allocation.
        """
    ),
    markdown("## 14. Create separate spatial K-fold manifests for RGB and multispectral data"),
    code(
        """
        from geodiff_gan.data.manifest import (
            build_within_tile_spatial_folds,
            load_manifest,
            write_manifest,
        )

        def create_fold_manifests(dataset):
            source_records = load_manifest(dataset["manifest"])
            fold_records, report = build_within_tile_spatial_folds(
                source_records,
                patch_size=PATCH_SIZE,
                folds=K_FOLDS,
            )
            root = CV_ROOT / "manifests" / dataset["name"]
            root.mkdir(parents=True, exist_ok=True)
            manifests = []
            for fold, records in enumerate(fold_records):
                path = root / f"fold_{fold:02d}.jsonl"
                write_manifest(path, records)
                manifests.append(path)
                counts = Counter(record.split for record in records)
                print(dataset["name"], "fold", fold, dict(counts))
            (root / "spatial_fold_report.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            return manifests, report

        RGB_FOLD_MANIFESTS, RGB_FOLD_REPORT = create_fold_manifests(RGB_DATA)
        MS_FOLD_MANIFESTS, MS_FOLD_REPORT = create_fold_manifests(MULTISPECTRAL_DATA)
        """
    ),
    markdown("## 15. Cross-validation helper"),
    code(
        """
        CV_RESULTS = {}

        def run_cross_validation(profile_name, enabled):
            spec = EXPERIMENT_SPECS[profile_name]
            manifests = MS_FOLD_MANIFESTS if spec["multispectral"] else RGB_FOLD_MANIFESTS
            profile_rows = []
            profile_results = {}
            for fold in CV_FOLDS_TO_RUN:
                fold_name = f"fold_{fold:02d}"
                fold_root = CV_ROOT / profile_name / fold_name
                result = train_experiment(
                    f"{profile_name}_{fold_name}",
                    spec,
                    manifest=manifests[fold],
                    root=fold_root,
                    enabled=enabled,
                )
                evaluation = evaluate_experiment(
                    f"{profile_name}_{fold_name}",
                    result,
                    select_fidelity=spec["fidelity"],
                )
                profile_results[fold] = {"result": result, "evaluation": evaluation}
                profile_rows.append({
                    "profile": profile_name,
                    "fold": fold,
                    **evaluation["metrics"],
                })
            table = pd.DataFrame(profile_rows)
            profile_root = CV_ROOT / profile_name
            profile_root.mkdir(parents=True, exist_ok=True)
            table.to_csv(profile_root / "fold_metrics.csv", index=False)
            if not table.empty:
                numeric = table.select_dtypes(include=[np.number]).columns
                summary = table[numeric].agg(["mean", "std"]).T.reset_index()
                summary.columns = ["metric", "mean", "std"]
                summary.to_csv(profile_root / "fold_mean_std.csv", index=False)
                display(table)
                display(summary)
            CV_RESULTS[profile_name] = profile_results
            return table
        """
    ),
    markdown("## 16. K-fold RGB-standard experiment"),
    code(
        """
        CV_RGB_STANDARD_TABLE = (
            run_cross_validation("rgb_standard", enabled=True)
            if RUN_CV_RGB_STANDARD
            else pd.DataFrame()
        )
        if not RUN_CV_RGB_STANDARD:
            print("Skipped. Set RUN_CV_RGB_STANDARD=True in Cell 1 to run it.")
        """
    ),
    markdown("## 17. K-fold RGB-fidelity experiment"),
    code(
        """
        CV_RGB_FIDELITY_TABLE = (
            run_cross_validation("rgb_fidelity", enabled=True)
            if RUN_CV_RGB_FIDELITY
            else pd.DataFrame()
        )
        if not RUN_CV_RGB_FIDELITY:
            print("Skipped. Set RUN_CV_RGB_FIDELITY=True in Cell 1 to run it.")
        """
    ),
    markdown("## 18. K-fold multispectral-fidelity experiment"),
    code(
        """
        CV_MULTISPECTRAL_TABLE = (
            run_cross_validation("multispectral_fidelity", enabled=True)
            if RUN_CV_MULTISPECTRAL_FIDELITY
            else pd.DataFrame()
        )
        if not RUN_CV_MULTISPECTRAL_FIDELITY:
            print("Skipped. Set RUN_CV_MULTISPECTRAL_FIDELITY=True in Cell 1 to run it.")
        """
    ),
    markdown("## 19. Aggregate and plot all available K-fold results"),
    code(
        """
        cv_tables = [
            table for table in (
                CV_RGB_STANDARD_TABLE,
                CV_RGB_FIDELITY_TABLE,
                CV_MULTISPECTRAL_TABLE,
            ) if not table.empty
        ]
        if cv_tables:
            CV_COMPARISON = pd.concat(cv_tables, ignore_index=True)
            CV_COMPARISON.to_csv(CV_ROOT / "all_fold_metrics.csv", index=False)
            metrics = [m for m in ("psnr", "ssim", "l1", "edge_f1", "lpips") if m in CV_COMPARISON]
            fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 5), squeeze=False)
            for axis, metric in zip(axes[0], metrics):
                groups = [
                    group[metric].dropna().to_numpy()
                    for _, group in CV_COMPARISON.groupby("profile")
                ]
                labels = [name for name, _ in CV_COMPARISON.groupby("profile")]
                axis.boxplot(groups, tick_labels=labels)
                axis.set_title(metric)
                axis.tick_params(axis="x", rotation=20)
                axis.grid(axis="y", alpha=0.25)
            cv_figure = SUITE_FIGURE_ROOT / "cross_validation_comparison.png"
            fig.tight_layout()
            fig.savefig(cv_figure, dpi=180, bbox_inches="tight")
            plt.show()
            display(DisplayImage(filename=str(cv_figure)))
        else:
            CV_COMPARISON = pd.DataFrame()
            print("No K-fold runs are enabled yet.")
        """
    ),
    markdown(
        """
        # Interactive analysis server

        The app reads saved patches and evaluation caches only. It does not retrain models.
        It provides dataset spectral inspection, model-specific result browsing, same-patch
        comparisons, error maps, experiment metrics, and available K-fold summaries.
        """
    ),
    markdown("## 20. Build and optionally launch the Gradio dataset/model explorer"),
    code(
        """
        import gradio as gr

        def to_rgb(image):
            image = image.detach().cpu().clamp(0, 1)
            return (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        def to_heatmap(image, cmap="turbo", vmin=None, vmax=None):
            image = image.detach().cpu().float().squeeze()
            if vmin is None:
                vmin = float(image.min())
            if vmax is None:
                vmax = float(torch.quantile(image.flatten(), 0.99))
            if vmax <= vmin:
                vmax = vmin + 1e-6
            normalized = ((image - vmin) / (vmax - vmin)).clamp(0, 1).numpy()
            colors = plt.get_cmap(cmap)(normalized)[..., :3]
            return (colors * 255).astype(np.uint8)

        def dataset_explorer(dataset_name, split, index):
            dataset = RGB_DATA if dataset_name == "rgb" else MULTISPECTRAL_DATA
            records = [r for r in read_jsonl(dataset["manifest"]) if r["split"] == split]
            if not records:
                raise gr.Error(f"No {split} records in {dataset_name}")
            index = int(index) % len(records)
            record = records[index]
            with np.load(record["patch"]) as patch:
                lr = chw(patch["lr"])
                hr = chw(patch["hr"])
                lr_ms = chw(patch["lr_ms"]) if "lr_ms" in patch.files else None
                valid = torch.from_numpy(patch["valid_mask_hr"][0]).bool()
            bicubic = F.interpolate(lr[None], size=hr.shape[-2:], mode="bicubic", align_corners=False)[0]
            displays = target_stretch([bicubic, hr], hr, valid)
            gallery = [(to_rgb(displays[0]), "Landsat RGB bicubic"), (to_rgb(displays[1]), "Sentinel target")]
            spectral = {
                "R": float(lr[0].mean()), "G": float(lr[1].mean()), "B": float(lr[2].mean())
            }
            if lr_ms is not None:
                false_color = unit_stretch(lr_ms[[3, 0, 1]])
                red, nir, swir1 = lr_ms[0], lr_ms[3], lr_ms[4]
                ndvi = (nir - red) / (nir + red).clamp_min(1e-6)
                ndbi = (swir1 - nir) / (swir1 + nir).clamp_min(1e-6)
                gallery.append((to_rgb(false_color), "NIR/R/G false color"))
                gallery.append((to_heatmap(ndvi, "RdYlGn", -1, 1), "NDVI"))
                gallery.append((to_heatmap(ndbi, "RdBu_r", -1, 1), "NDBI"))
                band_names = ("R", "G", "B", "NIR", "SWIR1", "SWIR2")
                for band_index, band_name in enumerate(band_names):
                    gallery.append((
                        to_heatmap(lr_ms[band_index], "viridis"),
                        f"Landsat {band_name}",
                    ))
                spectral.update({
                    "NIR": float(nir.mean()), "SWIR1": float(swir1.mean()),
                    "SWIR2": float(lr_ms[5].mean()), "NDVI_mean": float(ndvi.mean()),
                    "NDBI_mean": float(ndbi.mean()),
                })
            metadata = {**record, "patch": str(record["patch"]), "spectral_summary": spectral}
            return gallery, metadata

        def saved_model_gallery(experiment_name):
            evaluation = EXPERIMENT_EVALUATIONS[experiment_name]
            images = []
            for path in evaluation["saved_examples"]:
                images.append((str(path), path.stem))
            return images, evaluation["metrics"]

        def model_cache_gallery(experiment_name, cache_index):
            evaluation = EXPERIMENT_EVALUATIONS[experiment_name]
            caches = sorted(Path(evaluation["test_output"]).glob("*_uncertainty.npz"))
            if not caches:
                raise gr.Error(f"No saved test caches for {experiment_name}")
            cache_path = caches[int(cache_index) % len(caches)]
            with np.load(cache_path) as cache:
                prediction = chw(cache["mean"])
                base = chw(cache["base"])
                net_addition = chw(cache["net_addition"])
                decoder_residual = chw(cache["decoder_residual"])
                variance = torch.from_numpy(cache["variance"]).float()
                confidence = torch.from_numpy(cache["evidence_confidence"]).float()
                abstention = torch.from_numpy(cache["abstention_map"]).float()
                source_patch = Path(str(cache["source_patch"].item()))
            with np.load(source_patch) as patch:
                lr = chw(patch["lr"])
                hr = chw(patch["hr"])
                valid = torch.from_numpy(patch["valid_mask_hr"][0]).bool()
            bicubic = F.interpolate(
                lr[None], size=hr.shape[-2:], mode="bicubic", align_corners=False
            )[0].clamp(0, 1)
            displays = target_stretch([bicubic, base, prediction, hr], hr, valid)
            error = (prediction - hr).abs().mean(0)
            base_error = (base - hr).abs().mean(0)
            signed_limit = float(net_addition.abs().quantile(0.99).clamp_min(1e-6))
            residual_display = (0.5 + net_addition * (0.45 / signed_limit)).clamp(0, 1)
            decoder_limit = float(
                decoder_residual.abs().quantile(0.99).clamp_min(1e-6)
            )
            decoder_display = (
                0.5 + decoder_residual * (0.45 / decoder_limit)
            ).clamp(0, 1)
            error_limit = max(0.03, float(error[valid].quantile(0.99)))
            base_error_limit = max(0.03, float(base_error[valid].quantile(0.99)))
            gallery = [
                (to_rgb(displays[0]), "Landsat bicubic"),
                (to_rgb(displays[1]), "SwinIR base"),
                (to_rgb(displays[2]), f"{experiment_name} output"),
                (to_rgb(displays[3]), "Sentinel target"),
                (to_rgb(decoder_display), "Decoder residual (signed)"),
                (to_rgb(residual_display), "Final net addition (signed)"),
                (to_heatmap(error, "turbo", 0.0, error_limit), "Output absolute error"),
                (to_heatmap(base_error, "turbo", 0.0, base_error_limit), "Base absolute error"),
                (to_heatmap(variance, "magma"), "Stochastic variance"),
                (to_heatmap(confidence, "viridis", 0.0, 1.0), "Evidence confidence"),
                (to_heatmap(abstention, "inferno", 0.0, 1.0), "Abstention"),
            ]
            masked_l1 = float(error[valid].mean())
            masked_base_l1 = float(base_error[valid].mean())
            mse = float(((prediction - hr).pow(2).mean(0))[valid].mean())
            per_patch = {
                "experiment": experiment_name,
                "cache_index": int(cache_index) % len(caches),
                "cache": str(cache_path),
                "source_patch": str(source_patch),
                "output_masked_l1": masked_l1,
                "base_masked_l1": masked_base_l1,
                "output_psnr": float(-10 * np.log10(max(mse, 1e-12))),
                "output_improves_over_base_l1": masked_l1 < masked_base_l1,
                "mean_confidence": float(confidence.mean()),
                "mean_abstention": float(abstention.mean()),
                "mean_variance": float(variance.mean()),
            }
            return gallery, per_patch

        def compare_common_patch(common_index):
            key = COMMON_TEST_KEYS[int(common_index) % len(COMMON_TEST_KEYS)]
            record = TEST_MAPS["rgb"][key]
            with np.load(record["patch"]) as patch:
                lr = chw(patch["lr"])
                hr = chw(patch["hr"])
                valid = torch.from_numpy(patch["valid_mask_hr"][0]).bool()
            bicubic = F.interpolate(lr[None], size=hr.shape[-2:], mode="bicubic", align_corners=False)[0]
            outputs = {name: evaluate_common_patch(name, key)[0] for name in EXPERIMENT_RESULTS}
            images = [bicubic, *outputs.values(), hr]
            labels = ["Landsat bicubic", *outputs.keys(), "Sentinel target"]
            displays = target_stretch(images, hr, valid)
            gallery = []
            rows = []
            for image, display_image, label in zip(images, displays, labels):
                error = (image - hr).abs().mean(0)
                gallery.append((to_rgb(display_image), label))
                rows.append({"method": label, "masked_l1": float(error[valid].mean())})
            return gallery, pd.DataFrame(rows), {"identity": key, **record}

        with gr.Blocks(title="GeoDiff-GAN 3x Analysis") as ANALYSIS_APP:
            gr.Markdown("# GeoDiff-GAN Landsat-Sentinel 3x Analysis")
            with gr.Tab("Dataset explorer"):
                with gr.Row():
                    dataset_name = gr.Dropdown(["rgb", "multispectral"], value="rgb", label="Dataset")
                    dataset_split = gr.Dropdown(["train", "val", "test"], value="test", label="Split")
                    dataset_index = gr.Number(value=0, precision=0, label="Index")
                    dataset_button = gr.Button("Analyze patch")
                dataset_gallery = gr.Gallery(label="Sensor views", columns=3, height="auto")
                dataset_metadata = gr.JSON(label="Metadata and spectral evidence")
                dataset_button.click(
                    dataset_explorer,
                    [dataset_name, dataset_split, dataset_index],
                    [dataset_gallery, dataset_metadata],
                )
            with gr.Tab("Single-model results"):
                model_name = gr.Dropdown(list(EXPERIMENT_RESULTS), value="rgb_fidelity", label="Experiment")
                model_index = gr.Number(value=0, precision=0, label="Saved test-cache index")
                model_button = gr.Button("Analyze saved model output")
                model_gallery = gr.Gallery(label="Saved model internals", columns=3, height="auto")
                model_metrics = gr.JSON(label="Test metrics")
                model_button.click(
                    model_cache_gallery,
                    [model_name, model_index],
                    [model_gallery, model_metrics],
                )
            with gr.Tab("Same-patch comparison"):
                common_index = gr.Number(value=0, precision=0, label="Common test index")
                compare_button = gr.Button("Compare all models")
                compare_gallery = gr.Gallery(label="Aligned outputs", columns=3, height="auto")
                compare_table = gr.Dataframe(label="Per-image error")
                compare_metadata = gr.JSON(label="Patch identity and metadata")
                compare_button.click(
                    compare_common_patch,
                    common_index,
                    [compare_gallery, compare_table, compare_metadata],
                )
            with gr.Tab("Experiment metrics"):
                gr.Dataframe(EXPERIMENT_COMPARISON, label="Single-split metrics")
                if not CV_COMPARISON.empty:
                    gr.Dataframe(CV_COMPARISON, label="Cross-validation fold metrics")

        if RUN_GRADIO:
            ANALYSIS_APP.launch(
                share=False,
                inline=True,
                prevent_thread_lock=True,
                server_name="0.0.0.0",
            )
        else:
            print("Gradio app built but not launched. Set RUN_GRADIO=True in Cell 1.")
        """
    ),
    markdown("## 21. Export all reports without mixing or deleting source data"),
    code(
        """
        import zipfile
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive = Path("/kaggle/working") / f"geodiff_3x_experiment_suite_{timestamp}.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            for root in (EXPERIMENT_ROOT, CV_ROOT, SUITE_TABLE_ROOT, SUITE_FIGURE_ROOT):
                if not root.exists():
                    continue
                for path in root.rglob("*"):
                    if path.is_file():
                        handle.write(path, arcname=path.relative_to(SUITE_ROOT))
            for dataset in (RGB_DATA, MULTISPECTRAL_DATA):
                for key in ("manifest", "state"):
                    path = Path(dataset[key])
                    if path.exists():
                        handle.write(path, arcname=path.relative_to(SUITE_ROOT))
        print("Archive:", archive)
        print("Size GiB:", archive.stat().st_size / 2**30)
        print("Patch roots remain untouched:", RGB_DATA["patch_root"], MULTISPECTRAL_DATA["patch_root"])
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "kaggle": {"accelerator": "gpu", "dataSources": []},
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
print(f"Wrote {OUTPUT} with {len(cells)} cells")
