from __future__ import annotations

import ast
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "kaggle" / "GeoDiff_GAN_Kaggle_Landsat30m_Sentinel10m_3x.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    normalized = dedent(source).strip() + "\n"
    ast.parse(normalized)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": normalized.splitlines(keepends=True),
    }


cells = [
    markdown(
        """
        # GeoDiff-GAN: Real Landsat 30 m to Sentinel-2 10 m Super-Resolution

        This notebook trains the `SR-3x` branch on real, geographically paired observations:

        ```text
        Landsat 8/9 C2 L2 RGB [3, 128, 128] at 30 m
                            -> GeoDiff-GAN 3x
        Sentinel-2 L2A RGB  [3, 384, 384] at 10 m
        ```

        This is cross-sensor super-resolution, not synthetic 4x degradation. The notebook:

        1. discovers and pairs overlapping Landsat and Sentinel products by acquisition date;
        2. applies official metadata scaling and both sensors' quality masks;
        3. creates resumable, exactly aligned 3x NPZ patches;
        4. audits geographic splits, dates, masks, geometry, and registration;
        5. trains base, VAE, diffusion, and joint stages with resume and early stopping;
        6. evaluates masked PSNR, SSIM, edge F1, ERGAS, SAM, UIQI, sCC, and LR consistency;
        7. exports intermediate features, diffusion trajectories, spectra, wavelets, policies,
           uncertainty, and back-projection diagnostics.

        Start with `FAST_DEV_RUN=True` and one scene pair. Do not start a full experiment until
        the edge-overlay registration plot shows the same roads, rivers, and field boundaries.
        """
    ),
    markdown("## 1. Experiment controls"),
    code(
        r"""
        from pathlib import Path
        import os, sys, subprocess, shutil, json, time

        REPOSITORY_URL = "https://github.com/shashankjs2002/SI-SR-1.git"
        REPOSITORY_BRANCH = "SR-3x"
        REPOSITORY_DIR = Path("/kaggle/working/geodiff-gan-sr3x")

        # Point these to the attached Kaggle datasets. Recursive discovery is supported.
        # Keeping /kaggle/input scans all attached datasets and is the easiest first run.
        SENTINEL_INPUT = Path("/kaggle/input")
        LANDSAT_INPUT = Path("/kaggle/input")
        WORK_ROOT = Path("/kaggle/working/geodiff-ls-s2-3x")

        FAST_DEV_RUN = True
        RUN_DATA_PREPARATION = True
        REBUILD_PATCHES = False
        AUTO_EXTRACT_ARCHIVES_IF_NEEDED = True
        MAX_PAIRS = 1 if FAST_DEV_RUN else None

        PATCH_SIZE = 384
        PATCH_STRIDE = 288
        MAX_DAY_GAP = 3
        MINIMUM_OVERLAP_FRACTION = 0.10
        MINIMUM_VALID_FRACTION = 0.95
        BANDPASS_ADJUSTMENT = "none"  # Required for S2C; optional hls-oli for S2A/S2B.

        # Prefixes are matched against Sentinel product names. Keep a complete MGRS tile in one split.
        VALIDATION_SENTINEL_PREFIXES = []
        TEST_SENTINEL_PREFIXES = []
        UNMATCHED_SPLIT = "train" if FAST_DEV_RUN else "hash"

        STAGES_TO_RUN = ["base", "vae", "diffusion", "joint"]
        AUTO_RESUME_TRAINING = True
        TRAINING_PROGRESS_MODE = "compact"  # compact, tqdm, or quiet
        TRAINING_DIAGNOSTICS = False  # Dedicated debug cells are more storage-efficient.
        VALIDATION_LIMIT = 4 if FAST_DEV_RUN else 64
        RUN_FIDELITY_FINETUNE_ABLATIONS = False
        FIDELITY_FINETUNE_EPOCHS = 1 if FAST_DEV_RUN else 5

        EPOCHS = (
            {"base": 1, "vae": 1, "diffusion": 1, "joint": 1}
            if FAST_DEV_RUN
            else {"base": 20, "vae": 20, "diffusion": 60, "joint": 15}
        )
        STAGE_LEARNING_RATES = {
            "base": 1e-4,
            "vae": 1e-4,
            "diffusion": 1e-4,
            "joint": 2e-5,
        }

        EVALUATION_PROFILE = "dev" if FAST_DEV_RUN else "quick"
        EVALUATION_PROFILES = {
            "dev": {"limit": 4, "samples": 2, "steps": 2},
            "quick": {"limit": 20, "samples": 2, "steps": 10},
            "full": {"limit": 100, "samples": 8, "steps": 20},
        }
        EVALUATION_BACK_PROJECTION_STEPS = 3
        DEBUG_INDEX = 0
        VISUALIZATION_SAMPLES = 5
        RANDOM_SEED = 42

        PATCH_ROOT = WORK_ROOT / "patches"
        MANIFEST = WORK_ROOT / "manifest.jsonl"
        PAIRING_STATE = WORK_ROOT / "manifest.jsonl.pairing.json"
        CONFIG_ROOT = WORK_ROOT / "configs"
        RUN_ROOT = WORK_ROOT / "runs"
        EVALUATION_ROOT = WORK_ROOT / "evaluation"
        DEBUG_ROOT = WORK_ROOT / "debug"
        EXTRACTION_ROOT = WORK_ROOT / "extracted_inputs"
        for path in (WORK_ROOT, PATCH_ROOT, CONFIG_ROOT, RUN_ROOT, EVALUATION_ROOT, DEBUG_ROOT):
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
        print("Fast development run:", FAST_DEV_RUN)
        print("Pair limit:", MAX_PAIRS)
        """
    ),
    markdown("## 2. Clone the `SR-3x` branch and install the project"),
    code(
        r"""
        if REPOSITORY_DIR.exists() and (REPOSITORY_DIR / ".git").is_dir():
            print("Updating existing clone:", REPOSITORY_DIR)
            run(["git", "fetch", "origin", REPOSITORY_BRANCH], cwd=REPOSITORY_DIR)
            run(["git", "checkout", REPOSITORY_BRANCH], cwd=REPOSITORY_DIR)
            run(["git", "pull", "--ff-only", "origin", REPOSITORY_BRANCH], cwd=REPOSITORY_DIR)
        elif REPOSITORY_DIR.exists():
            required = REPOSITORY_DIR / "src" / "geodiff_gan"
            if not required.is_dir():
                raise RuntimeError(
                    f"{REPOSITORY_DIR} exists but is neither a Git clone nor a usable source tree. "
                    "Use a different REPOSITORY_DIR; this notebook will not delete it."
                )
            print("Using existing non-Git source tree:", REPOSITORY_DIR)
        else:
            run([
                "git", "clone", "--depth", "1", "--branch", REPOSITORY_BRANCH,
                REPOSITORY_URL, REPOSITORY_DIR,
            ])

        # Kaggle already supplies CUDA-enabled PyTorch. Do not reinstall torch.
        run([
            sys.executable, "-m", "pip", "install", "-q",
            "numpy>=1.26", "Pillow>=10", "PyYAML>=6", "tqdm>=4.66",
            "rasterio>=1.3", "pandas>=2", "matplotlib>=3.7",
        ])
        run([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"], cwd=REPOSITORY_DIR)

        source_root = (REPOSITORY_DIR / "src").resolve()
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        os.chdir(REPOSITORY_DIR)

        import geodiff_gan
        import torch

        print("Package:", Path(geodiff_gan.__file__).resolve())
        print("Python:", sys.version)
        print("PyTorch:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        print("GPU count:", torch.cuda.device_count())
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                print(index, properties.name, f"{properties.total_memory / 2**30:.1f} GiB")
        """
    ),
    markdown(
        """
        ## 3. Discover or extract the attached products

        Required Landsat layers from one product ID are `SR_B2`, `SR_B3`, `SR_B4`,
        `QA_PIXEL`, and `QA_RADSAT`. Keep `MTL.txt` and preferably `SR_QA_AEROSOL`.
        Sentinel inputs must be complete L2A `.SAFE` products.

        Archive extraction, when needed, writes only below `/kaggle/working`; attached
        `/kaggle/input` data is never modified.
        """
    ),
    code(
        r"""
        from geodiff_gan.data.sentinel import discover_safe_products
        from geodiff_gan.data.landsat_sentinel import discover_landsat_products

        def discover_inputs(sentinel_root, landsat_root):
            return discover_safe_products(sentinel_root), discover_landsat_products(landsat_root)

        sentinel_products, landsat_products = discover_inputs(SENTINEL_INPUT, LANDSAT_INPUT)
        if (not sentinel_products or not landsat_products) and AUTO_EXTRACT_ARCHIVES_IF_NEEDED:
            import hashlib
            archive_suffixes = (".zip", ".tar", ".tar.gz", ".tgz")
            archives = []
            for path in Path("/kaggle/input").rglob("*"):
                lower = path.name.lower()
                if path.is_file() and lower.endswith(archive_suffixes):
                    upper = path.name.upper()
                    if any(token in upper for token in ("LC08", "LC09", "MSIL2A", ".SAFE")):
                        archives.append(path)
            print(f"Found {len(archives)} relevant archive(s) for extraction")
            for archive in archives:
                identity = hashlib.sha256(str(archive).encode()).hexdigest()[:10]
                destination = EXTRACTION_ROOT / f"{archive.stem}_{identity}"
                marker = destination / ".complete"
                if marker.exists():
                    print("Already extracted:", archive.name)
                    continue
                destination.mkdir(parents=True, exist_ok=True)
                print("Extracting:", archive, "->", destination)
                shutil.unpack_archive(str(archive), str(destination))
                marker.touch()
            sentinel_candidates = discover_safe_products(EXTRACTION_ROOT)
            landsat_candidates = discover_landsat_products(EXTRACTION_ROOT)
            if not sentinel_products and sentinel_candidates:
                SENTINEL_INPUT = EXTRACTION_ROOT
                sentinel_products = sentinel_candidates
            if not landsat_products and landsat_candidates:
                LANDSAT_INPUT = EXTRACTION_ROOT
                landsat_products = landsat_candidates

        print(f"Canonical Sentinel products: {len(sentinel_products)}")
        for product in sentinel_products[:10]:
            print(" S2:", product)
        print(f"Complete Landsat products: {len(landsat_products)}")
        for product in landsat_products[:10]:
            print(" L8/9:", product.product_id, "at", product.directory)

        if not sentinel_products:
            raise FileNotFoundError("No complete Sentinel-2 L2A SAFE product was discovered.")
        if not landsat_products:
            raise FileNotFoundError(
                "No complete Landsat C2 L2 RGB+QA product was discovered. "
                "Check that all required same-prefix files were uploaded."
            )
        """
    ),
    markdown("## 4. Pairing preflight: date and geographic overlap"),
    code(
        r"""
        import pandas as pd
        from geodiff_gan.data.landsat_sentinel import pair_scenes

        pairs, unmatched_sentinel = pair_scenes(
            sentinel_products,
            landsat_products,
            max_day_gap=MAX_DAY_GAP,
            minimum_overlap_fraction=MINIMUM_OVERLAP_FRACTION,
        )
        pair_rows = [{
            "sentinel": pair.sentinel.name,
            "landsat": pair.landsat.product_id,
            "day_gap": pair.day_gap,
            "overlap_fraction": pair.overlap_fraction,
        } for pair in pairs]
        pair_table = pd.DataFrame(pair_rows)
        display(pair_table)
        print("Matched Sentinel products:", len(pairs))
        print("Unmatched Sentinel products:", len(unmatched_sentinel))
        for product in unmatched_sentinel[:20]:
            print(" no pair:", product.name)
        if not pairs:
            raise RuntimeError(
                "No scene pairs satisfy the date and overlap limits. Download an overlapping "
                "Landsat path/row closer to each Sentinel acquisition date."
            )
        """
    ),
    markdown("## 5. Create or resume exact 3x paired patches"),
    code(
        r"""
        if RUN_DATA_PREPARATION:
            command = [
                sys.executable, "-m", "geodiff_gan.cli.prepare_landsat_sentinel",
                "--sentinel-input", SENTINEL_INPUT,
                "--landsat-input", LANDSAT_INPUT,
                "--output", PATCH_ROOT,
                "--manifest", MANIFEST,
                "--state", PAIRING_STATE,
                "--patch-size", PATCH_SIZE,
                "--stride", PATCH_STRIDE,
                "--max-day-gap", MAX_DAY_GAP,
                "--minimum-overlap-fraction", MINIMUM_OVERLAP_FRACTION,
                "--minimum-valid-fraction", MINIMUM_VALID_FRACTION,
                "--bandpass-adjustment", BANDPASS_ADJUSTMENT,
                "--unmatched-split", UNMATCHED_SPLIT,
            ]
            for prefix in VALIDATION_SENTINEL_PREFIXES:
                command.extend(["--val-prefix", prefix])
            for prefix in TEST_SENTINEL_PREFIXES:
                command.extend(["--test-prefix", prefix])
            if MAX_PAIRS is not None:
                command.extend(["--max-pairs", MAX_PAIRS])
            if REBUILD_PATCHES:
                command.append("--rebuild")
            run(command, cwd=REPOSITORY_DIR)

        if not MANIFEST.exists():
            raise FileNotFoundError(
                f"Manifest not found: {MANIFEST}. Enable RUN_DATA_PREPARATION or attach a prepared dataset."
            )
        print("Manifest:", MANIFEST)
        print("Pairing state:", PAIRING_STATE)
        """
    ),
    markdown("## 6. Manifest, split, geometry, and mask audit"),
    code(
        r"""
        from collections import Counter, defaultdict
        import numpy as np

        records = [
            json.loads(line)
            for line in MANIFEST.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        split_counts = Counter(record["split"] for record in records)
        tile_counts = Counter(record["tile_id"] for record in records)
        pair_counts = Counter(
            (record.get("sentinel_product"), record.get("landsat_product"))
            for record in records
        )
        print("Patches:", len(records))
        print("Splits:", split_counts)
        print("MGRS tiles:", tile_counts)
        print("Scene pairs:", len(pair_counts))
        print("Day gaps:", Counter(record.get("day_gap") for record in records))

        split_by_tile = defaultdict(set)
        for record in records:
            split_by_tile[record["tile_id"]].add(record["split"])
        leakage = {tile: splits for tile, splits in split_by_tile.items() if len(splits) > 1}
        if leakage:
            raise RuntimeError(f"Geographic split leakage detected: {leakage}")

        geometry_rows = []
        for record in records[: min(20, len(records))]:
            patch = Path(record["patch"])
            if not patch.exists():
                raise FileNotFoundError(patch)
            with np.load(patch) as data:
                geometry_rows.append({
                    "patch": patch.name,
                    "lr": tuple(data["lr"].shape),
                    "hr": tuple(data["hr"].shape),
                    "valid_hr": float(data["valid_mask_hr"].mean()),
                    "valid_lr": float(data["valid_mask_lr"].mean()),
                })
                if data["hr"].shape[-2:] != (
                    data["lr"].shape[-2] * 3,
                    data["lr"].shape[-1] * 3,
                ):
                    raise RuntimeError(f"Invalid 3x geometry in {patch}")
        display(pd.DataFrame(geometry_rows))

        if not records:
            raise RuntimeError("No valid paired patches were produced.")
        if not split_counts.get("train"):
            raise RuntimeError(
                "There is no training split. For one-pair development use UNMATCHED_SPLIT='train'."
            )
        if not split_counts.get("val") or not split_counts.get("test"):
            print(
                "WARNING: validation/test is incomplete. This is acceptable only for pipeline debugging. "
                "A real experiment needs geographically held-out MGRS tiles."
            )
        """
    ),
    markdown("## 7. Visualize random aligned pairs (maximum three panels per row)"),
    code(
        r"""
        import random
        import matplotlib.pyplot as plt
        import torch
        from torch.nn import functional as F

        def chw(array):
            tensor = torch.from_numpy(array).float()
            if tensor.ndim == 3 and tensor.shape[-1] <= 16:
                tensor = tensor.permute(2, 0, 1)
            return tensor

        def shared_stretch(images, valid=None):
            values = images[-1]
            if valid is not None:
                selected = values[:, valid]
            else:
                selected = values.flatten()
            low = torch.quantile(selected, 0.02)
            high = torch.quantile(selected, 0.98).clamp_min(low + 1e-5)
            return [((image - low) / (high - low)).clamp(0, 1) for image in images]

        rng = random.Random(RANDOM_SEED)
        selected_records = rng.sample(records, min(VISUALIZATION_SAMPLES, len(records)))
        figure, axes = plt.subplots(
            len(selected_records), 3, figsize=(15, 4.6 * len(selected_records)), squeeze=False
        )
        for row, record in enumerate(selected_records):
            with np.load(record["patch"]) as data:
                lr = chw(data["lr"])
                hr = chw(data["hr"])
                valid = torch.from_numpy(data["valid_mask_hr"][0]).bool()
            lr_up = F.interpolate(
                lr[None], size=hr.shape[-2:], mode="bicubic", align_corners=False
            )[0].clamp(0, 1)
            lr_display, hr_display = shared_stretch([lr_up, hr], valid)
            difference = (lr_up - hr).abs().mean(0) * valid
            panels = (
                (lr_display, "Landsat 30 m (bicubic display only)", "rgb"),
                (hr_display, "Sentinel-2 10 m target", "rgb"),
                (difference, f"Absolute cross-sensor difference\nmean={float(difference[valid].mean()):.4f}", "heat"),
            )
            for column, (image, title, kind) in enumerate(panels):
                if kind == "rgb":
                    axes[row, column].imshow(image.permute(1, 2, 0))
                else:
                    axes[row, column].imshow(image, cmap="turbo", vmin=0, vmax=max(0.05, float(image.quantile(0.99))))
                axes[row, column].set_title(title, fontsize=10)
                axes[row, column].axis("off")
            axes[row, 0].set_ylabel(
                f"{record['tile_id']} | gap={record.get('day_gap')}d\n{Path(record['patch']).stem}",
                fontsize=9,
            )
        figure.tight_layout()
        plt.show()
        """
    ),
    markdown(
        """
        ## 8. Detailed registration and radiometry inspection

        In the edge overlay, Sentinel-only edges are yellow and Landsat-only edges are cyan.
        Matching features should appear nearly white. Consistent yellow/cyan double edges indicate spatial misregistration;
        do not treat such a scene pair as supervised ground truth.
        """
    ),
    code(
        r"""
        INSPECTION_INDEX = min(DEBUG_INDEX, len(records) - 1)
        inspection_record = records[INSPECTION_INDEX]
        with np.load(inspection_record["patch"]) as data:
            inspection_lr = chw(data["lr"])
            inspection_hr = chw(data["hr"])
            inspection_valid = torch.from_numpy(data["valid_mask_hr"][0]).bool()
            inspection_lr_valid = torch.from_numpy(data["valid_mask_lr"][0]).bool()
        inspection_lr_up = F.interpolate(
            inspection_lr[None], size=inspection_hr.shape[-2:], mode="bicubic", align_corners=False
        )[0].clamp(0, 1)

        def edge_map(image):
            gray = image.mean(0, keepdim=True)[None]
            dx = F.pad(gray[:, :, :, 1:] - gray[:, :, :, :-1], (0, 1, 0, 0))
            dy = F.pad(gray[:, :, 1:, :] - gray[:, :, :-1, :], (0, 0, 0, 1))
            edge = torch.sqrt(dx.square() + dy.square() + 1e-12)[0, 0]
            return edge / edge.quantile(0.99).clamp_min(1e-6)

        lr_edge = edge_map(inspection_lr_up).clamp(0, 1) * inspection_valid
        hr_edge = edge_map(inspection_hr).clamp(0, 1) * inspection_valid
        overlay = torch.stack((hr_edge, torch.maximum(hr_edge, lr_edge), lr_edge), dim=-1)
        difference = (inspection_lr_up - inspection_hr).abs().mean(0) * inspection_valid
        mask_display = inspection_valid.float()
        spectrum_lr = torch.fft.fftshift(torch.fft.fft2(inspection_lr_up.mean(0))).abs().log1p()
        spectrum_hr = torch.fft.fftshift(torch.fft.fft2(inspection_hr.mean(0))).abs().log1p()

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        detailed_panels = (
            (overlay, "Edge registration: yellow=S2-only, cyan=Landsat-only", "rgb_hwc"),
            (difference, "Absolute difference", "heat"),
            (mask_display, f"Joint valid mask: {float(mask_display.mean()):.2%}", "mask"),
            (spectrum_lr, "Landsat frequency spectrum", "spectrum"),
            (spectrum_hr, "Sentinel frequency spectrum", "spectrum"),
        )
        for axis, (image, title, kind) in zip(axes.flat, detailed_panels):
            if kind == "rgb_hwc":
                axis.imshow(image)
            elif kind == "heat":
                axis.imshow(image, cmap="turbo", vmin=0, vmax=max(0.05, float(image.quantile(0.99))))
            elif kind == "mask":
                axis.imshow(image, cmap="gray", vmin=0, vmax=1)
            else:
                axis.imshow(image, cmap="magma")
            axis.set_title(title)
            axis.axis("off")
        histogram_axis = axes.flat[-1]
        colors = ("red", "green", "blue")
        for channel, color in enumerate(colors):
            histogram_axis.hist(
                inspection_lr_up[channel][inspection_valid].numpy(), bins=80,
                alpha=0.35, density=True, color=color, linestyle="--", label=f"Landsat {color}"
            )
            histogram_axis.hist(
                inspection_hr[channel][inspection_valid].numpy(), bins=80,
                alpha=0.25, density=True, color=color, label=f"Sentinel {color}"
            )
        histogram_axis.set_title("Reflectance distributions")
        histogram_axis.set_xlim(0, 0.8)
        histogram_axis.legend(fontsize=7, ncol=2)
        fig.suptitle(
            f"{inspection_record['landsat_product']} -> {inspection_record['sentinel_product']}\n"
            f"gap={inspection_record['day_gap']} day(s)", fontsize=11
        )
        fig.tight_layout()
        plt.show()
        """
    ),
    markdown(
        """
        ## 9. Spatial, spectral, radiometric, and temporal resolution diagnostics

        These are four different sensor properties:

        - **Spatial:** ground size represented by a pixel.
        - **Spectral:** wavelength ranges measured by each band.
        - **Radiometric:** ability to distinguish intensity/reflectance differences.
        - **Temporal:** observation frequency and the date gap of this specific pair.

        The cross-sensor difference contains all four effects plus registration and noise.
        It must not be interpreted as pure missing spatial detail.
        """
    ),
    code(
        r"""
        import pandas as pd
        from matplotlib.patches import Patch

        resolution_summary = pd.DataFrame([
            {
                "dimension": "Spatial",
                "Landsat 8/9 OLI": "30 m RGB",
                "Sentinel-2 MSI": "10 m RGB",
                "experiment implication": "3x linear SR; 9x output samples",
            },
            {
                "dimension": "Spectral",
                "Landsat 8/9 OLI": "B4/B3/B2 RGB filters",
                "Sentinel-2 MSI": "B04/B03/B02 RGB filters",
                "experiment implication": "similar names, non-identical bandpasses",
            },
            {
                "dimension": "Radiometric",
                "Landsat 8/9 OLI": "12-bit acquisition; uint16 C2 L2",
                "Sentinel-2 MSI": "12-bit acquisition; uint16 reflectance product",
                "experiment implication": "different calibration, SNR, scaling and processors",
            },
            {
                "dimension": "Temporal",
                "Landsat 8/9 OLI": "16 days each; 8-day combined offset",
                "Sentinel-2 MSI": "nominal 5-day two-satellite revisit",
                "experiment implication": "use actual pair day gap, not mission revisit",
            },
        ])
        display(resolution_summary)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)

        spatial_axis = axes[0]
        spatial_axis.set_xlim(0, 30)
        spatial_axis.set_ylim(0, 30)
        spatial_axis.set_aspect("equal")
        spatial_axis.add_patch(
            plt.Rectangle((0, 0), 30, 30, fill=False, linewidth=4, edgecolor="#8c510a")
        )
        for coordinate in (10, 20):
            spatial_axis.axvline(coordinate, color="#2166ac", linewidth=2)
            spatial_axis.axhline(coordinate, color="#2166ac", linewidth=2)
        for row in range(3):
            for column in range(3):
                spatial_axis.text(
                    column * 10 + 5,
                    row * 10 + 5,
                    "10 m",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#2166ac",
                )
        spatial_axis.set_title("Spatial sampling\n1 Landsat 30 m pixel = 3 x 3 Sentinel 10 m samples")
        spatial_axis.set_xlabel("ground distance (m)")
        spatial_axis.set_ylabel("ground distance (m)")

        bandpasses = [
            ("Landsat blue B2", 450.0, 510.0, "#8c510a"),
            ("Sentinel blue B02", 490.0 - 65.0 / 2, 490.0 + 65.0 / 2, "#2166ac"),
            ("Landsat green B3", 530.0, 590.0, "#8c510a"),
            ("Sentinel green B03", 560.0 - 35.0 / 2, 560.0 + 35.0 / 2, "#2166ac"),
            ("Landsat red B4", 640.0, 670.0, "#8c510a"),
            ("Sentinel red B04", 665.0 - 30.0 / 2, 665.0 + 30.0 / 2, "#2166ac"),
        ]
        spectral_axis = axes[1]
        for index, (label, start, end, color) in enumerate(bandpasses):
            spectral_axis.barh(index, end - start, left=start, height=0.65, color=color, alpha=0.82)
        spectral_axis.set_yticks(range(len(bandpasses)), [value[0] for value in bandpasses])
        spectral_axis.invert_yaxis()
        spectral_axis.set_xlim(430, 700)
        spectral_axis.set_xlabel("wavelength (nm)")
        spectral_axis.set_title("Nominal RGB bandpass comparison\nrectangles are not full response functions")
        spectral_axis.grid(axis="x", alpha=0.25)
        spectral_axis.legend(
            handles=[
                Patch(color="#8c510a", label="Landsat 8/9 OLI"),
                Patch(color="#2166ac", label="Sentinel-2 MSI"),
            ],
            loc="lower right",
        )
        plt.show()

        print("Patch ground width: 128 x 30 m = 384 x 10 m = 3.84 km")
        print("Bandpass adjustment used for prepared targets:", BANDPASS_ADJUSTMENT)
        """
    ),
    code(
        r"""
        # Per-band agreement combines spectral, radiometric, spatial, temporal,
        # atmospheric, and registration effects. It is not a pure spectral test.
        valid_indices = torch.nonzero(inspection_valid.flatten(), as_tuple=False).flatten()
        maximum_scatter_points = 12000
        if len(valid_indices) > maximum_scatter_points:
            selected_positions = torch.linspace(
                0, len(valid_indices) - 1, maximum_scatter_points
            ).long()
            valid_indices = valid_indices[selected_positions]

        band_names = ("Red", "Green", "Blue")
        band_colors = ("#b2182b", "#1b7837", "#2166ac")
        spectral_rows = []
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        for channel, (band_name, color, axis) in enumerate(zip(band_names, band_colors, axes)):
            landsat_values = inspection_lr_up[channel].flatten()[valid_indices]
            sentinel_values = inspection_hr[channel].flatten()[valid_indices]
            difference_values = sentinel_values - landsat_values
            if landsat_values.std(unbiased=False) > 0 and sentinel_values.std(unbiased=False) > 0:
                correlation = float(torch.corrcoef(torch.stack((landsat_values, sentinel_values)))[0, 1])
            else:
                correlation = float("nan")
            spectral_rows.append({
                "band": band_name,
                "landsat_mean": float(landsat_values.mean()),
                "sentinel_mean": float(sentinel_values.mean()),
                "bias_sentinel_minus_landsat": float(difference_values.mean()),
                "mae": float(difference_values.abs().mean()),
                "pearson_r": correlation,
            })
            combined = torch.cat((landsat_values, sentinel_values))
            lower = max(0.0, float(torch.quantile(combined, 0.01)))
            upper = min(1.0, float(torch.quantile(combined, 0.99)))
            upper = max(upper, lower + 0.02)
            axis.hexbin(
                landsat_values.numpy(),
                sentinel_values.numpy(),
                gridsize=48,
                mincnt=1,
                cmap="magma",
            )
            axis.plot((lower, upper), (lower, upper), linestyle="--", color="white", linewidth=1.5)
            axis.set_xlim(lower, upper)
            axis.set_ylim(lower, upper)
            axis.set_aspect("equal")
            axis.set_title(
                f"{band_name}: r={correlation:.3f}\n"
                f"bias={float(difference_values.mean()):+.4f}, MAE={float(difference_values.abs().mean()):.4f}",
                color=color,
            )
            axis.set_xlabel("Landsat reflectance (bicubic grid)")
            axis.set_ylabel("Sentinel reflectance")
        fig.suptitle("Per-band cross-sensor agreement; dashed line is y = x")
        plt.show()
        display(pd.DataFrame(spectral_rows).round(6))
        """
    ),
    code(
        r"""
        from collections import Counter
        import random
        from tqdm.auto import tqdm

        radiometric_summary = pd.DataFrame([
            {
                "sensor": "Landsat 8/9 OLI",
                "instrument_bits": 12,
                "potential_acquisition_levels": 4096,
                "product_storage": "uint16",
                "reflectance_encoding": "DN x metadata multiplier + offset",
                "nominal_fallback_step": 0.0000275,
            },
            {
                "sensor": "Sentinel-2 MSI",
                "instrument_bits": 12,
                "potential_acquisition_levels": 4096,
                "product_storage": "uint16",
                "reflectance_encoding": "(DN + BOA offset) / quantification value",
                "nominal_fallback_step": 0.0001,
            },
        ])
        display(radiometric_summary)

        diagnostic_limit = min(200, len(records))
        temporal_rng = random.Random(RANDOM_SEED + 109)
        temporal_records = (
            records
            if len(records) <= diagnostic_limit
            else temporal_rng.sample(records, diagnostic_limit)
        )
        temporal_rows = []
        for record in tqdm(temporal_records, desc="resolution diagnostics", unit="patch"):
            with np.load(record["patch"]) as data:
                lr_value = chw(data["lr"])
                hr_value = chw(data["hr"])
                valid_value = torch.from_numpy(data["valid_mask_hr"][0]).bool()
            lr_up_value = F.interpolate(
                lr_value[None],
                size=hr_value.shape[-2:],
                mode="bicubic",
                align_corners=False,
            )[0].clamp(0, 1)
            patch_difference = (lr_up_value - hr_value).abs().mean(0)
            temporal_rows.append({
                "day_gap": int(record.get("day_gap", 0)),
                "pair_mae": float(patch_difference[valid_value].mean()),
                "tile_id": record["tile_id"],
            })
        temporal_frame = pd.DataFrame(temporal_rows)
        temporal_grouped = temporal_frame.groupby("day_gap")["pair_mae"].agg(
            ["count", "mean", "median", "std"]
        ).reset_index()
        display(temporal_grouped.round(6))

        signed_differences = inspection_hr - inspection_lr_up
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
        for channel, (band_name, color) in enumerate(zip(band_names, band_colors)):
            values = signed_differences[channel][inspection_valid]
            axes[0, 0].hist(
                values.numpy(),
                bins=100,
                density=True,
                histtype="step",
                linewidth=1.5,
                color=color,
                label=f"{band_name}: mean={float(values.mean()):+.4f}",
            )
        axes[0, 0].axvline(0, color="black", linestyle="--", linewidth=1)
        axes[0, 0].set_title("Radiometric cross-sensor difference\nSentinel - Landsat")
        axes[0, 0].set_xlabel("reflectance difference")
        axes[0, 0].set_ylabel("density")
        axes[0, 0].legend(fontsize=8)

        sensor_labels = ("Landsat OLI", "Sentinel MSI")
        x_positions = np.arange(2)
        axes[0, 1].bar(x_positions - 0.18, (12, 12), width=0.36, label="instrument acquisition bits")
        axes[0, 1].bar(x_positions + 0.18, (16, 16), width=0.36, label="product storage bits")
        axes[0, 1].set_xticks(x_positions, sensor_labels)
        axes[0, 1].set_ylim(0, 18)
        axes[0, 1].set_ylabel("bits")
        axes[0, 1].set_title("Radiometric acquisition is not product storage")
        axes[0, 1].legend(fontsize=8)
        for position in x_positions:
            axes[0, 1].text(position - 0.18, 12.25, "4096 levels", ha="center", fontsize=8)

        all_gap_counts = Counter(int(record.get("day_gap", 0)) for record in records)
        gap_values = sorted(all_gap_counts)
        axes[1, 0].bar(gap_values, [all_gap_counts[value] for value in gap_values], color="#4d9221")
        axes[1, 0].set_xticks(gap_values)
        axes[1, 0].set_xlabel("absolute acquisition gap (days)")
        axes[1, 0].set_ylabel("prepared patches")
        axes[1, 0].set_title("Temporal balance of the complete manifest")

        sampled_gaps = sorted(temporal_frame["day_gap"].unique())
        jitter_rng = np.random.default_rng(RANDOM_SEED + 109)
        for position, gap in enumerate(sampled_gaps):
            group = temporal_frame.loc[temporal_frame["day_gap"] == gap, "pair_mae"].to_numpy()
            jitter = jitter_rng.normal(0, 0.045, size=len(group))
            axes[1, 1].scatter(
                np.full(len(group), position) + jitter,
                group,
                s=12,
                alpha=0.35,
                color="#2166ac",
            )
            axes[1, 1].hlines(np.median(group), position - 0.24, position + 0.24, color="#b2182b", linewidth=3)
        axes[1, 1].set_xticks(range(len(sampled_gaps)), sampled_gaps)
        axes[1, 1].set_xlabel("absolute acquisition gap (days)")
        axes[1, 1].set_ylabel("Landsat-Sentinel pair MAE")
        axes[1, 1].set_title(f"Pair discrepancy by day gap\n{len(temporal_frame)} sampled patches; red=median")
        plt.show()

        if (
            temporal_frame["day_gap"].nunique() > 1
            and temporal_frame["pair_mae"].std(ddof=0) > 0
        ):
            temporal_correlation = temporal_frame[["day_gap", "pair_mae"]].corr().iloc[0, 1]
            print("Day-gap / pair-MAE correlation:", float(temporal_correlation))
        else:
            print("Day-gap / pair-MAE correlation is undefined (insufficient variation).")
        print("Pair MAE is cross-sensor discrepancy, not model error and not pure temporal change.")
        """
    ),
    markdown("## 10. Build runtime configs for all training stages"),
    code(
        r"""
        import copy
        import yaml
        from geodiff_gan.config import load_config

        template_path = REPOSITORY_DIR / "configs" / "landsat_sentinel_3x_small.yaml"
        default_path = REPOSITORY_DIR / "configs" / "default.yaml"
        runtime_config = load_config(template_path, default_path)
        runtime_config["data"].update({
            "manifest": str(MANIFEST),
            "captions": None,
            "input_mode": "paired",
            "target_key": "hr",
            "condition_key": None,
            "train_degradation_sampling": "fixed",
        })
        runtime_config["model"].update({
            "scale": 3,
            "input_channels": 3,
            "output_channels": 3,
            "base_upsample_mode": "resize_conv",
            "decoder_upsample_mode": "resize_conv",
            "use_text_conditioning": False,
        })
        runtime_config["training"].update({
            "batch_size": 1,
            "gradient_accumulation": 1 if FAST_DEV_RUN else 8,
            "num_workers": 2,
            "amp": True,
            "gradient_checkpointing": True,
            "auto_resume": AUTO_RESUME_TRAINING,
            "progress_mode": TRAINING_PROGRESS_MODE,
            "progress_updates_per_epoch": 2,
            "validate_every": 1,
            "validation_limit": VALIDATION_LIMIT,
            "keep_best_and_latest": True,
            "checkpoint_metric": "val_l1",
            "checkpoint_mode": "min",
            "early_stopping_patience": 5,
            "early_stopping_min_epochs": 5 if not FAST_DEV_RUN else 1,
            "train_back_projection_steps": 0,
        })
        runtime_config.setdefault("debug", {})
        runtime_config["debug"].update({
            "enabled": TRAINING_DIAGNOSTICS,
            "output_dir": str(WORK_ROOT / "training_debug"),
            "every_n_steps": 1 if FAST_DEV_RUN else 500,
            "max_exports_per_epoch": 1 if TRAINING_DIAGNOSTICS else 0,
            "print_tensor_stats": TRAINING_DIAGNOSTICS,
            "fail_on_nonfinite": True,
            "panel_size": 320,
            "save_tensors": False,
        })
        print(yaml.safe_dump(runtime_config, sort_keys=False))
        """
    ),
    markdown("## 11. Parameter report and complete 3x tensor-shape smoke test"),
    code(
        r"""
        from geodiff_gan.parameters import build_parameter_report
        from geodiff_gan.models.system import GeoDiffGAN

        parameter_report = build_parameter_report(runtime_config, patches=split_counts["train"], world_size=1)
        module_rows = []
        for name, values in parameter_report["core_modules"].items():
            module_rows.append({"module": name, **values})
        display(pd.DataFrame(module_rows).sort_values("scalar_parameters", ascending=False))
        stage_rows = []
        for name, values in parameter_report["training_stages"].items():
            stage_rows.append({"stage": name, **values})
        display(pd.DataFrame(stage_rows))

        shape_model = GeoDiffGAN.from_config(runtime_config).eval()
        sample_lr = torch.rand(1, 3, 128, 128)
        sample_hr = torch.rand(1, 3, 384, 384)
        with torch.no_grad():
            sample_base = shape_model.base(sample_lr)
            sample_features = shape_model.lr_encoder(sample_lr)
            sample_latent, _, _ = shape_model.vae.encode(sample_hr - sample_base, sample=False)
            sample_context = torch.zeros(1, 1, runtime_config["model"]["context_dim"])
            sample_mode = shape_model.mode_tensor("sr", 1, sample_lr.device)
            sample_mapped = shape_model.mapper(sample_latent, sample_features[1], sample_context, sample_mode)
            sample_decoded = shape_model.decoder(sample_mapped, sample_features)
        print("LR:", tuple(sample_lr.shape))
        print("Base:", tuple(sample_base.shape))
        print("LR features:", [tuple(value.shape) for value in sample_features])
        print("VAE latent:", tuple(sample_latent.shape))
        print("Mapper content:", tuple(sample_mapped.content.shape))
        print("Decoder residual:", tuple(sample_decoded.detail_residual.shape))
        assert sample_base.shape[-2:] == (384, 384)
        assert sample_decoded.detail_residual.shape[-2:] == (384, 384)
        del shape_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        """
    ),
    markdown("## 12. Train base, VAE, diffusion, and joint stages with automatic resume"),
    code(
        r"""
        def select_checkpoint(output_dir, stage):
            best = output_dir / f"{stage}_best.pt"
            latest = output_dir / f"{stage}_latest.pt"
            if best.exists():
                return best
            if latest.exists():
                return latest
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
            config_path = CONFIG_ROOT / f"{stage}.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            print(f"\n===== {stage}: epochs={EPOCHS[stage]} lr={STAGE_LEARNING_RATES[stage]} =====")
            run([sys.executable, "-m", "geodiff_gan.cli.train", "--config", config_path], cwd=REPOSITORY_DIR)
            checkpoint = select_checkpoint(output_dir, stage)
            if checkpoint is None:
                raise RuntimeError(f"No checkpoint produced for {stage}")
            print("Selected:", checkpoint)
            return config_path, checkpoint

        CHECKPOINTS = {}
        CONFIGS = {}
        previous_checkpoint = None
        for stage in STAGES_TO_RUN:
            config_path, previous_checkpoint = train_stage(stage, previous_checkpoint)
            CONFIGS[stage] = config_path
            CHECKPOINTS[stage] = previous_checkpoint
        CHECKPOINTS
        """
    ),
    markdown("## 13. Training curves and checkpoint inventory"),
    code(
        r"""
        from IPython.display import Image as DisplayImage, display

        checkpoint_rows = []
        for stage in STAGES_TO_RUN:
            stage_dir = RUN_ROOT / stage
            checkpoint_rows.append({
                "stage": stage,
                "selected": str(CHECKPOINTS[stage]),
                "best_exists": (stage_dir / f"{stage}_best.pt").exists(),
                "latest_exists": (stage_dir / f"{stage}_latest.pt").exists(),
            })
            curve = stage_dir / "training_curves.png"
            if curve.exists():
                print("Training curves:", stage)
                display(DisplayImage(filename=str(curve)))
            else:
                print(stage, "has no curve image yet; one epoch produces only a single plotted point.")
        display(pd.DataFrame(checkpoint_rows))
        """
    ),
    markdown("## 14. Masked evaluation and bicubic/base baselines"),
    code(
        r"""
        evaluation_settings = EVALUATION_PROFILES[EVALUATION_PROFILE]
        EVALUATION_LIMIT = evaluation_settings["limit"]
        EVALUATION_SAMPLES = evaluation_settings["samples"]
        EVALUATION_STEPS = evaluation_settings["steps"]
        EVALUATION_SPLIT = "test" if split_counts.get("test") else (
            "val" if split_counts.get("val") else "train"
        )
        final_stage = STAGES_TO_RUN[-1]
        evaluation_checkpoint = CHECKPOINTS[final_stage]
        evaluation_config = CONFIGS[final_stage]
        split_output = EVALUATION_ROOT / EVALUATION_SPLIT
        base_checkpoint = CHECKPOINTS.get("base")

        evaluation_command = [
            sys.executable, "-m", "geodiff_gan.cli.evaluate",
            "--config", evaluation_config,
            "--checkpoint", evaluation_checkpoint,
            "--output", split_output,
            "--split", EVALUATION_SPLIT,
            "--samples", EVALUATION_SAMPLES,
            "--steps", EVALUATION_STEPS,
            "--back-projection-steps", EVALUATION_BACK_PROJECTION_STEPS,
            "--mode", "sr",
            "--limit", EVALUATION_LIMIT,
            "--device", "cuda" if torch.cuda.is_available() else "cpu",
            "--progress", "compact",
            "--no-text",
        ]
        run(evaluation_command, cwd=REPOSITORY_DIR)

        baseline_path = EVALUATION_ROOT / f"{EVALUATION_SPLIT}_baselines.json"
        baseline_command = [
            sys.executable, "-m", "geodiff_gan.cli.baselines",
            "--config", evaluation_config,
            "--output", baseline_path,
            "--split", EVALUATION_SPLIT,
            "--limit", EVALUATION_LIMIT,
            "--device", "cuda" if torch.cuda.is_available() else "cpu",
            "--progress", "compact",
        ]
        if base_checkpoint is not None:
            baseline_command.extend(["--base-checkpoint", base_checkpoint])
        run(baseline_command, cwd=REPOSITORY_DIR)

        model_metrics = json.loads((split_output / "metrics.json").read_text(encoding="utf-8"))
        baseline_metrics = json.loads(baseline_path.read_text(encoding="utf-8"))
        method_labels = {
            "bicubic": "Landsat bicubic",
            "base": "SwinIR base",
        }
        metric_rows = [{"method": "GeoDiff-GAN final", **model_metrics}]
        for method, values in baseline_metrics.items():
            if isinstance(values, dict):
                metric_rows.append({"method": method_labels.get(method, method), **values})
        metric_table = pd.DataFrame(metric_rows)
        method_order = ["Landsat bicubic", "SwinIR base", "GeoDiff-GAN final"]
        metric_table["method"] = pd.Categorical(
            metric_table["method"],
            categories=method_order,
            ordered=True,
        )
        metric_table = metric_table.sort_values("method").reset_index(drop=True)
        preferred = [
            "method", "count", "l1", "psnr", "ssim", "edge_f1",
            "redegradation_l1", "ergas", "sam_degrees", "uiqi", "scc",
        ]
        display(metric_table[[column for column in preferred if column in metric_table.columns]].round(6))
        print(
            "Read this table as three separate systems: bicubic is interpolation only, "
            "SwinIR base is the deterministic radiometric SR branch, and GeoDiff-GAN final "
            "is base plus diffusion/GAN residual, evidence gating, and back-projection."
        )
        """
    ),
    markdown("## 15. Metric comparison plots"),
    code(
        r"""
        available_metrics = [
            metric for metric in (
                "psnr", "ssim", "edge_f1", "ergas", "sam_degrees",
                "uiqi", "scc", "redegradation_l1",
            )
            if metric in metric_table.columns and metric_table[metric].notna().any()
        ]
        if available_metrics:
            columns = 3
            rows_count = (len(available_metrics) + columns - 1) // columns
            fig, axes = plt.subplots(rows_count, columns, figsize=(15, 4 * rows_count), squeeze=False)
            palette = {
                "Landsat bicubic": "#9e9e9e",
                "SwinIR base": "#4f81bd",
                "GeoDiff-GAN final": "#70ad47",
            }
            for axis, metric in zip(axes.flat, available_metrics):
                values = metric_table[["method", metric]].dropna()
                labels = [str(value) for value in values["method"]]
                axis.bar(
                    labels,
                    values[metric],
                    color=[palette.get(label, "#8064a2") for label in labels],
                )
                direction = (
                    "higher is better"
                    if metric in ("psnr", "ssim", "edge_f1", "uiqi", "scc")
                    else "lower is better"
                )
                axis.set_title(f"{metric} ({direction})")
                axis.tick_params(axis="x", rotation=20)
                axis.grid(axis="y", alpha=0.25)
            for axis in axes.flat[len(available_metrics):]:
                axis.axis("off")
            fig.tight_layout()
            plt.show()
        else:
            print("No numeric metrics available for plotting.")
        """
    ),
    markdown(
        """
        ## 16. Side-by-side output for an evaluated patch

        The first panel is the original `128 x 128` Landsat patch rendered with
        nearest-neighbor display so its native 30 m pixels remain visible. Bicubic is
        shown separately and is used only to place Landsat on the `384 x 384` Sentinel
        grid for visual and numerical comparison; it does not create new 10 m evidence.
        The native panel is displayed at one-third the width and height of each `384 x 384`
        panel so the figure also communicates the `1:3` linear pixel-count ratio. This
        pixel-count-scaled layout is useful for inspecting array size, but the panels no
        longer use the same visual map scale even though they cover the same ground extent.
        """
    ),
    code(
        r"""
        evaluation_records = [record for record in records if record["split"] == EVALUATION_SPLIT]
        side_index = min(DEBUG_INDEX, len(evaluation_records) - 1)
        side_record = evaluation_records[side_index]
        side_path = Path(side_record["patch"])
        result_stem = "__".join((side_record["tile_id"], side_path.parent.name, side_path.stem))
        result_path = split_output / f"{result_stem}_uncertainty.npz"
        if not result_path.exists():
            candidates = sorted(split_output.glob("*_uncertainty.npz"))
            if not candidates:
                raise FileNotFoundError("No evaluated uncertainty outputs were found.")
            result_path = candidates[0]
            with np.load(result_path) as result_data:
                source_patch = str(result_data.get("source_patch", ""))
            matching = [record for record in evaluation_records if record["patch"] == source_patch]
            if matching:
                side_record = matching[0]
                side_path = Path(side_record["patch"])

        with np.load(side_path) as data:
            side_lr = chw(data["lr"])
            side_hr = chw(data["hr"])
            side_valid = torch.from_numpy(data["valid_mask_hr"][0]).bool()
        with np.load(result_path) as data:
            side_output = chw(data["mean"])
            cached_base_for_side = chw(data["base"]) if "base" in set(data.files) else None
        side_bicubic = F.interpolate(
            side_lr[None], size=side_hr.shape[-2:], mode="bicubic", align_corners=False
        )[0].clamp(0, 1)
        if cached_base_for_side is None:
            from geodiff_gan.models.system import GeoDiffGAN
            from geodiff_gan.training.checkpoint import load_checkpoint

            side_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            side_model = GeoDiffGAN.from_config(runtime_config).to(side_device).eval()
            load_checkpoint(evaluation_checkpoint, side_model, strict=False)
            with torch.inference_mode():
                cached_base_for_side = side_model.base(side_lr[None].to(side_device))[0].float().cpu()
            del side_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        side_images = [side_lr, side_bicubic, cached_base_for_side, side_output, side_hr]
        side_displays = shared_stretch(side_images, side_valid)

        titles = (
            "Landsat original 30 m\n128 x 128 native pixels",
            "Landsat bicubic display\n384 x 384 (no new information)",
            "SwinIR base\n384 x 384 deterministic",
            "GeoDiff-GAN 3x\n384 x 384",
            "Sentinel-2 target 10 m\n384 x 384",
        )
        fig = plt.figure(figsize=(24, 6.2), constrained_layout=True)
        grid = fig.add_gridspec(1, 5, width_ratios=(1, 3, 3, 3, 3))
        axes = [fig.add_subplot(grid[0, index]) for index in range(5)]
        for index, (image, title) in enumerate(zip(side_displays, titles)):
            axis = axes[index]
            axis.imshow(
                image.permute(1, 2, 0),
                interpolation="nearest" if index == 0 else "antialiased",
            )
            axis.set_title(title)
            axis.axis("off")
        fig.suptitle(
            f"{side_record['tile_id']} | gap={side_record['day_gap']} day(s) | "
            "panel dimensions scaled by pixel count (1:3)"
        )
        plt.show()

        comparison_errors = {
            "Landsat bicubic vs Sentinel": (side_bicubic - side_hr).abs().mean(0) * side_valid,
            "SwinIR base vs Sentinel": (cached_base_for_side - side_hr).abs().mean(0) * side_valid,
            "GeoDiff-GAN vs Sentinel": (side_output - side_hr).abs().mean(0) * side_valid,
        }
        valid_errors = torch.cat([error[side_valid] for error in comparison_errors.values()])
        error_vmax = max(0.05, float(torch.quantile(valid_errors, 0.99)))
        fig = plt.figure(figsize=(16, 5.2), constrained_layout=True)
        error_grid = fig.add_gridspec(1, 4, width_ratios=(1, 1, 1, 0.045))
        error_axes = [fig.add_subplot(error_grid[0, index]) for index in range(3)]
        color_axis = fig.add_subplot(error_grid[0, 3])
        for axis, (title, error) in zip(error_axes, comparison_errors.items()):
            plot = axis.imshow(error, cmap="turbo", vmin=0, vmax=error_vmax)
            axis.set_title(f"{title}\nMasked L1={float(error[side_valid].mean()):.4f}")
            axis.axis("off")
        fig.colorbar(plot, cax=color_axis, label="Absolute reflectance error")
        fig.suptitle("Error maps use the same color scale")
        plt.show()
        print("Native Landsat shape:", tuple(side_lr.shape))
        print("Sentinel/output shape:", tuple(side_hr.shape))
        print("Evaluation cache:", result_path)
        """
    ),
    markdown(
        """
        ## 17. Deterministic base and generated high-frequency contribution

        In SR mode, the decoder residual is high-pass filtered and evidence-gated before
        being added to the deterministic base. The signed residual uses gray for zero,
        brighter values for positive reflectance corrections, and darker values for
        negative corrections. It is amplified only for visualization. The magnitude map
        shows where the model adds the strongest detail regardless of sign.
        """
    ),
    code(
        r"""
        from geodiff_gan.models.blocks import high_pass

        with np.load(result_path) as data:
            cached_names = set(data.files)
            cached_base = chw(data["base"]) if "base" in cached_names else None
            cached_residual = (
                chw(data["decoder_residual"])
                if "decoder_residual" in cached_names
                else None
            )
            cached_net_addition = (
                chw(data["net_addition"])
                if "net_addition" in cached_names
                else side_output - cached_base
                if cached_base is not None
                else None
            )

        if cached_base is None:
            from geodiff_gan.models.system import GeoDiffGAN
            from geodiff_gan.training.checkpoint import load_checkpoint

            detail_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            detail_model = GeoDiffGAN.from_config(runtime_config).to(detail_device).eval()
            load_checkpoint(evaluation_checkpoint, detail_model, strict=False)
            with torch.inference_mode():
                cached_base = detail_model.base(side_lr[None].to(detail_device))[0].float().cpu()
            del detail_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if cached_net_addition is None:
            cached_net_addition = side_output - cached_base
        if cached_residual is None:
            # Older evaluation caches did not store the exact decoder residual. This
            # fallback isolates the high-frequency part of the final net correction.
            cached_residual = high_pass(cached_net_addition[None])[0]
            residual_source = "estimated from final output - base (rerun Cell 13 for exact decoder residual)"
        else:
            residual_source = "exact stochastic-mean evidence-gated decoder residual"

        base_plus_detail = (cached_base + cached_residual).clamp(0, 1)
        base_display, base_plus_display = shared_stretch(
            [cached_base, base_plus_detail, side_hr], side_valid
        )[:2]
        valid_residual_values = cached_residual[:, side_valid].abs()
        display_limit = float(torch.quantile(valid_residual_values, 0.99).clamp_min(1e-6))
        display_gain = 0.48 / display_limit
        signed_display = (0.5 + cached_residual * display_gain).clamp(0, 1)
        residual_magnitude = cached_residual.abs().mean(0) * side_valid
        magnitude_vmax = max(
            1e-5,
            float(torch.quantile(residual_magnitude[side_valid], 0.99)),
        )

        fig = plt.figure(figsize=(20, 5.4), constrained_layout=True)
        detail_grid = fig.add_gridspec(1, 5, width_ratios=(1, 1, 1, 1, 0.04))
        detail_axes = [fig.add_subplot(detail_grid[0, index]) for index in range(4)]
        detail_color_axis = fig.add_subplot(detail_grid[0, 4])
        detail_axes[0].imshow(base_display.permute(1, 2, 0))
        detail_axes[0].set_title("Deterministic base\n384 x 384")
        detail_axes[1].imshow(signed_display.permute(1, 2, 0))
        detail_axes[1].set_title(f"Signed generated high frequency\nvisual gain x{display_gain:.1f}")
        magnitude_plot = detail_axes[2].imshow(
            residual_magnitude,
            cmap="turbo",
            vmin=0,
            vmax=magnitude_vmax,
        )
        detail_axes[2].set_title(
            "High-frequency magnitude\n"
            f"mean={float(residual_magnitude[side_valid].mean()):.5f}"
        )
        detail_axes[3].imshow(base_plus_display.permute(1, 2, 0))
        detail_axes[3].set_title("Base + decoder high frequency\nbefore projection/abstention")
        for axis in detail_axes:
            axis.axis("off")
        fig.colorbar(
            magnitude_plot,
            cax=detail_color_axis,
            label="Absolute reflectance contribution",
        )
        fig.suptitle("Base branch and decoder detail contribution")
        plt.show()

        print("Residual source:", residual_source)
        print("Base shape:", tuple(cached_base.shape))
        print("Decoder residual shape:", tuple(cached_residual.shape))
        print("Decoder residual absolute mean:", float(residual_magnitude[side_valid].mean()))
        print("Final net addition absolute mean:", float(cached_net_addition[:, side_valid].abs().mean()))
        """
    ),
    markdown("## 18. Full intermediate diagnostic export"),
    code(
        r"""
        diagnostic_output = DEBUG_ROOT / f"{EVALUATION_SPLIT}_index_{DEBUG_INDEX}"
        run([
            sys.executable, "-m", "geodiff_gan.cli.debug",
            "--config", evaluation_config,
            "--checkpoint", evaluation_checkpoint,
            "--output", diagnostic_output,
            "--split", EVALUATION_SPLIT,
            "--index", DEBUG_INDEX,
            "--mode", "sr",
            "--steps", EVALUATION_STEPS,
            "--back-projection-steps", EVALUATION_BACK_PROJECTION_STEPS,
            "--diffusion-every", 1 if FAST_DEV_RUN else 5,
            "--save-tensors",
        ], cwd=REPOSITORY_DIR)

        diagnostic_images = (
            "overview.png",
            "stage_intermediates.png",
            "features.png",
            "tensor_histograms.png",
            "frequency_spectra.png",
            "policy_overlays.png",
            "edges_and_wavelets.png",
            "diffusion_trajectory.png",
            "projection_trajectory.png",
            "loss_breakdown.png",
        )
        for name in diagnostic_images:
            path = diagnostic_output / name
            if path.exists():
                print(name)
                display(DisplayImage(filename=str(path)))
        summary_path = diagnostic_output / "summary.txt"
        if summary_path.exists():
            print(summary_path.read_text(encoding="utf-8"))
        print("Complete diagnostic directory:", diagnostic_output)
        """
    ),
    markdown("## 19. Uncertainty, evidence, abstention, and validity maps"),
    code(
        r"""
        with np.load(result_path) as data:
            uncertainty_data = {name: data[name] for name in data.files if name != "source_patch"}

        uncertainty_panels = []
        for key, title, kind in (
            ("raw_mean", "Raw stochastic mean", "rgb"),
            ("mean", "Final abstention-aware output", "rgb"),
            ("variance", "Pixelwise sample variance", "heat"),
            ("evidence_confidence", "Evidence confidence", "heat"),
            ("abstention_map", "Abstention map", "heat"),
            ("valid_mask", "Joint valid mask", "mask"),
        ):
            if key in uncertainty_data:
                uncertainty_panels.append((uncertainty_data[key], title, kind))

        fig, axes = plt.subplots(2, 3, figsize=(15, 10), squeeze=False)
        for axis, (array, title, kind) in zip(axes.flat, uncertainty_panels):
            tensor = torch.from_numpy(array).float()
            if kind == "rgb":
                if tensor.ndim == 3 and tensor.shape[0] <= 4:
                    tensor = tensor.permute(1, 2, 0)
                axis.imshow(tensor.clamp(0, 1))
            else:
                tensor = tensor.squeeze()
                axis.imshow(tensor, cmap="gray" if kind == "mask" else "turbo")
            axis.set_title(title)
            axis.axis("off")
        for axis in axes.flat[len(uncertainty_panels):]:
            axis.axis("off")
        fig.tight_layout()
        plt.show()
        """
    ),
    markdown(
        """
        ## 20. Optional fidelity-focused and full-unfreeze fine-tuning ablations

        Your current numbers show that the deterministic SwinIR base can beat the final
        diffusion/GAN output on PSNR, SSIM, ERGAS, SAM, UIQI, and sCC. That means the joint stage is
        useful for LR consistency and some edges, but it may be pulling the image away from
        the paired Sentinel target. Run this section only after the normal pipeline finishes.

        Two controlled experiments are created:

        - `joint_fidelity`: starts from the existing joint checkpoint, reduces synthesis losses,
          and emphasizes Charbonnier, SSIM, and gradient losses.
        - `joint_unfrozen_fidelity`: same objective, but also makes the base and VAE trainable
          during joint fine-tuning.

        The second experiment is not automatically better. It can improve target fidelity, but
        it can also damage the conservative base branch if the data are misregistered or the
        Landsat/Sentinel date gap creates real land-cover differences.
        """
    ),
    code(
        r"""
        def build_ablation_config(name, trainable_modules=None):
            config = copy.deepcopy(runtime_config)
            output_dir = RUN_ROOT / name
            output_dir.mkdir(parents=True, exist_ok=True)
            config["training"].update({
                "stage": "joint",
                "epochs": FIDELITY_FINETUNE_EPOCHS,
                "learning_rate": min(float(STAGE_LEARNING_RATES["joint"]), 1e-5),
                "discriminator_learning_rate": 5e-6,
                "output_dir": str(output_dir),
                "init_checkpoint": str(evaluation_checkpoint),
                "resume": None,
                "auto_resume": AUTO_RESUME_TRAINING,
                "checkpoint_metric": "val_ssim",
                "checkpoint_mode": "max",
                "early_stopping_patience": 3 if FAST_DEV_RUN else 5,
            })
            config["training"]["loss_weights"] = {
                **config["training"].get("loss_weights", {}),
                "charbonnier": 2.0,
                "ssim": 1.0,
                "gradient": 0.25,
                "consistency": 0.5,
                "wavelet": 0.02,
                "perceptual": 0.02,
                "diffusion": 0.25,
                "evidence_calibration": 0.02,
                "adversarial": 0.0,
                "prompt_alignment": 0.0,
            }
            if trainable_modules is not None:
                config["training"]["trainable_modules"] = trainable_modules
            config_path = CONFIG_ROOT / f"{name}.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            return config_path, output_dir

        def train_and_evaluate_ablation(name, trainable_modules=None):
            config_path, output_dir = build_ablation_config(name, trainable_modules)
            print(f"\n===== {name}: fidelity fine-tune =====")
            run([sys.executable, "-m", "geodiff_gan.cli.train", "--config", config_path], cwd=REPOSITORY_DIR)
            checkpoint = select_checkpoint(output_dir, "joint")
            if checkpoint is None:
                raise RuntimeError(f"No checkpoint produced for {name}")
            output = EVALUATION_ROOT / name / EVALUATION_SPLIT
            command = [
                sys.executable, "-m", "geodiff_gan.cli.evaluate",
                "--config", config_path,
                "--checkpoint", checkpoint,
                "--output", output,
                "--split", EVALUATION_SPLIT,
                "--samples", EVALUATION_SAMPLES,
                "--steps", EVALUATION_STEPS,
                "--back-projection-steps", EVALUATION_BACK_PROJECTION_STEPS,
                "--mode", "sr",
                "--limit", EVALUATION_LIMIT,
                "--device", "cuda" if torch.cuda.is_available() else "cpu",
                "--progress", "compact",
                "--no-text",
            ]
            run(command, cwd=REPOSITORY_DIR)
            return {
                "name": name,
                "config": config_path,
                "checkpoint": checkpoint,
                "output": output,
                "metrics": json.loads((output / "metrics.json").read_text(encoding="utf-8")),
            }

        ABLATION_RESULTS = []
        if RUN_FIDELITY_FINETUNE_ABLATIONS:
            ABLATION_RESULTS.append(train_and_evaluate_ablation("joint_fidelity"))
            ABLATION_RESULTS.append(
                train_and_evaluate_ablation(
                    "joint_unfrozen_fidelity",
                    trainable_modules=[
                        "base",
                        "vae",
                        "lr_encoder",
                        "diffusion",
                        "mapper",
                        "decoder",
                    ],
                )
            )
        else:
            print("Skipped. Set RUN_FIDELITY_FINETUNE_ABLATIONS=True in Cell 1 to run.")
        ABLATION_RESULTS
        """
    ),
    markdown("## 21. Recompare original, SwinIR base, and fine-tuned ablations"),
    code(
        r"""
        comparison_rows = []
        for row in metric_table.to_dict("records"):
            comparison_rows.append({**row, "method": str(row["method"])})
        for result in ABLATION_RESULTS:
            comparison_rows.append({
                "method": result["name"],
                **result["metrics"],
            })
        comparison_table = pd.DataFrame(comparison_rows)
        preferred = [
            "method", "count", "l1", "psnr", "ssim", "edge_f1",
            "redegradation_l1", "ergas", "sam_degrees", "uiqi", "scc",
        ]
        display(comparison_table[[column for column in preferred if column in comparison_table.columns]].round(6))

        plot_metrics = [
            metric for metric in (
                "psnr", "ssim", "edge_f1", "ergas", "sam_degrees",
                "uiqi", "scc", "l1", "redegradation_l1",
            )
            if metric in comparison_table.columns and comparison_table[metric].notna().any()
        ]
        if plot_metrics:
            columns = 3
            rows_count = (len(plot_metrics) + columns - 1) // columns
            fig, axes = plt.subplots(rows_count, columns, figsize=(16, 4.2 * rows_count), squeeze=False)
            for axis, metric in zip(axes.flat, plot_metrics):
                values = comparison_table[["method", metric]].dropna()
                axis.bar(values["method"], values[metric], color="#4f81bd")
                direction = (
                    "higher is better"
                    if metric in ("psnr", "ssim", "edge_f1", "uiqi", "scc")
                    else "lower is better"
                )
                axis.set_title(f"{metric} ({direction})")
                axis.tick_params(axis="x", rotation=30)
                axis.grid(axis="y", alpha=0.25)
            for axis in axes.flat[len(plot_metrics):]:
                axis.axis("off")
            fig.tight_layout()
            plt.show()

        if ABLATION_RESULTS:
            visual_outputs = {
                "GeoDiff-GAN final": split_output,
                **{result["name"]: result["output"] for result in ABLATION_RESULTS},
            }
            panels = [
                (side_lr, "Landsat original 30 m"),
                (side_bicubic, "Landsat bicubic"),
                (cached_base_for_side, "SwinIR base"),
            ]
            for name, output_dir in visual_outputs.items():
                candidate = output_dir / result_path.name
                if candidate.exists():
                    with np.load(candidate) as data:
                        panels.append((chw(data["mean"]), name))
                else:
                    print("Missing visual cache:", candidate)
            panels.append((side_hr, "Sentinel target"))
            displays = shared_stretch([image for image, _ in panels], side_valid)
            columns = min(3, len(panels))
            rows_count = (len(panels) + columns - 1) // columns
            fig, axes = plt.subplots(rows_count, columns, figsize=(5 * columns, 5 * rows_count), squeeze=False)
            for axis, display_image, (source_image, title) in zip(axes.flat, displays, panels):
                axis.imshow(
                    display_image.permute(1, 2, 0),
                    interpolation="nearest" if title.startswith("Landsat original") else "antialiased",
                )
                error = float((source_image - side_hr).abs().mean(0)[side_valid].mean())
                axis.set_title(f"{title}\nMasked L1={error:.4f}")
                axis.axis("off")
            for axis in axes.flat[len(panels):]:
                axis.axis("off")
            fig.suptitle("Visual comparison after fidelity ablations")
            fig.tight_layout()
            plt.show()
        """
    ),
    markdown(
        """
        ## 22. Export model artifacts without deleting data

        The archive contains configs, checkpoints, evaluation, debug reports, manifest, and pairing
        state. Patch binaries are intentionally excluded because they can be many gigabytes and
        remain available under `WORK_ROOT` for the current Kaggle session.
        """
    ),
    code(
        r"""
        import zipfile
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = Path("/kaggle/working") / f"geodiff_landsat_sentinel_3x_{timestamp}.zip"
        export_items = [CONFIG_ROOT, RUN_ROOT, EVALUATION_ROOT, DEBUG_ROOT, MANIFEST, PAIRING_STATE]
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for item in export_items:
                if not item.exists():
                    continue
                if item.is_file():
                    archive.write(item, arcname=item.relative_to(WORK_ROOT))
                else:
                    for path in item.rglob("*"):
                        if path.is_file():
                            archive.write(path, arcname=path.relative_to(WORK_ROOT))
        print("Exported:", archive_path)
        print("Archive size GiB:", archive_path.stat().st_size / 2**30)
        print("Prepared patches were not deleted:", PATCH_ROOT)
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
OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
print(OUTPUT)
