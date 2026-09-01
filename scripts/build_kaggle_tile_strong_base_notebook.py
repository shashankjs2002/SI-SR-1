"""Build the progressive strong-base Landsat/Sentinel 3x Kaggle notebook."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
from textwrap import dedent
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "kaggle/GeoDiff_GAN_Kaggle_Tile_Residual_MoE_3x.ipynb"
OUTPUT = ROOT / "kaggle/GeoDiff_GAN_Kaggle_Tile_Strong_Base_MoE_3x.ipynb"
ARCHIVE = ROOT / "kaggle/geodiff_tile_strong_base_source.zip"


def text(cell):
    return "".join(cell["source"])


def set_markdown(cell, value):
    cell["source"] = dedent(value).strip().splitlines(True)


def set_code(cell, value):
    value = dedent(value).strip() + "\n"
    ast.parse(value)
    cell["source"] = value.splitlines(True)
    cell["execution_count"] = None
    cell["outputs"] = []


def find(cells, marker, cell_type=None):
    matches = [
        index for index, cell in enumerate(cells)
        if marker in text(cell) and (cell_type is None or cell["cell_type"] == cell_type)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one cell containing {marker!r}, found {matches}")
    return matches[0]


def new_markdown(value):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(value).strip().splitlines(True),
    }


def new_code(value):
    value = dedent(value).strip() + "\n"
    ast.parse(value)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": value.splitlines(True),
    }


def build():
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = copy.deepcopy(notebook["cells"])

    set_markdown(cells[0], """
    # GeoDiff 3x Strong Base V2 on real Landsat/Sentinel tiles

    This notebook preserves the earlier MoE experiment and starts a **new isolated
    suite**. It uses actual Landsat 8/9 Collection 2 Level-2 RGB at 30 m and aligned
    Sentinel-2 L2A RGB targets at 10 m. It does not use OLI2MSI, synthetic LR, RDN,
    captions, GAN loss, perceptual loss, or PixelShuffle.

    The deterministic base is strengthened before any diffusion comparison:

    1. Audit validation-only alignment and per-image radiometric upper bounds.
    2. Train a 48-channel, four-group, three-block residual attention base on
       aligned `32→96` crops.
    3. Continue the same checkpoint on `64→192` crops.
    4. Finish with `128→384` full stored patches and complete validation.
    5. Freeze that base and compare one expert, generic MoE and reliability MoE.

    The default full budget is approximately 275 training minutes plus preparation
    and evaluation. Time limits are checked at short checkpoint boundaries, so wall
    time can overrun slightly. Run the smoke profile first, then choose a new suite
    root and set `FAST_DEV_RUN=False`. The test set remains locked until validation
    choices are final. A 35 dB result is an experimental target, not a guarantee.

    Attach `geodiff_tile_strong_base_source.zip` as a private Kaggle dataset together
    with your Landsat and Sentinel products.
    """)

    controls = find(cells, "FAST_DEV_RUN = True", "code")
    set_code(cells[controls], """
    from pathlib import Path
    import os, sys, json, subprocess, time, shutil

    FAST_DEV_RUN = True
    REPOSITORY_BRANCH = "3x-continued"
    REPOSITORY_URL = "https://github.com/shashankjs2002/SI-SR-1.git"
    REPOSITORY_DIR = Path("/kaggle/working/geodiff-3x-continued-strong-base")
    SOURCE_ARCHIVE = None
    SUITE_ROOT = Path(
        "/kaggle/working/geodiff-tiles-strong-base-smoke"
        if FAST_DEV_RUN else "/kaggle/working/geodiff-tiles-strong-base-v2"
    )
    SENTINEL_INPUT = Path("/kaggle/input")
    LANDSAT_INPUT = Path("/kaggle/input")
    RESTORE_SUITE_FROM = None

    NUM_EXPERTS = 2
    TOP_K = 1
    INCLUDE_MULTISPECTRAL = True
    NEW_MULTISPECTRAL = False
    TRAIN_LR_CROP = 64
    PATCH_SIZE, PATCH_STRIDE = 384, 288
    MAX_DAY_GAP = 3
    MAX_PAIRS = 1 if FAST_DEV_RUN else None
    MINIMUM_VALID_FRACTION = 0.95
    RUN_DATA_PREPARATION = True
    AUTO_EXTRACT_ARCHIVES = True

    # The three phases continue one base checkpoint. Use a new SUITE_ROOT if changed.
    BASE_PHASES = (
        {"name": "crop32", "crop": 32, "epochs": 30, "minutes": 60,
         "learning_rate": 1e-4, "max_batches": 200},
        {"name": "crop64", "crop": 64, "epochs": 24, "minutes": 60,
         "learning_rate": 5e-5, "max_batches": 80},
        {"name": "full128", "crop": 128, "epochs": 16, "minutes": 30,
         "learning_rate": 1e-5, "max_batches": 24},
    )
    EPOCHS = {"base": 12, "vae": 6, "diffusion": 10, "joint": 6}
    SHARED_MINUTES = {"base": 50, "vae": 20}
    PER_EXPERIMENT_MINUTES = {"diffusion": 25, "joint": 10}

    # Old five profiles remain available but are disabled and use separate roots.
    RUN_EARLIER = {name: False for name in (
        "rgb_standard", "rgb_fidelity", "multispectral_fidelity",
        "rgb_harmonized_fidelity", "multispectral_guided_fidelity")}
    ALIGNMENT_AUDIT_LIMIT = 2 if FAST_DEV_RUN else None
    EVAL_LIMIT = 2 if FAST_DEV_RUN else None
    TEST_EVAL_LIMIT = 2 if FAST_DEV_RUN else None
    EVAL_SAMPLES = 1 if FAST_DEV_RUN else 4
    EVAL_STEPS = 2 if FAST_DEV_RUN else 20
    RUN_TEST_EVALUATION = False
    RANDOM_SEED = 42

    if NEW_MULTISPECTRAL and not INCLUDE_MULTISPECTRAL:
        raise ValueError("Six-band experiments require INCLUDE_MULTISPECTRAL=True")
    if not 1 <= TOP_K <= NUM_EXPERTS:
        raise ValueError("Require 1 <= TOP_K <= NUM_EXPERTS")
    SUITE_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT = SUITE_ROOT / "dataset"
    MANIFEST = DATA_ROOT / "manifest.jsonl"
    EXTRACTION_ROOT = SUITE_ROOT / "extracted_inputs"
    EXPERIMENT_ROOT = SUITE_ROOT / "experiments"
    FIGURE_ROOT = SUITE_ROOT / "figures"
    for directory in (DATA_ROOT, EXPERIMENT_ROOT, FIGURE_ROOT):
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
    print("Root:", SUITE_ROOT)
    print("Nominal main-study training budget:", main_minutes, "minutes")
    print("Extra baseline profiles enabled:", [k for k, v in RUN_EARLIER.items() if v])
    """)

    install = find(cells, "geodiff_tile_moe_source*.zip", "code")
    source = text(cells[install]).replace(
        "geodiff_tile_moe_source*.zip", "geodiff_tile_strong_base_source*.zip"
    )
    source = source.replace(
        "This source lacks the new experiments.",
        "This source lacks the strong-base experiment helpers.",
    )
    set_code(cells[install], source)

    setup = find(cells, "def save_state():", "code")
    set_code(cells[setup], """
    import copy, yaml
    RESULTS = {}

    def save_state():
        value = {"manifest": str(MANIFEST), "manifest_sha256": sha256(MANIFEST),
                 "results": RESULTS, "repository": str(REPOSITORY_DIR),
                 "source_sha256": source_digest}
        temporary = SUITE_ROOT / "suite_state.json.tmp"
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        temporary.replace(SUITE_ROOT / "suite_state.json")

    if (SUITE_ROOT / "suite_state.json").exists():
        state = json.loads((SUITE_ROOT / "suite_state.json").read_text())
        if state["manifest_sha256"] != sha256(MANIFEST):
            raise ValueError("Saved registry uses a different manifest")
        RESULTS.update(state["results"])
    save_state()

    def calibration_for(profile):
        if profile not in ("rgb_harmonized_fidelity", "multispectral_guided_fidelity"):
            return None
        key = "lr_ms" if profile.startswith("multispectral") else "lr"
        path = SUITE_ROOT / "calibration" / f"{key}_train_only.json"
        lock_json(path.with_suffix(".lock.json"), {"manifest_sha256": sha256(MANIFEST), "key": key})
        if not path.exists():
            run([sys.executable, "-m", "geodiff_gan.cli.fit_radiometric",
                 "--manifest", MANIFEST, "--output", path, "--condition-key", key,
                 "--scale", 3, "--maximum-patches", 50 if FAST_DEV_RUN else 1000,
                 "--seed", RANDOM_SEED], REPOSITORY_DIR)
        return path

    def make_config(profile, root, crop_size=None):
        crop_size = TRAIN_LR_CROP if crop_size is None else int(crop_size)
        config = experiment_config(
            REPOSITORY_DIR, profile, MANIFEST, root, experts=NUM_EXPERTS,
            top_k=TOP_K, crop_size=crop_size, calibration=calibration_for(profile),
            multispectral=NEW_MULTISPECTRAL, fast=FAST_DEV_RUN,
        )
        if profile in NEW_PROFILES:
            config["model"].update(
                base_architecture="residual_swin", base_embed_dim=48,
                base_groups=4, base_depth=3, base_heads=4, window_size=8,
                base_upsample_mode="resize_conv",
            )
            config["training"].update(
                batch_size=2, gradient_accumulation=8, weight_decay=0.0,
                validation_limit=None if not FAST_DEV_RUN else 2,
                early_stopping_patience=6, early_stopping_min_epochs=8,
            )
            config["experiment"].update(
                strong_base="residual_swin_48_g4_d3",
                progressive_crops=[32, 64, 128],
                psnr_first_base=True,
            )
        configured = config["training"].get("validation_limit")
        tile_count = len({r.tile_id for r in records})
        if configured is not None:
            config["training"]["validation_limit"] = max(int(configured), tile_count)
        return config

    def configure_psnr_base(config):
        config = copy.deepcopy(config)
        config["training"]["loss_weights"].update(
            mse=100.0, multiscale_mse=10.0, charbonnier=0.1, ssim=0.1,
            consistency=0.0, gradient=0.0, wavelet=0.0, perceptual=0.0,
            adversarial=0.0, radiometric=0.0, residual_supervision=0.0,
            base_guard=0.0,
        )
        return config

    def register(name, root, config, checkpoint):
        RESULTS[name] = {"root": str(root), "config": str(config),
                         "checkpoint": str(checkpoint),
                         "checkpoint_sha256": sha256(checkpoint)}
        save_state()

    def run_earlier(profile):
        if not RUN_EARLIER[profile]:
            print("Skipped extra-session profile:", profile)
            return
        if profile.startswith("multispectral") and not INCLUDE_MULTISPECTRAL:
            raise ValueError("Enable multispectral preparation first")
        root = EXPERIMENT_ROOT / (profile + "_resize_control")
        config = make_config(profile, root)
        parent = None
        for stage in ("base", "vae", "diffusion", "joint"):
            minutes = {**SHARED_MINUTES, **PER_EXPERIMENT_MINUTES}[stage]
            config_path, parent = run_stage(
                REPOSITORY_DIR, config, root, stage, parent=parent,
                epochs=EPOCHS[stage], minutes=minutes, fast=FAST_DEV_RUN,
            )
        register(profile + "_resize_control", root, config_path, parent)

    prefix = "ms" if NEW_MULTISPECTRAL else "rgb"
    BASE_ROOT = EXPERIMENT_ROOT / f"{prefix}_strong_base_full128"
    BASE_CONFIG = make_config("residual_base", BASE_ROOT, crop_size=128)
    report = []
    for profile in NEW_PROFILES:
        config = make_config(profile, EXPERIMENT_ROOT / f"{prefix}_{profile}")
        model = GeoDiffGAN.from_config(config)
        assert not any(isinstance(module, torch.nn.PixelShuffle) for module in model.modules())
        report.append({"profile": profile,
                       "parameters": sum(p.numel() for p in model.parameters()),
                       "base_parameters": sum(p.numel() for p in model.base.parameters()),
                       "experts": config["model"]["diffusion_experts"],
                       "top_k": config["model"]["diffusion_top_k"]})
        del model
    display(pd.DataFrame(report))
    """)

    build_index = find(cells, "## 7. Run registry", "markdown")
    cells[build_index:build_index] = [
        new_markdown("""
        ## 7. Validation-only reconstruction ceiling audit

        This cell does not change the dataset. It compares bicubic reconstruction with
        two target-assisted **diagnostic upper bounds** on validation only: a ±2 HR-pixel
        shift oracle and a per-image affine RGB oracle. These are not inference methods
        and their PSNR must never be reported as model performance. A mean gain above
        0.2 dB indicates that alignment or radiometry may cap the achievable base PSNR.
        """),
        new_code("""
        from geodiff_gan.data import SentinelPatchDataset

        audit_dataset = SentinelPatchDataset(
            MANIFEST, split="val", scale=3, augment=False, random_degradation=False,
            input_mode="paired", output_channels=3,
        )

        def masked_psnr(prediction, target, mask):
            denominator = (mask.sum() * target.shape[0]).clamp_min(1)
            mse = (((prediction - target) ** 2) * mask).sum() / denominator
            return float(-10 * torch.log10(mse.clamp_min(1e-12)))

        def shifted(image, mask, dy, dx):
            value = torch.roll(image, shifts=(dy, dx), dims=(-2, -1))
            valid = mask.clone()
            if dy > 0: valid[..., :dy, :] = 0
            if dy < 0: valid[..., dy:, :] = 0
            if dx > 0: valid[..., :, :dx] = 0
            if dx < 0: valid[..., :, dx:] = 0
            return value, valid

        rows = []
        limit = len(audit_dataset) if ALIGNMENT_AUDIT_LIMIT is None else min(len(audit_dataset), ALIGNMENT_AUDIT_LIMIT)
        for index in range(limit):
            sample = audit_dataset[index]
            lr, hr, mask = sample["lr_rgb"], sample["hr"], sample["valid_mask"][0]
            bicubic = F.interpolate(lr[None], size=hr.shape[-2:], mode="bicubic", align_corners=False)[0].clamp(0, 1)
            original = masked_psnr(bicubic, hr, mask)
            shift_scores = []
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    candidate, overlap = shifted(bicubic, mask, dy, dx)
                    shift_scores.append((masked_psnr(candidate, hr, overlap), dy, dx))
            best_shift, dy, dx = max(shift_scores)
            affine = bicubic.clone()
            valid = mask[0] > 0.5
            for channel in range(3):
                x, y = bicubic[channel][valid].double(), hr[channel][valid].double()
                slope = ((x - x.mean()) * (y - y.mean())).mean() / x.var(unbiased=False).clamp_min(1e-12)
                offset = y.mean() - slope * x.mean()
                affine[channel] = (bicubic[channel] * slope.float() + offset.float()).clamp(0, 1)
            affine_score = masked_psnr(affine, hr, mask)
            rows.append({"index": index, "tile": sample["tile_id"], "bicubic_psnr": original,
                         "shift_oracle_psnr": best_shift, "shift_gain": best_shift - original,
                         "best_dy": dy, "best_dx": dx,
                         "affine_oracle_psnr": affine_score, "affine_gain": affine_score - original})

        ceiling_audit = pd.DataFrame(rows)
        ceiling_audit.to_csv(SUITE_ROOT / "validation_reconstruction_ceiling_audit.csv", index=False)
        display(ceiling_audit)
        summary = ceiling_audit[["bicubic_psnr", "shift_gain", "affine_gain"]].mean()
        display(summary.to_frame("mean"))
        if summary["shift_gain"] > 0.2:
            print("WARNING: alignment appears to cap reconstruction. Inspect pair registration before claiming architecture gains.")
        if summary["affine_gain"] > 0.2:
            print("WARNING: radiometric mismatch appears important. Test train-only calibration as a separate ablation.")
        """),
    ]
    registry_index = find(cells, "## 7. Run registry", "markdown")
    set_markdown(cells[registry_index], "## 8. Run registry and bounded training helpers")

    base_md = find(cells, "## 8. Experiment A: train the updated base only", "markdown")
    set_markdown(cells[base_md], """
    ## 9. Train the progressive strong base

    Each phase starts from the previous phase's best checkpoint but writes to a new
    immutable directory. Validation always uses complete stored patches. Intermediate
    base checkpoints are registered for an honest validation comparison.
    """)
    base_code = find(cells, "BASE_CONFIG_PATH, BASE_CHECKPOINT = run_stage", "code")
    set_code(cells[base_code], """
    phase_parent = None
    phase_outputs = []
    for phase_index, phase in enumerate(BASE_PHASES):
        phase_root = EXPERIMENT_ROOT / f"{prefix}_strong_base_{phase['name']}"
        phase_config = configure_psnr_base(
            make_config("residual_base", phase_root, crop_size=phase["crop"])
        )
        config_path, checkpoint = run_stage(
            REPOSITORY_DIR, phase_config, phase_root, "base",
            parent=phase_parent, epochs=phase["epochs"], minutes=phase["minutes"],
            fast=FAST_DEV_RUN, learning_rate=phase["learning_rate"],
            max_batches_per_epoch=phase["max_batches"],
        )
        is_final = phase_index == len(BASE_PHASES) - 1
        name = f"{prefix}_residual_base" if is_final else f"{prefix}_{phase['name']}_residual_base"
        register(name, phase_root, config_path, checkpoint)
        phase_outputs.append({"phase": phase["name"], "crop": phase["crop"],
                              "checkpoint": str(checkpoint), "sha256": sha256(checkpoint)})
        phase_parent = checkpoint

    BASE_ROOT = Path(RESULTS[f"{prefix}_residual_base"]["root"])
    BASE_CONFIG_PATH = Path(RESULTS[f"{prefix}_residual_base"]["config"])
    BASE_CHECKPOINT = Path(RESULTS[f"{prefix}_residual_base"]["checkpoint"])
    (SUITE_ROOT / "strong_base_phases.json").write_text(json.dumps(phase_outputs, indent=2))
    display(pd.DataFrame(phase_outputs))
    print("Selected shared strong base:", BASE_CHECKPOINT)
    """)

    vae_md = find(cells, "## 9. Train the shared VAE", "markdown")
    set_markdown(cells[vae_md], "## 10. Freeze the strong base and train the shared residual representation")
    vae_code = find(cells, "VAE_ROOT = EXPERIMENT_ROOT", "code")
    source = text(cells[vae_code]).replace(
        'VAE_CONFIG = make_config("residual_base", VAE_ROOT)',
        'VAE_CONFIG = make_config("residual_base", VAE_ROOT, crop_size=TRAIN_LR_CROP)',
    )
    set_code(cells[vae_code], source)

    for cell in cells:
        if cell["cell_type"] == "markdown":
            value = text(cell)
            value = value.replace("## 10. Validate", "## 12. Validate")
            value = value.replace("## 11. Save comparison", "## 13. Save comparison")
            value = value.replace("## 12. Routing diagnostics", "## 14. Routing diagnostics")
            value = value.replace("## 13. Locked test", "## 15. Locked test")
            value = value.replace("## 14. Indexed comparison", "## 16. Indexed comparison")
            value = value.replace("## 15. Standalone", "## 17. Standalone")
            value = value.replace("## 16. Export", "## 18. Export")
            cell["source"] = value.splitlines(True)
        elif "VIEW_REPOSITORY =" in text(cell):
            value = text(cell).replace(
                'Path("/kaggle/working/geodiff-3x-continued-moe")',
                'Path("/kaggle/working/geodiff-3x-continued-strong-base")',
            ).replace(
                'Path("/kaggle/working/geodiff-tiles-moe-v1")',
                'Path("/kaggle/working/geodiff-tiles-strong-base-v2")',
            )
            set_code(cell, value)

    for index, cell in enumerate(cells):
        cell["id"] = f"strong-base-{index:03d}"
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
            ast.parse(text(cell))
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
