"""Build the 128 -> 384 TrustMoE OLI2MSI benchmark notebook."""
import ast
import copy
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "kaggle/GeoDiff_TrustMoE_Transformer_3x.ipynb"
OUTPUT = ROOT / "kaggle/GeoDiff_TrustMoE_OLI2MSI_128x128_3x.ipynb"


def set_cell(cell, text):
    text = dedent(text).strip() + "\n"
    if cell["cell_type"] == "code":
        ast.parse(text)
        cell["execution_count"] = None
        cell["outputs"] = []
    cell["source"] = text.splitlines(True)


def build():
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = copy.deepcopy(notebook["cells"])
    set_cell(cells[0], """
    # GeoDiff-TrustMoE on official OLI2MSI: 128 x 128 to 384 x 384

    This notebook tests the diffusion-free TrustMoE model on actual OLI2MSI
    Landsat-8 OLI / Sentinel-2 MSI RGB pairs. The model always receives a
    `3 x 128 x 128` LR tensor and predicts `3 x 384 x 384`.

    Official source frames are `160 x 160 -> 480 x 480`. Training reads a new
    random aligned `128 -> 384` crop per pair and epoch. Validation and all 100
    official test pairs use a fixed center crop. The official test set is untouched
    until the locked final-evaluation section. Five percent of the 5,225 official
    training pairs is assigned to validation by a deterministic source-ID hash.

    Radiometry is fixed dataset-wide: `clip(reflectance, 0, .3) / .3`, optionally
    rounded to the 8-bit grid used by the common OLI2MSI protocol. There is no
    per-image normalization, synthetic LR, target-time calibration, captioning,
    diffusion, VAE, second base pass or PixelShuffle.

    Every experiment gets a separate directory and shares one frozen base per seed.
    Checkpoints resume at completed epoch boundaries. A 35 dB result and publication
    novelty are hypotheses, not promises. Run the smoke mode first, then use a new
    suite root for the complete experiment.
    """)
    set_cell(cells[1], "## 1. Controls: official data, 128-pixel input, epochs and batch size")
    set_cell(cells[2], r'''
    from pathlib import Path
    import os, sys, json, shutil, subprocess, hashlib, time, zipfile

    FAST_DEV_RUN = False
    REPOSITORY_URL = 'https://github.com/shashankjs2002/SI-SR-1.git'
    REPOSITORY_BRANCH = '3x-continued'
    REPOSITORY_DIR = Path('/kaggle/working/geodiff-trust-moe-oli2msi-source')
    SUITE_ROOT = Path('/kaggle/working/geodiff-trust-moe-oli2msi-128-v1')
    KNOWN_DATA_ROOT = Path('/kaggle/input/datasets/twilight2002/oli2msi-thesis/OLI2MSI-dataset/OLI2MSI')
    OLI2MSI_DATA_ROOT = KNOWN_DATA_ROOT if KNOWN_DATA_ROOT.exists() else None
    RESTORE_SUITE_FROM = None
    IMPORT_BASE_CHECKPOINT = None

    EPOCHS = {'base': 30, 'residual': 15}
    BATCH_SIZE = 4
    TRAIN_LR_CROP = 128
    NUM_EXPERTS, TOP_K = 5, 2
    REGION_FRACTION = 0.5
    SEEDS = [42]
    RUN = dict(single_expert=True, dense_transformer=True, sparse_uniform=True,
               sparse_conv=True, sparse_transformer=True, sparse_no_trust=True,
               sparse_adversarial=True, adaptive_k=True)
    RUN_TEST_EVALUATION = True
    BASE_MODEL = dict(base_embed_dim=32, base_depth=2, base_groups=2, base_heads=4,
                      window_size=8, width=32, tile_size=8, halo=4)
    DISPLAY_MAX = 1.0
    DATASET_PROTOCOL = 'provided_benchmark'
    VALIDATION_PERCENT = 5
    QUANTIZE_TO_UINT8_GRID = True
    FAST_LIMIT = 16 if FAST_DEV_RUN else None

    if sys.version_info < (3, 10):
        raise RuntimeError('Use the current Python 3.10+ Kaggle kernel.')
    if not 1 <= TOP_K <= NUM_EXPERTS or not 0 < REGION_FRACTION <= 1:
        raise ValueError('Invalid expert count/top-k/region fraction')
    SUITE_ROOT.mkdir(parents=True, exist_ok=True)

    def run(command, cwd=None):
        environment = os.environ.copy()
        environment['PYTHONPATH'] = str(REPOSITORY_DIR / 'src') + os.pathsep + environment.get('PYTHONPATH', '')
        environment['PYTHONUNBUFFERED'] = '1'
        print('+', ' '.join(map(str, command)), flush=True)
        return subprocess.run(list(map(str, command)), cwd=cwd, env=environment, check=True)

    print('Suite:', SUITE_ROOT)
    print('Model geometry: 128 x 128 -> 384 x 384')
    print('Epochs:', EPOCHS, 'batch size:', BATCH_SIZE, 'experts:', NUM_EXPERTS, 'top-k:', TOP_K)
    ''')
    set_cell(cells[3], "## 2. Clone the exact branch, install it, restore outputs and verify GPU")
    set_cell(cells[4], r'''
    run([sys.executable, '-m', 'pip', 'install', '-q', 'rasterio', 'numpy', 'Pillow',
         'PyYAML', 'tqdm', 'pandas', 'matplotlib'])
    if not REPOSITORY_DIR.exists():
        run(['git', 'clone', '--depth', '1', '--single-branch', '--branch',
             REPOSITORY_BRANCH, REPOSITORY_URL, REPOSITORY_DIR])
    if not (REPOSITORY_DIR / '.git').is_dir():
        raise RuntimeError('REPOSITORY_DIR exists but is not a Git checkout; use another path.')
    branch = subprocess.check_output(['git', '-C', str(REPOSITORY_DIR), 'branch', '--show-current'], text=True).strip()
    if branch != REPOSITORY_BRANCH:
        raise RuntimeError(f'Expected {REPOSITORY_BRANCH}, found {branch}; use another repository path.')
    required = REPOSITORY_DIR / 'src/geodiff_gan/data/oli2msi_128.py'
    if not required.is_file():
        raise RuntimeError('Push the OLI2MSI TrustMoE notebook support to GitHub, then clone into a new path.')
    run([sys.executable, '-m', 'pip', 'install', '-q', '-e', REPOSITORY_DIR, '--no-deps'])
    sys.path.insert(0, str(REPOSITORY_DIR / 'src'))

    if RESTORE_SUITE_FROM:
        source = Path(RESTORE_SUITE_FROM)
        if source.is_file():
            expanded = SUITE_ROOT.parent / 'trust-moe-oli2msi-restore'
            expanded.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(source) as archive:
                for member in archive.infolist():
                    if not (expanded / member.filename).resolve().is_relative_to(expanded.resolve()):
                        raise ValueError('Unsafe path in restore ZIP: ' + member.filename)
                    if (member.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError('Restore ZIP symlinks are not accepted')
                archive.extractall(expanded)
            source = expanded
        if not source.is_dir():
            raise FileNotFoundError('RESTORE_SUITE_FROM must be a prior result ZIP or extracted suite directory.')
        for path in source.rglob('*'):
            if path.is_file():
                destination = SUITE_ROOT / path.relative_to(source)
                if not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, destination)

    import torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
    from IPython.display import display, FileLink
    from geodiff_gan.data.oli2msi_128 import find_oli2msi_layout, prepare_oli2msi_128
    from geodiff_gan.experiments.trust_moe import (
        make_config, prepare_manifest, write_json, digest_file, adopt_base, evaluate, dataset_for,
    )
    from geodiff_gan.experiments.trust_report import (
        METRICS, summarize, benchmark, compare_controls, paired_intervals, visualize, bundle_results,
    )
    if not torch.cuda.is_available():
        raise RuntimeError('Enable a Kaggle GPU accelerator.')
    print('Commit:', subprocess.check_output(['git', '-C', str(REPOSITORY_DIR), 'rev-parse', 'HEAD'], text=True).strip())
    print('Python:', sys.version, 'Torch:', torch.__version__, 'GPU:', torch.cuda.get_device_name(0))
    write_json(SUITE_ROOT / 'environment.json', dict(python=sys.version, torch=torch.__version__,
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), input=[128, 128], output=[384, 384]))
    ''')
    set_cell(cells[5], """
    ## 3. Discover and prepare official OLI2MSI pairs

    Attach a Kaggle dataset containing `train_lr`, `train_hr`, `test_lr`, and
    `test_hr`. `OLI2MSI_DATA_ROOT` may point to the dataset root, one split folder,
    or an individual TIFF. A full run requires exactly 5,225 official training and
    100 official test pairs. Missing/corrupt pairs stop the benchmark rather than
    silently reducing it.

    Prepared arrays are an internal restart cache for the current loader. Training
    sources remain full 160→480 so every epoch can draw a different aligned 128→384
    crop. Validation/test arrays are fixed center crops. The original attached TIFFs
    are read-only and never changed or deleted.
    """)
    set_cell(cells[6], r'''
    if OLI2MSI_DATA_ROOT is None:
        search = []
        for directory in Path('/kaggle/input').rglob('*'):
            if directory.is_dir() and ''.join(c for c in directory.name.casefold() if c.isalnum()) == 'trainlr':
                search.append(directory)
        roots = sorted({path.parent for path in search})
        if len(roots) != 1:
            raise ValueError(f'Set OLI2MSI_DATA_ROOT explicitly; candidate roots: {roots}')
        OLI2MSI_DATA_ROOT = roots[0]
    PREPARED_MANIFEST, DATA_CARD = prepare_oli2msi_128(
        OLI2MSI_DATA_ROOT, SUITE_ROOT / 'prepared_oli2msi', seed=42,
        validation_percent=VALIDATION_PERCENT, quantize=QUANTIZE_TO_UINT8_GRID,
        fast_limit=FAST_LIMIT, require_official_counts=not FAST_DEV_RUN)
    MANIFEST = SUITE_ROOT / 'runtime_manifest.jsonl'
    AUDIT = prepare_manifest(PREPARED_MANIFEST, MANIFEST, spatial_audit=False,
        minimum_test_fraction=0, disjoint_tiles=False)
    AUDIT['oli2msi_protocol'] = DATA_CARD
    write_json(SUITE_ROOT / 'dataset_audit.json', AUDIT)
    print(json.dumps(AUDIT, indent=2))
    if AUDIT['frames_hr'] != [384, 480]:
        raise RuntimeError(f'Expected train/eval HR frame sizes [384, 480], got {AUDIT["frames_hr"]}')
    ''')
    set_cell(cells[7], "## 4. Verify the exact 128 → 384 tensors before training")
    set_cell(cells[8], r'''
    from torch.nn import functional as F

    def preview_pair(index=0, split='test'):
        config = make_config(MANIFEST, SUITE_ROOT, profile='base', crop_size=TRAIN_LR_CROP)
        data = dataset_for(config, split)
        sample = data[int(index) % len(data)]
        lr, hr = sample['lr'], sample['hr']
        if tuple(lr.shape) != (3, 128, 128) or tuple(hr.shape) != (3, 384, 384):
            raise RuntimeError(f'Wrong model geometry: {tuple(lr.shape)} -> {tuple(hr.shape)}')
        bicubic = F.interpolate(lr[None], size=hr.shape[-2:], mode='bicubic', align_corners=False)[0].clamp(0, 1)
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        for axis, image, title in zip(axes, (lr, bicubic, hr),
                ('Landsat OLI LR 128 x 128', 'Bicubic 384 x 384', 'Sentinel-2 MSI HR 384 x 384')):
            axis.imshow(image.permute(1, 2, 0).clamp(0, 1), interpolation='nearest')
            axis.set_title(title); axis.axis('off')
        fig.tight_layout()
        folder = SUITE_ROOT / 'figures'; folder.mkdir(exist_ok=True)
        fig.savefig(folder / f'pair_{split}_{index}.png', dpi=160)
        plt.show()
        print('Stored ranges:', float(lr.min()), float(lr.max()), float(hr.min()), float(hr.max()))
    preview_pair(0, 'test')
    ''')

    # Correct dataset-specific prose retained from the generic experiment sections.
    for cell in cells[9:]:
        source = "".join(cell["source"])
        source = source.replace("Within-tile tests do not establish unseen-region generalization.",
            "The official center-crop test protocol does not establish generalization beyond OLI2MSI.")
        source = source.replace("Expert labels are not forced; examine collapse/utilization by scene class.",
            "OLI2MSI has no scene labels here; analyze expert collapse/utilization without semantic claims.")
        source = source.replace("""selected_indices, seen_classes = [], set()
    for i, record in enumerate(data.records):
        if record.scene_class not in seen_classes:
            seen_classes.add(record.scene_class); selected_indices.append(i)
        if len(selected_indices) >= 12:
            break""", "selected_indices = list(range(min(6, len(data))))")
        if cell["cell_type"] == "code":
            ast.parse(source)
            cell["execution_count"] = None
            cell["outputs"] = []
        cell["source"] = source.splitlines(True)
    for index, cell in enumerate(cells):
        cell["id"] = f"trust-oli128-{index:03d}"
    notebook["cells"] = cells
    OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    build()
