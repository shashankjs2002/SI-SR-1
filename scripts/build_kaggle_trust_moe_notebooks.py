"""Build the isolated diffusion-free study and a Kaggle paired-tile preparation notebook."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
from textwrap import dedent
import zipfile

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip().splitlines(True)})


def code(text):
    text = dedent(text).strip() + "\n"
    ast.parse(text)
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.splitlines(True)})


md("""
# GeoDiff-TrustMoE: direct sparse residual super-resolution

**No diffusion, VAE, denoising loop, second base pass, or PixelShuffle.**
The base reconstructs the image once. A small Transformer routes pooled LR regions
to efficient CNN residual experts. Only selected regions execute experts. A learned
RGB trust gate controls the correction. Unselected regions retain the base exactly.

This is an experimental hypothesis, not a guarantee of 35 dB or publication novelty.
Transformer/MoE satellite SR already exists; the study tests fidelity, spatial detail,
actual regional dispatch, and the resulting quality/latency trade-off.

**Run order:** prepare tiles in `Landsat_Sentinel_TrustMoE_Dataset_Preparation.ipynb`,
upload its exported folder/ZIP as a Kaggle dataset, then run this notebook on a GPU.
Also attach `geodiff_trust_moe_source.zip` from the updated local repository: local
changes are not automatically available on GitHub. Existing portable paired manifests
from the diverse-tile preparation notebook are also accepted.

Every experiment has its own folder and shares the same frozen base and input pairs.
Every epoch is a complete training-data pass. Increase an epoch count to continue from
`last.pt`; do not change the dataset/model/loss mid-run. Interrupted epochs are replayed
from the previous completed checkpoint. Save a Kaggle version **with outputs** before
ending a session, then attach/restore those outputs in the next session.
""")
md("## 1. Controls: epochs, batch size, data and model")
code(r'''
from pathlib import Path
import os, sys, json, shutil, subprocess, zipfile, hashlib, time

SUITE_ROOT = Path("/kaggle/working/geodiff-trust-moe-v1")
REPOSITORY_DIR = Path("/kaggle/working/geodiff-trust-moe-source")
SOURCE_ARCHIVE = None  # Auto-find geodiff_trust_moe_source.zip below /kaggle/input.
PREPARED_MANIFEST = None  # Or Path('/kaggle/input/.../manifest.jsonl').
RESTORE_SUITE_FROM = None  # Attached previous suite folder or downloaded results ZIP.
IMPORT_BASE_CHECKPOINT = None  # Optional compatible residual-Swin base, same data/range/splits.

DATASET_PROTOCOL = "own_tiles"  # "spatial_blocks" for new GEE India exports; "provided_benchmark" for official prepared benchmarks.
DISPLAY_MAX = 0.3  # Raw surface reflectance. Use 1.0 for clip(0,.3)/.3-normalized benchmarks.
EPOCHS = {"base": 30, "residual": 15}
BATCH_SIZE = 4
TRAIN_LR_CROP = 64  # Random aligned 64 -> 192 crops; validation/test use COMPLETE stored frames.
NUM_EXPERTS, TOP_K = 5, 2
REGION_FRACTION = 0.5  # Fraction of LR regions allowed to execute experts; an inference budget.
SEEDS = [42]  # For a final study repeat with [42, 123, 2026]; no test-based seed selection.
RUN = dict(single_expert=True, dense_transformer=True, sparse_uniform=True,
           sparse_conv=True, sparse_transformer=True, sparse_no_trust=True,
           sparse_adversarial=True, adaptive_k=True)
RUN_TEST_EVALUATION = True  # Executed only AFTER recording the validation-selected budget.
# Only change base dimensions to match an existing checkpoint, or for a NEW study.
BASE_MODEL = dict(base_embed_dim=32, base_depth=2, base_groups=2, base_heads=4,
                  window_size=8, width=32, tile_size=8, halo=4)

if sys.version_info < (3, 10):
    raise RuntimeError('Use a Python 3.10+ Kaggle kernel.')
if not 1 <= TOP_K <= NUM_EXPERTS or not 0 < REGION_FRACTION <= 1:
    raise ValueError('Invalid expert count/top-k/region fraction')
if DATASET_PROTOCOL not in ('own_tiles', 'spatial_blocks', 'provided_benchmark'):
    raise ValueError(DATASET_PROTOCOL)
SUITE_ROOT.mkdir(parents=True, exist_ok=True)

def safe_unzip(path, root):
    root = Path(root).resolve()
    with zipfile.ZipFile(path) as handle:
        for member in handle.infolist():
            if not (root / member.filename).resolve().is_relative_to(root):
                raise ValueError('Unsafe ZIP path: ' + member.filename)
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('ZIP symlinks are not accepted')
        handle.extractall(root)

if RESTORE_SUITE_FROM:
    source = Path(RESTORE_SUITE_FROM)
    if source.is_file():
        expanded = SUITE_ROOT.parent / 'trust-moe-restore'
        safe_unzip(source, expanded)
        source = expanded
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in source.rglob('*'):
        if path.is_file():
            destination = SUITE_ROOT / path.relative_to(source)
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
    print('Restored missing files; no current files were replaced.')

def run(command, cwd=None):
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(REPOSITORY_DIR / 'src') + os.pathsep + environment.get('PYTHONPATH', '')
    environment['PYTHONUNBUFFERED'] = '1'
    print('+', ' '.join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), cwd=cwd, env=environment, check=True)

print('Suite:', SUITE_ROOT, '\nEpochs:', EPOCHS, 'batch:', BATCH_SIZE)
print('No training time, maximum-batch, or update caps are used.')
''')
md("## 2. Install the attached source and check the GPU")
code(r'''
if not REPOSITORY_DIR.exists():
    matches = [Path(SOURCE_ARCHIVE)] if SOURCE_ARCHIVE else list(Path('/kaggle/input').rglob('geodiff_trust_moe_source.zip'))
    if len(matches) == 1:
        safe_unzip(matches[0], REPOSITORY_DIR)
    elif len(matches) > 1:
        raise ValueError(f'Set SOURCE_ARCHIVE to one of: {matches}')
    else:
        run(['git', 'clone', '--depth', '1', '--branch', '3x-continued',
             'https://github.com/shashankjs2002/SI-SR-1.git', REPOSITORY_DIR])
required = REPOSITORY_DIR / 'src/geodiff_gan/models/trust_moe.py'
if not required.is_file():
    raise RuntimeError('This source predates TrustMoE. Attach the supplied geodiff_trust_moe_source.zip and use a new REPOSITORY_DIR.')
run([sys.executable, '-m', 'pip', 'install', '-q', 'numpy', 'Pillow', 'PyYAML', 'tqdm', 'pandas', 'matplotlib'])
run([sys.executable, '-m', 'pip', 'install', '-q', '-e', REPOSITORY_DIR, '--no-deps'])
sys.path.insert(0, str(REPOSITORY_DIR / 'src'))
import torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, FileLink
from geodiff_gan.experiments.trust_moe import (
    make_config, prepare_manifest, write_json, digest_file, adopt_base, evaluate, dataset_for,
)
from geodiff_gan.experiments.trust_report import (
    METRICS, summarize, benchmark, compare_controls, paired_intervals, visualize, bundle_results,
)
print('Python:', sys.version, '\nTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
if not torch.cuda.is_available():
    raise RuntimeError('Enable a Kaggle GPU accelerator. This notebook will not silently train on CPU.')
print('GPU:', torch.cuda.get_device_name(0))
write_json(SUITE_ROOT / 'environment.json', dict(python=sys.version, torch=torch.__version__,
           cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), protocol=DATASET_PROTOCOL))
shutil.copy2(REPOSITORY_DIR / 'kaggle/GeoDiff_TrustMoE_Transformer_3x.ipynb', SUITE_ROOT / 'run_notebook.ipynb')
''')
md("""
## 3. Verify the prepared dataset and retain its fixed splits
For your tiles the audit requires at least 10% test pairs and disjoint geographic
regions inside every tile. Categories are metadata for analysis, **not forced expert
labels**. Directory-derived categories are weak scene labels, not pixel annotations.
No LR synthesis, per-image normalization, or target calibration occurs here.

For the new Earth Engine India exports use `DATASET_PROTOCOL='spatial_blocks'`.
Their geographic blocks belong to only one split; they do not put all three splits
inside every block. Retain `DISPLAY_MAX=0.3` for their raw reflectance arrays.

For a provided official benchmark, point to its existing *prepared paired NPZ manifest*,
set `DATASET_PROTOCOL='provided_benchmark'` and the appropriate `DISPLAY_MAX`.
That option preserves the official split without requiring 10% test or claiming
geographic independence. It does not discover or normalize raw OLI2MSI TIFFs.
""")
code(r'''
if PREPARED_MANIFEST is None:
    cards = list(Path('/kaggle/input').rglob('dataset_card.json'))
    candidates = sorted({p.parent / 'manifest.jsonl' for p in cards if (p.parent / 'manifest.jsonl').exists()})
    if not candidates:
        candidates = sorted(Path('/kaggle/input').rglob('manifest.jsonl'))
    if len(candidates) != 1:
        raise ValueError(f'Set PREPARED_MANIFEST to the prepared portable dataset manifest. Found {len(candidates)}: {candidates[:12]}')
    PREPARED_MANIFEST = candidates[0]
PREPARED_MANIFEST = Path(PREPARED_MANIFEST)
MANIFEST = SUITE_ROOT / 'runtime_manifest.jsonl'
AUDIT = prepare_manifest(PREPARED_MANIFEST, MANIFEST,
    spatial_audit=DATASET_PROTOCOL == 'own_tiles',
    minimum_test_fraction=0.10 if DATASET_PROTOCOL != 'provided_benchmark' else 0,
    disjoint_tiles=DATASET_PROTOCOL == 'spatial_blocks')
print(json.dumps(AUDIT, indent=2))
for name in ('dataset_card.json', 'source_pairs.csv', 'category_split_summary.csv', 'numerical_audit.csv'):
    path = PREPARED_MANIFEST.parent / name
    if path.exists():
        destination = SUITE_ROOT / 'dataset_info' / name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(path, destination)
from geodiff_gan.data.manifest import load_manifest
records = load_manifest(MANIFEST)
table = pd.DataFrame([dict(split=r.split, tile=r.tile_id, category=r.scene_class) for r in records])
display(pd.crosstab([table.category, table.tile], table.split))
print('Numerical data audit is not model selection. Do not inspect test predictions until the final section.')
''')
md("## 4. Visual sanity check using training pairs only")
code(r'''
def preview_pair(index=0, split='train'):
    config = make_config(MANIFEST, SUITE_ROOT, profile='base', crop_size=TRAIN_LR_CROP)
    data = dataset_for(config, 'val' if split == 'val' else 'train')
    sample = data[index]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, key, title in zip(axes[:2], ('lr', 'hr'), ('Original Landsat LR', 'Sentinel HR')):
        array = sample[key].numpy()
        shown = np.clip(array.transpose(1, 2, 0) / DISPLAY_MAX, 0, 1) ** (1 / 1.4)
        axis.imshow(shown, interpolation='nearest')
        axis.set_title(f'{title}: {array.shape[-1]} x {array.shape[-2]}')
    axes[2].imshow(sample['valid_mask'][0], vmin=0, vmax=1, cmap='gray')
    axes[2].set_title('Valid target pixels')
    for axis in axes:
        axis.axis('off')
    fig.tight_layout()
    folder = SUITE_ROOT / 'figures'; folder.mkdir(exist_ok=True)
    fig.savefig(folder / f'input_{split}_{index}.png', dpi=160)
    plt.show()
    print('Metric arrays unchanged. LR range:', sample['lr'].min().item(), sample['lr'].max().item())
preview_pair()
''')
md("""
## 5. Experiment helpers and base reconstruction
Train the base once per seed, then freeze it for every residual experiment. Optional
base import is strict: use a compatible residual-Swin checkpoint trained on **these
training pairs with the same numeric range and splits**. Import does not prove that
provenance; check it yourself. Diffusion/VAE modules are not instantiated on import.

Checkpoint selection is maximum validation PSNR. No test metrics affect training.
Changing `EPOCHS` is supported; changes to architecture, data, loss, or batch/crop
belong in a new experiment directory to avoid mixing incompatible runs.
""")
code(r'''
REGISTRY_PATH = SUITE_ROOT / 'registry.json'
REGISTRY = json.loads(REGISTRY_PATH.read_text()) if REGISTRY_PATH.exists() else {}

def best_path(seed, profile):
    return SUITE_ROOT / f'seed_{seed}' / profile / 'best.pt'

def train_profile(profile):
    for seed in SEEDS:
        root = SUITE_ROOT / f'seed_{seed}' / profile
        config = make_config(MANIFEST, root, profile=profile, seed=seed,
            epochs=EPOCHS['base' if profile == 'base' else 'residual'], batch_size=BATCH_SIZE,
            crop_size=TRAIN_LR_CROP, experts=NUM_EXPERTS, top_k=TOP_K,
            dataset_id=AUDIT['dataset_id'], parent=None if profile == 'base' else best_path(seed, 'base'),
            base_model=BASE_MODEL)
        if profile not in ('base', 'single_expert', 'dense_transformer'):
            config['model']['coverage'] = REGION_FRACTION
        root.mkdir(parents=True, exist_ok=True)
        path = root / 'request_config.json'
        write_json(path, config)
        if profile == 'base' and IMPORT_BASE_CHECKPOINT:
            if len(SEEDS) != 1:
                raise ValueError('One imported base is not multiple independent base-training seeds. Use one seed or train each base.')
            adopt_base(config, IMPORT_BASE_CHECKPOINT)
        else:
            run([sys.executable, '-m', 'geodiff_gan.cli.trust_moe', 'train', '--config', path])
        REGISTRY[f'{seed}/{profile}'] = str(best_path(seed, profile).relative_to(SUITE_ROOT))
        write_json(REGISTRY_PATH, REGISTRY)
    print('Saved:', profile)

train_profile('base')
''')

profiles = [
    ("single_expert", "Single-expert direct residual control", "One expert processes all regions. This tests whether multiple experts help at all; it has fewer parameters."),
    ("dense_transformer", "Dense-region Transformer MoE", "All spatial regions execute top-k experts. The expert modules and router match the sparse version; only spatial coverage differs."),
    ("sparse_uniform", "Uniform-region control", "The same regional budget is spread uniformly instead of selected by predicted difficulty."),
    ("sparse_conv", "Local-router control", "Replace global token attention with a depthwise-position convolution and token MLP. Experts and regional budget remain unchanged."),
    ("sparse_transformer", "Proposed contextual sparse residual MoE", "A Transformer predicts routes and base-error difficulty from inference-available cues. Selected tiles execute top-k experts; a per-band trust head accepts their residual."),
    ("sparse_no_trust", "Trust-gate ablation", "Remove the learned acceptance gate, retaining selected regions, expert routing, and the reconstruction/local-excess losses."),
    ("sparse_adversarial", "Adversarial-loss ablation", "Add a small conditional PatchGAN loss. This is an ablation, not an assumption that adversarial training improves fidelity."),
    ("adaptive_k", "Adaptive regional expert-count ablation", "Inactive regions use zero experts. Active regions receive 1..TOP_K slots by predicted-difficulty rank. This is an explicit rank policy, not a learned runtime optimizer."),
]
for number, (profile, title, description) in enumerate(profiles, 6):
    md(f"## {number}. {title}\n{description}")
    code(f"if RUN[{profile!r}]:\n    train_profile({profile!r})\nelse:\n    print('Skipped {profile}')")

md("## 14. Validation comparisons and training curves")
code(r'''
def evaluation_dir(seed, profile, split, coverage=None, top_k=None):
    checkpoint = best_path(seed, profile)
    identity = digest_file(checkpoint)[:12]
    tag = f'coverage_{coverage}_k_{top_k}'
    return SUITE_ROOT / f'seed_{seed}' / profile / 'evaluation' / split / identity / tag

VAL_EVALS = {}
for seed in SEEDS:
    for profile in ['base'] + [p for p, enabled in RUN.items() if enabled]:
        checkpoint = best_path(seed, profile)
        if not checkpoint.exists():
            raise FileNotFoundError(f'Run the {profile} training cell first: {checkpoint}')
        output = evaluation_dir(seed, profile, 'val')
        evaluate(checkpoint, output, 'val', manifest=MANIFEST)
        VAL_EVALS[f'{seed}/{profile}'] = output
VAL_TABLES = summarize(VAL_EVALS, SUITE_ROOT / 'reports/validation')
display(VAL_TABLES['overall'][['experiment', 'count', *METRICS]])
display(VAL_TABLES['expert_usage'])
from IPython.display import Image as DisplayImage
display(DisplayImage(filename=str(SUITE_ROOT / 'reports/validation/quality_comparison.png')))

histories = []
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for seed in SEEDS:
    for profile in ['base'] + [p for p, enabled in RUN.items() if enabled]:
        history = [json.loads(p.read_text()) for p in sorted((SUITE_ROOT / f'seed_{seed}' / profile / 'history').glob('*.json'))]
        for row in history:
            histories.append(dict(seed=seed, profile=profile, **row))
        if history:
            axes[0].plot([r['epoch'] for r in history], [r['val_psnr'] for r in history], label=f'{seed}/{profile}')
            axes[1].plot([r['epoch'] for r in history], [r['loss_total'] for r in history], label=f'{seed}/{profile}')
axes[0].set_title('Full-frame validation PSNR'); axes[1].set_title('Training objective (different losses are not directly comparable)')
axes[0].legend(fontsize=7); fig.tight_layout()
fig.savefig(SUITE_ROOT / 'figures/training_curves.png', dpi=170); plt.show()
pd.DataFrame(histories).to_csv(SUITE_ROOT / 'reports/training_history.csv', index=False)
''')
md("""
## 15. Parameters, measured latency and validation-only compute-budget ablations
Timing includes the base, router, expert dispatch, fusion and trust head. It excludes
disk I/O and transfer. CUDA is synchronized, AMP is consistent, and warm-up is excluded.
Record GPU and input dimensions before comparing with previous diffusion timings.
Spatial sparsity does not automatically mean a speedup: small tile kernels, halos,
attention, and gather/scatter overhead may outweigh saved expert work.

All models are timed at their trained budget. The proposed Transformer model is also
tested at 25%, 50%, 75%, and 100% regional coverage using its **trained top-k**.
Post-training budget changes are explicitly reported as ablations.
""")
code(r'''
TIMINGS, BUDGET_ROWS = [], []
for seed in SEEDS:
    for profile in ['base'] + [p for p, enabled in RUN.items() if enabled]:
        checkpoint = best_path(seed, profile)
        identity = digest_file(checkpoint)[:12]
        path = SUITE_ROOT / 'reports/timing' / f'{seed}_{profile}_{identity}.json'
        timings = benchmark(checkpoint, path, manifest=MANIFEST)
        TIMINGS.extend(dict(seed=seed, profile=profile, **row) for row in timings)
    if RUN['sparse_transformer']:
        for fraction in (0.25, 0.5, 0.75, 1.0):
            path = evaluation_dir(seed, 'sparse_transformer', 'val', fraction, TOP_K)
            metrics = evaluate(best_path(seed, 'sparse_transformer'), path, 'val',
                               coverage=fraction, top_k=TOP_K, manifest=MANIFEST, save_images=False)
            rows = json.loads((path / 'per_image.json').read_text())
            intervals = {r['metric']: r for r in paired_intervals(rows)}
            timing_path = path / 'timing.json'
            speed = benchmark(best_path(seed, 'sparse_transformer'), timing_path,
                              manifest=MANIFEST, coverage=fraction, top_k=TOP_K)[1]
            row = dict(seed=seed, coverage=fraction, top_k=TOP_K, **metrics,
                       mean_ms=speed['mean_ms'], images_per_second=speed['images_per_second'])
            for metric in ('psnr', 'ssim', 'edge_f1'):
                row[metric + '_ci_low'] = intervals[metric]['ci95_low']
            row['feasible'] = row['psnr_ci_low'] > -0.05 and row['ssim_ci_low'] > -0.001 and row['edge_f1_ci_low'] > 0
            BUDGET_ROWS.append(row)
TIMING_TABLE = pd.DataFrame(TIMINGS)
TIMING_TABLE.to_csv(SUITE_ROOT / 'reports/timing.csv', index=False)
display(TIMING_TABLE[['seed','profile','operation','parameters_total','parameters_used_on_this_frame','mean_ms','p95_ms','images_per_second','peak_allocated_mb']])
BUDGET_TABLE = pd.DataFrame(BUDGET_ROWS)
BUDGET_TABLE.to_csv(SUITE_ROOT / 'reports/validation_budget.csv', index=False)
display(BUDGET_TABLE)
''')
md("""
## 16. Record a validation decision before testing
Select a **single coverage budget across all requested seeds**, not the best test
result or best seed. The descriptive screen requires paired validation confidence
bounds above -0.05 dB PSNR, -0.001 SSIM, and zero Edge-F1 improvement for every seed.
Among passing budgets choose the fastest mean validation latency. These thresholds
are declared here before testing; they do not by themselves establish publishability.
Also inspect the saved tile-cluster intervals, especially when nearby patches correlate.

If none passes, report the failed screen and evaluate the predeclared training budget
for diagnosis. Do not relabel failure as success. Test model copies are immutable;
rerunning later does not silently replace a previously tested checkpoint.
""")
code(r'''
LOCK_PATH = SUITE_ROOT / 'test_plan.json'
if LOCK_PATH.exists():
    TEST_PLAN = json.loads(LOCK_PATH.read_text())
    print('Reusing the existing test plan. New training is not substituted into a completed test study.')
else:
    if BUDGET_TABLE.empty:
        raise RuntimeError('Train and validate sparse_transformer before final testing.')
    budgets = BUDGET_TABLE.groupby('coverage').agg(passing=('feasible', 'all'), count=('seed', 'nunique'), ms=('mean_ms','mean'))
    passing = budgets[budgets.passing & budgets['count'].eq(len(SEEDS))]
    chosen = float(passing.ms.idxmin()) if not passing.empty else REGION_FRACTION
    TEST_PLAN = dict(dataset_id=AUDIT['dataset_id'], protocol=DATASET_PROTOCOL,
        validation_screen_passed=not passing.empty, coverage=chosen, top_k=TOP_K,
        seeds=SEEDS, decisions_use='validation only', checkpoints={})
    for seed in SEEDS:
        for profile in ['base'] + [p for p, enabled in RUN.items() if enabled]:
            source = best_path(seed, profile)
            identity = digest_file(source)
            relative = Path('test_models') / f'{seed}_{profile}_{identity[:12]}' / 'best.pt'
            target = SUITE_ROOT / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source, target)
            if digest_file(target) != identity:
                raise RuntimeError('Test snapshot content differs')
            TEST_PLAN['checkpoints'][f'{seed}/{profile}'] = dict(path=str(relative), sha256=identity,
                coverage=chosen if profile == 'sparse_transformer' else None,
                top_k=TOP_K if profile == 'sparse_transformer' else None)
    write_json(LOCK_PATH, TEST_PLAN)
print(json.dumps(TEST_PLAN, indent=2))
if TEST_PLAN['dataset_id'] != AUDIT['dataset_id']:
    raise ValueError('Test plan belongs to different prepared data. Use its original dataset or a new SUITE_ROOT.')
''')
md("## 17. Fixed test-set evaluation of every control and the selected sparse model")
code(r'''
TEST_EVALS = {}
if RUN_TEST_EVALUATION:
    for name, choice in TEST_PLAN['checkpoints'].items():
        checkpoint = SUITE_ROOT / choice['path']
        if digest_file(checkpoint) != choice['sha256']:
            raise RuntimeError('Immutable test checkpoint changed: ' + name)
        output = SUITE_ROOT / 'evaluation/test' / name
        evaluate(checkpoint, output, 'test', coverage=choice['coverage'], top_k=choice['top_k'], manifest=MANIFEST)
        TEST_EVALS[name] = output
    TEST_TABLES = summarize(TEST_EVALS, SUITE_ROOT / 'reports/test')
    display(TEST_TABLES['overall'][['experiment', 'count', *METRICS]])
    display(TEST_TABLES['per_class'])
    display(TEST_TABLES['paired_intervals'])
    display(DisplayImage(filename=str(SUITE_ROOT / 'reports/test/quality_comparison.png')))
    for seed in TEST_PLAN['seeds']:
        group = {name: path for name, path in TEST_EVALS.items() if name.startswith(f'{seed}/')}
        comparison = compare_controls(group, f'{seed}/sparse_transformer',
                                      SUITE_ROOT / f'reports/test/controls_seed_{seed}.csv')
        display(comparison[comparison.metric.isin(['psnr','ssim','edge_f1','scc'])])
    across_seeds = TEST_TABLES['overall'].copy()
    across_seeds['profile'] = across_seeds.experiment.str.split('/').str[-1]
    seed_summary = across_seeds.groupby('profile')[list(METRICS)].agg(['mean', 'std'])
    seed_summary.to_csv(SUITE_ROOT / 'reports/test/seed_summary.csv')
    display(seed_summary)
    print('Single-seed standard deviations are unavailable, not zero. Test results cannot be used to retune this study.')
else:
    print('Test evaluation skipped.')
''')
md("""
## 18. Indexed LR/HR/model visualization and routing maps
Default split is test, maximum three panels per row. Every image uses the same display
stretch; original floating-point predictions remain in each evaluation's `images/*.npz`.
The original LR has fewer native pixels, although plots show the same geographic extent.
""")
code(r'''
def show_result(index=0, split='test', show_base=True, profiles=None, seed=None):
    seed = TEST_PLAN['seeds'][0] if seed is None else seed
    profiles = profiles or ['single_expert', 'dense_transformer', 'sparse_transformer']
    choices = {name.split('/', 1)[1]: choice for name, choice in TEST_PLAN['checkpoints'].items()
               if name.startswith(f'{seed}/') and name.split('/', 1)[1] in profiles}
    if not choices:
        raise ValueError('No trained model matches the requested profiles/seed')
    # Each checkpoint has its own budget. Render separately when the selected
    # proposed budget differs from its training budget, then save a joint comparison below.
    from geodiff_gan.experiments.trust_report import visualize_saved
    if split == 'test':
        roots = {name: SUITE_ROOT / 'evaluation/test' / f'{seed}/{name}' for name in choices}
        return visualize_saved(roots, MANIFEST, index=index, split=split, show_base=show_base,
            output_path=SUITE_ROOT / f'figures/test_seed{seed}_{index:06d}.png', display_max=DISPLAY_MAX)
    checkpoints = {name: SUITE_ROOT / choice['path'] for name, choice in choices.items()}
    return visualize(checkpoints, MANIFEST, index=index, split=split, show_base=show_base,
        output_path=SUITE_ROOT / f'figures/{split}_seed{seed}_{index:06d}.png', display_max=DISPLAY_MAX)

if RUN_TEST_EVALUATION:
    show_result(0, show_base=True)
# Later: show_result(index=25, profiles=['sparse_conv','sparse_transformer','sparse_adversarial'])
''')
md("""
## 19. Download the evidence and resume bundle
Contains per-image/class metrics, paired/tile bootstrap intervals, training curves,
route utilization, timings, LR/HR manifest references, prediction NPZs, selected figures,
configs, best/last checkpoints, test-plan snapshots, source, and this clean notebook.
It excludes raw satellite archives and prepared NPZ inputs, which remain in your
attached dataset. Never upload credentials. A bundle does not make the result a paper:
we must inspect control comparisons, failed experiments, data independence and speed.
""")
code(r'''
evidence = {
    'name': 'GeoDiff-TrustMoE', 'diffusion': False, 'pixel_shuffle': False,
    'status': 'experimental; no novelty/performance guarantee',
    'test_plan': TEST_PLAN, 'dataset': AUDIT,
    'limitations': ['Within-tile tests do not establish unseen-region generalization.',
        'Difficulty is predicted base error, not calibrated probabilistic uncertainty.',
        'Sparse expert execution has dense encoder/router/trust overhead.',
        'Expert labels are not forced; examine collapse/utilization by scene class.',
        'Latency uses validation input and excludes I/O; compare identical hardware/precision.',
        'Old diffusion metrics are incomparable unless data/range/split and metric protocol match.'],
}
write_json(SUITE_ROOT / 'research_context.json', evidence)
if RUN_TEST_EVALUATION:
    # Include a small, predetermined class-balanced set of actual LR/HR inputs
    # so the downloaded evidence is interpretable without downloading the dataset.
    reference = next(iter(TEST_PLAN['checkpoints'].values()))
    config = torch.load(SUITE_ROOT / reference['path'], map_location='cpu', weights_only=False)['config']
    config['manifest'] = str(MANIFEST)
    data = dataset_for(config, 'test')
    selected_indices, seen_classes = [], set()
    for i, record in enumerate(data.records):
        if record.scene_class not in seen_classes:
            seen_classes.add(record.scene_class); selected_indices.append(i)
        if len(selected_indices) >= 12:
            break
    example_rows = []
    for index in selected_indices:
        sample = data[index]
        path = SUITE_ROOT / 'example_inputs' / f'{index:06d}.npz'
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, lr=sample['lr'].numpy(), hr=sample['hr'].numpy(),
            valid_mask_hr=sample['valid_mask'].numpy(), valid_mask_lr=sample['valid_mask_lr'].numpy())
        record = data.records[index]
        example_rows.append(dict(index=index, tile_id=record.tile_id, scene_class=record.scene_class, patch=record.patch))
        show_result(index, show_base=True)
    write_json(SUITE_ROOT / 'example_inputs/index.json', example_rows)
for name in ('README.md', 'learning/GeoDiff_TrustMoE_Research_Protocol.md'):
    source = REPOSITORY_DIR / name
    if source.exists():
        target = SUITE_ROOT / 'documentation' / Path(name).name
        target.parent.mkdir(exist_ok=True); shutil.copy2(source, target)
ARCHIVE = Path('/kaggle/working') / f'geodiff_trust_moe_results_{time.strftime("%Y%m%d_%H%M%S")}.zip'
bundle_results(SUITE_ROOT, ARCHIVE, source_root=REPOSITORY_DIR, include_resume=True)
print(f'Results + resume bundle: {ARCHIVE} ({ARCHIVE.stat().st_size / 2**20:.1f} MiB)')
os.chdir('/kaggle/working')
display(FileLink(ARCHIVE.name, result_html_prefix='Download results: '))
print('Download this ZIP and the executed notebook. Save a Kaggle version with outputs to preserve last.pt between sessions.')
''')


def write_notebook(path, contents, prefix):
    contents = copy.deepcopy(contents)
    for i, cell in enumerate(contents):
        cell['id'] = f'{prefix}-{i:03d}'
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
            assert not cell['outputs'] and cell['execution_count'] is None
    notebook = dict(cells=contents, metadata=dict(kernelspec=dict(display_name='Python 3',
        language='python', name='python3'), language_info=dict(name='python', version='3.11')),
        nbformat=4, nbformat_minor=5)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=True) + '\n', encoding='utf-8')


def build_preparation():
    from build_colab_diverse_landsat_sentinel_dataset_notebook import cells as original
    preparation = copy.deepcopy(original)
    replacements = {
        'from google.colab import drive\ndrive.mount("/content/drive")\n': '',
        '/content/drive/MyDrive/thesis/landsat_sentinel_diverse_raw': '/kaggle/input',
        '/content/drive/MyDrive/thesis/geodiff_diverse_3x_build': '/kaggle/working/geodiff_diverse_3x_build',
        '/content/geodiff-3x-continued': '/kaggle/working/geodiff-trust-moe-source',
        '/content/geodiff_diverse_discovery': '/kaggle/working/geodiff_diverse_discovery',
        'geodiff_diverse_moe_source.zip': 'geodiff_trust_moe_source.zip',
        'Mount Drive and set controls': 'Set Kaggle paths and dataset controls',
        'Optional Drive path': 'Optional Kaggle input path',
        'MyDrive/thesis/landsat_sentinel_diverse_raw/': 'kaggle/input/your-raw-tiles/',
        'download from Google Drive or upload this file directly to Kaggle': 'download from the Kaggle Output panel',
        'INCLUDE_MULTISPECTRAL = True': 'INCLUDE_MULTISPECTRAL = False',
        'record.split == "test"': 'record.split == "train"',
        'hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()': 'file_sha256(ARCHIVE)',
        'display(FileLink(str(ARCHIVE)))': 'os.chdir("/kaggle/working")\ndisplay(FileLink(str(ARCHIVE.relative_to(Path.cwd()))))',
    }
    for cell in preparation:
        text = ''.join(cell['source'])
        for old, new in replacements.items():
            text = text.replace(old, new)
        if 'SOURCE_ARCHIVE = None' in text:
            text += '''\nsource_candidates = list(Path('/kaggle/input').rglob('geodiff_trust_moe_source.zip'))
if SOURCE_ARCHIVE is None and len(source_candidates) == 1:
    SOURCE_ARCHIVE = source_candidates[0]

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
'''
        if 'archives = [path' in text:
            text = text.replace('if path.is_file() and path.name.casefold().endswith(archive_suffixes)',
                                'if path.is_file() and path.name.casefold().endswith(archive_suffixes) and path.name != "geodiff_trust_moe_source.zip"')
        cell['source'] = text.splitlines(True)
    preparation[0]['source'] += ['\nKaggle: attach raw sensor tiles and the supplied source ZIP. GPU is not required.\n',
        'This notebook uses the existing paired preparation code and does not alter old datasets.\n',
        'Save outputs between sessions. The exported NPZ files contain both LR and HR; separate PNG/TIFF copies are unnecessary.\n',
        'Directory categories are weak scene labels. Inspect mixed tiles and supply overrides.\n']
    write_notebook(ROOT / 'kaggle/Landsat_Sentinel_TrustMoE_Dataset_Preparation.ipynb', preparation, 'trust-prepare')


def build():
    write_notebook(ROOT / 'kaggle/GeoDiff_TrustMoE_Transformer_3x.ipynb', cells, 'trust-study')
    build_preparation()
    archive = ROOT / 'kaggle/geodiff_trust_moe_source.zip'
    candidates = [ROOT / 'pyproject.toml', ROOT / 'README.md']
    for folder in ('src', 'scripts', 'configs'):
        candidates.extend(p for p in (ROOT / folder).rglob('*') if p.suffix in ('.py', '.yaml') and '__pycache__' not in p.parts)
    candidates.extend([ROOT / 'kaggle/GeoDiff_TrustMoE_Transformer_3x.ipynb',
                       ROOT / 'kaggle/Landsat_Sentinel_TrustMoE_Dataset_Preparation.ipynb',
                       ROOT / 'learning/GeoDiff_TrustMoE_Research_Protocol.md', ROOT / 'tests/test_trust_moe.py'])
    inventory = []
    candidates.extend((ROOT / 'learning/trust_moe').glob('*.md'))
    candidates.extend([ROOT / 'learning/GEE_India_Fixed_Pair_Datasets.md',
                       ROOT / 'colab/GEE_India_Tile_First_TIFF_Datasets.ipynb',
                       ROOT / 'tests/test_gee_tiles.py',
                       ROOT / 'colab/GEE_India_Landsat_Sentinel_Fixed_Datasets.ipynb',
                       ROOT / 'tests/test_gee_pairs.py'])
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(set(candidates)):
            if not path.is_file():
                continue
            data = path.read_bytes()
            name = path.relative_to(ROOT).as_posix()
            handle.writestr(name, data)
            inventory.append(dict(file=name, sha256=hashlib.sha256(data).hexdigest()))
        handle.writestr('source_inventory.json', json.dumps(inventory, indent=2))
    print(f'Built two notebooks and {archive} ({archive.stat().st_size / 2**20:.2f} MiB)')


if __name__ == '__main__':
    build()
