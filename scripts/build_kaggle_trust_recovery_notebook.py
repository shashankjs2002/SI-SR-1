"""Build the staged 128->384 OLI2MSI residual-recovery experiment."""
import ast
import copy
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "kaggle/GeoDiff_TrustMoE_OLI2MSI_Residual_Recovery_3x.ipynb"


def build():
    original = json.loads((ROOT / "kaggle/GeoDiff_TrustMoE_OLI2MSI_128x128_3x.ipynb").read_text())
    cells = copy.deepcopy(original["cells"][:9])

    def set_source(index, text):
        cells[index]["source"] = (dedent(text).strip() + "\n").splitlines(True)

    def add(text, code=False):
        cell = {"cell_type": "code" if code else "markdown", "metadata": {},
                "source": (dedent(text).strip() + "\n").splitlines(True)}
        if code:
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    set_source(0, """
    # GeoDiff-TrustMoE: residual recovery on OLI2MSI

    Your prior test table measured base PSNR 34.159966 dB and residual gains below
    0.00071 dB. This notebook checks the trained weights, tests residual learning,
    then compares HR-conditioned experts and routing that predicts correction benefit.

    **Protocol stays 128 x 128 actual Landsat LR -> 384 x 384 Sentinel HR.**
    Training draws aligned random crops from the 160->480 source frames. Validation
    uses 255 hash-selected training sources; test uses all 100 official test centers
    when validation_percent=5 and seed=42. The fixed clip(.0,.3)/.3 radiometry and
    optional uint8 grid match the previous notebook. These are center-crop results,
    not the paper's 32->96 results or full-frame 160->480 results.

    Train a useful single expert before comparing MoE. Experts see LR features plus
    signed HR base detail, with actual sparse tile dispatch. The gain router learns
    the observed fractional MSE reduction of a proposal. HR provides training labels
    only. There is still one frozen base pass, no diffusion and no PixelShuffle.

    Run top to bottom. Controls are epochs and batch size. Training resumes at the
    last completed epoch; completed evaluations are reused. Use a new suite for this
    changed objective. Final cells save exact outputs, comparisons and a download ZIP.
    Improvement and research novelty must be established by the resulting experiments.
    """)
    controls = "".join(cells[2]["source"])
    controls = controls.replace("geodiff-trust-moe-oli2msi-source", "geodiff-trust-recovery-source")
    controls = controls.replace("geodiff-trust-moe-oli2msi-128-v1", "geodiff-trust-recovery-oli128-v2")
    controls = controls.replace("IMPORT_BASE_CHECKPOINT = None", """OLD_SUITE_ROOT = Path('/kaggle/working/geodiff-trust-moe-oli2msi-128-v1')
# Set OLD_SUITE_ROOT to your attached previous outputs if this is a fresh session.
BASE_CHECKPOINTS = {42: None}  # Optional exact best.pt paths, one per requested seed.
IMPORT_BASE_CHECKPOINT = None""")
    controls = controls.replace("EPOCHS = {'base': 30, 'residual': 15}",
                                "EPOCHS = {'base': 20, 'diagnostic': 60, 'single': 15, 'residual': 10}")
    a, b = controls.index("RUN = dict("), controls.index("BASE_MODEL =")
    controls = controls[:a] + """RUN = dict(dense_gain=True, sparse_error=True, sparse_gain=True,
           sparse_no_trust=True, sparse_uniform=True, sparse_conv=False,
           sparse_adversarial=False, adaptive_k=False, legacy_expert_control=False)
RUN_TEST_EVALUATION = False  # Set True only after reviewing the validation cells.
DIAGNOSTIC_PAIRS = 8
REQUIRE_RECOVERY_SCREEN = True
""" + controls[b:]
    set_source(2, controls)
    install = "".join(cells[4]["source"])
    install = install.replace("src/geodiff_gan/data/oli2msi_128.py", "src/geodiff_gan/experiments/trust_recovery.py")
    install += """
from geodiff_gan.experiments.trust_moe import load_model
from geodiff_gan.experiments.trust_recovery import (
    RECOVERY_VERSION, recovery_config, audit_checkpoint, memorization_report, validation_screen,
)
assert RECOVERY_VERSION == 'trust-recovery-v2', 'Clone the updated 3x-continued branch.'
for name in ('figures', 'reports', 'diagnostics', 'source_receipts'):
    (SUITE_ROOT / name).mkdir(exist_ok=True)
write_json(SUITE_ROOT / 'source_receipts/revision.json', {
    'commit': subprocess.check_output(['git', '-C', str(REPOSITORY_DIR), 'rev-parse', 'HEAD'], text=True).strip(),
    'recovery_version': RECOVERY_VERSION,
})
if FAST_DEV_RUN:
    EPOCHS = {'base': 1, 'diagnostic': 2, 'single': 1, 'residual': 3}
    REQUIRE_RECOVERY_SCREEN = False
    RUN_TEST_EVALUATION = False
    print('SMOKE RUN: exercises plumbing; not a performance experiment.')
"""
    set_source(4, install)
    preview = "".join(cells[8]["source"]).replace("split='test'", "split='val'").replace("preview_pair(0, 'test')", "preview_pair(0, 'val')")
    set_source(8, preview)

    add("""
    ## 5. Reuse your trained base and audit the old residual

    The imported base is copied into this suite. All comparisons use exactly those
    frozen weights. If no base can be found, this cell trains one. Attach your prior
    Kaggle outputs and set OLD_SUITE_ROOT to avoid repeating base training.
    Epoch increases resume; architecture/loss changes need a new experiment directory.
    """)
    add(r'''
BASES, SINGLES, MODELS, CONFIGS = {}, {}, {}, {}

def launch(config):
    root = Path(config['root']); root.mkdir(parents=True, exist_ok=True)
    request = root / 'request_config.json'
    write_json(request, config)
    run([sys.executable, '-m', 'geodiff_gan.cli.trust_moe', 'train', '--config', request], cwd=REPOSITORY_DIR)
    checkpoint = root / 'best.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint

def read_checkpoint_config(checkpoint):
    return torch.load(checkpoint, map_location='cpu', weights_only=False)['config']

for seed in SEEDS:
    config = make_config(MANIFEST, SUITE_ROOT / f'seed_{seed}/base', profile='base',
        seed=seed, epochs=EPOCHS['base'], batch_size=BATCH_SIZE, crop_size=TRAIN_LR_CROP,
        experts=NUM_EXPERTS, top_k=TOP_K, dataset_id=AUDIT['dataset_id'], base_model=BASE_MODEL)
    config['training']['progress'] = 'compact'
    requested = BASE_CHECKPOINTS.get(seed)
    old = Path(requested) if requested else OLD_SUITE_ROOT / f'seed_{seed}/base/best.pt'
    if old.is_file():
        prior = read_checkpoint_config(old)
        if prior['dataset_id'] != AUDIT['dataset_id']:
            raise ValueError('Old base dataset differs. Check data protocol and manifest before importing.')
        # Honor the real base dimensions recorded by the trained checkpoint.
        for key in ('base_embed_dim', 'base_depth', 'base_groups', 'base_heads', 'window_size'):
            if key in prior['model']:
                config['model'][key] = prior['model'][key]
        BASES[seed] = adopt_base(config, old)
        print('Imported trained base:', old)
    else:
        print('No imported base found; training the base for this seed.')
        BASES[seed] = launch(config)
    MODELS[f'{seed}/base'] = BASES[seed]
    CONFIGS[f'{seed}/base'] = config
    write_json(SUITE_ROOT / f'seed_{seed}/base/lineage.json', {
        'source': str(old) if old.is_file() else 'trained in this suite',
        'source_sha256': digest_file(old) if old.is_file() else None,
        'checkpoint_sha256': digest_file(BASES[seed]),
    })
    old_expert = OLD_SUITE_ROOT / f'seed_{seed}/single_expert/best.pt'
    if old_expert.exists():
        previous = audit_checkpoint(old_expert, MANIFEST, SUITE_ROOT / f'diagnostics/old_single_{seed}.json')
        display(pd.DataFrame(previous['rows']))
        print('Reconstruction gradients:', previous.get('reconstruction_gradients'))
        print('Full objective gradients:', previous.get('full_loss_gradients'))
write_json(SUITE_ROOT / 'base_registry.json', {str(k): str(v.relative_to(SUITE_ROOT)) for k, v in BASES.items()})
''', True)
    add("""
    ## 6. Eight-image training diagnostic: can an expert learn a correction?

    This uses fixed TRAIN crops, full coverage, one expert, trust off and guard off.
    Its PSNR is memorization performance, never a validation/test score. It is not
    used to initialize the scientific runs. A failed screen stops the full study so
    you can share the diagnostic ZIP. No target is passed into the inference model.
    """)
    add(r'''
def new_config(seed, label, profile='single_expert', initializer=None, risk='gain', hr=True):
    dimensions = {k: v for k, v in CONFIGS[f'{seed}/base']['model'].items()
                  if k.startswith('base_') or k == 'window_size'}
    dimensions.update(width=BASE_MODEL['width'], tile_size=BASE_MODEL['tile_size'])
    return recovery_config(MANIFEST, SUITE_ROOT / f'seed_{seed}/{label}', parent=BASES[seed],
        initializer=initializer, profile=profile, seed=seed,
        epochs=EPOCHS['single'] if profile == 'single_expert' else EPOCHS['residual'],
        batch_size=BATCH_SIZE, crop_size=TRAIN_LR_CROP, experts=NUM_EXPERTS,
        top_k=TOP_K, dataset_id=AUDIT['dataset_id'], base_model=dimensions,
        risk_target=risk, coverage=REGION_FRACTION, hr_conditioning=hr)

DIAGNOSTICS = []
for seed in SEEDS:
    diagnostic = new_config(seed, 'diagnostic_overfit')
    diagnostic['training'].update(diagnostic_overfit_pairs=DIAGNOSTIC_PAIRS,
        epochs=EPOCHS['diagnostic'], batch_size=1, num_workers=0, learning_rate=1e-3,
        lr_epoch_decay=1.0)
    path = launch(diagnostic)
    report = memorization_report(path, SUITE_ROOT / f'diagnostics/memorization_{seed}.json')
    DIAGNOSTICS.append(report)
    display(pd.DataFrame(report['rows']))
    print('TRAIN-CROP diagnostic:', report['mean_psnr_gain'], 'dB; passed:', report['passed'])
if REQUIRE_RECOVERY_SCREEN and not all(r['passed'] for r in DIAGNOSTICS):
    archive = SUITE_ROOT.parent / (SUITE_ROOT.name + '_diagnostic.zip')
    bundle_results(SUITE_ROOT, archive, source_root=REPOSITORY_DIR)
    display(FileLink(str(archive.relative_to(Path('/kaggle/working')))))
    raise RuntimeError('Expert did not pass the train-crop learning check. Share diagnostics before running the full suite.')
''', True)
    add("""
    ## 7. Full training: one HR-conditioned residual expert

    This trains on all 4,970 designated training sources, using independent fresh
    residual initialization and the common frozen base. Its checkpoint is selected
    on the 255 validation sources. This is the initializer for every subsequent MoE.
    """)
    add(r'''
for seed in SEEDS:
    config = new_config(seed, 'single_recovery')
    CONFIGS[f'{seed}/single_recovery'] = config
    SINGLES[seed] = launch(config)
    MODELS[f'{seed}/single_recovery'] = SINGLES[seed]
''', True)
    add("""
    ## 8. Validate the single expert before training all routers

    Development screen: mean validation gain at least +0.01 dB, and over half the
    images improve. This detects useful learning; it is not a publication criterion.
    The report compares raw/EMA behavior and logs individual-module gradient norms.
    """)
    add(r'''
VAL_EVALS, SINGLE_SCREENS = {}, []

def eval_path(name, split, checkpoint, suffix='trained'):
    return SUITE_ROOT / 'evaluation' / split / name / (digest_file(checkpoint)[:12] + '_' + suffix)

for seed in SEEDS:
    for label, checkpoint in (('base', BASES[seed]), ('single_recovery', SINGLES[seed])):
        name = f'{seed}/{label}'
        output = eval_path(name, 'val', checkpoint)
        evaluate(checkpoint, output, 'val', manifest=MANIFEST)
        VAL_EVALS[name] = output
    rows = json.loads((VAL_EVALS[f'{seed}/single_recovery'] / 'per_image.json').read_text())
    screen = validation_screen(rows)
    SINGLE_SCREENS.append(screen)
    audit = audit_checkpoint(SINGLES[seed], MANIFEST, SUITE_ROOT / f'diagnostics/single_recovery_{seed}.json')
    display(pd.DataFrame(audit['rows']))
    display(pd.DataFrame(paired_intervals(rows)))
    print('Validation development screen:', screen)
write_json(SUITE_ROOT / 'reports/single_screen.json', SINGLE_SCREENS)
if REQUIRE_RECOVERY_SCREEN and not all(s['passed'] for s in SINGLE_SCREENS):
    raise RuntimeError('Single expert did not improve validation enough. Run the final bundle cell and share the evidence.')
''', True)
    add("""
    ## 9. Shared initializer and matched routing experiments

    Each MoE starts from the same learned encoder/single expert. Tiny independent RGB
    projection perturbations break symmetry. Two full-coverage warm-up epochs observe
    proposal benefit throughout the image; later epochs use the declared sparse budget.
    The router's auxiliary loss is prevented from updating the reconstruction encoder.
    Warm-up checkpoints cannot become the selected deployed checkpoint.

    `sparse_error` predicts base error; `sparse_gain` predicts realized proposal benefit.
    Both use identical architectures, data, losses, training duration, and inference
    budget except the risk label. Unexecuted tiles are excluded from gain supervision.
    This label remains an imperfect, changing estimate of recoverability, not an oracle.
    """)
    add(r'''
EXPERIMENTS = {
    'dense_gain': ('dense_transformer', 'gain'),
    'sparse_error': ('sparse_transformer', 'error'),
    'sparse_gain': ('sparse_transformer', 'gain'),
    'sparse_no_trust': ('sparse_no_trust', 'gain'),
    'sparse_uniform': ('sparse_uniform', 'gain'),
    'sparse_conv': ('sparse_conv', 'gain'),
    'sparse_adversarial': ('sparse_adversarial', 'gain'),
    'adaptive_k': ('adaptive_k', 'gain'),
}

def train_experiment(label):
    if not RUN.get(label, False):
        print('Skipped:', label)
        return
    profile, risk = EXPERIMENTS[label]
    for seed in SEEDS:
        config = new_config(seed, label, profile, initializer=SINGLES[seed], risk=risk)
        CONFIGS[f'{seed}/{label}'] = config
        MODELS[f'{seed}/{label}'] = launch(config)
    write_json(SUITE_ROOT / 'model_registry.json', {k: str(v.relative_to(SUITE_ROOT)) for k, v in MODELS.items()})
''', True)
    for label, title in (
        ("dense_gain", "10. Dense MoE control"),
        ("sparse_error", "11. Sparse base-error router control"),
        ("sparse_gain", "12. Sparse proposal-benefit router"),
        ("sparse_no_trust", "13. Remove trust"),
        ("sparse_uniform", "14. Uniform regional coverage control"),
        ("sparse_conv", "15. CNN router control (optional)"),
        ("sparse_adversarial", "16. Adversarial contribution (optional)"),
        ("adaptive_k", "17. Adaptive expert slots (optional)"),
    ):
        add("## " + title)
        add(f"train_experiment('{label}')", True)
    add("""
    ## 18. Legacy expert with the new curriculum (optional)

    This control isolates the HR-conditioned expert architecture. It receives the
    same single-expert loss and duration but has the old expert capacity and no HR
    cues. It is trained from scratch using the shared base, so it does not inherit
    an incompatible HR expert checkpoint. Report parameter differences explicitly.
    """)
    add(r'''
if RUN['legacy_expert_control']:
    for seed in SEEDS:
        config = new_config(seed, 'legacy_expert_control', hr=False)
        CONFIGS[f'{seed}/legacy_expert_control'] = config
        MODELS[f'{seed}/legacy_expert_control'] = launch(config)
''', True)
    add("## 19. Validation comparisons, routing, learning curves and measured runtime")
    add(r'''
TIMINGS, HISTORIES = [], []
for name, checkpoint in MODELS.items():
    output = eval_path(name, 'val', checkpoint)
    evaluate(checkpoint, output, 'val', manifest=MANIFEST)
    VAL_EVALS[name] = output
    for row in benchmark(checkpoint, output / 'timing.json', manifest=MANIFEST):
        TIMINGS.append(dict(experiment=name, **row))
    for history in sorted(checkpoint.parent.glob('history/epoch_*.json')):
        row = json.loads(history.read_text())
        diagnostics = row.pop('first_batch_diagnostics', {})
        row.pop('weighted_objective', None)
        row.pop('first_batch_gradient_norms', None)
        HISTORIES.append(dict(experiment=name, **row, **diagnostics))
VAL_TABLES = summarize(VAL_EVALS, SUITE_ROOT / 'reports/validation')
display(VAL_TABLES['overall'][['experiment', 'count', *METRICS, 'psnr_delta_vs_base', 'correction_abs_mean']])
display(VAL_TABLES['expert_usage'])
TIMING_TABLE = pd.DataFrame(TIMINGS)
TIMING_TABLE.to_csv(SUITE_ROOT / 'reports/timing.csv', index=False)
display(TIMING_TABLE[['experiment', 'operation', 'parameters_total', 'parameters_used_on_this_frame',
                     'mean_ms', 'p95_ms', 'images_per_second', 'peak_allocated_mb']])
history = pd.DataFrame(HISTORIES)
history.to_csv(SUITE_ROOT / 'reports/training_history.csv', index=False)
if not history.empty:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for name, group in history.groupby('experiment'):
        axes[0].plot(group.epoch, group.val_psnr, label=name)
        if 'correction_abs_mean' in group:
            axes[1].plot(group.epoch, group.correction_abs_mean, label=name)
    axes[0].set_title('Validation PSNR'); axes[1].set_title('First training batch: effective correction')
    axes[0].legend(fontsize=6); fig.tight_layout()
    fig.savefig(SUITE_ROOT / 'figures/learning.png', dpi=170); plt.show()
for seed in SEEDS:
    group = {k: v for k, v in VAL_EVALS.items() if k.startswith(f'{seed}/')}
    if f'{seed}/sparse_gain' in group:
        display(compare_controls(group, f'{seed}/sparse_gain', SUITE_ROOT / f'reports/validation/controls_{seed}.csv'))
write_json(SUITE_ROOT / 'model_registry.json', {k: str(v.relative_to(SUITE_ROOT)) for k, v in MODELS.items()})
''', True)
    add("""
    ## 20. Validation-only coverage ablation

    Trained top-k stays fixed. Evaluate 25/50/75/100% coverage to measure a quality vs
    computation curve. Final primary comparisons retain the trained coverage; these
    curves are diagnostic ablations. A reduced fraction is not proof of lower latency.
    """)
    add(r'''
BUDGET_ROWS = []
for seed in SEEDS:
    name = f'{seed}/sparse_gain'
    if name not in MODELS:
        continue
    for fraction in (0.25, 0.5, 0.75, 1.0):
        output = eval_path(name, 'val', MODELS[name], f'coverage_{fraction}')
        scores = evaluate(MODELS[name], output, 'val', coverage=fraction,
                          manifest=MANIFEST, save_images=False)
        timing = benchmark(MODELS[name], output / 'timing.json', manifest=MANIFEST, coverage=fraction)[1]
        BUDGET_ROWS.append(dict(seed=seed, coverage=fraction, **scores,
                               mean_ms=timing['mean_ms'], images_per_second=timing['images_per_second']))
budget = pd.DataFrame(BUDGET_ROWS)
budget.to_csv(SUITE_ROOT / 'reports/validation_coverage.csv', index=False)
display(budget)
''', True)
    add("""
    ## 21. Freeze the test plan after reviewing validation

    The proposed primary model is sparse_gain at the trained coverage and top-k.
    Record whether it improves over both the base and the strong single expert;
    failure is retained in the report. All enabled controls are tested. No checkpoint
    or budget is chosen by looking at test scores. Prior repeated OLI2MSI testing means
    a new geographically independent dataset is still needed for a strong final claim.
    """)
    add(r'''
plan_path = SUITE_ROOT / 'test_plan.json'
if plan_path.exists():
    TEST_PLAN = json.loads(plan_path.read_text())
    if TEST_PLAN['dataset_id'] != AUDIT['dataset_id']:
        raise ValueError('Test plan belongs to different data')
else:
    TEST_PLAN = dict(dataset_id=AUDIT['dataset_id'], protocol=DATA_CARD['protocol'],
                     primary='sparse_gain', coverage=REGION_FRACTION, top_k=TOP_K,
                     development_screens=SINGLE_SCREENS, checkpoints={})
    for name, source in MODELS.items():
        identity = digest_file(source)
        destination = SUITE_ROOT / 'test_models' / name / identity[:12] / 'best.pt'
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        if digest_file(destination) != identity:
            raise RuntimeError('Test checkpoint copy differs')
        TEST_PLAN['checkpoints'][name] = dict(path=str(destination.relative_to(SUITE_ROOT)), sha256=identity)
    write_json(plan_path, TEST_PLAN)
print(json.dumps(TEST_PLAN, indent=2))
''', True)
    add("## 22. Test all 100 official pairs using the frozen plan")
    add(r'''
TEST_EVALS = {}
if RUN_TEST_EVALUATION:
    if FAST_DEV_RUN:
        raise RuntimeError('Smoke results must not be reported as the official test benchmark')
    for name, item in TEST_PLAN['checkpoints'].items():
        checkpoint = SUITE_ROOT / item['path']
        if digest_file(checkpoint) != item['sha256']:
            raise RuntimeError('Test snapshot changed: ' + name)
        output = SUITE_ROOT / 'evaluation/test' / name
        scores = evaluate(checkpoint, output, 'test', manifest=MANIFEST)
        if scores['count'] != 100:
            raise RuntimeError('Expected all 100 official test pairs')
        TEST_EVALS[name] = output
    TEST_TABLES = summarize(TEST_EVALS, SUITE_ROOT / 'reports/test')
    display(TEST_TABLES['overall'][['experiment', 'count', *METRICS, 'psnr_delta_vs_base', 'correction_abs_mean']])
    display(TEST_TABLES['paired_intervals'])
    for seed in SEEDS:
        group = {k: v for k, v in TEST_EVALS.items() if k.startswith(f'{seed}/')}
        if f'{seed}/sparse_gain' in group:
            display(compare_controls(group, f'{seed}/sparse_gain', SUITE_ROOT / f'reports/test/controls_{seed}.csv'))
else:
    print('Test is disabled. After validation, set RUN_TEST_EVALUATION=True and rerun this cell.')
''', True)
    add("""
    ## 23. Indexed outputs, signed corrections and routing maps

    `show_result(index=25)` defaults to test. Use split='val' before opening test.
    Images use the protocol range directly (gamma=1), with one shared scale and a
    maximum of three columns. Correction plots use a shared symmetric color range;
    they are signed correction amplitudes, not an artificially enhanced prediction.
    """)
    add(r'''
from geodiff_gan.experiments.trust_report import visualize_saved

def show_result(index=0, split='test', show_base=True, profiles=None, seed=42):
    profiles = profiles or ['single_recovery', 'sparse_error', 'sparse_gain']
    registry = TEST_EVALS if split == 'test' else VAL_EVALS
    choices = {name.split('/', 1)[1]: path for name, path in registry.items()
               if name.startswith(f'{seed}/') and name.split('/', 1)[1] in profiles}
    if not choices:
        raise ValueError('Run the corresponding evaluation cell first.')
    destination = SUITE_ROOT / f'figures/{split}_{seed}_{index:06d}.png'
    figures = visualize_saved(choices, MANIFEST, index=index, split=split, show_base=show_base,
                               output_path=destination, display_max=1.0, gamma=1.0)
    corrections = []
    for name, path in choices.items():
        with np.load(path / 'images' / f'{index:06d}.npz', allow_pickle=False) as values:
            corrections.append((name, (values['prediction'] - values['base']).mean(0)))
    limit = max(1e-5, max(float(np.quantile(np.abs(a), .99)) for _, a in corrections))
    columns = min(3, len(corrections))
    fig, axes = plt.subplots(int(np.ceil(len(corrections)/columns)), columns,
                             figsize=(5*columns, 4*int(np.ceil(len(corrections)/columns))), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, (name, array) in zip(axes.flat, corrections):
        picture = ax.imshow(array, cmap='RdBu_r', vmin=-limit, vmax=limit)
        ax.set_title(name + ': mean signed RGB correction')
    fig.colorbar(picture, ax=list(axes.flat), shrink=.7, label='Protocol intensity units')
    fig.savefig(destination.with_name(destination.stem + '_corrections.png'), dpi=170)
    plt.show()
    return figures

show_result(0, split='val')
''', True)
    add("""
    ## 24. Download the evidence and restart bundle

    This cell can run even after a failed development screen. Download the ZIP and
    save the executed notebook. Reports distinguish training diagnostics, validation,
    test, failure, and measured runtime. Best/last checkpoints and code are included.
    Full source tiles and prepared input caches are excluded. Six paired TIFF previews
    are included for interpreting results without transferring the full dataset.
    """)
    add(r'''
import rasterio
from rasterio.transform import Affine

if 'BASES' in globals() and BASES:
    cfg = read_checkpoint_config(next(iter(BASES.values())))
    cfg['manifest'] = str(MANIFEST)
    split = 'test' if globals().get('TEST_EVALS') else 'val'
    data = dataset_for(cfg, split)
    folder = SUITE_ROOT / 'example_inputs'; folder.mkdir(exist_ok=True)
    for index in range(min(6, len(data))):
        sample = data[index]
        for key in ('lr', 'hr'):
            array = sample[key].numpy().astype('float32')
            # Benchmark crops have no verified georeferencing; do not invent a CRS.
            path = folder / f'{split}_{index:06d}_{key}.tif'
            with rasterio.open(path, 'w', driver='GTiff', width=array.shape[2], height=array.shape[1],
                               count=3, dtype='float32', compress='deflate', transform=Affine.identity()) as tif:
                tif.write(array)
                tif.update_tags(description='Protocol-normalized RGB example; no geographic CRS')
documentation = REPOSITORY_DIR / 'learning/trust_moe/07_residual_recovery_v2.md'
if documentation.exists():
    shutil.copy2(documentation, SUITE_ROOT / 'research_protocol.md')
notebook = REPOSITORY_DIR / 'kaggle/GeoDiff_TrustMoE_OLI2MSI_Residual_Recovery_3x.ipynb'
if notebook.exists():
    shutil.copy2(notebook, SUITE_ROOT / notebook.name)
archive = Path('/kaggle/working') / f'trust_recovery_results_{time.strftime("%Y%m%d_%H%M%S")}.zip'
bundle_results(SUITE_ROOT, archive, source_root=REPOSITORY_DIR, include_resume=True)
os.chdir('/kaggle/working')
print(f'Results and resume bundle: {archive} ({archive.stat().st_size / 2**20:.1f} MiB)')
display(FileLink(archive.name, result_html_prefix='Download results: '))
''', True)
    for i, cell in enumerate(cells):
        cell["id"] = f"trust-recovery-{i:03d}"
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            cell.update(execution_count=None, outputs=[])
    original["cells"] = cells
    OUTPUT.write_text(json.dumps(original, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    build()
