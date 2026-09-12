"""Build the 128->384 RGB/multispectral TrustMoE tile study notebook."""
import ast
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "kaggle/GeoDiff_TrustMoE_Landsat_Sentinel_RGB_Multispectral_128_3x.ipynb"


def cell(text, code=False):
    value = {
        "cell_type": "code" if code else "markdown",
        "metadata": {},
        "source": (dedent(text).strip() + "\n").splitlines(True),
    }
    if code:
        ast.parse("".join(value["source"]))
        value.update(execution_count=None, outputs=[])
    return value


def build():
    cells = [
        cell("""
        # TrustMoE study on real Landsat/Sentinel tiles: RGB and multispectral

        This notebook prepares aligned, real cross-sensor pairs from your raw
        Landsat 8/9 Collection-2 Level-2 and Sentinel-2 L2A products. Every model
        receives a `128 x 128` Landsat patch at 30 m and predicts a `384 x 384`
        Sentinel patch at 10 m.

        Two matched studies use the same source pairs and fixed spatial splits:

        - **RGB:** Red, Green and Blue input and output.
        - **Multispectral:** Red, Green, Blue, NIR, SWIR1 and SWIR2 input and output.

        Sentinel B11 and B12 are natively 20 m and are resampled onto the 10 m
        target grid. This enables band-aligned supervision but does not create
        genuine 10 m SWIR observations. The report therefore separates RGB
        spatial fidelity from six-band fidelity and spectral-index preservation.

        The ablations are dense Transformer routing, sparse Transformer routing,
        and sparse routing without the trust gate. Results include PSNR, SSIM,
        L1, Edge F1, ERGAS, SAM, UIQI, sCC, per-band PSNR, NDVI, NDWI and NDBI,
        paired RGB-vs-multispectral comparisons, routing/runtime evidence, indexed
        test visualization, and a downloadable restart/results archive.
        """),
        cell("## 1. Controls"),
        cell(r'''
        from pathlib import Path
        import hashlib, json, os, subprocess, sys, time, zipfile

        FAST_DEV_RUN = False
        REPOSITORY_URL = 'https://github.com/shashankjs2002/SI-SR-1.git'
        REPOSITORY_BRANCH = '3x-continued'
        REPOSITORY_DIR = Path('/kaggle/working/geodiff-trustmoe-ms-source')
        SUITE_ROOT = Path('/kaggle/working/geodiff-trustmoe-tiles-rgb-ms-128-v1')
        RAW_TILE_ROOT = Path('/kaggle/input')
        PREPARED_MANIFEST = None  # Optional six-band prepared manifest.
        RESTORE_SUITE_FROM = None  # Optional prior output directory or ZIP.

        EPOCHS = {'base': 30, 'residual': 12}
        BATCH_SIZE = 1
        TRAIN_LR_CROP = 128
        NUM_EXPERTS, TOP_K = 5, 2
        REGION_FRACTION = 0.5
        SEED = 42
        RUN = {'dense_transformer': True, 'sparse_transformer': True,
               'sparse_no_trust': True}
        RUN_TEST_EVALUATION = True

        PATCH_SIZE, PATCH_STRIDE = 384, 288
        MAX_DAY_GAP = 15
        MINIMUM_OVERLAP_FRACTION = 0.10
        MINIMUM_VALID_FRACTION = 0.95
        TRAIN_FRACTION, VALIDATION_FRACTION = 0.78, 0.10
        MINIMUM_VALIDATION_FRACTION, MINIMUM_TEST_FRACTION = 0.08, 0.10
        REBUILD_PREPARED_DATA = False
        DISPLAY_MAX = 0.3
        BASE_MODEL = dict(base_embed_dim=32, base_depth=2, base_groups=2,
                          base_heads=4, window_size=8, width=32, tile_size=8,
                          halo=6, expert_kind='hr_residual', expert_blocks=2)
        BAND_NAMES = ['red', 'green', 'blue', 'nir', 'swir1', 'swir2']
        MODALITIES = {
            'rgb': dict(condition_key='lr', target_key='hr', input_channels=3,
                        output_channels=3, band_names=BAND_NAMES[:3]),
            'multispectral': dict(condition_key='lr_ms', target_key='hr_ms',
                                  input_channels=6, output_channels=6,
                                  band_names=BAND_NAMES),
        }

        if FAST_DEV_RUN:
            EPOCHS = {'base': 1, 'residual': 4}
            RUN_TEST_EVALUATION = False
        if TRAIN_LR_CROP != 128 or PATCH_SIZE != 384:
            raise ValueError('This study is locked to 128 x 128 -> 384 x 384.')
        if not 1 <= TOP_K <= NUM_EXPERTS:
            raise ValueError('Require 1 <= TOP_K <= NUM_EXPERTS')
        SUITE_ROOT.mkdir(parents=True, exist_ok=True)

        def run(command, cwd=None):
            environment = os.environ.copy()
            environment['PYTHONPATH'] = str(REPOSITORY_DIR / 'src') + os.pathsep + environment.get('PYTHONPATH', '')
            environment['PYTHONUNBUFFERED'] = '1'
            print('+', ' '.join(map(str, command)), flush=True)
            return subprocess.run(list(map(str, command)), cwd=cwd, env=environment, check=True)

        print('Suite:', SUITE_ROOT)
        print('Geometry: 128 x 128 -> 384 x 384')
        print('Epochs:', EPOCHS, 'batch:', BATCH_SIZE, 'experts:', NUM_EXPERTS, 'top-k:', TOP_K)
        ''' , True),
        cell("## 2. Clone, install, optionally restore, and verify GPU"),
        cell(r'''
        run([sys.executable, '-m', 'pip', 'install', '-q', 'rasterio>=1.3',
             'numpy', 'Pillow', 'PyYAML', 'tqdm', 'pandas', 'matplotlib'])
        if not REPOSITORY_DIR.exists():
            run(['git', 'clone', '--depth', '1', '--single-branch', '--branch',
                 REPOSITORY_BRANCH, REPOSITORY_URL, REPOSITORY_DIR])
        if not (REPOSITORY_DIR / '.git').is_dir():
            raise RuntimeError('REPOSITORY_DIR exists but is not a Git checkout')
        branch = subprocess.check_output(
            ['git', '-C', str(REPOSITORY_DIR), 'branch', '--show-current'], text=True
        ).strip()
        if branch != REPOSITORY_BRANCH:
            raise RuntimeError(f'Expected {REPOSITORY_BRANCH}, found {branch}')
        required = REPOSITORY_DIR / 'src/geodiff_gan/spectral_indices.py'
        if not required.is_file():
            raise RuntimeError('Push the multispectral TrustMoE revision before running this notebook.')
        run([sys.executable, '-m', 'pip', 'install', '-q', '-e', REPOSITORY_DIR, '--no-deps'])
        sys.path.insert(0, str(REPOSITORY_DIR / 'src'))

        if RESTORE_SUITE_FROM:
            import shutil
            source = Path(RESTORE_SUITE_FROM)
            if source.is_file():
                restored = SUITE_ROOT.parent / 'restored_trustmoe_ms'
                restored.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(source) as archive:
                    for member in archive.infolist():
                        destination = (restored / member.filename).resolve()
                        if not destination.is_relative_to(restored.resolve()):
                            raise ValueError('Unsafe ZIP path: ' + member.filename)
                    archive.extractall(restored)
                source = restored
            for path in source.rglob('*'):
                if path.is_file():
                    destination = SUITE_ROOT / path.relative_to(source)
                    if not destination.exists():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, destination)

        import numpy as np
        import pandas as pd
        import torch
        import matplotlib.pyplot as plt
        from torch.nn import functional as F
        from IPython.display import display, FileLink
        from geodiff_gan.data.manifest import load_manifest
        from geodiff_gan.experiments.trust_moe import (
            make_config, prepare_manifest, write_json, digest_file, evaluate,
            dataset_for, load_model,
        )
        from geodiff_gan.experiments.trust_report import benchmark, bundle_results
        from geodiff_gan.spectral_indices import spectral_index_images

        if not torch.cuda.is_available():
            raise RuntimeError('Enable a Kaggle GPU accelerator.')
        for name in ('configs', 'figures', 'reports', 'evaluations'):
            (SUITE_ROOT / name).mkdir(exist_ok=True)
        commit = subprocess.check_output(
            ['git', '-C', str(REPOSITORY_DIR), 'rev-parse', 'HEAD'], text=True
        ).strip()
        environment = dict(commit=commit, branch=branch, python=sys.version,
                           torch=torch.__version__, cuda=torch.version.cuda,
                           gpu=torch.cuda.get_device_name(0), input=[6, 128, 128],
                           output=[6, 384, 384])
        write_json(SUITE_ROOT / 'environment.json', environment)
        print(json.dumps(environment, indent=2))
        ''', True),
        cell("""
        ## 3. Prepare real six-band tile pairs

        Raw inputs remain read-only. The cache is restart-safe. Both modalities use
        one manifest, so their train, validation and test patch identities are exact
        matches. Use a new `SUITE_ROOT` when changing preparation settings.
        """),
        cell(r'''
        if PREPARED_MANIFEST is None:
            build_root = SUITE_ROOT / 'data'
            raw_manifest = build_root / 'manifest_raw.jsonl'
            patch_root = build_root / 'patches'
            state = build_root / 'pairing_state.json'
            quarantine = build_root / 'quarantine.jsonl'
            command = [
                sys.executable, '-m', 'geodiff_gan.cli.prepare_landsat_sentinel',
                '--sentinel-input', RAW_TILE_ROOT,
                '--landsat-input', RAW_TILE_ROOT,
                '--output', patch_root,
                '--manifest', raw_manifest,
                '--state', state,
                '--quarantine', quarantine,
                '--patch-size', PATCH_SIZE,
                '--stride', PATCH_STRIDE,
                '--max-day-gap', MAX_DAY_GAP,
                '--minimum-overlap-fraction', MINIMUM_OVERLAP_FRACTION,
                '--minimum-valid-fraction', MINIMUM_VALID_FRACTION,
                '--bandpass-adjustment', 'none',
                '--include-multispectral',
                '--split-strategy', 'within-tile-spatial',
                '--train-fraction', TRAIN_FRACTION,
                '--validation-fraction', VALIDATION_FRACTION,
                '--minimum-validation-fraction', MINIMUM_VALIDATION_FRACTION,
                '--minimum-test-fraction', MINIMUM_TEST_FRACTION,
            ]
            if FAST_DEV_RUN:
                command += ['--max-pairs', 1]
            if REBUILD_PREPARED_DATA:
                command.append('--rebuild')
            run(command, REPOSITORY_DIR)
            PREPARED_MANIFEST = raw_manifest

        PREPARED_MANIFEST = Path(PREPARED_MANIFEST)
        if not PREPARED_MANIFEST.is_file():
            raise FileNotFoundError(PREPARED_MANIFEST)

        required = {'lr', 'hr', 'lr_ms', 'hr_ms', 'valid_mask_lr', 'valid_mask_hr'}
        source_records = load_manifest(PREPARED_MANIFEST, resolve_paths=True)
        for record in source_records:
            with np.load(record.patch, allow_pickle=False) as values:
                missing = required - set(values.files)
                if missing:
                    raise ValueError(f'{record.patch}: missing {sorted(missing)}; rebuild with --include-multispectral')
                if values['lr'].shape != (3, 128, 128) or values['hr'].shape != (3, 384, 384):
                    raise ValueError(f'{record.patch}: wrong RGB geometry')
                if values['lr_ms'].shape != (6, 128, 128) or values['hr_ms'].shape != (6, 384, 384):
                    raise ValueError(f'{record.patch}: wrong multispectral geometry')
                if not np.allclose(values['lr'], values['lr_ms'][:3]):
                    raise ValueError(f'{record.patch}: RGB LR is not the first three multispectral bands')
                if not np.allclose(values['hr'], values['hr_ms'][:3]):
                    raise ValueError(f'{record.patch}: RGB HR is not the first three multispectral bands')

        MANIFEST = SUITE_ROOT / 'runtime_manifest.jsonl'
        AUDIT = prepare_manifest(PREPARED_MANIFEST, MANIFEST, spatial_audit=True,
                                 minimum_test_fraction=MINIMUM_TEST_FRACTION)
        AUDIT.update(model_geometry='128->384', bands=BAND_NAMES,
                     swir_target_note='Sentinel B11/B12 resampled from native 20 m to 10 m grid')
        write_json(SUITE_ROOT / 'dataset_audit.json', AUDIT)
        print(json.dumps(AUDIT, indent=2))
        display(pd.DataFrame([dict(split=r.split, tile=r.tile_id,
                                   scene_class=r.scene_class,
                                   day_gap=r.day_gap) for r in load_manifest(MANIFEST)]))
        ''', True),
        cell("## 4. Training-only visual and spectral audit"),
        cell(r'''
        def study_config(modality, profile, root, parent=None):
            options = MODALITIES[modality]
            config = make_config(
                MANIFEST, root, profile=profile, seed=SEED,
                epochs=EPOCHS['base' if profile == 'base' else 'residual'],
                batch_size=BATCH_SIZE, crop_size=TRAIN_LR_CROP,
                experts=NUM_EXPERTS, top_k=TOP_K,
                dataset_id=AUDIT['dataset_id'], parent=parent,
                base_model=BASE_MODEL, **options,
            )
            config['model']['coverage'] = 1.0 if profile == 'dense_transformer' else REGION_FRACTION
            config['training']['progress'] = 'compact'
            config['study'] = dict(modality=modality, label=f'{modality}_{profile}',
                                   geometry='128->384', bands=options['band_names'])
            return config

        def preview_training_pair(index=0):
            config = study_config('multispectral', 'base', SUITE_ROOT / 'preview')
            sample = dataset_for(config, 'train')[int(index) % AUDIT['counts']['train']]
            lr, hr = sample['lr'], sample['hr']
            lr_up = F.interpolate(lr[None], size=hr.shape[-2:], mode='bicubic', align_corners=False)[0]
            figure, axes = plt.subplots(2, 3, figsize=(15, 9))
            panels = [(lr[:3], 'Landsat RGB 30 m | 128 x 128'),
                      (lr_up[:3], 'Bicubic display | 384 x 384'),
                      (hr[:3], 'Sentinel RGB 10 m | 384 x 384')]
            for axis, (image, title) in zip(axes[0], panels):
                axis.imshow(np.clip(image.permute(1, 2, 0) / DISPLAY_MAX, 0, 1) ** (1 / 1.4))
                axis.set_title(title); axis.axis('off')
            for axis, channel, title in zip(axes[1], (3, 4, 5), ('NIR', 'SWIR1', 'SWIR2')):
                axis.imshow(hr[channel], cmap='gray', vmin=0, vmax=DISPLAY_MAX)
                axis.set_title(f'Sentinel {title} target'); axis.axis('off')
            figure.tight_layout()
            figure.savefig(SUITE_ROOT / f'figures/train_audit_{index}.png', dpi=170)
            plt.show()

            lr_indices = spectral_index_images(lr_up[None], BAND_NAMES)
            hr_indices = spectral_index_images(hr[None], BAND_NAMES)
            figure, axes = plt.subplots(2, 3, figsize=(15, 8))
            for column, name in enumerate(('ndvi', 'ndwi', 'ndbi')):
                axes[0, column].imshow(lr_indices[name][0], cmap='RdYlGn', vmin=-1, vmax=1)
                axes[0, column].set_title(f'LR bicubic {name.upper()}')
                axes[1, column].imshow(hr_indices[name][0], cmap='RdYlGn', vmin=-1, vmax=1)
                axes[1, column].set_title(f'HR target {name.upper()}')
            for axis in axes.flat: axis.axis('off')
            figure.tight_layout()
            figure.savefig(SUITE_ROOT / f'figures/train_indices_{index}.png', dpi=170)
            plt.show()
        preview_training_pair(0)
        ''', True),
        cell("## 5. Training helper and registry"),
        cell(r'''
        REGISTRY_PATH = SUITE_ROOT / 'registry.json'
        REGISTRY = json.loads(REGISTRY_PATH.read_text()) if REGISTRY_PATH.exists() else {}

        def train_model(modality, profile, parent=None):
            label = f'{modality}_{profile}'
            root = SUITE_ROOT / 'models' / label
            config = study_config(modality, profile, root, parent=parent)
            path = SUITE_ROOT / 'configs' / f'{label}.json'
            write_json(path, config)
            run([sys.executable, '-m', 'geodiff_gan.cli.trust_moe', 'train', '--config', path])
            checkpoint = root / 'best.pt'
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            REGISTRY[label] = str(checkpoint.relative_to(SUITE_ROOT))
            write_json(REGISTRY_PATH, REGISTRY)
            return checkpoint

        def registered(label):
            path = SUITE_ROOT / REGISTRY[label]
            if not path.is_file():
                raise FileNotFoundError(path)
            return path
        ''', True),
        cell("## 6. Train/resume the RGB base"),
        cell("RGB_BASE = train_model('rgb', 'base')", True),
        cell("## 7. Train/resume the six-band multispectral base"),
        cell("MS_BASE = train_model('multispectral', 'base')", True),
        cell("## 8. Dense Transformer MoE controls"),
        cell(r'''
        if RUN['dense_transformer']:
            train_model('rgb', 'dense_transformer', RGB_BASE)
            train_model('multispectral', 'dense_transformer', MS_BASE)
        ''', True),
        cell("## 9. Sparse Transformer MoE"),
        cell(r'''
        if RUN['sparse_transformer']:
            train_model('rgb', 'sparse_transformer', RGB_BASE)
            train_model('multispectral', 'sparse_transformer', MS_BASE)
        ''', True),
        cell("## 10. Sparse MoE without trust"),
        cell(r'''
        if RUN['sparse_no_trust']:
            train_model('rgb', 'sparse_no_trust', RGB_BASE)
            train_model('multispectral', 'sparse_no_trust', MS_BASE)
        ''', True),
        cell("## 11. Validation: all models and matched metric tables"),
        cell(r'''
        VALIDATION = {}
        for label in sorted(REGISTRY):
            output = SUITE_ROOT / 'evaluations' / 'val' / label
            scores = evaluate(registered(label), output, split='val', device='cuda')
            VALIDATION[label] = output
            scores['label'] = label
            write_json(output / 'metrics.json', scores)

        def metric_table(evaluations):
            rows = []
            for label, root in evaluations.items():
                values = json.loads((Path(root) / 'metrics.json').read_text())
                modality, profile = label.split('_', 1)
                row = dict(label=label, modality=modality, profile=profile,
                           count=int(values['count']))
                for key in ('psnr', 'ssim', 'l1', 'edge_f1', 'ergas', 'sam_degrees',
                            'uiqi', 'scc', 'rgb_psnr', 'rgb_ssim', 'rgb_l1',
                            'rgb_edge_f1',
                            'ndvi_mae', 'ndwi_mae', 'ndbi_mae'):
                    if key in values: row[key] = values[key]
                rows.append(row)
            return pd.DataFrame(rows).sort_values(['modality', 'profile'])

        VAL_TABLE = metric_table(VALIDATION)
        VAL_TABLE.to_csv(SUITE_ROOT / 'reports/validation_metrics.csv', index=False)
        display(VAL_TABLE.round(6))
        ''', True),
        cell("## 12. Validation plots: spatial fidelity and spectral indices"),
        cell(r'''
        def save_metric_plots(table, split):
            values = table.copy()
            values['rgb_comparable_psnr'] = values['rgb_psnr'].fillna(values['psnr'])
            values['rgb_comparable_ssim'] = values['rgb_ssim'].fillna(values['ssim'])
            values['rgb_comparable_edge_f1'] = values['rgb_edge_f1'].fillna(values['edge_f1'])
            figure, axes = plt.subplots(1, 3, figsize=(17, 5))
            for axis, key, title in zip(
                axes,
                ('rgb_comparable_psnr', 'rgb_comparable_ssim', 'rgb_comparable_edge_f1'),
                ('RGB PSNR (dB)', 'RGB SSIM', 'RGB Edge F1'),
            ):
                axis.bar(values.label, values[key])
                axis.set_title(title); axis.tick_params(axis='x', rotation=70)
                axis.grid(axis='y', alpha=.25)
            figure.tight_layout()
            figure.savefig(SUITE_ROOT / f'figures/{split}_spatial_metrics.png', dpi=180)
            plt.show()

            ms = values[values.modality.eq('multispectral')]
            if not ms.empty:
                figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
                for axis, name in zip(axes, ('ndvi', 'ndwi', 'ndbi')):
                    axis.bar(ms.profile, ms[f'{name}_mae'])
                    axis.set_title(f'{name.upper()} MAE vs Sentinel target')
                    axis.tick_params(axis='x', rotation=45); axis.grid(axis='y', alpha=.25)
                figure.tight_layout()
                figure.savefig(SUITE_ROOT / f'figures/{split}_index_mae.png', dpi=180)
                plt.show()
        save_metric_plots(VAL_TABLE, 'validation')
        ''', True),
        cell("""
        ## 13. Freeze the test plan

        Run this only after the validation comparison is accepted. The test plan
        stores every checkpoint hash; changing a model requires a new suite root.
        """),
        cell(r'''
        TEST_PLAN = {
            'dataset_id': AUDIT['dataset_id'],
            'test_count': AUDIT['counts']['test'],
            'geometry': '128->384',
            'models': {label: {'path': REGISTRY[label],
                               'sha256': digest_file(registered(label))}
                       for label in sorted(REGISTRY)},
        }
        plan_path = SUITE_ROOT / 'test_plan.json'
        if plan_path.exists() and json.loads(plan_path.read_text()) != TEST_PLAN:
            raise ValueError('Frozen test plan changed. Use a new SUITE_ROOT.')
        write_json(plan_path, TEST_PLAN)
        print(json.dumps(TEST_PLAN, indent=2))
        ''', True),
        cell("## 14. Evaluate every frozen model on the complete test split"),
        cell(r'''
        TEST = {}
        if RUN_TEST_EVALUATION:
            for label in sorted(TEST_PLAN['models']):
                output = SUITE_ROOT / 'evaluations' / 'test' / label
                scores = evaluate(registered(label), output, split='test', device='cuda')
                if scores['count'] != TEST_PLAN['test_count']:
                    raise RuntimeError(f'{label}: incomplete test evaluation')
                scores['label'] = label
                write_json(output / 'metrics.json', scores)
                TEST[label] = output
            TEST_TABLE = metric_table(TEST)
            TEST_TABLE.to_csv(SUITE_ROOT / 'reports/test_metrics.csv', index=False)
            display(TEST_TABLE.round(6))
            save_metric_plots(TEST_TABLE, 'test')
        else:
            print('Test evaluation is disabled. Review validation, then set RUN_TEST_EVALUATION=True.')
        ''', True),
        cell("## 15. RGB versus multispectral paired study"),
        cell(r'''
        if RUN_TEST_EVALUATION:
            paired_rows = []
            for profile in ('base', 'dense_transformer', 'sparse_transformer', 'sparse_no_trust'):
                rgb_label, ms_label = f'rgb_{profile}', f'multispectral_{profile}'
                if rgb_label not in TEST or ms_label not in TEST:
                    continue
                rgb_rows = pd.read_json(Path(TEST[rgb_label]) / 'per_image.json')
                ms_rows = pd.read_json(Path(TEST[ms_label]) / 'per_image.json')
                if not rgb_rows.patch.map(Path).map(lambda p: p.name).equals(
                        ms_rows.patch.map(Path).map(lambda p: p.name)):
                    raise RuntimeError(f'Pair ordering differs for {profile}')
                delta = ms_rows.rgb_psnr - rgb_rows.psnr
                paired_rows.append(dict(
                    profile=profile,
                    rgb_psnr=rgb_rows.psnr.mean(),
                    multispectral_rgb_psnr=ms_rows.rgb_psnr.mean(),
                    rgb_psnr_delta_ms_minus_rgb=delta.mean(),
                    fraction_ms_rgb_better=float((delta > 0).mean()),
                    multispectral_six_band_psnr=ms_rows.psnr.mean(),
                    multispectral_sam_degrees=ms_rows.sam_degrees.mean(),
                ))
            RGB_MS_STUDY = pd.DataFrame(paired_rows)
            RGB_MS_STUDY.to_csv(SUITE_ROOT / 'reports/rgb_vs_multispectral.csv', index=False)
            display(RGB_MS_STUDY.round(6))
        ''', True),
        cell("## 16. Spectral-index preservation and improvement over LR"),
        cell(r'''
        if RUN_TEST_EVALUATION:
            index_rows = []
            for label, root in TEST.items():
                if not label.startswith('multispectral_'):
                    continue
                values = pd.read_json(Path(root) / 'per_image.json')
                for name in ('ndvi', 'ndwi', 'ndbi'):
                    improvement = values[f'{name}_mae_improvement_vs_bicubic']
                    index_rows.append(dict(
                        model=label, index=name.upper(), count=len(values),
                        model_mae=values[f'{name}_mae'].mean(),
                        lr_bicubic_mae=values[f'bicubic_{name}_mae'].mean(),
                        mae_improvement=improvement.mean(),
                        fraction_improved=float((improvement > 0).mean()),
                        model_bias=values[f'{name}_bias'].mean(),
                        model_correlation=values[f'{name}_correlation'].mean(),
                        lr_correlation=values[f'bicubic_{name}_correlation'].mean(),
                    ))
            INDEX_STUDY = pd.DataFrame(index_rows)
            INDEX_STUDY.to_csv(SUITE_ROOT / 'reports/spectral_index_study.csv', index=False)
            display(INDEX_STUDY.round(6))
        ''', True),
        cell("## 17. Per-band PSNR"),
        cell(r'''
        if RUN_TEST_EVALUATION:
            rows = []
            for label, root in TEST.items():
                values = json.loads((Path(root) / 'metrics.json').read_text())
                _, config = load_model(registered(label), 'cpu')
                for band in config['data']['band_names']:
                    rows.append(dict(model=label, band=band,
                                     psnr=values[f'{band}_psnr']))
            BAND_TABLE = pd.DataFrame(rows)
            BAND_TABLE.to_csv(SUITE_ROOT / 'reports/per_band_psnr.csv', index=False)
            display(BAND_TABLE.pivot(index='model', columns='band', values='psnr').round(4))
        ''', True),
        cell("## 18. Visualize any test index"),
        cell(r'''
        def _display_rgb(array):
            return np.clip(array[:3].transpose(1, 2, 0) / DISPLAY_MAX, 0, 1) ** (1 / 1.4)

        def show_test_index(index=0, profile='sparse_transformer'):
            if not RUN_TEST_EVALUATION:
                raise RuntimeError('Run the test evaluation first.')
            rgb_label = f'rgb_{profile}'
            ms_label = f'multispectral_{profile}'
            if rgb_label not in TEST or ms_label not in TEST:
                raise KeyError(f'Profile not evaluated: {profile}')
            _, ms_config = load_model(registered(ms_label), 'cpu')
            sample = dataset_for(ms_config, 'test')[int(index)]
            lr, hr = sample['lr'].numpy(), sample['hr'].numpy()
            lr_up = F.interpolate(sample['lr'][None], size=sample['hr'].shape[-2:],
                                  mode='bicubic', align_corners=False)[0].numpy()
            with np.load(Path(TEST[rgb_label]) / 'images' / f'{index:06d}.npz') as values:
                rgb_prediction = values['prediction']
            with np.load(Path(TEST[ms_label]) / 'images' / f'{index:06d}.npz') as values:
                ms_prediction = values['prediction']

            figure, axes = plt.subplots(2, 3, figsize=(15, 10), squeeze=False)
            panels = [(lr, 'Original Landsat | 128 x 128'),
                      (lr_up, 'Landsat bicubic | 384 x 384'),
                      (rgb_prediction, f'RGB {profile}'),
                      (ms_prediction, f'Multispectral {profile} shown as RGB'),
                      (hr, 'Sentinel target | 384 x 384')]
            for axis in axes.flat: axis.axis('off')
            for axis, (array, title) in zip(axes.flat, panels):
                axis.imshow(_display_rgb(array), interpolation='nearest')
                axis.set_title(title)
            figure.suptitle(f'Test index {index}: equal geographic footprint')
            figure.tight_layout()
            figure.savefig(SUITE_ROOT / f'figures/test_{index:04d}_{profile}_rgb.png', dpi=180)
            plt.show()

            figure, axes = plt.subplots(3, 3, figsize=(14, 13), squeeze=False)
            for row, (channel, band) in enumerate(((3, 'NIR'), (4, 'SWIR1'), (5, 'SWIR2'))):
                for axis, array, source in zip(
                    axes[row], (lr_up[channel], ms_prediction[channel], hr[channel]),
                    ('LR bicubic', 'Multispectral SR', 'Sentinel target'),
                ):
                    axis.imshow(array, cmap='gray', vmin=0, vmax=DISPLAY_MAX)
                    axis.set_title(f'{source} {band}'); axis.axis('off')
            figure.tight_layout()
            figure.savefig(SUITE_ROOT / f'figures/test_{index:04d}_{profile}_bands.png', dpi=180)
            plt.show()

            lr_indices = spectral_index_images(torch.from_numpy(lr_up)[None], BAND_NAMES)
            model_indices = spectral_index_images(torch.from_numpy(ms_prediction)[None], BAND_NAMES)
            hr_indices = spectral_index_images(torch.from_numpy(hr)[None], BAND_NAMES)
            figure, axes = plt.subplots(3, 3, figsize=(14, 13), squeeze=False)
            for row, name in enumerate(('ndvi', 'ndwi', 'ndbi')):
                for axis, mapping, source in zip(
                    axes[row], (lr_indices, model_indices, hr_indices),
                    ('LR bicubic', 'Multispectral SR', 'Sentinel target'),
                ):
                    image = axis.imshow(mapping[name][0], cmap='RdYlGn', vmin=-1, vmax=1)
                    axis.set_title(f'{source} {name.upper()}'); axis.axis('off')
                figure.colorbar(image, ax=axes[row].tolist(), shrink=.75)
            figure.suptitle(f'Test index {index}: spectral-index comparison')
            figure.savefig(SUITE_ROOT / f'figures/test_{index:04d}_{profile}_indices.png', dpi=180)
            plt.show()

            rgb_row = json.loads((Path(TEST[rgb_label]) / 'records' / f'{index:06d}.json').read_text())
            ms_row = json.loads((Path(TEST[ms_label]) / 'records' / f'{index:06d}.json').read_text())
            display(pd.DataFrame([
                dict(model=rgb_label, psnr=rgb_row['psnr'], ssim=rgb_row['ssim'],
                     edge_f1=rgb_row['edge_f1']),
                dict(model=ms_label, psnr=ms_row['psnr'], rgb_psnr=ms_row['rgb_psnr'],
                     ssim=ms_row['ssim'], edge_f1=ms_row['edge_f1'],
                     rgb_edge_f1=ms_row['rgb_edge_f1'],
                     ndvi_mae=ms_row['ndvi_mae'], ndwi_mae=ms_row['ndwi_mae'],
                     ndbi_mae=ms_row['ndbi_mae']),
            ]).round(6))
            return dict(patch=sample['patch'], tile=sample['tile_id'],
                        landsat=sample['landsat_product'], sentinel=sample['sentinel_product'])

        if RUN_TEST_EVALUATION:
            display(show_test_index(0, 'sparse_transformer'))
        ''', True),
        cell("## 19. Routing, parameters and measured inference time"),
        cell(r'''
        EFFICIENCY = []
        for label in sorted(REGISTRY):
            checkpoint = registered(label)
            model, config = load_model(checkpoint, 'cpu')
            parameters = sum(value.numel() for value in model.parameters())
            base_parameters = sum(value.numel() for value in model.base.parameters())
            optimized_parameters = (base_parameters if config['profile'] == 'base'
                                    else parameters - base_parameters)
            del model
            timing = benchmark(checkpoint, SUITE_ROOT / f'reports/timing_{label}.json',
                               repeats=3 if FAST_DEV_RUN else 10,
                               warmup=1 if FAST_DEV_RUN else 3, device='cuda')
            full = next(row for row in timing if row['operation'] == 'full')
            EFFICIENCY.append(dict(model=label, parameters=parameters,
                optimized_parameters=optimized_parameters, mean_ms=full['mean_ms'],
                images_per_second=full['images_per_second'],
                peak_gpu_memory_mb=full['peak_allocated_mb'],
                active_fraction=full['active_fraction'],
                expert_tile_calls=full['mean_timed_expert_tile_calls']))
        EFFICIENCY = pd.DataFrame(EFFICIENCY)
        EFFICIENCY.to_csv(SUITE_ROOT / 'reports/efficiency.csv', index=False)
        display(EFFICIENCY.round(4))
        ''', True),
        cell("## 20. Download results and restart state"),
        cell(r'''
        archive = SUITE_ROOT.parent / f'{SUITE_ROOT.name}-results-and-resume.zip'
        archive = bundle_results(SUITE_ROOT, archive, source_root=REPOSITORY_DIR,
                                 include_resume=True)
        print('Archive:', archive, 'size MB:', archive.stat().st_size / 2**20)
        display(FileLink(str(archive)))
        ''', True),
    ]
    for index, value in enumerate(cells):
        value["id"] = f"trustmoe-rgb-ms-128-{index:03d}"
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
    print(OUTPUT)


if __name__ == "__main__":
    build()
