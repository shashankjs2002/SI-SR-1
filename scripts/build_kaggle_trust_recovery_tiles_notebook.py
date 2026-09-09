"""Build the raw-tile/prepared-manifest TrustMoE residual-recovery notebook."""
import ast
import copy
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "kaggle/GeoDiff_TrustMoE_OLI2MSI_Residual_Recovery_3x.ipynb"
OUTPUT = ROOT / "kaggle/GeoDiff_TrustMoE_Tiles_Residual_Recovery_3x.ipynb"


def cell(text, code=False):
    value = {"cell_type": "code" if code else "markdown", "metadata": {},
             "source": (dedent(text).strip() + "\n").splitlines(True)}
    if code:
        ast.parse("".join(value["source"]))
        value.update(execution_count=None, outputs=[])
    return value


def build():
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    recovery = copy.deepcopy(notebook["cells"][9:])
    cells = [
        cell("""
        # GeoDiff-TrustMoE residual recovery on your Landsat/Sentinel tiles

        This notebook accepts either:

        1. Raw Landsat 8/9 Collection-2 Level-2 products and Sentinel-2 L2A SAFE
           products or archives anywhere under `/kaggle/input`; or
        2. The portable prepared dataset created by
           `Landsat_Sentinel_TrustMoE_Dataset_Preparation.ipynb`.

        Raw products are paired by geographic overlap and acquisition date. The
        resulting examples contain actual Landsat 30 m RGB inputs and actual
        Sentinel-2 10 m RGB targets. Stored geometry is 128x128 -> 384x384. Fixed
        within-tile spatial regions provide train, validation and at least 10% test
        pairs. Existing preparation state resumes and newly attached products are
        included. Invalid pairs are written to a quarantine log.

        The study first tests whether one residual expert can learn, then compares
        dense, uniform, base-error-routed, and correction-benefit-routed models. The
        target HR image supplies training labels only and never enters inference.
        Every model uses the same frozen base and fixed splits. There is one base
        pass, no diffusion, VAE, PixelShuffle, or synthetic LR generation.

        Keep test evaluation disabled until validation is reviewed. Results include
        per-category and per-tile reports, paired and tile-cluster intervals, routing,
        parameter counts, measured runtime, visualizations, exact predictions, and a
        restart/download ZIP. Run cells from top to bottom.
        """),
        cell("## 1. Controls: paths, epochs, batch size, experts, and scene labels"),
        cell(r'''
        from pathlib import Path
        import hashlib, json, os, shutil, subprocess, sys, tarfile, time, zipfile

        FAST_DEV_RUN = False
        REPOSITORY_URL = 'https://github.com/shashankjs2002/SI-SR-1.git'
        REPOSITORY_BRANCH = '3x-continued'
        REPOSITORY_DIR = Path('/kaggle/working/geodiff-trust-recovery-tiles-source')
        SUITE_ROOT = Path('/kaggle/working/geodiff-trust-recovery-tiles-v2')
        RAW_TILE_ROOT = Path('/kaggle/input')
        PREPARED_MANIFEST = None  # Optional exact /kaggle/input/.../manifest.jsonl.
        RESTORE_SUITE_FROM = None  # Prior Kaggle output folder or downloaded ZIP.
        OLD_SUITE_ROOT = Path('/kaggle/working/geodiff-trust-moe-v1')
        BASE_CHECKPOINTS = {42: None}  # Optional prior best.pt path per seed.

        EPOCHS = {'base': 30, 'diagnostic': 60, 'single': 15, 'residual': 10}
        BATCH_SIZE = 4
        NUM_EXPERTS, TOP_K = 5, 2
        REGION_FRACTION = 0.5
        SEEDS = [42]  # Final study: [42, 123, 2026].
        RUN = dict(dense_gain=True, sparse_error=True, sparse_gain=True,
                   sparse_no_trust=True, sparse_uniform=True, sparse_conv=False,
                   sparse_adversarial=False, adaptive_k=False,
                   legacy_expert_control=False)
        RUN_TEST_EVALUATION = False
        REQUIRE_RECOVERY_SCREEN = True
        DIAGNOSTIC_PAIRS = 8

        # Used only when a class cannot be inferred from directories such as urban,
        # agriculture, forest, water, barren, grassland, industrial, or mixed.
        CATEGORY_BY_TILE = {
            # '44QLL': 'urban',
        }
        CATEGORY_BY_PRODUCT = {
            # 'S2A_MSIL2A_202...': 'agriculture',
        }

        # Fixed data protocol. Change only for a new prepared-dataset version.
        PATCH_SIZE, PATCH_STRIDE = 384, 288
        MAX_DAY_GAP = 3
        MINIMUM_OVERLAP_FRACTION = 0.10
        MINIMUM_VALID_FRACTION = 0.95
        TRAIN_FRACTION, VALIDATION_FRACTION = 0.78, 0.10
        MINIMUM_VALIDATION_FRACTION, MINIMUM_TEST_FRACTION = 0.08, 0.10
        TRAIN_LR_CROP = 64
        DISPLAY_MAX = 0.3
        DATASET_PROTOCOL = 'own_tiles_spatial_v1'
        BASE_MODEL = dict(base_embed_dim=32, base_depth=2, base_groups=2,
                          base_heads=4, window_size=8, width=32, tile_size=8)
        MAX_PAIRS = 1 if FAST_DEV_RUN else None
        REBUILD_PREPARED_DATA = False

        if sys.version_info < (3, 10):
            raise RuntimeError('Use a Python 3.10+ Kaggle kernel.')
        if not 1 <= TOP_K <= NUM_EXPERTS or not 0 < REGION_FRACTION <= 1:
            raise ValueError('Invalid expert count/top-k/region fraction')
        if TRAIN_FRACTION + VALIDATION_FRACTION >= 1:
            raise ValueError('Train and validation fractions must leave test data')
        SUITE_ROOT.mkdir(parents=True, exist_ok=True)

        def safe_extract(path, root):
            root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=True)
            def checked(name):
                target = (root / name).resolve()
                if not target.is_relative_to(root):
                    raise ValueError('Unsafe archive member: ' + name)
            if zipfile.is_zipfile(path):
                with zipfile.ZipFile(path) as archive:
                    for member in archive.infolist():
                        checked(member.filename)
                        if (member.external_attr >> 16) & 0o170000 == 0o120000:
                            raise ValueError('Archive symlinks are not accepted')
                    archive.extractall(root)
            else:
                with tarfile.open(path) as archive:
                    for member in archive.getmembers():
                        checked(member.name)
                        if not (member.isfile() or member.isdir()):
                            raise ValueError('Only regular files/directories are accepted')
                    archive.extractall(root, filter='data')

        if RESTORE_SUITE_FROM:
            source = Path(RESTORE_SUITE_FROM)
            if source.is_file():
                expanded = SUITE_ROOT.parent / 'trust-recovery-tiles-restore'
                safe_extract(source, expanded); source = expanded
            if not source.is_dir():
                raise FileNotFoundError(source)
            for path in source.rglob('*'):
                if path.is_file():
                    destination = SUITE_ROOT / path.relative_to(source)
                    if not destination.exists():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, destination)
            print('Restored missing files without replacing current files.')

        def run(command, cwd=None):
            environment = os.environ.copy()
            environment['PYTHONPATH'] = str(REPOSITORY_DIR / 'src') + os.pathsep + environment.get('PYTHONPATH', '')
            environment['PYTHONUNBUFFERED'] = '1'
            print('+', ' '.join(map(str, command)), flush=True)
            return subprocess.run(list(map(str, command)), cwd=cwd, env=environment, check=True)

        print('Suite:', SUITE_ROOT)
        print('Epochs:', EPOCHS, 'batch:', BATCH_SIZE, 'experts:', NUM_EXPERTS, 'top-k:', TOP_K)
        ''', True),
        cell("## 2. Clone the branch, install it, and verify the GPU"),
        cell(r'''
        if not REPOSITORY_DIR.exists():
            run(['git', 'clone', '--depth', '1', '--single-branch', '--branch',
                 REPOSITORY_BRANCH, REPOSITORY_URL, REPOSITORY_DIR])
        if not (REPOSITORY_DIR / '.git').is_dir():
            raise RuntimeError('REPOSITORY_DIR is not the requested Git checkout')
        branch = subprocess.check_output(['git', '-C', str(REPOSITORY_DIR), 'branch', '--show-current'], text=True).strip()
        if branch != REPOSITORY_BRANCH:
            raise RuntimeError(f'Expected {REPOSITORY_BRANCH}, found {branch}; use another path.')
        required = REPOSITORY_DIR / 'src/geodiff_gan/experiments/trust_recovery.py'
        if not required.is_file():
            raise RuntimeError('Push the current 3x-continued changes before running this notebook.')
        run([sys.executable, '-m', 'pip', 'install', '-q', 'numpy', 'rasterio>=1.3',
             'Pillow', 'PyYAML', 'tqdm', 'pandas', 'matplotlib'])
        run([sys.executable, '-m', 'pip', 'install', '-q', '-e', REPOSITORY_DIR, '--no-deps'])
        sys.path.insert(0, str(REPOSITORY_DIR / 'src'))

        import torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
        from IPython.display import display, FileLink
        from geodiff_gan.data.diverse import (
            annotate_scene_classes, infer_scene_class, normalize_scene_class,
            validate_category_splits, write_dataset_card,
        )
        from geodiff_gan.data.manifest import load_manifest, write_manifest
        from geodiff_gan.data.sentinel import discover_safe_products, tile_id_from_product
        from geodiff_gan.data.landsat_sentinel import discover_landsat_products, pair_scenes
        from geodiff_gan.experiments.trust_moe import (
            make_config, prepare_manifest, write_json, digest_file, adopt_base,
            evaluate, dataset_for, load_model,
        )
        from geodiff_gan.experiments.trust_recovery import (
            RECOVERY_VERSION, recovery_config, audit_checkpoint,
            memorization_report, validation_screen,
        )
        from geodiff_gan.experiments.trust_report import (
            METRICS, summarize, benchmark, compare_controls, paired_intervals,
            visualize, bundle_results,
        )
        if RECOVERY_VERSION != 'trust-recovery-v2':
            raise RuntimeError('Wrong source revision')
        if not torch.cuda.is_available():
            raise RuntimeError('Enable a Kaggle GPU accelerator.')
        for name in ('figures', 'reports', 'diagnostics', 'source_receipts'):
            (SUITE_ROOT / name).mkdir(exist_ok=True)
        commit = subprocess.check_output(['git', '-C', str(REPOSITORY_DIR), 'rev-parse', 'HEAD'], text=True).strip()
        write_json(SUITE_ROOT / 'source_receipts/revision.json', dict(commit=commit, branch=branch))
        print('Commit:', commit, 'Torch:', torch.__version__, 'GPU:', torch.cuda.get_device_name(0))
        if FAST_DEV_RUN:
            EPOCHS = {'base': 1, 'diagnostic': 2, 'single': 1, 'residual': 3}
            REQUIRE_RECOVERY_SCREEN = RUN_TEST_EVALUATION = False
            print('SMOKE RUN: pipeline verification only.')
        ''', True),
        cell("""
        ## 3. Find a prepared manifest or prepare the raw tiles

        If exactly one portable paired manifest is attached, it is used directly.
        Otherwise the cell extracts raw product archives, discovers complete products,
        pairs them, and creates restart-safe patches under the suite. Source archives
        remain read-only. Added products are included on the next run. Set
        `PREPARED_MANIFEST` when Kaggle input contains multiple datasets.
        """),
        cell(r'''
        if PREPARED_MANIFEST is None:
            cards = list(RAW_TILE_ROOT.rglob('dataset_card.json'))
            candidates = sorted({p.parent / 'manifest.jsonl' for p in cards
                                 if (p.parent / 'manifest.jsonl').exists()})
            if len(candidates) == 1:
                PREPARED_MANIFEST = candidates[0]
                print('Using portable prepared dataset:', PREPARED_MANIFEST)
            elif len(candidates) > 1:
                raise ValueError(f'Set PREPARED_MANIFEST; found: {candidates[:12]}')

        if PREPARED_MANIFEST is None:
            build = SUITE_ROOT / 'data'
            extract_root = build / 'extracted_products'
            patch_root = build / 'patches'
            raw_manifest = build / 'raw_manifest.jsonl'
            classified_manifest = build / 'classified_manifest.jsonl'
            quarantine = build / 'quarantine.jsonl'
            for directory in (extract_root, patch_root):
                directory.mkdir(parents=True, exist_ok=True)
            suffixes = ('.zip', '.tar', '.tar.gz', '.tgz')
            archives = [p for p in RAW_TILE_ROOT.rglob('*') if p.is_file()
                        and p.name.casefold().endswith(suffixes)]
            print('Raw archives:', len(archives))
            for number, archive in enumerate(sorted(archives), 1):
                relative = archive.parent.relative_to(RAW_TILE_ROOT)
                identity = hashlib.sha256((str(archive.relative_to(RAW_TILE_ROOT)) +
                                           str(archive.stat().st_size)).encode()).hexdigest()[:12]
                destination = extract_root / relative / identity
                marker = destination / '.complete.json'
                if marker.exists():
                    continue
                print(f'Extracting {number}/{len(archives)}: {archive.name}')
                safe_extract(archive, destination)
                marker.write_text(json.dumps({'archive': str(archive),
                                               'size': archive.stat().st_size}, indent=2))

            sentinels, landsats = {}, {}
            for root in (RAW_TILE_ROOT, extract_root):
                for product in discover_safe_products(root):
                    sentinels.setdefault(product.name.casefold(), product)
                for product in discover_landsat_products(root):
                    landsats.setdefault(product.product_id.casefold(), product)
            sentinels, landsats = list(sentinels.values()), list(landsats.values())
            if not sentinels or not landsats:
                raise RuntimeError('No complete Sentinel-2 L2A SAFE and Landsat C2 L2 products found')
            pairs, unmatched = pair_scenes(sentinels, landsats, max_day_gap=MAX_DAY_GAP,
                                           minimum_overlap_fraction=MINIMUM_OVERLAP_FRACTION)

            def category_for(product):
                tile = tile_id_from_product(product)
                if tile in CATEGORY_BY_TILE:
                    return normalize_scene_class(CATEGORY_BY_TILE[tile])
                matches = [(prefix, value) for prefix, value in CATEGORY_BY_PRODUCT.items()
                           if product.name.casefold().startswith(prefix.casefold())]
                return (normalize_scene_class(max(matches, key=lambda x: len(x[0]))[1])
                        if matches else infer_scene_class(product))

            pair_categories, pair_rows = {}, []
            for pair in pairs:
                category = category_for(pair.sentinel)
                pair_categories[pair.sentinel.name.casefold()] = category
                pair_rows.append(dict(scene_class=category or 'UNLABELED',
                    tile_id=tile_id_from_product(pair.sentinel), sentinel=pair.sentinel.name,
                    landsat=pair.landsat.product_id, day_gap=pair.day_gap,
                    overlap=pair.overlap_fraction))
            pair_table = pd.DataFrame(pair_rows)
            display(pair_table)
            if pair_table.empty or pair_table.scene_class.eq('UNLABELED').any():
                display(pair_table[pair_table.scene_class.eq('UNLABELED')])
                raise ValueError('Label every source using class directories or CATEGORY overrides')
            pair_table.to_csv(build / 'source_pairs.csv', index=False)
            print('Products:', len(sentinels), 'Sentinel,', len(landsats), 'Landsat')
            print('Compatible pairs:', len(pairs), 'unmatched Sentinel:', len(unmatched))
            print('Pairs by class:', pair_table.scene_class.value_counts().to_dict())

            discovery = SUITE_ROOT / 'data/discovery'
            for sensor, products in (('sentinel', sentinels), ('landsat', landsats)):
                root = discovery / sensor; root.mkdir(parents=True, exist_ok=True)
                for product in products:
                    identity = product.name if sensor == 'sentinel' else product.product_id
                    destination = root / identity
                    if sensor == 'sentinel':
                        if not destination.exists():
                            destination.symlink_to(Path(product).resolve(), target_is_directory=True)
                    else:
                        destination.mkdir(parents=True, exist_ok=True)
                        for source in product.files.values():
                            link = destination / Path(source).name
                            if not link.exists():
                                link.symlink_to(Path(source).resolve())
            command = [sys.executable, '-m', 'geodiff_gan.cli.prepare_landsat_sentinel',
                '--sentinel-input', discovery / 'sentinel', '--landsat-input', discovery / 'landsat',
                '--output', patch_root, '--manifest', raw_manifest,
                '--state', build / 'pairing_state.json', '--quarantine', quarantine,
                '--patch-size', PATCH_SIZE, '--stride', PATCH_STRIDE,
                '--max-day-gap', MAX_DAY_GAP,
                '--minimum-overlap-fraction', MINIMUM_OVERLAP_FRACTION,
                '--minimum-valid-fraction', MINIMUM_VALID_FRACTION,
                '--bandpass-adjustment', 'none', '--split-strategy', 'within-tile-spatial',
                '--train-fraction', TRAIN_FRACTION, '--validation-fraction', VALIDATION_FRACTION,
                '--minimum-validation-fraction', MINIMUM_VALIDATION_FRACTION,
                '--minimum-test-fraction', MINIMUM_TEST_FRACTION]
            if MAX_PAIRS is not None:
                command += ['--max-pairs', MAX_PAIRS]
            if REBUILD_PREPARED_DATA:
                command.append('--rebuild')
            run(command, REPOSITORY_DIR)
            records = annotate_scene_classes(load_manifest(raw_manifest),
                category_by_tile=CATEGORY_BY_TILE,
                category_by_product={**CATEGORY_BY_PRODUCT,
                    **{k: v for k, v in pair_categories.items() if v}}, require_all=True)
            write_manifest(classified_manifest, records)
            category_rows = validate_category_splits(records,
                minimum_test_fraction=MINIMUM_TEST_FRACTION, require_each_split=True)
            pd.DataFrame(category_rows).to_csv(build / 'category_split_summary.csv', index=False)
            write_dataset_card(build, records, preparation=dict(patch_size=PATCH_SIZE,
                stride=PATCH_STRIDE, train_fraction=TRAIN_FRACTION,
                validation_fraction=VALIDATION_FRACTION, minimum_test_fraction=MINIMUM_TEST_FRACTION,
                max_day_gap=MAX_DAY_GAP, minimum_valid_fraction=MINIMUM_VALID_FRACTION,
                input='actual Landsat 30 m RGB', target='actual Sentinel-2 10 m RGB'))
            PREPARED_MANIFEST = classified_manifest

        PREPARED_MANIFEST = Path(PREPARED_MANIFEST)
        MANIFEST = SUITE_ROOT / 'runtime_manifest.jsonl'
        AUDIT = prepare_manifest(PREPARED_MANIFEST, MANIFEST, spatial_audit=True,
                                 minimum_test_fraction=MINIMUM_TEST_FRACTION)
        DATA_CARD = {'protocol': DATASET_PROTOCOL, 'source_manifest': str(PREPARED_MANIFEST)}
        AUDIT['input'] = 'actual Landsat 8/9 Collection-2 L2 RGB at 30 m'
        AUDIT['target'] = 'actual Sentinel-2 L2A RGB at 10 m'
        write_json(SUITE_ROOT / 'dataset_audit.json', AUDIT)
        print(json.dumps(AUDIT, indent=2))
        if AUDIT['frames_hr'] != [PATCH_SIZE]:
            raise RuntimeError(f'Expected stored HR {PATCH_SIZE}, found {AUDIT["frames_hr"]}')
        records = load_manifest(MANIFEST)
        table = pd.DataFrame([dict(split=r.split, tile=r.tile_id,
                                   category=r.scene_class) for r in records])
        display(pd.crosstab([table.category, table.tile], table.split))
        print('Test fraction:', AUDIT['counts']['test'] / sum(AUDIT['counts'].values()))
        print('Do not inspect test predictions before the frozen-plan section.')
        ''', True),
        cell("## 4. Visual check using training data only"),
        cell(r'''
        from torch.nn import functional as F

        def preview_pair(index=0):
            config = make_config(MANIFEST, SUITE_ROOT, profile='base', crop_size=TRAIN_LR_CROP)
            data = dataset_for(config, 'train')
            sample = data[int(index) % len(data)]
            lr, hr = sample['lr'], sample['hr']
            bicubic = F.interpolate(lr[None], size=hr.shape[-2:], mode='bicubic',
                                    align_corners=False)[0].clamp(0, 1)
            fig, axes = plt.subplots(1, 4, figsize=(17, 4))
            values = ((lr, 'Original Landsat 30 m'), (bicubic, 'Bicubic display'),
                      (hr, 'Sentinel-2 target 10 m'), (sample['valid_mask'], 'Valid mask'))
            for axis, (array, title) in zip(axes, values):
                if array.shape[0] == 1:
                    axis.imshow(array[0], vmin=0, vmax=1, cmap='gray')
                else:
                    axis.imshow(np.clip(array.permute(1, 2, 0) / DISPLAY_MAX, 0, 1) ** (1 / 1.4),
                                interpolation='nearest')
                axis.set_title(f'{title}\n{array.shape[-2]} x {array.shape[-1]}')
                axis.axis('off')
            fig.tight_layout()
            fig.savefig(SUITE_ROOT / f'figures/train_pair_{index}.png', dpi=170)
            plt.show()
            record = data.records[int(index) % len(data)]
            print('Class:', record.scene_class, 'tile:', record.tile_id,
                  'source:', record.source_product)
            print('Stored range LR:', float(lr.min()), float(lr.max()),
                  'HR:', float(hr.min()), float(hr.max()))
        preview_pair(0)
        ''', True),
    ]

    for copied in recovery:
        source = "".join(copied["source"])
        replacements = {
            "on OLI2MSI": "on your tile dataset",
            "all 4,970 designated training sources": "the complete fixed training split",
            "the 255 validation sources": "the complete fixed validation split",
            "all 100 fixed test pairs": "the complete fixed test split",
            "all 100 official pairs": "the complete fixed test split",
            "all 100 official test pairs": "the complete fixed test split",
            "OLI2MSI has become a development benchmark": "this test split must remain locked",
            "Prior repeated OLI2MSI testing means": "Keep this tile test split locked.",
            "a new geographically independent dataset is still needed for a strong final claim.":
                "A geographically independent test region is still needed for a strong final claim when current splits share source tiles.",
            "Expected all 100 official test pairs": "Expected every fixed test pair",
            "scores['count'] != 100": "scores['count'] != AUDIT['counts']['test']",
            "protocol=DATA_CARD['protocol']": "protocol=DATASET_PROTOCOL",
            "the OLI2MSI study": "the tile study",
            "official test center": "fixed test pair",
            "fixed center": "fixed",
            "kaggle/GeoDiff_TrustMoE_OLI2MSI_Residual_Recovery_3x.ipynb":
                "kaggle/GeoDiff_TrustMoE_Tiles_Residual_Recovery_3x.ipynb",
            "# Benchmark crops have no verified georeferencing; do not invent a CRS.":
                "# Prepared arrays do not retain the source crop transform; do not invent a CRS.",
            "Protocol-normalized RGB example; no geographic CRS":
                "Prepared RGB example; source crop georeferencing unavailable",
        }
        for old, new in replacements.items():
            source = source.replace(old, new)
        if copied["cell_type"] == "code":
            ast.parse(source)
            copied.update(execution_count=None, outputs=[])
        copied["source"] = source.splitlines(True)
        cells.append(copied)

    # Tile examples retain GeoTIFF arrays but cannot inherit source georeferencing
    # after the paired/cropped NPZ representation; this limitation stays explicit.
    for i, current in enumerate(cells):
        current["id"] = f"trust-recovery-tiles-{i:03d}"
        if current["cell_type"] == "code":
            ast.parse("".join(current["source"]))
            current.update(execution_count=None, outputs=[])
    notebook["cells"] = cells
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    build()
