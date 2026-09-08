"""Generate a TIFF-only batch export and local crop workflow."""
import ast
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]


def build():
    cells = []
    def add(kind, text):
        source = dedent(text).strip() + '\n'
        cell = dict(cell_type=kind, metadata={}, source=source.splitlines(True), id=f'tile-{len(cells):03}')
        if kind == 'code':
            ast.parse(source)
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add('markdown', '''
    # India scene-overlap TIFF exports, then local paired crops

    This workflow exports LARGE Landsat/Sentinel overlap rectangles once, not one
    Earth Engine request per training patch. Exports run on Earth Engine and arrive
    in Drive; local raster windows produce the paired TIFF datasets. It is not an
    untouched full-scene download: nonoverlapping pixels are masked, and Sentinel
    is aligned to the Landsat grid. No synthetic LR and no NPZ files.

    Use a NEW output directory. This sampling strategy differs from point-first
    collection. Existing data is never deleted. Balanced weak classes and fixed
    geographic splits remain, but IDs differ from the point-first dataset.

    | Total pairs | Train | Common val | Common test |
    |---:|---:|---:|---:|
    |2000|1200|200|600|
    |4000|3200|200|600|
    |6000|5200|200|600|

    Set aside substantial Drive storage. A roughly 100 km overlap can require
    around 2 GB uncompressed for four HR float bands alone. Multiple tiles and
    patch archives can exceed a free Drive quota. No automatic deletion is used.
    Batch exports may queue; a speedup is plausible, not guaranteed.
    ''')
    add('code', '''
    import sys, subprocess, json, time
    from pathlib import Path
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'earthengine-api',
                    'rasterio', 'pyproj', 'numpy', 'pandas', 'matplotlib'], check=True)
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT = 'YOUR_EARTH_ENGINE_CLOUD_PROJECT_ID'
    REPO = Path('/content/geodiff-tile-first')
    OUTPUT = Path('/content/drive/MyDrive/thesis/gee_india_tile_first_v1')
    DRIVE_FOLDER = 'geodiff_india_tile_exports_v1'  # Must be unique in your Drive.
    TILE_DIRECTORY = Path('/content/drive/MyDrive') / DRIVE_FOLDER
    MAX_TILES = 30
    CANDIDATE_LIMIT = 150
    MAX_ACTIVE_EXPORTS = 2
    if not REPO.exists():
        subprocess.run(['git', 'clone', '--depth', '1', '--branch', '3x-continued',
            'https://github.com/shashankjs2002/SI-SR-1.git', str(REPO)], check=True)
    if not (REPO / 'src/geodiff_gan/data/gee_tiles.py').is_file():
        raise RuntimeError('Push the updated code to GitHub first; then use a new REPO directory.')
    sys.path.insert(0, str(REPO / 'src'))
    import ee
    from geodiff_gan.data.gee_pairs import IndiaPairConfig, IndiaPairDownloader, export_geotiff_datasets
    from geodiff_gan.data.gee_tiles import plan_tiles, submit_exports, crop_tiles
    ee.Authenticate()
    ee.Initialize(project=PROJECT)
    CONFIG = IndiaPairConfig()
    downloader = IndiaPairDownloader(OUTPUT, CONFIG)
    ''')
    add('markdown', '''
    ## Plan overlap tiles
    Candidate acquisition dates and masks match the point-first defaults. Scene
    selection is a bounded discovery heuristic, not proof of national coverage.
    The plan is saved and reused. Thirty overlaps may not fill every category quota;
    inspect coverage before starting an expensive export run. If inadequate, use
    a new study root with a larger plan; do not change completed test selections.
    ''')
    add('code', '''
    import pandas as pd
    from IPython.display import display
    PLAN = plan_tiles(downloader, MAX_TILES, CANDIDATE_LIMIT)
    display(pd.DataFrame(PLAN)[['tile_id', 'landsat_date', 'sentinel_date', 'discovery_class', 'width', 'height']])
    gib = sum(t['width'] * t['height'] * (4 * 4 + 9 * 4 * 4 + 4) for t in PLAN) / 2**30
    print(f'Approximate uncompressed tile bytes: {gib:.1f} GiB, excluding paired crops/ZIPs.')
    print('Inspect Drive quota before submitting. This notebook never deletes data.')
    ''')
    add('markdown', '''
    ## Submit exports and resume the queue
    Each overlap has three exports: LR RGB+valid, HR RGB+valid, and land-cover codes.
    Only two active tasks are requested by default, respecting other account tasks.
    Task IDs are saved. Rerun this cell after a Colab restart; submitted Earth Engine
    tasks can continue without the notebook. Failed tasks are displayed, not silently
    resubmitted. Resolve permission/quota/export errors before retrying a failed task.
    Do not use duplicate Drive folders with the same name.
    ''')
    add('code', '''
    WATCH_QUEUE = True
    while True:
        states = submit_exports(downloader, PLAN, DRIVE_FOLDER, MAX_ACTIVE_EXPORTS)
        frame = pd.DataFrame(states)
        display(frame[['prefix', 'state']] if not frame.empty else frame)
        if not frame.empty and frame['state'].isin(['FAILED', 'CANCELLED', 'UNKNOWN']).any():
            display(frame)
            raise RuntimeError('An export needs attention. Completed TIFFs and task receipts are preserved.')
        if len(states) == 3 * len(PLAN) and all(s['state'] == 'COMPLETED' for s in states):
            break
        if not WATCH_QUEUE:
            break
        time.sleep(60)
    ''')
    add('markdown', '''
    ## Crop locally without more Earth Engine image requests
    Wait for files to appear in mounted Drive. Crops are nonoverlapping within each
    scene grid: 128x128 LR and 384x384 HR. Spatial 40 km blocks with a 500 m guard
    separate splits. Different acquisition grids can overlap within a split; these
    patches are not all statistically independent. WorldCover labels are weak 2021
    labels. Clouds, invalid pixels and insufficient class coverage are rejected.
    This cell is restartable and never changes fixed selection IDs after completion.
    ''')
    add('code', '''
    ROWS = crop_tiles(OUTPUT, TILE_DIRECTORY, CONFIG)
    display(pd.crosstab(pd.DataFrame(ROWS).scene_class, pd.DataFrame(ROWS).split))
    ''')
    add('code', '''
    import rasterio
    import numpy as np
    import matplotlib.pyplot as plt
    def show_pair(index=0, split='train'):
        row = [r for r in ROWS if r['split'] == split][index]
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for axis, key in zip(axes, ['lr', 'hr']):
            with rasterio.open(OUTPUT / row[f'{key}_path']) as src:
                axis.imshow(np.clip(src.read().transpose(1, 2, 0) / .3, 0, 1), interpolation='nearest')
                axis.set_title(f'{key.upper()} {src.width}x{src.height}')
                axis.axis('off')
        plt.tight_layout()
        plt.show()
    show_pair()
    ''')
    add('code', '''
    for archive in export_geotiff_datasets(OUTPUT, CONFIG):
        print(archive, f'{archive.stat().st_size/2**30:.2f} GiB')
    print('Download the three GeoTIFF ZIPs through Drive, then upload them to Kaggle.')
    ''')
    add('markdown', '''
    ## Limits and provenance
    No live exports were tested by the code author without your Earth Engine project.
    Review the per-tile TIFFs and pair previews before training. The existing NPZ-based
    training loader is not compatible with this TIFF manifest without adaptation.
    Source attribution: USGS Landsat, Copernicus Sentinel, ESA WorldCover and FAO GAUL.
    Review source licenses, including GAUL non-commercial restrictions.

    [Earth Engine batch export and grid documentation](https://developers.google.com/earth-engine/guides/exporting_images)
    ''')
    path = ROOT / 'colab/GEE_India_Tile_First_TIFF_Datasets.ipynb'
    path.write_text(json.dumps(dict(cells=cells, nbformat=4, nbformat_minor=5,
        metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python'))), indent=1)+'\n')
    print(path)


if __name__ == '__main__':
    build()
