"""Build the Earth Engine India paired-download notebook; no live EE calls here."""
import ast
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(dict(cell_type="markdown", metadata={}, source=dedent(text).strip().splitlines(True)))


def code(text):
    text = dedent(text).strip() + "\n"
    ast.parse(text)
    cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=text.splitlines(True)))


md('''
# Earth Engine: fixed India Landsat/Sentinel datasets

Download **actual paired observations**, not synthetic LR. Every pair contains RGB
Landsat 30 m `128x128` and Sentinel-2 10 m `384x384`, masks and acquisition metadata.
This notebook uses the Earth Engine Python API in Colab, not manually started
Code Editor export tasks. You need your own authorized Earth Engine Cloud project.

| Total pairs | Training | Validation (same IDs) | Test (same IDs) |
|---:|---:|---:|---:|
| 2,000 | 1,200 | 200 | 600 |
| 4,000 | 3,200 | 200 | 600 |
| 6,000 | 5,200 | 200 | 600 |

Training sets are nested. Counts mean pairs, so 2,000 pairs contain 2,000 LR and
2,000 HR arrays. Test is at least 10% even for the largest set; its proportion in
smaller sets is deliberately larger. Exact 80/10/10 for each size would not retain
the same test IDs/count across sizes.

Sampling uses five weak WorldCover categories across the GAUL India region:
forest, agriculture, urban, barren/sparse vegetation and permanent water. Categories
are balanced in every split. WorldCover labels are from 2021, not verified labels
at the 2023-2025 acquisition dates. Results represent this sampled distribution,
not an unbiased survey of all land cover in India.

Cloud/quality rejection may leave insufficient pairs. In that case the script
reports a shortfall; it never duplicates images or leaks split boundaries to fill it.
Do not assume the first execution will necessarily obtain all 6,000 valid pairs.
''')
md('''
## 1. Install dependencies, clone GitHub, and set your project
No source ZIP upload is needed. The downloader code must first be committed and
pushed to the `3x-continued` GitHub branch. An existing checkout is reused without
overwriting local edits or automatically changing its revision.
''')
code('''
import sys, subprocess
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'earthengine-api', 'requests', 'pyproj', 'numpy', 'Pillow', 'PyYAML', 'tqdm', 'pandas', 'matplotlib'], check=True)
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import json, shutil

EE_PROJECT = 'YOUR_EARTH_ENGINE_CLOUD_PROJECT_ID'
REPOSITORY_URL = 'https://github.com/shashankjs2002/SI-SR-1.git'
REPOSITORY_BRANCH = '3x-continued'
REPOSITORY = Path('/content/geodiff-gee-github')
OUTPUT = Path('/content/drive/MyDrive/thesis/gee_india_fixed_v1')
OUTPUT.mkdir(parents=True, exist_ok=True)
if not REPOSITORY.exists():
    subprocess.run(['git', 'clone', '--depth', '1', '--single-branch', '--branch',
                    REPOSITORY_BRANCH, REPOSITORY_URL, str(REPOSITORY)], check=True)
if not (REPOSITORY / '.git').is_dir():
    raise RuntimeError('REPOSITORY is not a Git checkout. Choose a new REPOSITORY path; existing files are preserved.')
def git_value(*args):
    return subprocess.check_output(['git', '-C', str(REPOSITORY), *args], text=True).strip()
if git_value('remote', 'get-url', 'origin') != REPOSITORY_URL:
    raise RuntimeError('Existing checkout has a different origin. Choose a new REPOSITORY path.')
if git_value('branch', '--show-current') != REPOSITORY_BRANCH:
    raise RuntimeError('Existing checkout is on another branch. Choose a new REPOSITORY path.')
if not (REPOSITORY / 'src/geodiff_gan/data/gee_pairs.py').exists():
    raise RuntimeError('Downloader is missing. Commit and push the new code to GitHub, then clone into a new REPOSITORY path.')
print('Repository revision:', git_value('rev-parse', 'HEAD'))
sys.path.insert(0, str(REPOSITORY / 'src'))
from geodiff_gan.data.gee_pairs import IndiaPairConfig, IndiaPairDownloader, export_datasets
CONFIG = IndiaPairConfig()
CONFIG.validate()
print('Output:', OUTPUT)
print('Filesystem free GiB:', round(shutil.disk_usage(OUTPUT).free / 2**30, 2))
print('Also check your Google Drive account quota: filesystem free space may not reflect it.')
''')
md('''
## 2. Authenticate with Earth Engine
Register your project and enable the Earth Engine API before this step. Authentication
is handled by Google's library; never paste tokens into the notebook or result archive.
Downloads consume Earth Engine computation and account quotas. Do not run multiple
copies to evade quotas. No GPU is required.
''')
code('''
import ee
if EE_PROJECT == 'YOUR_EARTH_ENGINE_CLOUD_PROJECT_ID':
    raise ValueError('Set EE_PROJECT to your own authorized Cloud project ID.')
ee.Authenticate()
ee.Initialize(project=EE_PROJECT)
print('Earth Engine connection:', ee.Number(1).getInfo())
downloader = IndiaPairDownloader(OUTPUT, CONFIG)
''')
md('''
## 3. Inspect fixed sampling sites, then collect/resume
The first call samples and saves candidate coordinates. A seeded hash assigns 40 km
equal-area geographic blocks to train/val/test before collection. Each crop must fit
inside its block with a 500 m guard. All dates at a location stay in the same split.
There are at most two accepted date pairs per sampled location.

Landsat QA_PIXEL/QA_RADSAT and Sentinel SCL masks are applied. Nearest-date candidates
must be within three days, have at least 95% joint validity and at least 50% patch
coverage of the weak requested class. Individual acquisitions are used, not median
composites. No PSNR-based pair filtering is used.

The LR grid retains the selected Landsat scene's 30 m projection and pixel origin.
The HR grid subdivides it exactly into 10 m cells. Sentinel RGB is bilinearly
reprojected, so this is aligned 10 m data, not its untouched native pixel array.
This does not solve residual sensor registration or spectral-response differences.
''')
code('''
import pandas as pd
from IPython.display import display
sites = downloader.candidates()
site_table = pd.DataFrame(sites)
display(pd.crosstab(site_table.scene_class, site_table.split))
site_table.to_csv(OUTPUT / 'candidate_sites.csv', index=False)
print('Sites:', len(sites), 'blocks:', site_table.block.nunique())
''')
code('''
# Rerun this cell after a connection/session interruption, with the SAME OUTPUT.
# Completed NPZs, scene lists and acceptance/rejection receipts are preserved.
SELECTION = downloader.collect()
print('Fixed selections saved:', OUTPUT / 'fixed_selection.json')
''')
md('## 4. Check counts and visualize a training pair')
code('''
import numpy as np
import matplotlib.pyplot as plt
rows = downloader.accepted()
table = pd.DataFrame(rows)
display(pd.crosstab(table.scene_class, table.split))
table.drop(columns=['grid']).to_csv(OUTPUT / 'accepted_pairs.csv', index=False)

def show_pair(index=0, split='train'):
    selected = [r for r in rows if r['split'] == split]
    row = selected[index]
    with np.load(OUTPUT / row['npz'], allow_pickle=False) as data:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for axis, key in zip(axes[:2], ['lr', 'hr']):
            image = data[key]
            axis.imshow(np.clip(image.transpose(1, 2, 0) / 0.3, 0, 1) ** (1 / 1.4), interpolation='nearest')
            axis.set_title(f'{key.upper()} | {image.shape[-1]} x {image.shape[-2]}')
        axes[2].imshow(data['valid_mask_hr'][0], cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('Joint validity')
        for axis in axes: axis.axis('off')
        fig.suptitle(f"{row['scene_class']} | {row['landsat_date']} / {row['sentinel_date']} | {row['gap_seconds']/86400:.2f} days")
        fig.tight_layout()
        fig.savefig(OUTPUT / f'preview_{split}_{index}.png', dpi=150)
        plt.show()
    return row

show_pair(0)
''')
md('''
## 5. Export three portable Kaggle datasets
These ZIPs contain their own fixed manifests and actual NPZs under `train/`, `val/`
and `test/`. Every NPZ contains both `lr` and `hr`; separate TIFF/PNG copies are
unnecessary. The master files remain in OUTPUT, so no downloaded data is deleted.
Writing all three ZIPs uses extra storage. Upload one ZIP at a time if your Drive
quota is small; the code never automatically deletes old files to make room.
''')
code('''
EXPORT_SIZES = (2000, 4000, 6000)
archives = export_datasets(OUTPUT, CONFIG, sizes=EXPORT_SIZES)
for archive in archives:
    print(archive, f'{archive.stat().st_size / 2**30:.2f} GiB')
print('Download these ZIP files from the OUTPUT folder in Google Drive, then upload each as a Kaggle dataset.')
''')
md('''
## 6. Optional browser download and training settings
For large ZIPs, downloading through Google Drive is more reliable than Colab's browser
transfer. Run the following cell for one archive when needed. For each training run,
attach only the intended dataset or set PREPARED_MANIFEST explicitly. Use a separate
SUITE_ROOT for each size; never mix checkpoints between data-size experiments.

```python
DATASET_PROTOCOL = 'spatial_blocks'
DISPLAY_MAX = 0.3
PREPARED_MANIFEST = Path('/kaggle/input/YOUR_DATASET/manifest.jsonl')
SUITE_ROOT = Path('/kaggle/working/trust_moe_india_2000')  # or 4000 / 6000
```

Use the updated TrustMoE training notebook. Preserve the collection config,
candidate sites, scene lists and fixed_selection.json. A seed alone cannot preserve a
live satellite catalog forever. More data can change model quality; it does not guarantee
that the 6,000-pair model is best. Only held-out results can establish that.
''')
code('''
DOWNLOAD_TO_BROWSER = False
DOWNLOAD_SIZE = 2000
if DOWNLOAD_TO_BROWSER:
    from google.colab import files
    files.download(str(OUTPUT / f'india_pairs_{DOWNLOAD_SIZE}.zip'))
''')


def build():
    for i, cell in enumerate(cells):
        cell['id'] = f'gee-india-{i:03d}'
    path = ROOT / 'colab/GEE_India_Landsat_Sentinel_Fixed_Datasets.ipynb'
    notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3',
        language='python', name='python3'), language_info=dict(name='python', version='3.11')),
        nbformat=4, nbformat_minor=5)
    path.write_text(json.dumps(notebook, indent=1) + '\n', encoding='utf-8')
    print(path)


if __name__ == '__main__':
    build()
