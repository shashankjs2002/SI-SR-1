# Fixed India Landsat/Sentinel datasets from Earth Engine

## What is being created?

The [Colab notebook](../colab/GEE_India_Landsat_Sentinel_Fixed_Datasets.ipynb)
downloads real Landsat 8/9 and Sentinel-2 observations through the Earth Engine
Python API. It does not manufacture LR by shrinking Sentinel images.

An "image" in the requested dataset count means a **pair**: one RGB LR array and
one RGB HR array. Defaults are 128 x 128 at 30 m and 384 x 384 at 10 m, covering
the same approximately 3.84 km square. The dates are 2023-2025, with at most three
days between sensors. These are individual observations, not temporal composites.

| Dataset | Train pairs | Fixed validation pairs | Fixed test pairs |
|---|---:|---:|---:|
| 2,000 | 1,200 | 200 | 600 |
| 4,000 | 3,200 | 200 | 600 |
| 6,000 | 5,200 | 200 | 600 |

The 1,200 training pairs are a subset of the 3,200, which are a subset of the 5,200.
Validation and test IDs are identical across all three. Thus test is 30%, 15% and
10%, respectively. This deliberately holds evaluation constant when measuring
the effect of adding training data. It is not three independent 80/10/10 splits.

## Run it

1. Commit and push the downloader changes to the `3x-continued` branch of
   `https://github.com/shashankjs2002/SI-SR-1.git`. No source ZIP upload is needed.
2. Open the Colab notebook above. CPU runtime is sufficient.
3. Set `EE_PROJECT` to your registered Earth Engine Cloud project ID. Authorize
   Drive and Earth Engine when the notebook asks. Project setup is described in
   [Google's authentication guide](https://developers.google.com/earth-engine/guides/auth).
4. Keep `OUTPUT` on Drive. Run the candidate inspection, collection and preview cells.
5. Run the ZIP export cell. Download the three ZIPs through Drive and upload each
   as a separate Kaggle dataset. An optional browser download cell is included.

Setup clones that GitHub branch into `/content/geodiff-gee-github`. On reruns it
checks the origin/branch and reuses the checkout without resetting or pulling over
local changes. The commit ID is printed. To use newly pushed code, select a new
`REPOSITORY` directory and restart the runtime if the old module was already imported.
Keep `OUTPUT` unchanged to preserve downloaded pairs.

For a configured local Python environment, the equivalent script is
[download_gee_india_pairs.py](../scripts/download_gee_india_pairs.py):

```bash
python -m pip install -e . --no-deps
python -m pip install earthengine-api requests pyproj rasterio
python scripts/download_gee_india_pairs.py --project YOUR_PROJECT --root /YOUR/PERSISTENT/FOLDER --authenticate
```

The rest of the repository's dependencies must already be installed for local use.
Do not put authentication tokens in source files or dataset archives.

## How pairing and quality control work

- RGB Landsat uses SR_B4/B3/B2 with `DN * 0.0000275 - 0.2`. Invalid, cloudy,
  shadowed, snowy and saturated observations are masked using QA_PIXEL and
  QA_RADSAT. [Landsat catalog](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2)
- RGB Sentinel uses B4/B3/B2 scaled by 0.0001. Only SCL vegetation, bare-ground
  and water classes are accepted, with a 30 m mask erosion around invalid areas.
  Harmonized surface reflectance avoids mixing the processing-baseline offset.
  [Sentinel catalog](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)
- Both sensors must have at least 95% joint valid coverage. Arrays store raw
  reflectance in [0,1], with masks; negative or greater-than-one observations are
  excluded, not stretched into valid data. Previews alone use a shared display stretch.
- Candidate scenes have limited scene cloud cover; Sentinel candidates are ranked
  by time gap before cloud percentage. This favors available clear observations,
  so the dataset is not an unbiased sample of every season or weather condition.
- No pair is accepted or rejected according to model PSNR. Temporal change and
  spectral-response differences can still occur and must be discussed in results.

The native Landsat projection, pixel origin and 30 m spacing define the LR crop.
Sentinel is bilinearly reprojected onto the corresponding 10 m subgrid. This keeps
exact 3x array geometry without upsampling Landsat before storing it. Sentinel is
therefore aligned, not an untouched native array. Specifying the affine transform
avoids the origin shifts that can arise from exporting with scale alone.
[Earth Engine grid guidance](https://developers.google.com/earth-engine/guides/exporting_images)

Grid alignment does not prove subpixel co-registration. Inspect roads, shorelines
and stable features for remaining displacement before interpreting reconstruction
errors as purely model errors. There is no automatic registration correction here.

## How fixed splits prevent leakage

Coordinates are divided into 40 km blocks in EPSG:6933. A seeded hash assigns each
block to one split before downloading pairs. Crops must lie inside their assigned
block with a 500 m guard. All acquisition dates at a location remain together.
At most two date pairs per sampled location are accepted.

Different splits cannot share one of these blocks. This is geographic-block
separation, not separation by entire Landsat scene or Sentinel tile. Within a
split, nearby crops may overlap and two dates may be correlated. Statistical
analysis should consider block-level resampling, not treat every patch as fully
independent. A single large source scene can cover multiple disjoint blocks.

The five balanced weak categories are forest, agriculture, urban, barren/sparse
vegetation, and permanent water (the internal key is `water_wetland`, but the
selected WorldCover code 80 is permanent water only). Each accepted patch needs
at least 50% coverage of its requested category. These are **WorldCover 2021 weak
labels**, not verified semantic labels at acquisition time. Mixed landscapes and
land-cover change require inspection. Do not claim the router's experts have
learned these categories merely because sampling used them.
[WorldCover catalog](https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200)

## Resume and exact counts

Persistent output contains:

```text
collection_config.json      configuration lock
candidate_sites/            fixed sampled coordinates
scene_lists/                cached acquisition candidates
master/                    accepted paired NPZ files
accepted/                  one metadata receipt per completed pair
rejected/                  quality/guard rejection reasons
fixed_selection.json        exact IDs for all three datasets
india_pairs_2000.zip
india_pairs_4000.zip
india_pairs_6000.zip
```

After a session restart, rerun setup/authentication and collection using the SAME
OUTPUT. Completed pairs are skipped. Failed requests are retried with backoff;
repeated failures stop with previous work preserved. Files are written through
temporary files before their completion receipts. No cleanup deletes data.

A seed alone cannot freeze a changing satellite catalog, so preserve the cached
coordinates, scene lists and final selection. Changing collection settings requires
a different output root. If an accepted NPZ is missing, restore it from backup.

Exact counts are conditional on finding enough valid candidates in every category
and split. If all candidates are exhausted, the script reports the shortfall rather
than duplicating pairs. Network failures can be resumed. A genuine candidate
shortfall requires a new collection root with more candidates or a broader date
window; preserve the original collection. No runtime or successful 6,000-pair
download is guaranteed. Large national sampling queries can also hit service limits.

The downloader uses small NPY requests, one patch at a time, rather than thousands
of manually launched export tasks. [Download API limits](https://developers.google.com/earth-engine/apidocs/ee-image-getdownloadurl)
and [account quotas](https://developers.google.com/earth-engine/guides/usage) still apply.

## Portable format and training

**Default export is now GeoTIFF**, in `india_pairs_<size>_geotiff.zip`:

```text
train/LR/<class>/<pair_id>.tif
train/HR/<class>/<pair_id>.tif
val/LR/<class>/<pair_id>.tif
val/HR/<class>/<pair_id>.tif
test/LR/<class>/<pair_id>.tif
test/HR/<class>/<pair_id>.tif
pairs.jsonl
pair_ids.json
dataset_card.json
```

TIFFs contain three float32 RGB reflectance bands, CRS, affine transform and an
internal validity mask. Valid zero reflectance is not treated as nodata. LR and
HR filenames match. No gamma correction, display stretching or uint8 conversion
is applied. Existing master NPZs are reused: there is no need to download again.
The new ZIP names leave earlier NPZ ZIPs untouched. Interrupted ZIP creation can
be rerun without collecting data again.

The existing training loader still requires NPZ. Optional compatible exports use
`export_datasets(OUTPUT, CONFIG, sizes=(2000, 4000, 6000))`, or CLI `--format npz`.
The following manifest/settings instructions apply to that **NPZ export**, not
directly to the GeoTIFF ZIP. Both exports contain the same fixed pair IDs.

Each ZIP has `manifest.jsonl`, `dataset_card.json`, pair IDs, source metadata and
`train/<class>/*.npz`, `val/<class>/*.npz`, `test/<class>/*.npz`.
Each NPZ contains `lr`, `hr`, `valid_mask_lr`, `valid_mask_hr`. Separate HR/LR TIFF
or PNG copies are not needed by the current model. This saves duplicate storage
and keeps the real sensor pair together. Manifest paths are relative to the ZIP root.

Master pairs and ZIPs coexist. The three ZIPs duplicate shared pairs intentionally
so each dataset is self-contained. Check Drive's account quota, not just filesystem
free space. Export one size at a time when needed; nothing is deleted automatically.

In the updated [TrustMoE training notebook](../kaggle/GeoDiff_TrustMoE_Transformer_3x.ipynb):

```python
DATASET_PROTOCOL = 'spatial_blocks'
PREPARED_MANIFEST = Path('/kaggle/input/YOUR_DATASET/manifest.jsonl')
DISPLAY_MAX = 0.3
SUITE_ROOT = Path('/kaggle/working/trust_moe_india_2000')
```

Use distinct run roots for 2000/4000/6000 and keep the same model, seeds and training
policy for data-size comparisons. Full epochs on larger data imply more optimizer
updates: report this extra training cost rather than calling the compute identical.
Select settings on validation, then report the common test once per locked experiment.
All numeric metrics use stored reflectance, not display-enhanced images. They are
not directly comparable to OLI2MSI clip03-normalized metrics without matching protocols.

## Attribution and verification status

Retain USGS, Copernicus and WorldCover attribution and review redistribution terms.
India selection uses [FAO GAUL 2015](https://developers.google.com/earth-engine/datasets/catalog/FAO_GAUL_2015_level0),
whose catalog has a non-commercial license restriction; review this before use
outside an academic/non-commercial project. The boundary is not packaged as a
separate vector file, and it is not a legal assertion about disputed borders.

Local tests exercise selection, geometry, portable exports and training-manifest
validation. Live Earth Engine collection requires your project and authorization;
it has not been executed in this workspace. More data or a Transformer router does
not by itself establish novelty, generalization or a publishable performance claim.
