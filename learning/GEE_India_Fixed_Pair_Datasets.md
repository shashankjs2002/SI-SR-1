# Fixed India Landsat/Sentinel datasets from Earth Engine

## Faster Alternative: Export Overlaps Once

Use [the tile-first Colab notebook](../colab/GEE_India_Tile_First_TIFF_Datasets.ipynb)
when per-patch Earth Engine requests are too slow. It exports a large scene-overlap
rectangle for each chosen Landsat/Sentinel pair, then uses local TIFF windows to
create patches. Three batch tasks per overlap replace per-patch image requests.
The third export supplies land-cover labels; sensor TIFFs include a validity band.
Final paired TIFFs contain RGB with an internal validity mask.

It is not a raw full-scene archive: only the common overlap is relevant, Sentinel
is aligned to the native Landsat grid, and pixels outside overlap/India are masked.
Budget substantial Drive space; batch jobs can queue, so no speedup is guaranteed.
The notebook shows an uncompressed storage estimate before submitting jobs.

Use a new OUTPUT. This changes the sampling population and must not be mixed with
an existing point-first experiment. Scene plans and task IDs are saved; cropping
processes only the ready prefix of the plan so export completion order cannot
change selection. Rerun setup, queue monitoring and cropping after interruption.
The same 600 test and 200 validation pairs are shared across the new three sizes.
Insufficient class quotas cause a clear shortfall, never duplication. Existing
NPZ-based training code still needs a TIFF-aware loader for either TIFF workflow.

Source: [Earth Engine batch exports and explicit grids](https://developers.google.com/earth-engine/guides/exporting_images).

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
master/                    accepted LR/HR GeoTIFF files
accepted/                  one metadata receipt per completed pair
rejected/                  quality/guard rejection reasons
fixed_selection.json        exact IDs for all three datasets
india_pairs_2000_geotiff.zip
india_pairs_4000_geotiff.zip
india_pairs_6000_geotiff.zip
```

After a session restart, rerun setup/authentication and collection using the SAME
OUTPUT. Completed pairs are skipped. Failed requests are retried with backoff;
repeated failures stop with previous work preserved. Files are written through
temporary files before their completion receipts. No cleanup deletes data.

A seed alone cannot freeze a changing satellite catalog, so preserve the cached
coordinates, scene lists and final selection. Changing collection settings requires
a different output root. If an accepted TIFF is missing, restore it from backup.

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
is applied. Master storage also uses TIFFs: `master/LR/` and `master/HR/`.
There is no NPZ export or NPZ cache in this workflow. Downloaded arrays are
processed in memory before being written as TIFF.

Use the new default OUTPUT `gee_india_tiff_v1` when starting this version. An
older collection containing NPZ receipts is rejected rather than silently reused.
No old files are deleted. Preserve old collections separately.

Interrupted collection skips pairs only after both TIFFs and their metadata receipt
have been written. Interrupted ZIP export can be rerun without collecting again.
ZIPs contain TIFF imagery and JSON metadata only. They duplicate common holdouts
intentionally so each of the three datasets is self-contained.

The existing TrustMoE training notebook currently expects NPZ manifests. It cannot
directly read this TIFF layout; a TIFF-aware paired loader is required. Do not point
that notebook at `pairs.jsonl` and assume compatibility. No optional NPZ export is
provided in this version.

Keep the same fixed train/val/test IDs in any downstream loader. Use separate run
roots per dataset size and select settings on validation. Full epochs on larger
datasets imply more optimizer updates, so report the additional training cost.
Numeric metrics must use stored reflectance, not enhanced display images.

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
