# Kaggle Usage

Import `GeoDiff_GAN_Kaggle.ipynb` into Kaggle, enable Internet, select a GPU
accelerator, and attach a dataset containing extracted Sentinel-2 L2A `.SAFE`
directories.

The notebook is configured to clone
`https://github.com/shashankjs2002/SI-SR-1.git`.

The notebook:

1. Clones this repository without modifying Kaggle's CUDA PyTorch installation.
2. Finds `.SAFE` products below the selected `/kaggle/input` directory.
3. Assigns configurable SAFE prefixes to validation and test, then extracts cloud-filtered RGB
   patches into `/kaggle/working`.
4. Quarantines zero-filled edge/corner patches without deleting them.
5. Visualizes five accepted samples with clean and noisy LR inputs.
6. Visualizes the exact samples removed by the edge/corner filter.
7. Optionally captions patches with `Qwen/Qwen3-VL-8B-Instruct`.
8. Writes runtime YAML with calibrated mild degradation and optional tensor diagnostics.
9. Runs one or all stages using one GPU or `torchrun`, with compact progress and automatic
   same-stage checkpoint resume.
10. Displays per-stage training curves and detailed intermediate visualizations.
11. Evaluates validation and test separately, then exports checkpoints and metrics.

The notebook separates run length from model capacity:

- `FAST_DEV_RUN` controls data limits, epochs, sampling steps, and diagnostic frequency.
- `MODEL_SIZE` selects `"xs"` (0.765M), `"small"` (12.14M), `"medium"` (21.13M), or
  `"large"` (81.86M).

It defaults to small for both development and non-dev runs. Small retains the large architecture's
modules, four-level diffusion hierarchy, 768-dimensional SigLIP conditioning, training stages,
losses, gates, projection, and uncertainty logic with narrower channels. Select XS explicitly for
the fastest pipeline-only check.

The first cell also exposes:

```python
TEST_SAFE_PREFIXES = ["CHHATARPUR1"]
VAL_SAFE_PREFIXES = ["CHHATARPUR2"]
UNMATCHED_SAFE_SPLIT = "train"
VALIDATE_EVERY = 1
VALIDATION_LIMIT = 4 if FAST_DEV_RUN else 64
EVALUATION_LIMIT = 4 if FAST_DEV_RUN else 100
AUTO_RESUME_TRAINING = True
TRAINING_PROGRESS_MODE = "compact"
PROGRESS_UPDATES_PER_EPOCH = 2
TRAINING_DIAGNOSTICS = False
```

`compact` emits only milestone updates and one summary per epoch. Use `tqdm` for live notebook
bars or `quiet` for minimal logs. Training diagnostics are disabled by default because per-batch
tensor statistics can exhaust Kaggle's cell output; use the dedicated diagnostic cell instead.

With `AUTO_RESUME_TRAINING=True`, rerunning the training cell finds the numerically newest
`<stage>_epoch_*.pt` under that stage's run directory and continues from the next epoch. Checkpoints
include generator, discriminators, both optimizers, and AMP scaler state and are written atomically.
This resumes only files still present under `/kaggle/working`; save a notebook version or Kaggle
Dataset before a session expires.

Run and generated-config directories include `MODEL_SIZE`, for example `runs/small/base`. This
prevents automatic resume from loading a shape-incompatible checkpoint from another capacity.

Prefix matching is case-insensitive. `MAX_TILES` limits only unmatched/training SAFE products;
explicit validation and test products remain selected. If an old manifest lacks `source_product`,
delete the old manifest and patch directory once and rerun preparation.

`RUN_DATA_PREPARATION=True` scans for newly attached SAFE products on every run. Previously
completed products are skipped using `/kaggle/working/geodiff-gan-output/preparation-state.json`.
Set it to `False` to reuse the current manifest without scanning. If the manifest does not exist,
the notebook prepares it automatically. Set `REBUILD_PATCHES=True` only when deliberately changing
patch extraction settings and regenerating selected products.

If the Kaggle dataset contains `CITY_...SAFE/S2...SAFE`, the outer directory is a label wrapper,
not a second product. The notebook reports both the raw `.SAFE` directory count and the canonical
product count, ignores wrappers for extraction, and still uses the outer city label for split
prefixes.

Set `SENTINEL_INPUT` to the exact dataset root. Using `/kaggle/input` scans every attached dataset,
which can legitimately produce more canonical products than the number in one dataset.

Do not change `MODEL_SIZE` between training stages in one run. Stage-to-stage checkpoint transfer
requires identical architecture dimensions.

The preparation, filtering, and visualization cells reconstruct their required
paths after a Kaggle kernel restart. Run the clone/install cell again only when
the package is no longer importable or when pulling newer repository code.

The diagnostic cell displays stage intermediates, activation histograms, Fourier spectra, policy
overlays, edge/wavelet comparisons, diffusion and projection trajectories, and stores a compressed
tensor archive for the selected patch.

A single MGRS tile can exercise the pipeline in `FAST_DEV_RUN`, but all patches
will remain in one tile-level split. Research runs require enough geographically
separated SAFE products to populate train, validation, and test splits. Prefix rules are rejected
when they assign one MGRS tile to multiple splits.

Super-resolution is not a classification problem. The notebook reports validation/test L1, PSNR,
SSIM, edge F1, LR re-degradation error, optional LPIPS/DISTS, and uncertainty calibration metrics.
