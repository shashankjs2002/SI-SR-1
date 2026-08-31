# Residual-base and mixture-of-experts experiments on our tile pairs

## Scope

Branch: `3x-continued`. Dataset: our real Landsat 8/9 C2 L2 and Sentinel-2 L2A
products. This experiment does not use OLI2MSI, RDN, captions, or a second base pass.
The new notebook is `kaggle/GeoDiff_GAN_Kaggle_Tile_Residual_MoE_3x.ipynb`.

This is an implementable research hypothesis, not a claim that 35 dB or publication
novelty has been achieved. No new satellite test results have been measured locally.

## Experiment matrix

| Run | Base | Diffusion | Router supervision | Final residual acceptance |
| --- | --- | --- | --- | --- |
| Residual base | New residual-group Swin-style base | Not used for the output | None | Zero residual, baseline only |
| Single expert | Same frozen base | Shared UNet + one small velocity head | Denoising | Existing evidence gate |
| Generic MoE | Same frozen base | Shared UNet + N small velocity heads | Denoising + load balancing | Existing evidence gate |
| Reliability MoE | Same frozen base | Identical N-head architecture | Also base-relative quality and expert ranking | Existing gate times learned image-level acceptance |

`NUM_EXPERTS` is the number of trainable heads; `TOP_K` is the number selected for
each image at each diffusion timestep. Defaults are 2 and 1. They are different
settings. Experts do not have predefined labels such as urban/water/forest: any
specialization has to be demonstrated using utilization and error analyses.

The five earlier RGB-standard, RGB-fidelity, multispectral-fidelity, RGB-harmonized,
and multispectral-guided profiles are available as optional separate re-runs. Their
new names end in `_resize_control`. Historical files are untouched. These are not
exact historical reproductions: the new controls use a smaller shared UNet, disable
natural-image perceptual and GAN losses, and use resize-convolution throughout.
Never combine historical scores with the new results as if the protocol were identical.

Set `NEW_MULTISPECTRAL=True` to conduct the new four-way study with six-band
Landsat conditioning. All four then share that same input setting. The deterministic
base still receives RGB only. Use another suite root for a separate RGB-vs-MS study.

## What changed in the base?

The previous compact base already had residual connections. Simply adding a skip
connection is not a new research contribution. The opt-in `residual_swin` base has:

- local attention and MLP residual connections;
- residual groups with scaled convolutional residuals;
- a global shallow-feature skip;
- shifted-window attention masks and learned relative-position biases;
- a zero-initialized RGB correction over a bicubic anchor;
- resize-convolution rather than PixelShuffle.

This is a compact Swin-style implementation, not a claim of reproducing official
SwinIR weights or architecture exactly. Zero initialization makes the initial base
equal to bicubic; the base then learns both radiometric and spatial corrections.

The optional resize-convolution setting also covers the residual VAE and diffusion
UNet, not just the RGB decoder. Every model created by the new notebook is checked
for **zero `torch.nn.PixelShuffle` modules**. Legacy constructors retain their old
defaults solely so old checkpoints can still be instantiated correctly.

## Where are the experts?

Gaussian forward noising remains unchanged:

`z_t = alpha_t * z_0 + sigma_t * epsilon`, with `epsilon ~ N(0, I)`.

The experts predict corrections to a shared **velocity prediction** in the reverse
denoising model. They do not choose arbitrary noise distributions. Changing forward
noising without changing the target and reverse sampler would invalidate this model.

The shared UNet runs once per denoising step. Each head is a small 1x1/3x3 convolutional
adapter. Training evaluates all heads for auxiliary supervision. Inference dispatches
only the selected heads for each batch item. The shared UNet is still the dominant
cost: top-1 does not imply an N-fold speed improvement.

The router is a separately parameterized trainable module inside the diffusion
module. Its inputs are pooled LR features, timestep embeddings, raw Landsat RGB or
six-band values, base high-frequency energy, and downsampled-base minus Landsat
RGB. That last quantity is a model discrepancy feature, **not** proof that the base
is wrong: different sensors can genuinely have different radiometry.

No Sentinel target, target error, tile identifier, or validation/test label enters
the router's inference inputs. The generic and reliability variants use the same
input features and have matching parameter counts, isolating the extra supervision
and acceptance mechanism rather than giving one model privileged inference data.

## How is reliability learned?

For each training image, timestep and expert, decode a candidate residual with the
frozen mapper/decoder. Compute its masked MSE and the frozen base's masked MSE.
Let `g_i = MSE(base, target) - MSE(candidate_i, target)`.

- Quality target: `sigmoid(g_i / temperature)`, supervised using BCE-with-logits.
- Ranking target: `softmax(-MSE(candidate_i, target) / temperature)`.
- Router score: selection probability multiplied by predicted quality, normalized
  before top-k selection.
- Final acceptance: selected qualities averaged using the final-timestep routing
  weights. This multiplies the existing spatial evidence gate.

All oracle targets are detached and use only **training** HR. Invalid pixels are
excluded. No reconstruction gradients pass through oracle labels. All experts also
receive a weak velocity loss; a warm-up epoch blends all heads, and load balancing
discourages collapse. Dead experts are reported rather than hidden.

The output has one base pass:

`Y = clip(B + A_image * E(x,y) * high_pass(R(x,y)), 0, 1)`.

`A_image` is image-level router acceptance, not a per-pixel expert mask. `E` is the
existing spatial evidence map. This implementation must not be described as local
per-band trust, pixelwise MoE dispatch, or a new physical sensor uncertainty model.

**Important approximation:** router targets during diffusion training are decoded
single-noisy-timestep proposals, not full DDIM trajectories. During joint refinement,
the mapper/decoder instead receive actual frozen-diffusion 8-step samples. The router
and LR encoder remain frozen then. Decoder changes can make prior quality estimates
less calibrated. Test this with final utilization/acceptance and base-relative scores;
a sampled-router recalibration stage is future work, not an implemented guarantee.

## Dataset preparation and splits

The notebook calls the existing `geodiff_gan.cli.prepare_landsat_sentinel` CLI. It
discovers products, matches date and footprints, scales surface reflectance using
metadata, combines cloud/invalid masks and projects both observations onto aligned
grids. It stores actual Landsat LR, not downsampled Sentinel substitutes.

- Stored paired patches: 128x128 at 30 m -> 384x384 at 10 m.
- Random training crops: 32x32 -> 96x96 by default; use 64 -> 192 for a later run.
- Masks, RGB and all additional bands receive exactly the same crop/flip/rotation.
- Cropping is random on each access; unused pixels are not permanently discarded.
- Validation/test use whole stored patches, never random training crops.
- A fixed, seeded tile-interleaved manifest prevents capped validation from using
  only the first city's patches. It keeps every record and preserves all split labels.
  Stage validation uses at least one image per tile. Final validation screening defaults
  to 64 patches for time savings; the real-run test limit is independently `None` (all).
- Spatial train/val/test regions occur in every tile, with guard exclusions between
  splits. Approximate target fractions are 80/10/10. Guard regions and valid masks
  change the exact counts. The split validator fails if a tile has insufficient coverage.

This is within-tile spatial generalization, not unseen-city generalization. Nearby
patches can still be correlated. Before any publication claim, add multiple tiles,
multiple seeds and a genuinely held-out geographic evaluation.

Landsat multispectral conditioning uses `[R,G,B,NIR,SWIR1,SWIR2]`, all at native
30 m. This is not the earlier Sentinel-only setup that resampled some 20 m bands.
The target is always Sentinel RGB at 10 m. Cross-sensor color differences, geometric
misregistration and actual date changes can limit the achievable pixel fidelity.

## Time, checkpoint and experiment isolation

Default main training caps: shared base 50 minutes, shared VAE 30 minutes, then
40-minute diffusion plus 20-minute joint caps for each of the three residual runs.
Total nominal cap: 260 minutes, plus preparation/evaluation and short boundary
overruns. A cap is checked after a checkpoint chunk (at most 120 batches); this is
not a hard real-time guarantee of a six-hour wall-clock total.

An epoch here can be a capped shuffled subset, not a full dataset sweep. Record the
number of batches/updates, seed, wall time and early-stopping condition. The random
loader is recreated/reseeded each chunk. Generic and reliability runs share settings,
but actual updates can differ when time caps or early stopping trigger. An equal-update
follow-up comparison is needed if the reliability run is compute-limited earlier.

Every experiment has its own manifest snapshot, config locks, stage directories,
best/latest checkpoint, completed-stage receipts, validation/test caches and figures.
The prepared arrays are shared to save disk space. SHA-256 provenance and exact
frozen-base tensor comparisons prevent the stale-base lineage issue from earlier runs.
Changing N, dataset, code or parent weights requires a new experiment root.

A completed checkpoint epoch is resumable; interruption before that save loses only
the active chunk. Kaggle itself can delete session storage: save a notebook version
with its outputs and attach that saved output to resume. The small progress-report ZIP
is intentionally not a complete training backup. It excludes credentials and raw data.
After restoring frozen manifests and NPZs, raw-product discovery/extraction is skipped.

## What should be reported?

Use identical validation/test manifests, raw reflectance range, masks, seeds, sampling
steps and sample count for all runs. The final residual scale is 1.0. Only the base-only
control uses scale 0. Report L1, PSNR, SSIM, edge F1, ERGAS, SAM, UIQI, sCC and the
delta against each model's own frozen base. LR re-degradation error is diagnostic:
cross-sensor data need not match a simplified synthetic degradation operator exactly.

Also report utilization, dead experts, acceptance distribution, memory/parameters,
actual training time, and fraction of images beating the base. The included paired
patch bootstrap is descriptive, not an independence-corrected confidence interval
for geographic generalization. The full test cell is off until validation review.

Plots use a common fixed display stretch and do not modify saved prediction arrays
or metrics. Original native LR is displayed first; equal panel sizes indicate equal
geographic footprint, not equal pixel counts. Indexed views support optional bases,
error maps and at most three panels per row. The read-only Gradio viewer does not
load models; after a kernel restart its standalone cell is sufficient if files remain.

## Novelty boundaries and relevant prior work

Residual connections, MoE routing, configurable expert counts, load balancing and
diffusion experts already exist. Do not claim that combining these alone is novel.
The hypothesis worth testing is whether **base-relative cross-sensor reconstruction
quality supervision for routed residual generation** provides a reproducible gain
over parameter-matched generic routing under the same input protocol.

Relevant prior work identified during the design discussion:

- [SwinIR](https://arxiv.org/abs/2108.10257): residual transformer restoration base.
- [Swin2-MoSE](https://arxiv.org/abs/2404.18924): remote-sensing SR with MoE and gating.
- [Switch-DiT](https://arxiv.org/abs/2403.09176): diffusion MoE and timestep/task routing.
- [MoEDiff-SR](https://arxiv.org/abs/2504.07308): diffusion SR experts with supervised selection, in MRI.
- [Distortion-aware parameter-shared MoE](https://doi.org/10.3390/s26072186): multispectral remote-sensing SR routing.

This is not an exhaustive priority search. A paper needs direct comparisons and
ablations, a precise distinction from these approaches, and measured benefits.
If the reliability branch loses to the generic branch or base, retain and report
that negative result rather than choosing test-specific scales or changing normalization.

## Local verification

- 79 unit/integration tests passed, including the new four-stage miniature training
  pipeline, both evaluation CLIs, frozen-base checks, sparse/dense expert parity,
  crop/mask alignment, balanced manifests and saved-result viewing.
- All 24 code cells in the 49-cell notebook pass Python AST parsing; execution
  counts and cell outputs are empty.
- The 62-file source ZIP was checked against the current source tree; it includes
  no checkpoints, NPZ imagery or Kaggle credentials.
- `git diff --check` passed. The existing `.gitignore` edit was left untouched.
- Full Kaggle GPU training and evaluation on the user's real tiles were not run
  locally. The browser-facing Gradio app was not launched: the local Gradio install
  lacks `python_multipart`. The notebook installs Gradio with its dependencies;
  its underlying cached-image/metadata reader was tested without a server.
