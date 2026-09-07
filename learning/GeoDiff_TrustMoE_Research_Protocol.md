# GeoDiff-TrustMoE: research protocol and implementation notes

Revision: 2026-09-07. Branch: `3x-continued`.

## What this revision does

It adds a separate, diffusion-free model. The old GeoDiff-GAN implementation and
checkpoints remain available for comparison; they are not overwritten or silently
converted. A frozen residual-Swin reconstruction base is followed by direct regional
residual experts, a contextual Transformer router, and an RGB trust gate.

There is no diffusion sampling, VAE, latent mapper, second Swin pass, or PixelShuffle
in this model. Removing the denoising loop removes its repeated forward passes; it
does **not** prove that the new model is faster or more accurate. Both are measured.

## Research question, not a novelty certificate

Can a direct, base-error-guided regional residual MoE improve structural fidelity
without meaningful PSNR/SSIM degradation, while requiring less measured inference
time than a matched all-region residual MoE?

The following prior work makes broad claims such as "first Transformer MoE for
satellite SR" or "first distortion-guided expert router" indefensible:

| Primary source | Existing overlap | Consequence for our claims |
|---|---|---|
| [Swin2-MoSE, arXiv 2024 / journal version](https://arxiv.org/abs/2404.18924) | Transformer-block MoE, learned expert merging, remote-sensing SR including OLI2MSI | Adding attention and experts is not itself a contribution. |
| [Heterogeneous MoE for RS-SR, 2025](https://arxiv.org/abs/2502.09654) | Heterogeneous expert groups and pixel-wise dual routing | Scene diversity and per-location expert selection already have precedent. |
| [Distortion-Aware Routing and Parameter-Shared MoE, 2026](https://doi.org/10.3390/s26072186) | Frequency/edge/noise cues, distortion-conditioned expert assignment, efficient multispectral restoration | This is especially close; a full-method comparison and matched reproduction are needed before a novelty claim. |

Our narrower hypothesis combines a frozen reconstruction reference, an explicit
regional execution budget, actual gathered-tile expert execution, and supervised
RGB residual acceptance. No exhaustive search proves this combination is unique.
The current notebook supplies internal ablations, **not reproductions of these
external papers**. Add matched external baselines before claiming SOTA.

## Forward path

```text
Actual Landsat LR ----------------------> residual-Swin base B ------------------+
       |                                      |                               |
       +--> LR/base/HF/discrepancy cues --> shared LR encoder                   |
                                               |                               |
                                    pool into LR-region tokens                 |
                                               |                               |
                                    contextual Transformer                     |
                                        /              \                       |
                             predicted difficulty   expert logits              |
                                        |              |                       |
                              select spatial budget + top-k                    |
                                               |                               |
                               gather only selected tiles + halos              |
                                               |                               |
                                    efficient RGB experts                      |
                                               |                               |
                                      weighted tile assembly R                 |
                                               |                               |
                         LR features + R + discrepancy --> RGB trust T         |
                                               |                               |
                                   SR = clamp(B + T * R, 0, 1) <----------------+
```

For region j, A_j is zero when bypassed, and g_jk are normalized selected expert
weights. R_j = A_j sum_k(g_jk E_k(F_j)). The default region is 8x8 LR pixels,
with a four-LR-pixel halo on every side. At 3x scale its output core is 24x24 HR.
Only the core is assembled; halos supply convolutional context.

The model gathers tiles per selected expert and runs those gathered tiles in
bounded batches. It does not execute all experts on a full feature map and mask
the outputs afterward. Expert-forward hooks test this property. The encoder,
token router, base, and trust head still run densely. Halo duplication, small
kernel launches, attention and scatter costs must be included in timing.

Spatial bypass is **blockwise**, not arbitrary independent pixel dispatch.
The trust gate is spatial and RGB-specific. No positivity or improvement guarantee
follows from T in [0,1]: a small harmful residual can still reduce PSNR. At zero
coverage or zero residual scale, all experts are skipped and output equals B exactly.

## What the router knows

It receives LR observations, the base projected to the LR grid, base high-frequency
energy, base-to-LR discrepancy, and LR local variation. It never receives Sentinel HR
at inference. Its difficulty target during training is masked log-MSE of the frozen
base in each region. This is predicted reconstruction difficulty, **not a calibrated
probabilistic uncertainty estimate**. Cross-sensor date or radiometric mismatch can
also cause high error; the router cannot prove it has found missing recoverable detail.

The top-k expert choice is discrete. Selected softmax weights use detached forward
normalization as a surrogate gradient so top-1 is not gradient-dead. A load-balance
auxiliary objective and random route exploration encourage expert use. These do not
guarantee specialization or prevent collapse. Inspect per-class assignment tables
and routing maps; do not name experts urban/forest/water without evidence.

The optional adaptive-K experiment uses a declared rank rule allocating one through
TOP_K slots among selected regions. It is not a learned optimal stopping policy.
Inactive regions always use zero experts. With TOP_K=1 it reduces to fixed-K routing.

## Training and losses

1. Train the deterministic base with masked MSE and a small Charbonnier term.
2. Freeze that exact EMA-selected base. Train encoder, router, experts and trust
   directly in image space. There are no separate diffusion/joint stages.
3. Select each checkpoint by full-frame validation PSNR. Run regional-budget
   ablations on validation only, then record immutable test checkpoint copies.

Residual objectives are masked MSE, Charbonnier, SSIM, gradient, Haar-wavelet and
low-frequency radiometric error; positive local excess-MSE over the base; an ungated
proposal objective; local trust-label regression; difficulty regression; and load
balance. The separate adversarial experiment adds a small conditional PatchGAN loss.
The default profiles have no discriminator or learned natural-image perceptual loss.
All weights are written to each experiment's JSON config and component losses are
logged every epoch. Their numerical weights alone do not measure their gradient influence.

For channel c, the trust target is a 9x9 local ridge-stabilized coefficient:

```text
T*_c = clip(sum(mask * R_c * (HR_c - B_c)) /
            (sum(mask * R_c^2) + ridge), 0, 1)
```

The implementation uses local averages rather than sums and a matching fixed ridge.
Labels are detached from autograd. The trust predictor, not the oracle, is used at
validation/test. The local excess penalty discourages worse-than-base corrections;
it is not a mathematical constraint that every patch must improve.

An epoch is one complete shuffled pass, including the last partial batch. AMP,
gradient clipping and EMA are supported. Checkpoints include optimizer, scaler,
EMA, epoch and optional discriminator state. Increasing epochs is allowed. Training
resumes from the last completed epoch; an interrupted partial epoch is replayed with
epoch-addressed RNG. Changing batch/crop/architecture/loss/data starts another run.
There are no training-minute, maximum-batch, or optimizer-update budget controls.

## Dataset and numeric protocol

Use the new Kaggle preparation notebook for actual Landsat Collection 2 L2 and
Sentinel L2A products. It retains the existing pairing, QA masks, date-gap checks,
geospatial grid alignment, spatial split guards, and quarantine log. It does not
create LR by downsampling HR. Registration error and temporal change remain possible;
inspect the pairs rather than assuming reprojection makes them perfectly aligned.

Exported files are portable `split/scene_class/tile/pair/*.npz`, each containing LR,
HR and masks. A separate image copy is unnecessary. The target split is 78/10/12;
the export checks nonempty class splits and at least 10% test per represented class.
Spatial guards remove overlapping cross-split patches. The training audit repeats
geometry, finite/range, duplicate-content and within-tile isolation checks.

Within-tile spatial tests measure held-out areas of seen tiles, not transfer to unseen
geographic regions. Multiple date pairs in one tile are not independent sites.
Category directory names are weak scene labels; a tile can contain mixed land cover.
For generalization claims add disjoint held-out tiles and inspect geographic overlap
between neighboring tiles. Those claims are not established by this notebook.

Training uses aligned random 64->192 crops by default. Validation/test use entire
stored 128->384 tile patches, or full 160->480 frames for an already-prepared OLI2MSI
manifest. No automatic scaling or quantization is applied by the new trainer.
Do not compare raw-reflectance PSNR with clip(0,.3)/.3 or quantized benchmark PSNR.
DISPLAY_MAX affects visualizations only: .3 for raw reflectance, 1 for normalized
benchmark tensors. Shared display stretch must not be mistaken for calibration.

## Experiment matrix and interpretation

| Experiment | Question |
|---|---|
| Bicubic / shared base | Does the learned correction beat unchanged input interpolation and its own base? |
| Single expert, full coverage | Is MoE better than a direct residual CNN control? Capacity is not matched. |
| Transformer MoE, full coverage | Does actual spatial sparsity save time relative to the same expert bank? |
| Uniform spatial coverage | Is predicted difficulty better than distributing the same budget uniformly? |
| Local router | Is global token attention needed? This control has different router capacity. |
| Sparse Transformer | Does the proposed combination improve the quality/time trade-off? |
| No trust gate | Does acceptance help fidelity beyond the local excess loss? |
| Adversarial loss | Does a small adversarial term improve structure, or hurt fidelity? |
| Adaptive regional K | Does rank-based variable expert count offer a useful budget trade-off? |

All controls use the same training pairs, seeds, epochs, crop size and base. This is
an equal-epoch ablation, not an equal-FLOP or equal-time study. Run at least three
seeds for a final analysis, examine convergence, and report a stronger deterministic
baseline if the present base is undertrained. A better data mix does not theoretically
guarantee superiority over another model.

## Evidence to keep and claims to avoid

Report all test-pair L1, PSNR, SSIM, Edge F1, ERGAS, RGB SAM, UIQI, sCC, and LR
redegradation error. RGB SAM is not a full multispectral reconstruction result.
Cross-sensor redegradation is a diagnostic, not exact physical observation consistency.
QNR needs the appropriate PAN/reference setup and is not computed here.

Save per-image and per-class metrics, paired bootstrap differences versus base and
controls, and tile-cluster intervals when multiple tiles exist. Pair bootstrap can
be overconfident for neighboring patches. One tile cannot establish a population-level
confidence interval. Inspect Edge-F1 threshold sensitivity, seam artifacts, routing
collapse, spectra, failures and qualitative examples before making a paper claim.

Validation budget screening uses the declared margins -0.05 dB PSNR, -0.001 SSIM and
positive Edge-F1 improvement. Those screens are descriptive and subject to selection
bias; the locked test comparison is the confirmation step. Test failures stay in the
report. If the test set has already influenced prior model development, disclose that
it is no longer a pristine holdout and obtain additional unseen data.

Timing is a synchronized resident-model benchmark with warm-up, mean/median/p95,
AMP/GPU/version/shape records and allocator peak memory. The full model remains resident
even during its base-only timing; compare memory using that stated scope. Parameter
counts distinguish total modules from experts used on a particular input. Top-k
parameter fractions are not FLOPs estimates. No unmeasured speedup is asserted.

Download the result ZIP and executed notebook. The ZIP retains model/config lineage,
test decisions, source, metric arrays, plots, example inputs and best/last checkpoints;
it excludes full raw/prepared datasets and credentials. Later paper writing must be
based on those measured outcomes, including negative ablations, not the target result.
