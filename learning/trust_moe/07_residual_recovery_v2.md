# Residual recovery: what changed and what this experiment can establish

## Starting evidence

The submitted `outs/trustmoe-on-oli2msi.ipynb` contains 4,970 training, 255
validation, and 100 test pairs. Its single expert trained for 15 epochs, whereas
several other profiles trained for 3-5 epochs. The experiments therefore did not
have equal training duration. The single-expert validation plateau still shows
that merely extending the shortest runs is not a sufficient explanation.

Selected values transcribed from the supplied test screenshot:

| Profile | PSNR | SSIM | Edge F1 | PSNR minus base |
|---|---:|---:|---:|---:|
| Base | 34.159966 | 0.914357 | 0.656170 | 0 |
| Single expert | 34.160665 | 0.914366 | 0.656222 | 0.000699 |
| Dense Transformer | 34.160647 | 0.914362 | 0.656192 | 0.000681 |
| Sparse Transformer | 34.160386 | 0.914361 | 0.656190 | 0.000420 |
| Sparse no trust | 34.160442 | 0.914361 | 0.656208 | 0.000476 |
| Sparse adversarial | 34.160672 | 0.914363 | 0.656214 | 0.000706 |
| Bicubic | 31.967629 | 0.878670 | 0.208575 | -2.192337 |

These gains are practically tiny. They do not establish a useful residual, routing,
or trust contribution. Calling them numerical noise is not proven without repeat
measurements: tiny changes can be deterministic. Likewise a correction smaller
than 1/255 can still affect floating-point PSNR. Use correction/error ratios,
gradients, per-image deltas, and repeated seeds, rather than one absolute cutoff.

The `NaN` standard deviations describe a one-run group. They do not measure the
spread across the 100 images. The notebook exports paired image intervals and, when
there are enough distinct identifiers, tile-cluster intervals separately.

## What code inspection established

1. The old expert processes LR-grid features, bilinearly upsamples them, then
   predicts RGB using one convolution. Its cues contain downsampled absolute
   high-frequency energy; signed HR phase is unavailable to the expert.
2. The expert's RGB projection starts with standard deviation 0.001 and its
   `tanh` output is multiplied by 0.1. This deliberately small initialization can
   make early corrections and feature gradients very small.
3. Direct proposal Charbonnier has weight 0.25; local excess-MSE guard has weight
   100 from the first epoch. Coefficients alone do not establish relative influence:
   the raw terms and their gradients differ. The new audit records both.
4. The error predictor and expert reconstruction share an encoder. Auxiliary risk
   gradients can compete with the weak reconstruction signal.
5. Base error is used to prioritize regions. Large cross-sensor error can come from
   temporal change or misalignment and may be unrepairable from LR observations.

These observations motivate changes; they do not prove which one caused the remote
run's plateau. The old checkpoint audit inspects raw versus EMA output, the actual
checkpoint epoch, gradients, expert load, and effective correction before retraining.

## Architecture

The model retains a single frozen residual-Swin base B. On selected regions it
computes a residual R, accepts it through an RGB gate T, and returns:

`SR = clip(B + T * R, 0, 1)`.

HR-conditioned experts receive three RGB cue groups: B, its signed high-pass
component, and B minus bicubic LR. They also receive features computed from LR and
the existing inference cues. Two residual blocks process LR features; a reduced
channel HR head processes the upsampled features and these nine HR cue channels.
No HR target enters the forward method. The output remains bounded by
`0.1 * tanh(...)`; the diagnostic reports the target residual's 99th percentile
so saturation can be investigated rather than raising this limit blindly.

Experts run only on dispatched tiles and halos. HR cues, the encoder, and router
are shared dense overhead. The halo is six LR pixels for two expert residual
blocks. This adds work; a lower active fraction does not promise lower wall time.
The benchmark measures the full resident forward pass, including dispatch.

The optional `expert_kind="legacy"` preserves the old expert's parameter layout.
Existing V1 checkpoints still load strictly. V2 uses `expert_kind="hr_residual"`.
There is no diffusion, VAE, second base pass, or PixelShuffle in either path.

At the default width 32 and five experts, the legacy model has 205,387 parameters
and the HR-conditioned model has 339,627. The common base is 76,531 parameters;
the new experts total 231,055 (46,211 each). These are resident model parameters,
not FLOPs or per-pixel active parameters. A local 128x128 forward produced exactly
384x384 and 256 expert-tile calls at 50% coverage with top-k=2. GPU latency remains
to be measured on Kaggle.

## Predicting useful correction

The new region-ranking target is the detached realized benefit of an ungated
proposal on valid pixels in tile j:

`gain_j = clip((MSE(B, HR)_j - MSE(clip(B+R), HR)_j) / (MSE(B, HR)_j + 1e-6), -1, 1)`.

A positive label means the current proposal reduces error. A negative label means
it damages reconstruction. This differs from asking which region has the largest
base error. The router learns to predict these labels using inference-time cues.

Bypassed regions have no observed proposal. They are excluded from gain regression,
not assigned a false zero label. Full-coverage warm-up and exploratory training
dispatch expose more regions. Evaluation uses neither exploration nor HR labels.
The gain label changes as the experts learn; it is not an estimate of the best
possible reconstruction or calibrated probabilistic uncertainty. The experiment
must measure whether it generalizes beyond these partial observations.

Expert-choice probabilities learn through reconstruction and load balance. We do
not supply semantic expert labels or compute an oracle optimal expert at inference.
The error-router control has the same model and budget as the gain-router model;
only the region-ranking training target differs.

## Training sequence

1. Import the prior trained base with strict shape and dataset checks. If it is
   unavailable, train a base. Record the parent checkpoint identity.
2. Memorize eight fixed TRAIN crops using one expert, full coverage, trust disabled,
   and no guard. Require at least +0.05 dB mean gain on those same crops and over
   half improving. This is a debugging screen, not validation evidence. Its weights
   never initialize the study and the evaluator refuses its checkpoint.
3. Train a fresh single expert on the entire designated training split. Select it
   by validation PSNR. A development screen requires at least +0.01 dB mean gain
   and more than half of validation pairs improving before the MoE study proceeds.
4. Clone that expert and shared encoder into all MoE variants, with small independent
   RGB projection perturbations to break exact symmetry. Preserve the common base.
5. Use two full-coverage, trust-bypassed warm-up epochs for MoE training. Then deploy
   the configured region fraction and ramp trust, guard, and optional adversarial
   weights over three epochs. Only epochs evaluated after warm-up can be selected.
6. Compare every enabled profile on validation and record the test plan. The primary
   model is `sparse_gain` at its trained budget. Coverage sweeps are separate ablations.
7. Evaluate the declared checkpoints on all 100 fixed test pairs once. Report failed
   comparisons as well as successes, including comparison against the single expert.

The active proposal now gets masked MSE weight 100 and Charbonnier weight 2. The
final image has MSE weight 100, with smaller structural and radiometric objectives.
The final guard weight is 5 for gated/MoE variants, and zero for the single recovery
expert. The direct ungated proposal loss prevents a closed trust gate from removing
the expert's reconstruction supervision. These are experimental settings, not
validated optimal weights.

The new configuration detaches the reconstruction encoder features at the router
input. Routing losses can train the router but cannot overwhelm this encoder.
EMA starts with a shorter effective history and approaches decay 0.99. Skipped AMP
optimizer steps do not update EMA. Compact epoch logs include correction amplitude;
history files include module gradients, loss terms, and learning rate.

## Controls and required comparisons

| Run | Question |
|---|---|
| Base | What does the common deterministic model already achieve? |
| Single recovery | Does one useful residual expert improve it? |
| Dense gain | Does adding experts help over the single recovery network? |
| Sparse error | How does the prior base-error ranking work with learned experts? |
| Sparse gain | Does predicted benefit improve allocation at the same budget? |
| Sparse no trust | Does RGB acceptance help or just suppress useful correction? |
| Sparse uniform | Does learned region selection beat fixed regional coverage? |
| Sparse CNN router, optional | Is Transformer context actually necessary? |
| Adversarial, optional | Does the added discriminator improve the desired trade-off? |
| Adaptive k, optional | Does variable expert count improve the measured budget curve? |
| Legacy expert recovery, optional | How much comes from the new expert architecture? |

All MoE controls use the same single-expert initializer, base, split, crops, seed,
epochs, and batch size. Dense and single controls deliberately use different
computation; report their parameter counts and measured time rather than calling
them equal-compute comparisons. Enabling three or more seeds is needed to measure
training variability. Latency must be measured on the same GPU and precision.

## What could be a research contribution?

The defensible research question is whether an observed-benefit predictor allocates
sparse residual computation more effectively than base-error ranking in actual
Landsat-to-Sentinel reconstruction, while a learned RGB acceptance gate preserves
fidelity. This is a hypothesis about the combination and its measured behavior.
It is not evidence of novelty by itself. Residual blocks, Transformers, MoE,
curriculum learning, warm starts, and sparse masks are established techniques.

Related primary sources checked on 2026-09-09:

- [Swin2-MoSE](https://arxiv.org/abs/2404.18924) replaces Transformer feed-forward
  processing with expert mixtures and evaluates remote-sensing SR including OLI2MSI.
- [Heterogeneous Mixture of Experts for Remote Sensing Image Super-Resolution](https://arxiv.org/abs/2502.09654)
  studies heterogeneous experts and pixel-adaptive dual routing.
- [S2Transformer](https://pmc.ncbi.nlm.nih.gov/articles/PMC12473788/)
  studies dynamic sparse region activation for remote-sensing SR. Sparse regions
  and Transformer routing alone cannot be claimed as a first contribution.
- [Distortion-Aware Routing and Parameter-Shared MoE](https://doi.org/10.3390/s26072186)
  further overlaps in routing and multispectral expert computation.

This targeted review does not certify that gain-based regional residual routing is
new. A complete related-work comparison and reproducible positive ablations are
still required before a paper claim. Multiple prior looks at OLI2MSI test results
also mean it has become a development benchmark; the planned independent geographic
dataset is important for a stronger final evaluation.

## Running and sharing

Use `kaggle/GeoDiff_TrustMoE_OLI2MSI_Residual_Recovery_3x.ipynb` after the updated
`3x-continued` source is available on GitHub. Set `OLD_SUITE_ROOT` to the attached
previous run, or provide exact `BASE_CHECKPOINTS`. Set epochs and batch size in cell
1. There are no time, step, or update limits in the notebook controls. Enable test
evaluation only after reviewing validation. A failed learning screen still allows
the final bundle cell to run.

For your own data, use
`kaggle/GeoDiff_TrustMoE_Tiles_Residual_Recovery_3x.ipynb`. It accepts raw Landsat
C2 L2 and Sentinel-2 L2A products/archives under `/kaggle/input`, or a portable
prepared manifest. Raw inputs are paired, quarantined, and divided into fixed
within-tile spatial train/validation/test regions with at least 10% test pairs.
Scene classes come from directory names or explicit tile/product overrides and are
used for reporting, not as forced router labels.

The notebook saves metrics per image, bootstrap comparisons, module-gradient audits,
correction/target ratios, routing utilization, training history, synchronized latency,
parameter counts, plots, example LR/HR TIFFs, code, and best/last checkpoints. Exact
floating-point predictions remain in result NPZs; these are evaluation artifacts,
not a new exported training-dataset format. TIFF examples are protocol-normalized
RGB and have no invented geographic CRS. Prepared input caches are excluded from
the download; recreate them deterministically when restoring the bundle.

Local tests establish execution and learnability on synthetic fixtures. They cannot
establish improved OLI2MSI PSNR, calibrated routing, GPU runtime, or publication merit.
