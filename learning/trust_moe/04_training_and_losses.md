# 04. Training and Losses

Source: [experiment trainer](../../src/geodiff_gan/experiments/trust_moe.py) and
[residual objectives](../../src/geodiff_gan/training/trust_losses.py).

## Two Training Stages

| Stage | Updated parameters | Frozen parameters |
|---|---|---|
| Base | Residual-Swin reconstruction base | Outer encoder/router/experts/trust |
| Residual | Encoder, router, experts and trust | The selected base |

The optional adversarial profile additionally trains a discriminator. It is not
used for SR inference. There is no diffusion stage and no noisy latent training.
The current model does not use captions.

Every residual experiment imports the same selected base for a given seed. That
base must remain fixed, or comparisons confound base changes and residual changes.
The complete inference checkpoint contains the base and outer branch together;
it does not need to load separate stage files for inference.

## Current Objective Weights

| Loss | Default residual weight | Purpose and limitation |
|---|---:|---|
| Masked MSE | 100 | Penalize pixel error; directly related to PSNR at fixed range |
| Charbonnier | 0.5 | Smooth absolute-error-like reconstruction objective |
| 1 - SSIM | 0.1 | Encourage local structural agreement |
| Gradient | 0.2 | Encourage agreement in spatial changes, not just intensity |
| Wavelet | 0.1 | Compare multiband frequency responses |
| Radiometric | 0.05 | Encourage low-frequency numerical agreement |
| Local excess-MSE guard | 100 | Penalize neighborhoods worse than the frozen base |
| Ungated proposal | 0.25 | Train raw proposals even when trust attenuates them |
| Trust regression | 0.01 | Learn the detached per-band acceptance label |
| Difficulty regression | 0.02 | Learn which regions have larger base error |
| Load balance | 0.01 | Discourage concentrated expert assignments |
| Adversarial | 0, or 0.002 in its ablation | Encourage critic-compatible detail; can harm fidelity |

The base stage uses `100*MSE + 0.5*Charbonnier`. The total loss values of different
profiles are not directly comparable. A coefficient of 100 does not imply 100 times
the influence: raw loss scales and gradients differ. Inspect the logged components.

## Masks and Local Guard

Invalid reference pixels are excluded from masked reconstruction objectives. The
guard pools `(final_squared_error - base_squared_error) * mask` locally and penalizes
the positive part. It is a soft penalty, not an enforced inequality at each pixel.
The adversarial path replaces invalid pixels with the same base values in real
and fake images. These masks do not establish that every unmasked pixel is perfectly
cloud-free or temporally consistent.

## Batch Size and Scene Mixing

The notebook default batch is four. Gradients are averaged across the examples;
this is normal training, not a requirement that desert and forest choose the same
expert. Routes are predicted separately per region of each image. Scene diversity
inside a batch can help or hurt optimization depending on balance and data quality.
This trainer does not expose gradient accumulation. It performs one optimizer step
per batch, including the final partial batch, with no maximum-step or time cap.

## EMA, Checkpoints and Resume

AdamW uses zero weight decay and default learning rate 1e-4. Gradients are clipped,
CUDA AMP is available, and EMA decay is 0.999. EMA is a moving average of model
weights, not an additional model architecture. Validation evaluates EMA weights.

`last.pt` stores the most recently completed epoch, raw weights, EMA, optimizer,
scaler and optional discriminator. `best.pt` is selected by validation PSNR.
Increasing the epoch count resumes training. An interrupted partial epoch is replayed
from the previous checkpoint with epoch-addressed RNG. It is not batch-level resume.

Changes to architecture, data, loss, batch or crop require a new experiment directory.
If base training changes the parent checkpoint, do not resume an old residual run
against that new parent. Restore its original parent or start a new residual study.

## Inference Versus Validation

In evaluation mode there is no route exploration. The model performs one base pass
and one direct residual computation. HR is only used afterward to calculate metrics.
No trust oracle, captions, diffusion samples or discriminator is used to make SR.

## Check Yourself

Explain why a validation loss can decrease while a different metric worsens. Then
explain why changing training losses after resuming is not the same experiment.

Next: [Parameters and experiments](05_parameters_and_experiments.md).
