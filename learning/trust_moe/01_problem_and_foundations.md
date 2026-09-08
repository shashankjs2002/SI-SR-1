# 01. Problem and Foundations

## The Task in Plain Language

Landsat observes a location at a coarser spatial sampling than Sentinel-2 RGB. We
learn to predict a Sentinel-like image from an actual Landsat observation. The goal
is not merely to enlarge the array: it is to estimate useful fine-scale structure.

The default example is RGB `128 x 128` at 30 m to RGB `384 x 384` at 10 m.
Both cover approximately `3,840 m x 3,840 m`. A 3x linear scale means nine times
as many pixels, not three times as much measured information.

## Why SR When Sentinel Exists?

Sentinel imagery supplies paired training references. A learned mapping might later
be useful when a suitable Sentinel observation is unavailable. That intended use
does not prove transfer to historical Landsat, other seasons, clouds, or new regions.
Those uses require separate tests. A reconstruction is not a new satellite measurement.

## Four Sources of Difference

| Dimension | Meaning here | Why it matters |
|---|---|---|
| Spatial | Pixel sampling, sensor blur and alignment | A boundary can shift or become mixed inside an LR pixel |
| Spectral | Different band response functions | Corresponding RGB bands do not measure identical wavelengths |
| Radiometric | Reflectance scale, atmospheric correction and illumination | Equal surfaces can have different numerical brightness |
| Temporal | Different acquisition times | Crops, water and construction can genuinely change |

Sentinel is therefore a cross-sensor reference, not a perfectly simultaneous,
spectrally identical ground truth. Reprojection aligns grids; it does not guarantee
subpixel registration or remove temporal differences.

## Why Use a Base Plus Correction?

Write `B = base(LR)` and `SR = clamp(B + T * R, 0, 1)`.
The base handles the broad reconstruction. The outer branch proposes a correction
R, and T controls how much to accept. This separates base quality from correction
quality and lets us ask whether the extra branch is actually useful.

R is a signed RGB residual, not an image of new objects. It can change brightness,
color and texture. It is **not constrained to contain only high frequencies**.
Calling it a detail branch expresses its purpose, not a strict frequency guarantee.

## What Changed From the Previous Model?

| Earlier diffusion system | Current direct model |
|---|---|
| Residual latent diffusion and repeated denoising | Direct RGB residual prediction |
| VAE and mapper stages | No VAE or latent mapper |
| Multiple stochastic samples can be averaged | One deterministic evaluation pass |
| Image-level/latent expert behavior in the previous experiments | Regional expert dispatch |
| Sampling variance diagnostics | Predicted base difficulty and trust, not sample variance |

The old code remains in the repository for comparisons. The current notebook does
not instantiate that old pipeline. Removing diffusion does not imply that all CNNs
were replaced with Transformers: the base uses window attention, the router uses
token attention, and the experts remain small CNNs.

## Understanding the Data Values

A raw reflectance of 0.12 differs from 0.40 after dividing by 0.3. PSNR depends on
the error scale and declared data range. Never compare results with different
normalization, quantization, crops or masks as if they used the same protocol.
Display stretching is only for viewing; metric arrays must remain unchanged.

## Check Yourself

1. Why does 128-to-384 preserve the geographic footprint?
2. Can a sharper boundary still be in the wrong place?
3. Why can the LR/HR pair differ even after both are georeferenced?
4. Does the residual branch have access to the target at inference? No.

Next: [Architecture and shapes](02_architecture_and_shapes.md).
