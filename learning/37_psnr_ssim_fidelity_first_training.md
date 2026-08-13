# Fidelity-First Landsat 30 m to Sentinel-2 10 m Super-Resolution

## 1. Research decision

The deterministic SwinIR base has consistently produced higher PSNR and SSIM than
the complete diffusion-GAN output. The current acceptance criteria are therefore:

1. Final validation PSNR must exceed base PSNR.
2. Final validation SSIM must not decrease.
3. Improvement must hold on geographically isolated tiles.
4. Test data must not select losses or inference settings.

The architecture is unchanged. SwinIR, residual VAE, latent diffusion, GeoMapper,
evidence gate, and residual decoder remain present. Only objectives, optimization,
checkpoint selection, and inference controls change.

## 2. Why MSE is required

For images in `[0, 1]`:

```text
PSNR = -10 log10(MSE)
```

Minimizing MSE directly maximizes PSNR. Charbonnier is robust but behaves more like
L1. SSIM compares local luminance, contrast, and structure, so explicit `1 - SSIM`
is retained.

| Loss | Purpose | Initial weight |
|---|---|---:|
| Charbonnier | Robust pixel reconstruction | 1.0 |
| MSE | Direct PSNR objective | 50.0 |
| Multiscale MSE | Native and coarse fidelity | 25.0 |
| `1 - SSIM` | Local structural fidelity | 0.5 |
| Radiometric | Per-band mean and contrast | 0.1 |
| Residual supervision | Correct high-frequency correction | 0.5 |
| Base guard | Penalize regression below base | 100.0 |
| Gradient | Weak edge support | 0.05 |
| Wavelet | Weak frequency support | 0.01 |

Weights are not directly comparable because raw loss magnitudes differ. Inspect
their weighted contributions before tuning.

## 3. Disabled synthesis pressure

These terms are zero during fidelity-first training:

- **Adversarial:** can reward realistic but incorrect texture.
- **Perceptual:** does not directly optimize pixel fidelity.
- **Cross-sensor consistency:** the current blur/downsample operator does not model
  all OLI versus MSI spectral and radiometric differences.
- **Training back-projection:** an uncalibrated operator can pull a correct
  Sentinel-like output toward Landsat radiometry.

They are retained for later one-variable ablations.

## 4. New safeguards

### 4.1 Base guard

For each sample:

```text
L_guard = max(0, MSE(final, HR) - MSE(base, HR) + margin)
```

It is zero when the final branch improves the frozen base and positive when the
branch causes regression.

### 4.2 Residual supervision

The required correction is:

```text
target detail = high_pass(Sentinel HR - SwinIR base)
```

The evidence-gated residual is supervised against this target instead of receiving
only an indirect final-image loss.

### 4.3 Improvement-aware confidence

The evidence target compares local squared error before and after adding ungated
detail. Confidence should be high only where the candidate improves the base.

### 4.4 Conservative joint learning rates

The joint learning rate is `1e-5`, multiplied by:

```text
diffusion 0.25, LR encoder 0.50, GeoMapper 1.00, decoder 1.00
```

Base and VAE remain frozen. Full unfreezing is an ablation because it can destroy
the strongest existing reconstruction.

Validation PSNR also drives `ReduceLROnPlateau`: after two non-improving checks,
all module-group learning rates are multiplied by 0.5 down to `1e-7`. This permits
coarse early movement and smaller late corrections without a long fixed schedule.

### 4.5 Sampled validation

Old joint validation denoised a latent derived from the true target residual. Real
inference starts from noise. Validation can now run fixed-seed DDIM sampling and
checkpoint on sampled `val_psnr`. It also reports:

- `val_psnr_gain_vs_base`
- `val_ssim_gain_vs_base`
- `val_base_improvement_rate`

Require positive gains and an improvement rate clearly above 0.5.

## 5. Inference ablations

### Residual strength

```text
output(alpha) = base + alpha * (final - base)
alpha in {0, 0.25, 0.5, 0.75, 1}
```

- Best `alpha=0`: generated detail has no fidelity value.
- Best `0<alpha<1`: residual direction helps but magnitude is excessive.
- Best `alpha=1`: full residual is calibrated.

### Back-projection

Use zero steps by default. Compare `0`, `1`, and `3` on validation only. Falling
PSNR with more steps demonstrates forward-model mismatch.

### Sampling

After selecting alpha, compare DDIM steps `{10, 20, 50}` and sample counts
`{1, 4, 8}`. Averaging may improve PSNR by suppressing stochastic texture but can
smooth structure. Retain the cheapest statistically equivalent setting.

## 6. Data improvements with greatest leverage

No loss overcomes incorrect supervision. Inspect and filter:

1. Prefer same-day pairs and report them separately from 1-3 day pairs.
2. Require cloud, shadow, saturation, and validity masks from both sensors.
3. Estimate registration from Landsat and downsampled-Sentinel gradients.
4. Apply only physically justified OLI-to-MSI bandpass adjustment.
5. Quarantine large per-band bias, contrast, or regression outliers.
6. Exclude fire, flood, harvest, construction, and moving-cloud changes.
7. Report urban, agriculture, water, forest, and barren land separately.

Do not independently normalize each target patch or tune harmonization on test
data. That can artificially increase PSNR and break reflectance interpretation.

## 7. Experiment sequence

1. Re-evaluate the existing joint checkpoint with projection zero and all alpha
   values.
2. If every nonzero alpha loses, train the new fidelity experiment in its separate
   output directory.
3. Train base with MSE and SSIM until validation PSNR plateaus.
4. Train VAE/mapper/decoder with residual supervision.
5. Train diffusion, then use its lower joint learning rate.
6. Joint fine-tune without GAN, perceptual, consistency, or projection.
7. Select by sampled validation PSNR and verify SSIM.
8. Tune alpha and sampling on validation.
9. Evaluate the selected configuration once on untouched test tiles.
10. Reintroduce one synthesis loss at a time only after final beats base.

## 8. Interpretation

Do not move to a larger model merely because final loses to base. If the best alpha
remains zero, investigate sampled residuals and pair quality first. A valid research
outcome may be that SwinIR is the fidelity model while diffusion-GAN belongs to a
separate perceptual mode. The complete system is not superior reconstruction until
held-out PSNR and SSIM demonstrate it.
