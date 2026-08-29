## 1. Main Difference

The previous OLI2MSI model was primarily:

```text
Landsat LR
  → compact SwinIR base
  → diffusion residual
  → GeoMapper
  → residual decoder
  → evidence gate
  → base + generated residual
```

The new model is:

```text
Landsat LR
  ├─→ Fidelity-SwinIR deterministic base
  └─→ LR encoder → latent diffusion → GeoMapper → residual decoder
                                                    ↓
                                      evidence-gated residual
                                                    ↓
                              base-referenced trust controller
                                                    ↓
                          accepted residual + deterministic base
                                                    ↓
                                      Sentinel-resolution output
```

The main objective changed from **generating plausible high-frequency detail** to **adding detail only when it is predicted to improve the deterministic base**.

---

# 2. Fidelity-SwinIR Base

## Previous Base

The old OLI2MSI model used a compact SwinIR-style base:

- Embedding width: `32`
- Transformer depth: `4`
- Relatively shallow feature extraction
- Direct SR reconstruction
- Limited capacity for cross-sensor radiometric mapping
- Its test result was approximately:

```text
PSNR: 33.765 dB
SSIM: 0.9091
```

Because the final GeoDiff output was only about `0.019 dB` better, the deterministic base was already doing almost all useful reconstruction.

## New Fidelity-SwinIR

The new base uses:

| Property | New value |
|---|---:|
| Input | `3 × 160 × 160` |
| Output | `3 × 480 × 480` |
| Embedding width | `72` |
| Transformer blocks | `12` |
| Attention heads | `6` |
| Window size | `8 × 8` |
| Residual groups | `3` |
| Blocks per group | `4` |
| Upsampling | Resize-convolution |
| Parameters | `797,337` |

### Internal Flow

```text
Landsat RGB
3 × 160 × 160
      ↓
3×3 shallow convolution
72 × 160 × 160
      ↓
Residual Swin Group 1
4 shifted-window blocks
      ↓
Residual Swin Group 2
4 shifted-window blocks
      ↓
Residual Swin Group 3
4 shifted-window blocks
      ↓
LayerNorm + 3×3 convolution
      ↓
Long residual connection
      ↓
3× resize-convolution
72 × 480 × 480
      ↓
RGB detail head
3 × 480 × 480
```

## Nested Residual Connections

There are two levels of residual connections.

### Group Residual

Each four-block group computes:

```text
group_output = group_input + Conv(SwinBlocks(group_input))
```

This improves gradient flow and lets the transformer learn corrections instead of reconstructing the complete representation.

### Global Residual

After all three groups:

```text
deep_features = shallow_features + transformed_features
```

This protects low-frequency information from being damaged by deep processing.

---

# 3. Bicubic-Anchored Reconstruction

The new base does not predict an unrestricted HR image.

It first constructs:

```text
bicubic_anchor = Bicubic(Landsat RGB)
```

Then it predicts only a correction:

```text
base_output = bicubic_anchor + learned_detail
```

The final RGB detail convolution is initialized with zero weights and bias. Therefore, before training:

```text
learned_detail = 0
base_output = bicubic_anchor
```

This is important because the model starts from a valid reconstruction instead of adding random untrained patterns.

---

# 4. Radiometric Calibration Head

Landsat OLI and Sentinel-2 MSI do not have identical spectral response functions. Even correctly aligned observations can have differences in:

- Brightness
- Channel balance
- Atmospheric response
- Illumination
- Sensor calibration

The new base contains a scene-level calibration head:

```text
shallow features
      ↓
global average pooling
      ↓
fully connected layers
      ↓
RGB gain + RGB bias
```

It predicts six values:

```text
gain_R, gain_G, gain_B
bias_R, bias_G, bias_B
```

The anchor becomes:

```text
calibrated_anchor = bicubic_anchor × gain + bias
```

The implementation constrains these values:

```text
gain = 1 + 0.1 × tanh(predicted_gain)
bias = 0.1 × tanh(predicted_bias)
```

The head is zero-initialized, so initially:

```text
gain = 1
bias = 0
```

It therefore starts as an identity transformation and learns only systematic Landsat-to-Sentinel differences.

This is safer than per-image min-max normalization, which can artificially inflate PSNR and destroy physical comparability.

---

# 5. Complete Resize-Convolution Architecture

The new publication variant contains **no PixelShuffle modules**.

Resize-convolution is used in:

- Fidelity-SwinIR base
- Residual VAE decoder
- Diffusion U-Net decoder
- GAN/residual decoder

The operation is:

```text
feature map
    ↓
bilinear spatial resize
    ↓
3×3 convolution
```

This was introduced because previous residual images showed periodic grid-like structures. PixelShuffle can create phase-dependent checkerboard or lattice artifacts when its channel groups are not learned uniformly.

The new model was verified to contain:

```text
PixelShuffle modules: 0
```

---

# 6. Residual VAE Changes

The VAE represents the detail missing from the deterministic base:

```text
target_residual = Sentinel HR − Fidelity-SwinIR base
```

Its geometry is:

```text
Residual HR: 3 × 480 × 480
              ↓
Latent:       4 × 60 × 60
```

The spatial compression factor is `8`.

The decoder now uses resize-convolution instead of PixelShuffle:

```text
4 × 60 × 60
    ↓
128 × 60 × 60
    ↓
128 × 120 × 120
    ↓
64 × 240 × 240
    ↓
32 × 480 × 480
    ↓
3 × 480 × 480
```

VAE parameters:

```text
1,727,563
```

---

# 7. Updated Diffusion U-Net

The latent diffusion model operates on:

```text
4 × 60 × 60
```

Its channel widths are:

```text
64 → 128 → 192 → 256
```

It receives:

- Noisy residual latent
- Timestep
- Landsat LR features
- Mode token
- Optional degradation information
- Optional text context, disabled in this experiment

The diffusion U-Net now uses resize-convolution during upsampling.

Its output is a velocity estimate:

```text
predicted v: 4 × 60 × 60
```

Diffusion parameters:

```text
13,790,596
```

During joint fidelity refinement, training focuses on the lower `35%` of diffusion timesteps. Extremely noisy states are avoided because deployment requires accurate reconstruction, not unrestricted synthesis from almost pure noise.

---

# 8. LR Encoder

The Landsat image is encoded at four resolutions:

| Feature | Shape |
|---|---|
| `f160` | `32 × 160 × 160` |
| `f80` | `64 × 80 × 80` |
| `f40` | `128 × 40 × 40` |
| `f20` | `128 × 20 × 20` |

These features provide:

- Landsat spatial structure
- Large-scale land-cover information
- Edges and boundaries
- Evidence that constrains diffusion
- Skip connections for the residual decoder

LR encoder parameters:

```text
1,111,296
```

---

# 9. GeoMapper and Residual Decoder

The GeoMapper combines:

```text
denoised latent + Landsat features
```

It produces:

- Spatial content: `64 × 60 × 60`
- Evidence confidence: `1 × 60 × 60`
- Edit permission: `1 × 60 × 60`
- Layer-wise style vectors

GeoMapper parameters:

```text
898,306
```

The decoder uses channel stages:

```text
64 → 48 → 32 → 24
```

and reconstructs:

```text
raw residual: 3 × 480 × 480
```

Decoder parameters:

```text
336,422
```

The raw residual is high-pass filtered:

```text
detail_residual = raw_residual − low_pass(raw_residual)
```

This prevents the diffusion branch from changing large-scale brightness and color that should remain controlled by the deterministic base.

---

# 10. Base-Referenced Trust Controller

This is the main new research component.

The previous model used:

```text
output = base + evidence × residual
```

The problem was that evidence confidence did not directly answer:

> Will this residual improve the base against the target?

A region could receive high evidence confidence even when the residual pointed in the wrong direction.

The new output is:

```text
output = base + evidence × trust × residual
```

The trust controller receives:

- GeoMapper content
- Landsat LR features
- Evidence confidence
- Candidate residual energy
- High-frequency energy of the deterministic base

It predicts:

```text
trust map: 1 × 480 × 480
```

Trust is constrained to:

```text
0 ≤ trust ≤ 1
```

It is initialized at:

```text
trust = 0.25
```

This matches the earlier experiment where a global residual scale of `0.25` performed best.

Trust-controller parameters:

```text
75,041
```

---

# 11. Local Least-Squares Trust Target

During training, the target HR image is available. Therefore, the optimal residual coefficient can be calculated locally.

Let:

```text
b = deterministic base
r = candidate evidence-gated residual
y = Sentinel target
```

The final prediction is:

```text
ŷ = b + τr
```

The locally optimal coefficient is:

```text
τ* = clip(
    mean(r · (y − b)) /
    [mean(r²) + ε],
    0,
    1
)
```

Interpretation:

- `τ* ≈ 0`: residual is harmful or points in the wrong direction.
- `τ* ≈ 0.25`: only a small part should be accepted.
- `τ* ≈ 1`: most of the generated residual is useful.
- Negative coefficients are clipped to zero because SR mode must abstain rather than reverse generated detail.

The trust controller learns to predict this optimal coefficient without having the target during inference.

---

# 12. Spatial Base-Guard Loss

The previous base guard compared one average error for the complete image:

```text
MSE(final, target) versus MSE(base, target)
```

This had a weakness. A small improvement over a large smooth area could hide severe damage around roads, buildings, or field boundaries.

The new local guard computes:

```text
local_error_final = local_MSE(final, target)
local_error_base  = local_MSE(base, target)
```

Then penalizes:

```text
ReLU(local_error_final − local_error_base)
```

This means generated detail is penalized wherever it locally performs worse than the deterministic base.

---

# 13. Training Changes

## Stage 1: Fidelity-SwinIR Base

Optimized primarily for reconstruction:

- MSE
- Multiscale MSE
- Charbonnier
- SSIM
- Small gradient loss
- Radiometric loss

The MSE weight is intentionally high because the current objective is PSNR.

## Stage 2: Residual VAE and Decoder

Learns:

```text
Sentinel target − trained base
```

The trust controller is also introduced here.

## Stage 3: Latent Diffusion

- V-prediction objective
- Diffusion checkpoint selected using minimum validation diffusion loss
- Previous fallback bug that could select the maximum validation loss was corrected

## Stage 4: Fidelity Joint Refinement

The following remain frozen:

- Fidelity-SwinIR base
- Diffusion U-Net
- LR encoder

The following are refined:

- GeoMapper
- Residual decoder
- Trust controller

This prevents joint training from damaging already-good base and diffusion checkpoints.

GAN and natural-image perceptual losses are disabled for this experiment because they generally improve perceived sharpness at the cost of PSNR.

---

# 14. Validation and Checkpoint Changes

The new pipeline also fixes experimental issues that were not architectural but affected results.

### Fixed Validation Noise

Previously, a different diffusion seed could be used each epoch. A checkpoint might therefore appear better because it received an easier random sample.

The new validation uses fixed random seeds.

### Correct Diffusion Checkpoint Direction

If `val_psnr` was unavailable during diffusion training, the old fallback could use validation loss while retaining:

```text
mode = max
```

It could therefore select the checkpoint with the highest loss.

Fallback loss metrics now use:

```text
mode = min
```

### Checkpoint Lineage

Each downstream checkpoint stores the SHA-256 fingerprint of its parent checkpoint.

This prevents combinations such as:

```text
new base + old VAE + unrelated diffusion + old joint checkpoint
```

### Residual Scale Selection

Validation tests:

```text
0.00, 0.25, 0.50, 0.75, 1.00, 1.25
```

Including `0.00` is important because it represents the deterministic base. If all generated residuals are harmful, validation can safely select the base instead.

---

# 15. Parameter Distribution

| Module | Parameters |
|---|---:|
| Fidelity-SwinIR base | 797,337 |
| Residual VAE | 1,727,563 |
| LR encoder | 1,111,296 |
| Diffusion U-Net | 13,790,596 |
| GeoMapper | 898,306 |
| Residual decoder | 336,422 |
| Trust controller | 75,041 |
| **Total core model** | **18,736,561** |

---

# 16. What Is Actually Novel?

The following components are **not individually novel**:

- SwinIR
- Residual diffusion
- High-pass filtering
- Evidence gating
- Uncertainty estimation
- Resize-convolution
- Residual learning

The provisional contribution is:

> A target-supervised, base-referenced local least-squares residual acceptance mechanism with a local excess-risk guard for paired cross-sensor satellite super-resolution.

More simply:

> The diffusion model proposes detail, but a learned spatial controller is trained using the mathematically optimal local coefficient that determines whether that detail improves the deterministic base.

This is different from:

- Gating an external reference
- Using uncertainty only to change diffusion noise
- Gating latent frequency features
- Applying one global residual scale

However, publication novelty is established only after literature comparison and ablations.

---

# 17. Expected Improvement

The previous result was:

```text
SwinIR base:     33.7650 dB
Final GeoDiff:   33.7837 dB
Improvement:     +0.0187 dB
```

The new model attempts to improve this through:

1. A substantially stronger deterministic anchor.
2. Complete `5,225`-pair training instead of a small partial dataset.
3. Full `160×160 → 480×480` protocol.
4. Correct checkpoint selection.
5. Stable validation seeds.
6. Local rather than global residual acceptance.
7. Guaranteed validation comparison against residual scale zero.
8. Removal of PixelShuffle artifacts.
9. PSNR-focused losses.
10. Protection of the deterministic base during joint training.

These changes make `35+ dB` a reasonable experimental target, but only the complete official OLI2MSI test run can confirm it.