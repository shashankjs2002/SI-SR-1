# OLI2MSI Fidelity Loss Configuration

The total reconstruction loss is approximately:

\[
L = 50L_{\text{MSE}} + 25L_{\text{MS-MSE}}
+ 1L_{\text{Charbonnier}} + 0.5L_{\text{SSIM}}
\]

The weights are **not percentages**. Each loss has a different numerical scale, so `50` does not necessarily mean 50% of total loss.

## 1. MSE: `50.0`

\[
L_{\text{MSE}}=\frac{1}{N}\sum(\hat{I}-I)^2
\]

MSE squares every pixel error between prediction and Sentinel target.

**Role**

- Directly optimizes PSNR because:

\[
PSNR=-10\log_{10}(MSE)
\]

- Enforces accurate RGB reflectance values.
- Penalizes large errors heavily.

**Impact**

- Usually increases PSNR.
- Improves radiometric fidelity.
- Can produce smooth or averaged textures when multiple HR solutions are possible.
- Strongly discourages hallucinated details.

Its high weight is intentional because the present objective prioritizes PSNR.

---

## 2. Multiscale MSE: `25.0`

The images are compared at multiple resolutions, currently approximately:

```text
1×: original resolution
2×: average-pooled resolution
4×: average-pooled resolution
```

\[
L_{\text{MS-MSE}}
=\frac{1}{3}(L_1+L_2+L_4)
\]

**Role**

- Full resolution checks individual pixels and fine structures.
- Lower resolutions check broad terrain, colour and land-cover structure.
- Prevents the model from improving local pixels while introducing large-scale brightness drift.

**Impact**

- Improves coarse spatial structure.
- Stabilizes training.
- Reduces colour and low-frequency artifacts.
- Helps PSNR and often SSIM.
- May add additional smoothness when weighted too strongly.

This is especially useful for cross-sensor Landsat–Sentinel pairs because illumination and broad radiometric patterns matter.

---

## 3. Charbonnier: `1.0`

\[
L_{\text{Charb}}=
\frac{1}{N}\sum\sqrt{(\hat{I}-I)^2+\epsilon^2}
\]

It is a smooth and numerically stable approximation of L1 loss.

**Role**

- Measures absolute reconstruction error.
- Is less dominated by a few large-error pixels than MSE.
- Provides stable gradients around zero.

**Impact**

- Preserves edges better than using MSE alone.
- Makes training more robust to limited sensor misregistration, atmospheric variation and outliers.
- Reduces excessive sensitivity to a few abnormal pixels.

MSE provides strong PSNR optimization, while Charbonnier prevents that optimization from being controlled entirely by large outliers.

---

## 4. SSIM: `0.5`

SSIM compares local:

- luminance;
- contrast;
- structural correlation.

The optimized loss is generally:

\[
L_{\text{SSIM}}=1-SSIM(\hat{I},I)
\]

**Role**

- Encourages roads, boundaries and terrain patterns to have structures similar to the target.
- Evaluates local relationships instead of only independent pixels.

**Impact**

- Improves SSIM.
- Reduces structural blurring.
- Helps retain locally coherent features.
- Too much SSIM can permit small radiometric errors because structurally similar images may have different pixel values.

The moderate `0.5` weight supports structure without overpowering pixel fidelity.

---

## 5. Consistency: `0.0`

Consistency would degrade the generated HR output back to LR and compare it with the Landsat input:

\[
L_{\text{consistency}} =
\lVert D(\hat{I}_{HR})-I_{LR}\rVert
\]

It is disabled.

**Why disabled**

Landsat and Sentinel are different sensors. A simple synthetic degradation operator cannot exactly reproduce:

- Landsat spectral response;
- atmospheric conditions;
- sensor PSF/MTF;
- temporal differences;
- radiometric calibration.

A strong consistency loss could force Sentinel-like output to reproduce Landsat-specific colour and blur.

**Impact of zero**

- Avoids imposing an inaccurate sensor model.
- Allows the output to match the Sentinel target.
- Redegradation error may be worse.
- The model has less explicit LR evidence enforcement.

---

## 6. Perceptual: `0.0`

Perceptual loss normally compares features extracted by a pretrained network such as VGG:

\[
L_{\text{perceptual}}
=
\lVert\phi(\hat{I})-\phi(I)\rVert
\]

It is disabled.

**Why disabled**

- VGG is trained on natural photographs, not satellite reflectance.
- It may favor visually pleasing textures over correct satellite pixels.
- It commonly reduces PSNR while improving visual sharpness.

**Impact of zero**

- Better focus on PSNR and pixel accuracy.
- Avoids natural-image feature bias.
- Output may appear smoother.
- Fine perceptual texture recovery may be weaker.

---

## 7. Adversarial: `0.0`

Adversarial loss comes from a discriminator that distinguishes generated images from real Sentinel images.

**Role when enabled**

- Encourages sharp and realistic textures.
- Can generate high-frequency details.
- Improves perceptual realism.

**Why disabled**

GAN-generated texture can be plausible but geographically incorrect. It commonly:

- reduces PSNR;
- reduces pixel correspondence;
- produces false structures;
- complicates stable training.

**Impact of zero**

- No discriminator is optimized.
- Lower hallucination risk.
- More conservative reconstruction.
- Better PSNR/SSIM orientation.
- Less sharp than a GAN-optimized result.

## Overall Behaviour

This configuration says:

> Reconstruct the Sentinel target as accurately as possible at pixel and structural levels, while avoiding perceptual or adversarial hallucination.

Expected result:

- high PSNR;
- high SSIM;
- accurate colour and broad structure;
- stable training;
- relatively conservative high-frequency detail;
- potentially smoother output than GAN/perceptual variants.

The most important diagnostic is the **weighted contribution** of every active loss. For example, an MSE of `0.0005` contributes:

```text
50 × 0.0005 = 0.025
```

while an SSIM loss of `0.05` contributes:

```text
0.5 × 0.05 = 0.025
```

Therefore, despite weights `50` and `0.5`, both can influence training equally.
