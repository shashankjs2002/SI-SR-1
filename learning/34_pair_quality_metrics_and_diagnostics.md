# 34 - Pair Quality Metrics and Diagnostic Interpretation

## Learning objectives

This chapter teaches how to answer the questions supervisors ask about every
number and image:

- What does it mean?
- Why do we calculate it?
- How is it calculated?
- What does a high or low value tell us?
- What can it **not** tell us?

## 1. First rule: calculate on reflectance, not display images

All numeric comparisons must use the stored `[0,1]` reflectance tensors before
percentile stretching, contrast enhancement, or colormap conversion.

Display stretching changes pixel values to improve visibility. It is valid for
figures but invalid for PSNR, MAE, SSIM, or radiometric bias.

The second rule is to use the validity mask. Let:

- \(X\) be a prediction or enlarged Landsat image;
- \(Y\) be the Sentinel reference;
- \(M\in\{0,1\}\) be the valid-pixel mask;
- \(C=3\) be the RGB channel count.

Only pixels where \(M=1\) belong in the denominator.

## 2. Absolute mean error: the most frequently misunderstood number

The masked mean absolute error, also called L1 error or MAE, is:

\[
MAE(X,Y)=
\frac{
\sum_{n,c,i,j}M_{n,i,j}|X_{n,c,i,j}-Y_{n,c,i,j}|
}{
C\sum_{n,i,j}M_{n,i,j}
}.
\]

### Plain-language meaning

It is the average absolute reflectance difference per valid color-channel sample.

If `MAE = 0.020`, the two tensors differ by an average of `0.020` reflectance
units on the normalized `[0,1]` scale. It does **not** mean:

- 2 m of spatial error;
- exactly 2% relative error;
- 98% accuracy;
- that 2% of pixels are wrong.

Relative percentage would require division by a defined reference magnitude and
be unstable over dark pixels.

### What is "absolute mean error of the same image pair"?

For the dataset inspection plot, Landsat is bicubically enlarged to `384 x 384`
only so it can be compared element-wise with Sentinel:

\[
E_{pair}=MAE(\operatorname{bicubic}(Landsat),Sentinel).
\]

This is a **pair discrepancy**, not a model error. It includes:

1. genuine spatial-resolution difference;
2. spectral response difference;
3. atmospheric and illumination difference;
4. resampling effects;
5. temporal change;
6. registration error.

For model evaluation:

\[
E_{model}=MAE(GeoDiffGAN(Landsat),Sentinel).
\]

Improvement over bicubic is:

\[
\Delta E=E_{bicubic}-E_{model}.
\]

Positive \(\Delta E\) means the model reduced average absolute discrepancy.

### Why absolute value?

Without absolute value, positive and negative errors cancel. An image that is too
bright in one region and equally too dark in another could have near-zero signed
mean error despite being inaccurate.

## 3. Signed bias and per-band error

MAE loses the direction of error. Signed bias restores it:

\[
Bias_c=\frac{\sum M(X_c-Y_c)}{\sum M}.
\]

- positive bias: the first image is brighter than the target in that band;
- negative bias: it is darker;
- near-zero bias: no overall shift, but local errors may still be large.

Per-band MAE can reveal a spectral mismatch hidden by one RGB average. For
example, a large blue-band error with smaller red and green errors may indicate
haze, water, or sensor bandpass differences.

## 4. RMSE and PSNR

Mean squared error is:

\[
MSE=\frac{\sum M(X-Y)^2}{C\sum M}.
\]

Squaring gives extra weight to large errors. Root mean squared error is
\(RMSE=\sqrt{MSE}\).

For normalized images with maximum value 1:

\[
PSNR=10\log_{10}\left(\frac{1}{MSE}\right)
=-10\log_{10}(MSE).
\]

Higher PSNR is better. A 3 dB increase corresponds to approximately halving MSE,
not halving MAE.

### What PSNR tells us

It measures pixel-level fidelity and strongly penalizes large errors.

### What PSNR does not tell us

It does not prove that generated texture is realistic, that edges are correctly
located, or that the output is physically valid. Small registration shifts can
reduce PSNR even when both images look reasonable.

## 5. SSIM

SSIM compares local luminance, contrast, and structure. Its ideal value is 1.
Higher is better.

The repository computes SSIM only over windows that are effectively valid under
the mask. This prevents clouds or borders from improving or damaging the score.

SSIM is more tolerant of some brightness changes than PSNR, but it is still a
full-reference metric and remains sensitive to cross-sensor alignment.

## 6. Edge F1

The implementation:

1. converts RGB to grayscale;
2. computes horizontal and vertical gradients;
3. thresholds strong target edges at the 90th percentile with a minimum threshold;
4. permits a one-pixel matching tolerance;
5. calculates precision, recall, and F1.

\[
Precision=\frac{matched\ predicted\ edges}{predicted\ edges},
\]

\[
Recall=\frac{matched\ target\ edges}{target\ edges},
\]

\[
F1=\frac{2PR}{P+R}.
\]

Higher is better.

- High precision, low recall: output is conservative and misses many real edges.
- Low precision, high recall: output creates excessive or false edges.
- High F1: strong balance of target-edge recovery and false-edge control.

Edge F1 is valuable because a smooth image can have acceptable PSNR but fail to
recover roads, field boundaries, and building structures.

## 7. LR re-degradation error

The model output is passed through the current differentiable sensor approximation
and compared with the real Landsat input:

\[
E_{LR}=MAE(D_{3x}(\hat{H}),L).
\]

Lower is better. It asks:

> If the predicted HR image were observed through our approximate 30 m sensor
> model, would it reproduce the Landsat evidence?

This is an evidence-consistency test. It does not verify that HR details are
correct because many different HR images can map to nearly the same LR image.

The current paired operator uses a nominal Gaussian blur and area downsampling.
It is not a complete calibrated per-band Landsat OLI image-formation model, so
this metric must be described as **approximate LR consistency**.

## 8. LPIPS and DISTS

| Metric | Preferred direction | Main sensitivity |
|---|---|---|
| LPIPS | Lower | Deep-feature perceptual difference |
| DISTS | Lower | Structure and texture similarity |

They can better reflect perceptual appearance than PSNR, but their pretrained
features were not designed specifically for calibrated satellite reflectance.
Use them as complementary metrics, not physical truth.

For masked evaluation, invalid prediction pixels are replaced by target pixels
before LPIPS/DISTS. This prevents invalid regions from contributing, although a
deep network's receptive field can still make mask boundaries imperfect.

## 9. Dataset-pair diagnostics in the Kaggle notebook

### Landsat bicubic panel

**What it is:** Real `128 x 128` Landsat enlarged to `384 x 384` for display.

**Why use it:** It establishes the no-learning baseline and lets us visually
compare the same grid size.

**What it tells us:** Large-scale radiometry and coarse structure available in
the input.

**What it does not tell us:** Bicubic interpolation does not create measured 10 m
detail.

### Sentinel target panel

**What it is:** Native `384 x 384` Sentinel reference patch.

**Why use it:** It supplies the supervised target and finer spatial structures.

**Caution:** It is a different sensor observation, not a mathematically exact
ground truth for Landsat.

### Absolute-difference heatmap

The notebook displays:

\[
E(i,j)=M(i,j)\frac{1}{3}\sum_c|X(c,i,j)-Y(c,i,j)|.
\]

- blue: small absolute discrepancy;
- green/yellow: moderate discrepancy;
- red: large discrepancy under the displayed color range.

The heatmap color is not a class label and not a probability. Its maximum is
scaled using the 99th percentile with a minimum display range, so compare the
printed numeric MAE and colorbar rule, not color alone.

### Edge-registration overlay

The current RGB overlay is assembled as:

```text
red channel   = Sentinel edge strength
green channel = maximum of Sentinel and Landsat edge strength
blue channel  = Landsat edge strength
```

Therefore:

- matching edges appear white;
- Sentinel-only edges appear yellow;
- Landsat-only edges appear cyan;
- parallel yellow/cyan double edges indicate spatial displacement.

A random mixture can reflect resolution differences. A consistent displacement
in the same direction across roads and boundaries suggests misregistration.

### Validity mask

- white: included in losses and metrics;
- black: ignored because at least one sensor is invalid.

A 95% valid fraction is a data-quality statement, not a registration score.

### Reflectance histograms

These show the distribution of each RGB channel for Landsat and Sentinel over
valid pixels.

- similar shapes with a shift: possible radiometric bias;
- different spread: contrast, atmospheric, land-cover, or spectral difference;
- multiple peaks: mixed land-cover classes;
- strong clipping at 0 or 1: scaling, saturation, or clipping concern.

Histograms discard spatial arrangement. Two badly misregistered images can have
nearly identical histograms.

### Frequency spectra

The centered Fourier magnitude plot describes spatial-frequency energy:

- bright center: strong low-frequency content;
- energy farther from center: finer detail and sharp edges;
- lines or repeated peaks: directional or periodic structures;
- grid-like peaks in generated residuals: possible upsampling artifact.

Sentinel should normally contain more high-frequency energy than Landsat, but
more high-frequency energy is not automatically better. Noise and hallucinated
texture also increase it.

## 10. Copy-paste pair-statistics cell

```python
import numpy as np
import torch
from torch.nn import functional as F

with np.load(records[DEBUG_INDEX]["patch"]) as data:
    lr = torch.from_numpy(data["lr"]).float()
    hr = torch.from_numpy(data["hr"]).float()
    valid = torch.from_numpy(data["valid_mask_hr"][0]).bool()

lr_up = F.interpolate(
    lr[None], size=hr.shape[-2:], mode="bicubic", align_corners=False
)[0].clamp(0, 1)

absolute = (lr_up - hr).abs()
signed = lr_up - hr
masked_values = valid[None].expand_as(absolute)

mae = absolute[masked_values].mean()
rmse = signed[masked_values].square().mean().sqrt()
bias = [signed[c][valid].mean() for c in range(3)]
band_mae = [absolute[c][valid].mean() for c in range(3)]

print("pair MAE:", float(mae))
print("pair RMSE:", float(rmse))
print("RGB signed bias:", [float(value) for value in bias])
print("RGB per-band MAE:", [float(value) for value in band_mae])
print("valid fraction:", float(valid.float().mean()))
```

## 11. How to accept or reject a pair

Do not use one universal MAE threshold before observing the dataset distribution.
Instead:

1. require valid geometry and metadata;
2. require day gap within the selected limit;
3. require at least 95% joint patch validity;
4. calculate MAE, RMSE, signed bias, and per-band MAE for every patch;
5. inspect distribution outliers by scene pair;
6. inspect edge overlays for systematic displacement;
7. inspect histograms for severe radiometric mismatch;
8. reject or quarantine problematic **scene pairs**, not only convenient patches;
9. document every rejection rule before final evaluation.

Patch-level cherry-picking after seeing test results would bias evaluation.

## 12. Common supervisor questions

### If MAE is low, is the pair good?

Not necessarily. Smooth or dark scenes can have low MAE despite misregistration,
and histograms can match without spatial alignment. Use edge overlays and per-band
bias as well.

### If the heatmap is red, is the dataset wrong?

Not automatically. Red means high local discrepancy under that plot's scale. It
may represent legitimate fine detail, radiometric difference, temporal change,
or registration error. Diagnose the spatial pattern.

### Why does Sentinel show more detail than Landsat?

Sentinel samples at 10 m while Landsat samples at 30 m, and their optical transfer
functions differ. Fine structures are attenuated or mixed inside Landsat pixels.

### Why can model MAE beat bicubic but still look wrong?

MAE rewards average pixel closeness. A model can reduce broad radiometric error
while producing smooth or incorrect local texture. Check edge F1, perceptual
metrics, residuals, spectra, and visual alignment.

### Is `1 - MAE` accuracy?

No. Regression has no natural classification-style accuracy here. Report MAE,
PSNR, SSIM, edge F1, perceptual metrics, and consistency separately.

## Mastery checklist

- [ ] I can derive masked MAE and explain its denominator.
- [ ] I distinguish pair discrepancy from model reconstruction error.
- [ ] I can interpret `MAE=0.02` without calling it 98% accuracy.
- [ ] I know when signed bias is more informative than MAE.
- [ ] I can explain why PSNR and edge F1 may disagree.
- [ ] I can explain every panel in the pair-inspection figure.
- [ ] I know the exact colors of the registration overlay.
- [ ] I understand why low LR consistency does not prove correct HR detail.

Next: [35 - Training, Debugging, Evaluation, and Supervisor Questions](35_training_debugging_evaluation_viva.md).
