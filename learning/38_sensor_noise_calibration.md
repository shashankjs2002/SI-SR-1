# 38 - Sensor Noise Calibration

## Learning Objectives

- distinguish raw-sensor noise from processed L2A residuals;
- build analytic and Monte Carlo oracle variance targets;
- design a restricted covariance estimator;
- prevent variance inflation and error absorption;
- keep covariance fixed during the inner reconstruction solve.

## 1. Raw noise versus processed reflectance

For a raw photon-counting detector, a Poisson-Gaussian approximation is often:

\[
y_i\sim
\mathcal{N}
\left(
\mu_i,\alpha\mu_i+\beta
\right).
\]

Here:

- \(\alpha\mu_i\) approximates signal-dependent shot noise;
- \(\beta\) approximates signal-independent read noise.

Sentinel-2 L2A RGB is not raw detector output. It has undergone:

- radiometric calibration;
- atmospheric correction;
- resampling and tiling;
- quantization;
- masking;
- possible clipping and product-specific processing.

Therefore, the paper should use:

> sensor-calibrated heteroscedastic residual model

unless the Poisson-Gaussian assumption is validated in the exact product
domain being studied.

## 2. Synthetic simulator targets

For the current synthetic pipeline, the known quantities include:

- clean LR \(\mu\);
- normalized blur parameter;
- Gaussian-noise parameter;
- Poisson-strength parameter;
- quantization parameter;
- realized observed LR \(y\).

The realized residual is:

\[
\eta=y-\mu.
\]

The analytic variance proxy is:

\[
s_i^2
\approx
\sigma_g^2
+
\frac{\mu_i}{p}
+
\frac{1}{12(L-1)^2}.
\]

The exact parameter ranges are selected by degradation severity in
[degradation.py](../src/geodiff_gan/models/degradation.py).

## 3. Monte Carlo oracle

At fixed \(\mu,\theta\), draw \(K\) independent observations:

\[
y_i^{(k)}\sim\mathcal{S}(\mu_i,\theta),
\qquad k=1,\ldots,K.
\]

Estimate:

\[
\widehat{s_i^2}
=
\frac{1}{K-1}
\sum_{k=1}^{K}
\left(
y_i^{(k)}-\bar{y}_i
\right)^2.
\]

Per-pixel estimates require many draws and substantial storage. A practical
alternative bins pixels by:

- spectral band;
- clean-LR intensity;
- Gaussian parameter;
- Poisson parameter;
- quantization strength.

Estimate one variance per bin. This creates a stable oracle lookup table and
also reveals whether the affine approximation is adequate.

## 4. Initial calibration model

The safest first model predicts band-specific positive coefficients:

\[
\alpha_b=\operatorname{softplus}(a_b)+\epsilon,
\qquad
\beta_b=\operatorname{softplus}(r_b)+\epsilon.
\]

Use a denoised or deterministic LR mean estimate \(\hat{\mu}\):

\[
\hat{s}_{b,i}^2
=
\alpha_b\hat{\mu}_{b,i}+\beta_b.
\]

The coefficients may be conditioned on degradation metadata:

\[
[\alpha,\beta]
=
g_\psi(\theta,\operatorname{pool}(f_{\mathrm{LR}})).
\]

Start with low capacity. A full-resolution unconstrained variance map can
memorize reconstruction error and make physical interpretation weak.

## 5. Covariance estimator inputs

Allowed inference inputs:

- observed LR \(y\);
- degradation metadata \(\theta\);
- LR encoder features;
- band identity;
- acquisition/product metadata available at deployment.

Forbidden inference inputs:

- HR target;
- clean synthetic LR;
- realized ground-truth noise;
- test-set calibration statistics.

These forbidden values may supervise training or define oracle arms, but they
cannot enter the deployed estimator.

## 6. Keep covariance outside the inner solve

The first implementation should compute:

\[
\widehat{\Sigma}
=
\operatorname{stopgrad}
\left[
g_\psi(y,\theta)
\right]
\]

for the projection solve.

This means:

1. the calibration network predicts covariance;
2. covariance is detached for the inner optimizer;
3. the logistic-proximal layer solves a fixed-covariance convex problem;
4. the calibration network is trained with its own NLL and supervision;
5. optional advanced implicit coupling is deferred.

If \(\Sigma\) is a live function of \(x\), the likelihood contains:

\[
\frac{(y-Cx)^2}{\alpha(Cx)+\beta}
+
\log[\alpha(Cx)+\beta],
\]

and the previous global convexity guarantee does not automatically apply.

## 7. Calibration losses

### 7.1 Noise NLL

For known synthetic clean LR:

\[
\mathcal{L}_{\mathrm{noise\_nll}}
=
\frac12
\sum_i
\left[
\frac{(y_i-\mu_i)^2}{\hat{s}_i^2}
+
\log\hat{s}_i^2
\right].
\]

### 7.2 Log-variance supervision

When oracle variance is available:

\[
\mathcal{L}_{\logvar}
=
\left\|
\log\hat{s}^2-\log s_*^2
\right\|_1.
\]

Log scale prevents high-variance pixels from dominating.

### 7.3 Parameter supervision

When the simulator parameters map cleanly to physical coefficients:

\[
\mathcal{L}_{\mathrm{param}}
=
\lVert\hat{\alpha}-\alpha_*\rVert_1
+
\lVert\hat{\beta}-\beta_*\rVert_1.
\]

Use this only when the target mapping is correctly derived.

### 7.4 Spatial regularization

If predicting a variance map:

\[
\mathcal{L}_{\mathrm{smooth}}
=
\lVert\nabla\log\hat{s}^2\rVert_1.
\]

This can suppress pixelwise error memorization, but it should not erase real
land-cover or signal-dependent variation.

## 8. Floors, ceilings, and numerical safety

Use:

\[
s_{\min}^2
\le
\hat{s}_i^2
\le
s_{\max}^2.
\]

Implementation options:

- `softplus(raw) + variance_floor`;
- bounded sigmoid between configured limits;
- clamped log variance.

Record:

- fraction at the floor;
- fraction at the ceiling;
- minimum, median, mean, and maximum variance;
- values by band and degradation severity.

Saturation at either boundary indicates a modeling or optimization problem.

## 9. Error absorption tests

Train and evaluate under:

1. correct blur operator;
2. intentionally mismatched blur;
3. shifted quantization range;
4. unseen noise severity;
5. spatial misregistration.

If predicted variance rises whenever the operator is wrong, the estimator is
absorbing model error rather than isolating observation noise. This may still
be operationally useful, but it must be described as residual uncertainty, not
sensor noise.

## 10. Calibration outputs to save

For every evaluated sample, save:

- predicted variance map;
- predicted standard deviation map;
- clean LR, when synthetic;
- observed LR;
- realized noise;
- standardized residual;
- band and degradation parameters;
- projection residual;
- solver iteration count.

These arrays support later reliability analysis without rerunning inference.

## Exercises

1. Derive the analytic variance proxy from Gaussian, Poisson, and uniform
   quantization assumptions.
2. Explain why L2A reflectance should not automatically be called raw
   Poisson-Gaussian data.
3. Design intensity bins for a Monte Carlo oracle.
4. Explain why the covariance estimator must not see clean LR at inference.
5. Describe an experiment that detects operator-error absorption.

## Mastery Checklist

- [ ] I distinguish raw sensor noise from processed residual uncertainty.
- [ ] I can construct analytic and Monte Carlo oracle variance.
- [ ] I understand why the initial covariance model should be restricted.
- [ ] I know why covariance is detached during the first inner solve.
- [ ] I can diagnose variance inflation and error absorption.

Next: [39 - SensorCal-LogisticProx Architecture](39_sensorcal_architecture.md).
