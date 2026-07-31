# 36 - Probabilistic Foundations of Logistic-Proximal Reconstruction

## Learning Objectives

- interpret the bounded logistic objective as a MAP-like estimator;
- derive the Gaussian likelihood and log-determinant term;
- distinguish realized noise, variance, and predicted residual;
- understand fixed, oracle, and estimated covariance;
- explain why one noisy sample is insufficient to verify variance pixel by
  pixel.

## 1. Completed model prediction

Let a conditional SR model produce logits:

\[
v\in\mathbb{R}^{B\times C\times H\times W}.
\]

Its unconstrained bounded image is:

\[
p=\sigma(v).
\]

For an existing model that outputs an image \(p\in[0,1]\) rather than logits,
define:

\[
v=\operatorname{logit}
\left(
\operatorname{clip}(p,\epsilon,1-\epsilon)
\right).
\]

Use an explicit \(\epsilon\), such as \(10^{-5}\), to prevent infinite logits.

## 2. Bernoulli KL as a bounded proximity measure

The elementwise divergence:

\[
D_{\mathrm{Bern}}(x\|p)
=
x\log\frac{x}{p}
+
(1-x)\log\frac{1-x}{1-p}
\]

is not claiming that reflectance pixels are physically Bernoulli random
variables. It is a mathematical Bregman divergence whose domain is bounded:

\[
0<x<1.
\]

It penalizes movement from \(p\) while respecting the geometry induced by
binary entropy. Near 0 and 1, its gradient becomes steep, discouraging invalid
values without an external clamp.

Its derivative with respect to \(x\) is:

\[
\frac{\partial D_{\mathrm{Bern}}}{\partial x}
=
\operatorname{logit}(x)-\operatorname{logit}(p)
=
\operatorname{logit}(x)-v.
\]

## 3. Gaussian LR likelihood

Assume an LR observation model:

\[
y=Cx+\eta,
\qquad
\eta\sim\mathcal{N}(0,\Sigma).
\]

For fixed positive-definite \(\Sigma\), the negative log likelihood is:

\[
-\log p(y\mid x,\Sigma)
=
\frac12(y-Cx)^\top\Sigma^{-1}(y-Cx)
+
\frac12\log|\Sigma|
+
\frac{m}{2}\log(2\pi).
\]

If \(\Sigma\) is fixed during optimization over \(x\), the last two terms are
constant with respect to \(x\). They still matter when learning \(\Sigma\).

## 4. Why the log determinant is necessary

Suppose a variance estimator can increase every variance value. Without
\(\log|\Sigma|\), the quadratic term can be made arbitrarily small by making
\(\Sigma\) arbitrarily large.

For diagonal variance \(s_i^2\):

\[
\mathcal{L}_{\mathrm{NLL}}
=
\frac12\sum_i
\left[
\frac{r_i^2}{s_i^2}
+
\log s_i^2
\right],
\qquad
r=y-Cx.
\]

The first term rewards larger variance; the second penalizes it. Their balance
prevents trivial unbounded inflation, although it does not by itself guarantee
that all residual error is genuine sensor noise.

## 5. MAP-like reconstruction

The fixed-covariance logistic-proximal estimate is:

\[
\hat{x}
=
\arg\min_{0<x<1}
D_{\mathrm{Bern}}(x\|\sigma(v))
+
\frac12
(y-Cx)^\top\Sigma^{-1}(y-Cx).
\]

Interpretation:

- the completed conditional model supplies a learned image prior centered at
  \(\sigma(v)\);
- the likelihood measures compatibility with the LR observation;
- \(\Sigma\) controls how strongly each LR measurement should be trusted.

Small variance gives strong data consistency. Large variance permits greater
deviation from observed LR because the measurement is considered less
reliable.

## 6. Three covariance regimes

### 6.1 Fixed covariance

\[
\Sigma=\sigma_0^2 I.
\]

This is simple, stable, and serves as the prior-art floor. It cannot adapt to
different degradation strengths or signal levels.

### 6.2 Oracle covariance

\[
\Sigma=\Sigma_{\mathrm{oracle}}(\mu,\theta).
\]

The simulator knows the degradation parameters and clean LR. This arm asks:

> Would correct noise information help if it were available?

If the answer is no, an estimator cannot create a useful gain.

### 6.3 Estimated covariance

\[
\widehat{\Sigma}
=
g_\psi(y,\theta).
\]

This arm asks:

> Can a calibration model infer enough of the useful covariance information
> from observable inputs?

The estimator must not use HR target information at inference.

## 7. Realized residual versus variance

For one observation:

\[
\eta_{\mathrm{realized}}=y-\mu.
\]

Variance is an expectation over repeated observations:

\[
\operatorname{Var}(\eta_i\mid\mu,\theta)
=
\mathbb{E}[\eta_i^2\mid\mu,\theta].
\]

One sample with a large residual does not prove high variance, and one sample
near zero does not prove low variance. Calibration must be evaluated over many
pixels or repeated simulated draws grouped by comparable predicted variance.

## 8. Standardized residuals

For diagonal predicted standard deviation \(\hat{s}_i\), define:

\[
z_i
=
\frac{y_i-(C\hat{x})_i}{\hat{s}_i}.
\]

Under a correctly specified independent Gaussian model, aggregated \(z\)
should approximately have:

\[
\mathbb{E}[z]\approx0,
\qquad
\operatorname{Var}(z)\approx1.
\]

Also inspect:

- quantile-quantile plots;
- tail frequency;
- spatial autocorrelation;
- band-specific statistics;
- degradation-severity bins.

Identity covariance after whitening is a model diagnostic, not guaranteed
truth. Quantization, clipping, spatially correlated processing, and model bias
can violate the Gaussian assumption.

## 9. Calibration versus reconstruction uncertainty

Two uncertainties must not be mixed:

1. **Observation uncertainty:** uncertainty in LR measurement \(y\), represented
   by \(\Sigma\).
2. **Posterior reconstruction uncertainty:** variation in plausible HR outputs
   given the same LR observation.

Observation covariance does not by itself describe all ambiguity in missing HR
detail. Stochastic diffusion samples are still required to measure posterior
diversity.

## 10. Identifiability warning

A covariance model can absorb:

- sensor noise;
- incorrect blur;
- registration error;
- atmospheric residual;
- backbone bias;
- true scene change;
- quantization error.

The log determinant prevents infinite variance but cannot label the source of
each residual. Synthetic supervision, degradation metadata, restricted model
capacity, and operator-mismatch experiments are necessary to make the
interpretation credible.

## Exercises

1. Derive the derivative of Bernoulli KL with respect to \(x\).
2. Explain why \(\log|\Sigma|\) is constant for the inner solve but necessary
   for training a covariance estimator.
3. Distinguish realized residual from conditional variance.
4. Explain how large variance changes data-consistency strength.
5. Give three non-noise effects that a learned covariance might absorb.

## Mastery Checklist

- [ ] I can interpret each term in the logistic-proximal objective.
- [ ] I understand why Bernoulli KL is a bounded divergence, not a sensor claim.
- [ ] I know the difference between fixed, oracle, and estimated covariance.
- [ ] I can explain standardized residual diagnostics.
- [ ] I understand the remaining identifiability risk.

Next: [37 - Six-Arm Controlled Pilot](37_six_arm_pilot.md).
