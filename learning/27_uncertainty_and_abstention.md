# 27 - Stochastic Uncertainty and Abstention

## Purpose

Diffusion can generate multiple plausible HR residuals for the same LR input. Their disagreement
estimates ambiguity. GeoDiff-GAN uses this uncertainty to blend unreliable stochastic output back
toward the deterministic base.

## Multi-Sample Inference

Generate \(N\) outputs:

\[
\hat x^{(1)},\ldots,\hat x^{(N)}
\sim
p_\theta(x\mid y,c,\theta_d,m).
\]

The sample mean is:

\[
\bar x
=
\frac1N\sum_{n=1}^{N}\hat x^{(n)}.
\]

A pixelwise uncertainty map can be the channel-averaged variance:

\[
U(x,y)
=
\frac{1}{3}
\sum_{c=1}^{3}
\frac1N
\sum_{n=1}^{N}
\left(
\hat x_c^{(n)}-\bar x_c
\right)^2.
\]

This is epistemic/generative disagreement under sampling. It is not a calibrated physical sensor
uncertainty by default.

## Agreement from Uncertainty

The system converts uncertainty into agreement:

\[
A
=
\exp\left(-\frac{U}{s_u}\right),
\]

where \(s_u\) is `uncertainty_scale`, defaulting to `0.0025`.

Properties:

```text
U = 0       -> A = 1
U increases -> A decreases exponentially
```

## Evidence and Uncertainty Confidence

The GeoMapper evidence map \(C_E\) is resized to HR. Final reported confidence is:

\[
C_F=C_E\odot A.
\]

The abstention map is:

\[
M_A=1-C_F.
\]

High abstention means either:

- LR evidence confidence is low;
- stochastic samples disagree;
- both.

## Output Blending

The uncertain stochastic image is blended toward the deterministic base:

\[
x_{\text{abstained}}
=
x_{\text{base}}
+A_{\text{eff}}
\odot
\left(
x_{\text{stochastic}}-x_{\text{base}}
\right).
\]

With confidence floor \(f\):

\[
A_{\text{eff}}
=
f+(1-f)A.
\]

At \(f=0\):

- agreement 1 keeps the stochastic output;
- agreement 0 returns the base.

This is a continuous abstention policy rather than a hard pixel mask.

## Why Blend to the Base?

The deterministic base:

- is reproducible;
- is trained with reconstruction losses;
- carries LR-aligned low-frequency structure;
- is less likely to hallucinate sample-specific texture.

Abstention does not imply the base is correct. It means the system prefers the safer deterministic
estimate when generative alternatives disagree.

## Calibration Evaluation

Useful measures include:

\[
\operatorname{corr}(U,|x-\hat x|),
\]

\[
\operatorname{corr}(C_F,-|x-\hat x|).
\]

For selective prediction, sort pixels by confidence and evaluate error at retained coverage:

\[
R(q)
=
\mathbb E
\left[
|x-\hat x|
\mid
C_F\text{ in top }q
\right].
\]

A useful confidence estimate should produce lower error as coverage decreases.

## Limitations

Low sample variance does not guarantee correctness. All samples may share the same learned bias.
High variance may represent harmless texture alternatives rather than structural error.

Uncertainty depends on:

- number of samples;
- sampling steps;
- guidance scale;
- random seeds;
- decoder sensitivity;
- whether deterministic DDIM trajectories diversify enough from different initial noise.

## Failure Modes

| Symptom | Interpretation |
|---|---|
| uncertainty near zero everywhere | sample collapse |
| uncertainty high everywhere | weak conditioning |
| confidence-error correlation near zero | uncalibrated confidence |
| abstention removes all details | scale too small or evidence too low |
| high-confidence errors | shared model bias |
| uncertainty changes with batch ordering | reproducibility defect |

## Recommended Protocol

Use at least 8 samples for reporting. Fix seeds and report:

```text
mean prediction
variance/uncertainty map
confidence-error correlation
selective L1 at 80% coverage
sample diversity
runtime per sample
```

## Implementation

See `apply_uncertainty_abstention` in
[`models/system.py`](../src/geodiff_gan/models/system.py) and sampling aggregation in
[`cli/evaluate.py`](../src/geodiff_gan/cli/evaluate.py).

