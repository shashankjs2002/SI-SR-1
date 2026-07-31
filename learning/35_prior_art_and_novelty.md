# 35 - Prior Art, GLinSAT Equivalence, and Honest Novelty

## Learning Objectives

- derive the exact equivalence between hard Bernoulli projection and GLinSAT;
- distinguish terminal placement from a novel constraint mechanism;
- organize the closest prior-art clusters;
- formulate a contribution that can survive technical review;
- avoid novelty claims that have already been falsified.

## 1. Why this chapter exists

Several attractive descriptions of the proposed method are not novel:

- differentiable constrained output layers already exist;
- logistic-entropy projection under bounded linear equality exists;
- terminal projection after generative sampling exists;
- Poisson-Gaussian and heteroscedastic noise estimation exist;
- diffusion measurement consistency is an active, crowded field.

The paper must absorb this prior art rather than hide it.

## 2. Exact GLinSAT equivalence

Let:

\[
p=\sigma(v)
\]

and define the hard bounded projection:

\[
\min_{0<x<1,\;Cx=y}
D_{\mathrm{KL}}
\left(
\operatorname{Bern}(x)
\middle\|
\operatorname{Bern}(p)
\right).
\]

For one element:

\[
D_{\mathrm{Bern}}(x\|p)
=
x\log\frac{x}{p}
+
(1-x)\log\frac{1-x}{1-p}.
\]

Using:

\[
\log\frac{p}{1-p}=v,
\]

expand:

\[
D_{\mathrm{Bern}}(x\|p)
=
x\log x+(1-x)\log(1-x)-vx-\log(1-p).
\]

The final term is constant with respect to \(x\). Therefore:

\[
\arg\min_{Cx=y}
D_{\mathrm{Bern}}(x\|\sigma(v))
=
\arg\min_{Cx=y}
\left[
-v^\top x
+
\sum_j
\left(
x_j\log x_j
+
(1-x_j)\log(1-x_j)
\right)
\right].
\]

GLinSAT solves the bounded logistic-entropy problem:

\[
\min_{0\le x\le u,\;Ax=b}
-c^\top x
+
\frac{1}{\theta}
\sum_j
\left[
\frac{x_j}{u_j}\log\frac{x_j}{u_j}
+
\left(1-\frac{x_j}{u_j}\right)
\log\left(1-\frac{x_j}{u_j}\right)
\right].
\]

Set:

\[
u=1,\qquad A=C,\qquad b=y,\qquad c=v,\qquad\theta=1.
\]

The problems are identical term for term. The hard Bernoulli projection must
therefore be treated as a GLinSAT-class baseline, not as a new layer.

Primary reference:

- [GLinSAT, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/dd73f39426a03131c38c8d943153d44b-Abstract-Conference.html)

Related positive linear satisfiability:

- [LinSATNet](https://arxiv.org/abs/2407.13917)

## 3. Noisy slack does not create a hard fiber

Consider:

\[
\min_{x,\eta}
D_{\mathrm{Bern}}(x\|\sigma(v))
+
\frac12\eta^\top\Sigma^{-1}\eta
\quad
\text{subject to }Cx+\eta=y.
\]

Because \(\eta\) is unrestricted:

\[
\eta=y-Cx.
\]

Elimination gives:

\[
\min_{0<x<1}
D_{\mathrm{Bern}}(x\|\sigma(v))
+
\frac12(y-Cx)^\top\Sigma^{-1}(y-Cx).
\]

Thus:

- \(Cx=y\) is not enforced;
- \(x\) is not confined to a lower-dimensional fiber;
- the equality \(Cx+\eta=y\) is true by definition of the residual;
- the scientific value must come from the noise model and calibration, not
  from calling the decomposition exact.

## 4. Terminal projection is not novel

Constraint timing can include:

- projection only at the final output;
- projection at every reverse step;
- adaptive projection schedules.

Terminal placement may be computationally attractive, but it is a design
choice, not a priority claim. A recent explicit treatment is:

- [Enforcing Constraints in Generative Sampling via Adaptive Correction
  Scheduling](https://arxiv.org/abs/2605.11214)

The paper should compare timing only if timing is an experimental question. It
should not use final-only placement as the headline contribution.

## 5. Six prior-art clusters

### 5.1 Differentiable optimization layers

These methods place optimization problems inside neural networks and
differentiate through their solutions:

- OptNet;
- DC3;
- GLinSAT;
- LinSATNet;
- other KKT or implicit layers.

The proposed solver belongs to this lineage.

### 5.2 Information projection and Bregman geometry

The Bernoulli KL objective is an information projection under linear
constraints. Relevant foundations include:

- Bregman projections;
- Csiszar information projections;
- generalized iterative scaling;
- entropy-regularized constrained optimization.

The information geometry is established mathematics.

### 5.3 Diffusion inverse problems

Established methods enforce or guide measurement consistency during sampling,
including:

- DPS;
- MCG;
- DDRM;
- related posterior and projected diffusion methods.

These are not equivalent to a supervised conditional output layer, but they
are necessary context and baselines where feasible.

### 5.4 Projected and constrained diffusion

Examples include:

- Projected Diffusion Models;
- hard-constraint guidance;
- linear-constraint conditional diffusion;
- adaptive correction scheduling.

The relevant distinction is experimental setting and mechanism, not a claim
that diffusion constraints were previously absent.

### 5.5 Noise-aware restoration

Established work includes:

- Poisson-Gaussian camera noise;
- heteroscedastic Gaussian likelihoods;
- blind noise-parameter estimation;
- calibrated aleatoric uncertainty;
- physics-based raw-image noise simulation.

An affine variance law is not itself new.

### 5.6 Remote-sensing harmonization and consistency

Remote sensing already uses:

- deterministic conservative resampling;
- sensor MTF/PSF characterization;
- spectral harmonization;
- degradation-consistency losses;
- multiresolution reconstruction protocols.

The proposed contribution must show a measurable remote-sensing benefit.

## 6. Defensible contribution statement

A conservative working statement is:

> We conduct a controlled study of bounded consistency mechanisms for
> conditional satellite super-resolution and introduce a sensor-calibrated
> covariance estimator used by a fixed-covariance logistic-proximal output
> layer. The estimator is trained from LR evidence and degradation metadata,
> while the inner solve remains convex. We evaluate reconstruction, noise
> recovery, uncertainty calibration, stochastic diversity, geographic
> generalization, and solver cost against matched soft, Euclidean, fixed,
> oracle, and hard-consistency baselines.

This statement does not claim:

- invention of Bernoulli projection;
- invention of signal-dependent noise;
- invention of terminal projection;
- guaranteed improvement.

## 7. What could still be scientifically useful

The paper can contribute through:

1. A fair same-backbone comparison that is missing in satellite SR.
2. An oracle-versus-estimated covariance decomposition.
3. Calibration and diversity analysis rather than PSNR-only ranking.
4. Tile-level geographic statistics.
5. A careful study of operator mismatch and processed-reflectance noise.
6. Open implementation and reproducible splits.

Negative results can be useful scientifically, but a flat comparison is harder
to publish as a full journal paper. The pilot exists to discover this early.

## 8. Claims table

| Claim | Status before experiments |
|---|---|
| Hard Bernoulli projection is new | False |
| Terminal placement is new | False |
| Affine Poisson-Gaussian variance is new | False |
| Fixed covariance gives a convex inner solve | Derivable fact |
| Estimated covariance improves calibration | Untested hypothesis |
| Gains generalize geographically | Untested hypothesis |
| The study is suitable for JSTARS | Conditional on results and execution |

## Exercises

1. Expand Bernoulli KL and recover the GLinSAT objective.
2. Explain why free \(\eta\) makes the noisy equality tautological.
3. Give one difference between zero-shot posterior sampling and supervised
   conditional SR.
4. Rewrite a novelty claim as a falsifiable empirical hypothesis.
5. Explain why a new module name cannot establish novelty.

## Mastery Checklist

- [ ] I can derive the exact GLinSAT equivalence.
- [ ] I know why noisy consistency is not a hard fiber.
- [ ] I do not claim terminal projection as novel.
- [ ] I can organize the six relevant prior-art clusters.
- [ ] I can state the contribution without hiding prior work.

Next: [36 - Probabilistic Foundations of Logistic-Proximal Reconstruction](36_probabilistic_foundations.md).
