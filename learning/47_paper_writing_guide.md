# 47 - Paper Structure, Figures, Tables, and Claims

## Learning Objectives

- build a coherent applied-methods manuscript;
- place novelty and prior art accurately;
- map each research question to evidence;
- design interpretable figures and tables;
- write limitations that protect scientific credibility.

## 1. Working title

A conservative title:

> Sensor-Calibrated Bounded Consistency Layers for Conditional Satellite
> Super-Resolution

An explicitly comparative title:

> Hard, Soft, and Noise-Calibrated Consistency in Conditional Satellite
> Super-Resolution: A Controlled Study

Avoid title claims such as:

- first exact diffusion projection;
- physically guaranteed super-resolution;
- novel Bernoulli I-projection;
- true 40 m Sentinel-2 RGB reconstruction.

## 2. One-sentence paper story

> We compare matched bounded consistency mechanisms under a controlled
> synthetic Sentinel-2 degradation protocol and test whether sensor-calibrated
> covariance improves uncertainty calibration over fixed covariance without
> unacceptable loss of reconstruction quality, diversity, or runtime.

Every section should support this sentence.

## 3. Manuscript structure

### 3.1 Introduction

Explain:

- satellite SR is ill posed;
- LR observations contain uncertain evidence;
- consistency strength should depend on observation reliability;
- existing SR work often emphasizes point reconstruction;
- the paper studies the calibration-quality trade-off.

End with precise contributions, not a list of modules.

### 3.2 Related work

Cover:

1. satellite SR and synthetic degradation;
2. diffusion inverse problems;
3. constrained/projected diffusion;
4. differentiable optimization and satisfiability layers;
5. information projection;
6. heteroscedastic and Poisson-Gaussian noise estimation;
7. remote-sensing sensor harmonization and uncertainty.

GLinSAT must be discussed as the closest hard-layer mechanism.

### 3.3 Problem formulation

Define:

\[
\mu=C_\theta x,
\qquad
y=\mu+\eta.
\]

State:

- 40 m RGB is synthetic;
- \(C_\theta\) is the controlled clean operator;
- L2A residuals are not assumed to be raw sensor counts;
- test tiles are geographically isolated.

### 3.4 Methods

Describe:

- shared backbone;
- six arms;
- fixed/oracle/estimated covariance;
- calibration head;
- fixed-covariance dual solver;
- training and checkpointing.

Mark established mechanisms and proposed additions.

### 3.5 Experiments

Include:

- data and tile splits;
- degradation ranges;
- backbone checkpoint;
- arm hyperparameters;
- seeds;
- metrics;
- pre-registered gates;
- hardware and runtime.

### 3.6 Results

Order:

1. isolated solver correctness;
2. oracle usefulness;
3. covariance estimation;
4. six-arm pilot;
5. geographic generalization;
6. operator mismatch;
7. runtime;
8. failure cases.

### 3.7 Discussion and limitations

Address:

- synthetic observation model;
- covariance identifiability;
- diagonal-Gaussian approximation;
- limited regions;
- solver overhead;
- posterior interpretation;
- real sensor transfer.

## 4. Contributions section

Use a form like:

1. We provide a controlled same-backbone comparison of soft, Euclidean,
   hard-logistic, fixed-noise logistic, oracle-noise logistic, and
   sensor-calibrated logistic consistency for satellite SR.
2. We develop and evaluate a restricted LR covariance estimator conditioned on
   observable image evidence and degradation metadata while preserving a
   fixed-covariance convex inner solve.
3. We introduce an evaluation protocol covering noise recovery, uncertainty
   calibration, row/null diversity, operator mismatch, geographic isolation,
   and solver efficiency in addition to standard reconstruction metrics.

Only retain Contribution 2 if implementation and results support it.

## 5. Required figures

### Figure 1: study architecture

Show:

- shared backbone;
- calibration branch;
- six output arms;
- common evaluation.

### Figure 2: degradation and covariance targets

Show:

- HR;
- clean LR;
- observed LR;
- realized residual;
- analytic oracle variance;
- Monte Carlo variance.

### Figure 3: solver geometry

Explain:

- completed prediction;
- hard equality;
- fixed noisy consistency;
- calibrated noisy consistency.

### Figure 4: calibration results

Include:

- reliability curve;
- coverage;
- standardized residual;
- oracle-gap recovery.

### Figure 5: qualitative reconstruction

Use identical crops and scales for:

- backbone;
- fixed;
- hard;
- oracle;
- SensorCal;
- target.

### Figure 6: diversity

Compare pre/post:

- total variation;
- row-space variation;
- null-space variation;
- uncertainty-error relationship.

### Figure 7: failure cases

Include:

- oversmoothing;
- noise copying;
- operator mismatch;
- variance saturation;
- nonconvergence.

## 6. Required tables

| Table | Content |
|---|---|
| 1 | Related-work mechanism comparison |
| 2 | Dataset, regions, tiles, and patches |
| 3 | Six-arm reconstruction and calibration results |
| 4 | Oracle-gap and noise-recovery results |
| 5 | Row/null diversity and stochastic uncertainty |
| 6 | Operator-mismatch robustness |
| 7 | Parameters, iterations, latency, memory, failures |
| 8 | Ablations of covariance inputs and model capacity |

## 7. Results language

Use:

> On the held-out tiles, SensorCal reduced tile-averaged calibration error
> relative to fixed covariance while satisfying the pre-registered PSNR
> guardrail.

Avoid:

> SensorCal reconstructs the true missing details.

Use:

> The estimated covariance recovered a specified fraction of the oracle
> calibration improvement under the synthetic degradation protocol.

Avoid:

> The model learned physical sensor noise.

unless independent calibration data supports that interpretation.

## 8. Abstract template

```text
Satellite super-resolution is ill posed, and consistency with a noisy
low-resolution observation can either preserve evidence or reproduce
measurement error. We study [six mechanisms] using a shared conditional
super-resolution backbone and geographically isolated Sentinel-2 tiles under
a controlled synthetic degradation protocol. We propose [restricted
calibration component], which estimates [covariance] from [allowed inputs]
while retaining a fixed-covariance convex output solve. We evaluate
reconstruction, noise recovery, calibration, stochastic diversity, operator
mismatch, and runtime. [Insert only measured results.] These findings show
[bounded conclusion], while remaining limited to [limitations].
```

Never write result numbers before the final locked evaluation.

## 9. Discussion questions

The discussion should answer:

- Why did oracle covariance help or fail?
- Which regions benefited?
- Did calibration trade away high-frequency detail?
- Did the estimator absorb operator mismatch?
- How much diversity survived?
- Was the iterative solver worth its cost?
- What changes are needed for real cross-sensor data?

## 10. Submission positioning

IEEE JSTARS is plausible if:

- the empirical finding is clear;
- GLinSAT and other baselines are correct;
- tile splits are leak-free;
- calibration adds value beyond relabeling known objectives;
- code and settings are reproducible;
- limitations are explicit.

TGRS or ISPRS JPRS generally require stronger method novelty or broader
evidence. Venue choice should follow the final contribution, not precede it.

## Exercises

1. Write a three-bullet contributions section without using "first" or "novel."
2. Map each primary research question to one table or figure.
3. Rewrite an overclaim about physical noise.
4. Draft one limitations paragraph.
5. Explain why failure cases belong in the main paper.

## Mastery Checklist

- [ ] I can tell the paper story in one sentence.
- [ ] I position GLinSAT as close prior art.
- [ ] Every contribution maps to a controlled result.
- [ ] I know the required figures and tables.
- [ ] I can write bounded claims and explicit limitations.

Next: [48 - Reproducibility and Submission Readiness](48_reproducibility_and_submission.md).
