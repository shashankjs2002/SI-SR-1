# 45 - Statistical Design and Go/No-Go Gates

## Learning Objectives

- identify the correct statistical unit;
- define paired comparisons across methods;
- pre-register success and failure criteria;
- separate pilot decisions from final journal evidence;
- avoid post-hoc metric selection.

## 1. The statistical unit

Patches from one satellite tile share:

- geography;
- acquisition conditions;
- land-cover patterns;
- atmospheric processing;
- overlapping pixels.

They are not independent experimental units.

For final inference, aggregate patch metrics within each held-out tile:

\[
m_t^A
=
\frac{1}{N_t}
\sum_{i\in t}
m_i^A.
\]

Compare methods using tile-level paired differences:

\[
d_t=m_t^A-m_t^B.
\]

The number of test tiles, not merely the number of patches, controls geographic
evidence.

## 2. Paired design

Every method sees the same:

- HR patch;
- LR noise realization;
- degradation parameters;
- backbone prediction;
- evaluation mask.

Pairing removes substantial observation-level variability.

For each tile and metric, store:

\[
(m_t^{\mathrm{soft}},
m_t^{\mathrm{euclid}},
m_t^{\mathrm{hard}},
m_t^{\mathrm{fixed}},
m_t^{\mathrm{oracle}},
m_t^{\mathrm{sensorcal}}).
\]

## 3. Direction of improvement

Define direction before analysis:

| Metric | Better direction |
|---|---|
| PSNR, SSIM, edge F1 | Higher |
| L1, LPIPS, DISTS | Lower |
| Gaussian NLL, CRPS | Lower |
| Coverage error | Lower |
| Noise MAE/RMSE | Lower |
| Runtime, memory | Lower |
| Solver convergence | Higher |

For standardized residual mean and variance, the target is proximity to 0 and
1, not simply higher or lower.

## 4. Pre-registration

Before running the test set, write:

- primary baseline;
- primary calibration metric;
- primary reconstruction guardrail;
- diversity guardrail;
- maximum acceptable runtime overhead;
- minimum convergence rate;
- statistical aggregation method;
- handling of failed samples;
- seed count;
- stop conditions.

Use placeholders until pilot variance is known:

```text
Primary hypothesis:
SensorCal improves tile-level calibration metric C over LogisticProx-Fixed.

Fidelity guardrail:
The paired change in reconstruction metric R must remain within delta_R.

Diversity guardrail:
Null-space variance retention must be at least rho_N of the fixed baseline.

Operational guardrail:
Median added latency must remain below delta_T and convergence above rho_C.
```

Choose numerical thresholds from domain requirements or a small development
set, not after observing final test results.

## 5. Hierarchical hypotheses

Use a sequence:

### H1: covariance information is useful

\[
\text{Oracle better than Fixed on calibration}.
\]

If H1 fails, stop.

### H2: covariance is estimable

\[
\text{SensorCal recovers a pre-registered fraction of Oracle gain}.
\]

If H2 fails, redesign or stop.

### H3: calibration does not destroy reconstruction

\[
\text{SensorCal satisfies fidelity guardrails}.
\]

### H4: result generalizes geographically

\[
\text{paired benefit persists across complete held-out tiles}.
\]

This order prevents a large study when the underlying information is not
useful.

## 6. Confidence intervals

For enough held-out tiles, use a paired tile bootstrap:

1. sample test tiles with replacement;
2. compute mean paired difference;
3. repeat many times;
4. report percentile or bias-corrected interval;
5. retain the point estimate and tile count.

When there are very few test tiles, confidence intervals are unstable.
Emphasize per-tile results and scope claims narrowly.

## 7. Multiple seeds

There are two sources of variability:

1. geographic variability across tiles;
2. optimization variability across training seeds.

Report both. A practical structure is:

- train three SensorCal seeds;
- evaluate each on identical test observations;
- report per-seed tile means;
- summarize between-seed variation;
- avoid treating all seed-tile combinations as independent without a suitable
  hierarchical analysis.

## 8. Multiple comparisons

Six arms and many metrics create many possible comparisons. Avoid searching
for whichever metric becomes significant.

Specify:

- one primary comparison: SensorCal versus Fixed;
- one primary calibration metric;
- one reconstruction guardrail;
- secondary analyses labeled exploratory;
- correction or careful interpretation for many confirmatory tests.

The oracle arm is diagnostic and does not compete as a deployable model.

## 9. Practical significance

A tiny statistically stable change may be operationally irrelevant.

Discuss:

- absolute metric change;
- relative oracle-gap recovery;
- added latency;
- map-level visual impact;
- downstream use implications;
- failure concentration by land cover.

Do not use statistical significance as a substitute for scientific importance.

## 10. Pilot versus final evidence

| Property | Pilot | Full study |
|---|---|---|
| Regions | One held-out region | Multiple held-out regions |
| Purpose | Kill or continue | Support paper claims |
| Backbone | One small/common model | Selected model plus key ablation |
| Seeds | At least three for calibrator | At least three |
| Metrics | Primary calibration and guardrails | Full suite |
| Operator mismatch | Small diagnostic | Systematic ablation |
| Claim scope | Internal decision | Explicit geographic scope |

## 11. Go decision

Proceed when all are true:

- Oracle clearly improves the primary calibration outcome.
- SensorCal closes a meaningful pre-registered fraction of the oracle gap.
- Reconstruction guardrails pass.
- Diversity guardrail passes.
- Result is directionally stable across seeds.
- Solver convergence and runtime pass operational thresholds.
- Diagnostics do not show variance inflation or operator-error absorption.

## 12. No-go decision

Stop or redesign when any critical condition holds:

- Oracle and Fixed are equivalent within practical tolerance.
- SensorCal cannot outperform Fixed.
- NLL improves only through severe oversmoothing.
- test benefit is driven by one tile;
- predicted variance saturates;
- solver failures are frequent;
- result depends on patch leakage;
- only post-hoc metrics appear favorable.

## 13. Reporting negative outcomes

A failed pilot can still become:

- a thesis diagnostic chapter;
- an internal engineering report;
- an ablation in a broader GeoDiff-GAN paper;
- evidence against unnecessary uncertainty modules.

Do not force a journal story from a falsified hypothesis.

## Exercises

1. Explain why patches are not the final statistical unit.
2. Write a primary hypothesis and one fidelity guardrail.
3. Distinguish statistical and practical significance.
4. Explain the oracle-gap recovery concept.
5. Give three conditions that force a no-go decision.

## Mastery Checklist

- [ ] I use paired tile-level comparisons.
- [ ] I pre-register primary metrics and guardrails.
- [ ] I separate geographic and seed variability.
- [ ] I know the hierarchical H1-H4 decision order.
- [ ] I will stop rather than reinterpret a failed pilot.

Next: [46 - Solver and Calibration Diagnostics](46_diagnostics_and_visualization.md).
