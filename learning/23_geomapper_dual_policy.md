# 23 - GeoMapper and Dual-Policy Gating

## Purpose

GeoMapper converts a denoised diffusion latent into spatial decoder content, four layer-wise style
vectors, an evidence-confidence map, and an edit-permission map.

It separates two different questions:

1. **Evidence confidence:** how much generated detail is supported in SR mode?
2. **Edit permission:** where may prompt-driven synthetic changes be applied in edit mode?

## Inputs and Outputs

For the small preset:

```text
denoised latent:       B x 4 x 64 x 64
LR feature f64:        B x 48 x 64 x 64
text context:          B x T x 768
mode:                  B
content output:        B x 48 x 64 x 64
evidence confidence:   B x 1 x 64 x 64
edit permission:       B x 1 x 64 x 64
four style vectors:    4 x (B x 96)
```

## Content Mapping

The latent and LR evidence are concatenated:

\[
M_0
=
\operatorname{Conv}_{3\times3}
\left(
[z_0;f_{64}]
\right).
\]

Four residual blocks produce:

\[
M=R_4(R_3(R_2(R_1(M_0)))).
\]

This is the spatial content tensor consumed by the decoder.

## Text Context Map

Token embeddings are mean pooled:

\[
\bar c=\frac{1}{T}\sum_{i=1}^{T}c_i.
\]

A linear projection converts text into mapper channels:

\[
c_M=W_c\bar c.
\]

It is broadcast over the 64x64 grid:

\[
C_M(x,y)=c_M.
\]

Text alone has no spatial position. Spatial control arises only when this global context is
multiplied by evidence or permission maps.

## Evidence Confidence

LR evidence is projected:

\[
E_{LR}=W_Ef_{64}.
\]

The confidence network is:

\[
C
=
\sigma
\left(
W_2\operatorname{SiLU}
\left(
W_1[M;E_{LR}]
\right)
\right).
\]

Thus:

\[
C\in[0,1]^{B\times1\times64\times64}.
\]

The map uses both generative content and observed LR features. It is not uncertainty by itself; it
is a learned confidence policy trained against local reconstruction accuracy.

## Edit Permission

Before mode masking:

\[
P_{\text{raw}}
=
\sigma
\left(
V_2\operatorname{SiLU}
\left(
V_1[M;C_M;C]
\right)
\right).
\]

Let \(m=0\) for SR and \(m=1\) for edit:

\[
P=mP_{\text{raw}}.
\]

Therefore:

\[
P=0\quad\text{in SR mode}.
\]

This hard mode multiplication is the key safety boundary preventing edit residuals from entering
ordinary SR.

## Prompt Injection

The prompt-strength map is:

\[
S_p
=
0.15C(1-m)+Pm.
\]

In SR mode:

\[
S_p^{SR}=0.15C.
\]

Text semantics are weak and evidence constrained.

In edit mode:

\[
S_p^{edit}=P.
\]

Prompt influence follows the learned edit-permission map.

Spatial content becomes:

\[
\widetilde M=M+C_M\odot S_p.
\]

## Style Generation

Spatial content and prompt strength are globally pooled:

\[
\bar M=\operatorname{GAP}(\widetilde M),
\qquad
\bar S_p=\operatorname{GAP}(S_p).
\]

The style input is:

\[
s_{in}
=
[\bar M;\bar c\odot\bar S_p].
\]

Each decoder stage has an independent MLP:

\[
s_l
=
W_{l,2}
\operatorname{SiLU}
\left(
W_{l,1}s_{in}
\right),
\qquad l=0,1,2,3.
\]

For the small model, \(s_l\in\mathbb R^{B\times96}\).

## Evidence Calibration Loss

The ungated SR candidate is:

\[
x_u=x_{\text{base}}+H(r_{\text{detail}}).
\]

Local error is:

\[
e(x,y)
=
\operatorname{AvgPool}_{9\times9}
\left(
\frac{1}{3}\sum_c|x_u-x_{\text{target}}|
\right).
\]

The calibration target decreases with error, conceptually:

\[
C^*(x,y)\approx\exp(-e(x,y)/\tau).
\]

The evidence loss trains \(C\) to represent local reliability rather than merely shrinking all
residuals.

The `small_12tile_improved` experiment adds a centered cosine-correlation term between predicted
confidence and the local accuracy target. This explicitly supervises spatial ranking and
discourages the nearly uniform low-confidence shortcut observed in the six-tile run.

## Evidence Versus Uncertainty

Evidence confidence is predicted from one forward pass. Stochastic uncertainty is estimated from
variation across multiple diffusion samples. The implementation combines them multiplicatively:

\[
C_{\text{final}}
=
C_{\text{evidence}}\odot\exp(-U/s_u).
\]

## Failure Modes

| Symptom | Meaning |
|---|---|
| confidence near zero everywhere | global abstention shortcut |
| confidence near one everywhere | unsupported residual authority |
| very low confidence std | poor spatial selectivity |
| nonzero edit permission in SR | policy-boundary failure |
| prompt dominates SR content | excessive semantic injection |
| style vectors constant | style-head collapse |

## Required Ablations

Compare:

- no evidence gate;
- evidence gate without calibration loss;
- no edit gate;
- shared single gate;
- full dual-policy mapping;
- no text context;
- no LR evidence projection.

## Implementation

See `GeoMapper` in [`models/generator.py`](../src/geodiff_gan/models/generator.py).
