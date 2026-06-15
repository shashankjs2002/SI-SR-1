# 25 - Dual-Discriminator GAN

## Purpose

GeoDiff-GAN uses two discriminator families during joint training:

1. a conditional multi-scale PatchGAN for spatial realism;
2. a Haar-wavelet discriminator for high-frequency realism.

Neither discriminator is used during inference.

## Conditional PatchGAN

The LR condition is resized to the HR candidate:

\[
y_\uparrow
=
\operatorname{BilinearResize}(y,\operatorname{size}(x)).
\]

The input is:

\[
q=[x;y_\uparrow]\in\mathbb R^{B\times6\times H\times W}.
\]

For small-preset discriminator width \(C=24\):

```text
6 x H x W
-> 24 x H/2 x W/2
-> 48 x H/4 x W/4
-> 96 x H/8 x W/8
-> 96 x approximately H/8 x W/8
-> 1-channel patch logits
```

Every convolution uses spectral normalization:

\[
\bar W=\frac{W}{\sigma_{\max}(W)}.
\]

This constrains the layer's largest singular value and stabilizes adversarial training.

## Patch Decision

The discriminator outputs a grid:

\[
D_p(x,y)\in\mathbb R^{B\times1\times h\times w}.
\]

Each logit judges a local receptive field rather than the complete image. PatchGAN is suited to
texture and local structural realism but does not guarantee global geographic correctness.

## Multi-Scale Discriminator

Three PatchGANs inspect progressively downsampled images:

\[
D_{MS}(x,y)
=
\{
D_0(x,y),
D_1(P_2x,P_2y),
D_2(P_4x,P_4y)
\},
\]

where \(P_2\) is 2x average pooling.

The finest scale judges local texture. Coarser scales judge larger structures with a larger
effective receptive field.

## Haar-Wavelet Decomposition

For each 2x2 pixel block:

\[
LL=\frac{x_{00}+x_{01}+x_{10}+x_{11}}{2},
\]

\[
LH=\frac{-x_{00}-x_{01}+x_{10}+x_{11}}{2},
\]

\[
HL=\frac{-x_{00}+x_{01}-x_{10}+x_{11}}{2},
\]

\[
HH=\frac{x_{00}-x_{01}-x_{10}+x_{11}}{2}.
\]

Only high-frequency bands are used:

\[
W_H(x)=[LH;HL;HH]\in\mathbb R^{B\times9\times H/2\times W/2}.
\]

The LR image is area-resized to the same size and repeated three times:

\[
y_W\in\mathbb R^{B\times9\times H/2\times W/2}.
\]

The wavelet discriminator receives:

\[
[W_H(x);y_W]\in\mathbb R^{B\times18\times H/2\times W/2}.
\]

It then applies another conditional PatchGAN.

## Why Wavelets?

RGB PatchGAN can reward realistic color and local texture while ignoring whether directional
high-frequency bands match natural satellite statistics. Wavelet discrimination directly exposes:

- horizontal detail;
- vertical detail;
- diagonal/fine texture;
- checkerboard or ringing artifacts.

It still judges realism, not paired correctness.

## Hinge Loss

For real target \(x_r\) and generated output \(x_f\):

\[
\mathcal L_D
=
\mathbb E[\max(0,1-D(x_r,y))]
+
\mathbb E[\max(0,1+D(x_f,y))].
\]

The generator adversarial loss is:

\[
\mathcal L_G^{adv}
=
-\mathbb E[D(x_f,y)].
\]

Losses from all multi-scale logits and the wavelet discriminator are combined. The generator
adversarial coefficient is deliberately low because scientific fidelity has priority over texture
realism.

## Conditionality

The discriminator does not ask only:

> Does this look like a satellite image?

It asks:

> Does this HR patch look realistic given this LR observation?

However, concatenating an upsampled LR condition does not prove the discriminator uses it.
Condition-shuffling ablations are required.

## Failure Modes

| Symptom | Interpretation |
|---|---|
| discriminator loss near zero | discriminator overwhelms generator |
| discriminator remains random | insufficient capacity or learning |
| sharp but wrong roads/buildings | realism exceeds evidence constraint |
| wavelet detail becomes noisy | high-frequency discriminator rewards texture noise |
| periodic grid persists | discriminator cannot detect or penalize phase pattern |
| training oscillates | adversarial weight/LR imbalance |

## Required Ablations

- no adversarial training;
- PatchGAN only;
- wavelet discriminator only;
- both discriminators;
- single versus three scales;
- shuffled LR condition;
- alternative spectral or FFT discriminator.

## Implementation

See [`models/discriminators.py`](../src/geodiff_gan/models/discriminators.py) and Haar operations in
[`models/blocks.py`](../src/geodiff_gan/models/blocks.py).
