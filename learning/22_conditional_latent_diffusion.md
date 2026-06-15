# 22 - Conditional Latent Diffusion

## Purpose

Latent diffusion models the ambiguous residual distribution:

\[
p(z_0\mid y,c,\theta,m),
\]

conditioned on:

- LR spatial features \(y\);
- optional text tokens \(c\);
- degradation metadata \(\theta\);
- operating mode \(m\in\{SR,edit\}\).

It generates a 4x64x64 residual latent, not a full 512x512 RGB image.

## Forward Diffusion

The scheduler uses a cosine noise schedule with 1,000 training steps:

\[
\bar\alpha_t
=
\frac{
\cos^2\left(
\frac{t/T+s}{1+s}\frac{\pi}{2}
\right)
}{
\cos^2\left(
\frac{s}{1+s}\frac{\pi}{2}
\right)
},
\qquad s=0.008.
\]

The noisy latent is:

\[
z_t
=
\sqrt{\bar\alpha_t}z_0+
\sqrt{1-\bar\alpha_t}\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
\]

## Velocity Prediction

The target velocity is:

\[
v_t
=
\sqrt{\bar\alpha_t}\epsilon
-
\sqrt{1-\bar\alpha_t}z_0.
\]

The U-Net predicts:

\[
\hat v_\phi
=
f_\phi(z_t,t,c,\theta,m,f_{LR}).
\]

The clean latent estimate is:

\[
\hat z_0
=
\sqrt{\bar\alpha_t}z_t
-
\sqrt{1-\bar\alpha_t}\hat v_\phi.
\]

Velocity prediction balances signal and noise targets across timesteps better than predicting only
noise in many latent-diffusion settings.

## Small-Preset U-Net Shapes

Widths are `[48, 96, 144, 192]`.

```text
latent z_t:       4x64x64
LR condition:    48x64x64
concatenation:   52x64x64
input conv:      48x64x64
down level 0:    48x64x64
down level 1:    96x32x32   + cross-attention
down level 2:   144x16x16   + cross-attention
down level 3:   192x8x8
middle:         192x8x8      + cross-attention
up level 2:     144x16x16    + cross-attention
up level 1:      96x32x32    + cross-attention
up level 0:      48x64x64
output velocity:  4x64x64
```

## Scalar Conditioning

Time, degradation, and mode embeddings are added:

\[
h_{\text{cond}}
=
h_t(t)+h_\theta(\theta)+E_m(m).
\]

For the small model, their common dimension is:

\[
d_{\text{cond}}=4\times48=192.
\]

Each conditioned residual block projects this vector into channel-wise scale and shift:

\[
[\gamma,\beta]=W h_{\text{cond}},
\]

\[
\widetilde F
=
\operatorname{GN}(F)\odot(1+\gamma)+\beta.
\]

This lets timestep, sensor degradation, and SR/edit mode modify every spatial location without
destroying spatial coordinates.

## Cross-Attention

At selected resolutions, image features become queries and text tokens become keys/values:

\[
Q=W_Q\operatorname{Flatten}(F),
\quad
K=W_Kc,
\quad
V=W_Vc,
\]

\[
\operatorname{CA}(F,c)
=
F+
\operatorname{reshape}
\left[
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_h}}
\right)V
\right].
\]

Cross-attention at 32x32 and 16x16 gives prompt context without the cost of attention at full HR.

## U-Net Skip Connections

Encoder features are concatenated with matching decoder resolutions:

\[
F^{up}_l
=
U_l
\left(
[\,\operatorname{Up}(F^{up}_{l+1});F^{down}_l\,]
\right).
\]

Skips preserve spatial information lost during downsampling.

## DDIM Sampling

Inference starts from:

\[
z_T\sim\mathcal N(0,I).
\]

For a selected sequence of 20 timesteps, the model estimates \(\hat z_0\) and noise:

\[
\hat\epsilon
=
\sqrt{1-\bar\alpha_t}z_t+
\sqrt{\bar\alpha_t}\hat v_t.
\]

The deterministic DDIM-style update is:

\[
z_{t'}
=
\sqrt{\bar\alpha_{t'}}\hat z_0+
\sqrt{1-\bar\alpha_{t'}}\hat\epsilon.
\]

No additional random noise is injected between sampling steps.

## Classifier-Free Guidance

With conditional and null predictions:

\[
\hat v_{\text{CFG}}
=
\hat v_{\varnothing}
+w(\hat v_c-\hat v_{\varnothing}).
\]

- \(w=1\): ordinary conditional prediction;
- \(w>1\): stronger prompt influence;
- excessive \(w\): hallucination, oversaturation, or structural drift.

## Training Objective

\[
\mathcal L_{\text{diff}}
=
w_{\text{SNR}}(t)
\left\|
\hat v_\phi-v_t
\right\|_2^2.
\]

The decoder is frozen during the dedicated diffusion stage so the latent target remains stable.

## Gradient Checkpointing

During training, U-Net levels can be recomputed in backward instead of storing all activations.
This reduces memory at the cost of additional compute and does not alter the mathematical model.

## Failure Modes

| Symptom | Meaning |
|---|---|
| latent std never decreases | denoising failure |
| final latent nearly identical for all inputs | conditional collapse |
| prompt changes geometry in SR mode | excessive text authority |
| output ignores degradation vector | sensor conditioning unused |
| samples vary wildly | weak conditioning or unstable decoder |
| samples never vary | diffusion prior ignored |

## Implementation

See [`models/diffusion.py`](../src/geodiff_gan/models/diffusion.py) and conditioned blocks in
[`models/blocks.py`](../src/geodiff_gan/models/blocks.py).
