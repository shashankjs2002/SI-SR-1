# 03. Routing, Experts and Trust

## Four Different Controls

| Quantity | Question it answers | Granularity |
|---|---|---|
| Region coverage | Where may extra computation run? | Selected LR blocks |
| Top-k | How many experts handle each selected block? | Per block |
| Mixture weights | How are selected expert proposals combined? | Per block, per expert |
| RGB trust | How much correction is accepted? | Per HR pixel and RGB channel |

These are not interchangeable. A trust value of zero does not save the expert
computation already spent. Spatial bypass saves that computation before execution.

## Difficulty Is Not Ground Truth at Inference

During training, the router's scalar target is log10 of local masked base MSE.
For example, MSE 0.0001 has score -4, while MSE 0.001 has score -3. The larger
score means greater base error. Inference uses a predicted score from LR/base cues.

High difficulty can reflect an edge, texture, misregistration, radiometric mismatch
or temporal change. The predictor does not distinguish recoverable information from
every other error source. Calling it calibrated uncertainty would be inaccurate.

## Discrete Selection and Learning

The model chooses the highest-scoring regions, then top-k expert logits inside them.
The hard regional ranking is detached; difficulty learns from its separate loss.
Expert gates use softmax probabilities masked by discrete assignments and normalized
by a detached selected sum. This is a surrogate gradient: it preserves a training
signal for top-1, rather than differentiating through the discrete top-k decision.

Route exploration and a load-balance auxiliary loss encourage expert use. Exploration
is used only while training. Expert weights are not manually labelled urban/forest/
water. Similar experts and unused experts are possible and must be measured.

## What Makes Execution Sparse?

Selected feature blocks are gathered from an unfolded view and grouped by expert.
Only those blocks are sent through that expert. Predictions are multiplied by their
mixture weights and accumulated into the corresponding output blocks.

An 8x8 LR core has a four-pixel halo on every side, making a 16x16 expert input.
The halo provides context, but duplicates work. After upsampling, only the 24x24 HR
core is retained. Halos do not guarantee the absence of seams: neighboring blocks
may choose different experts. Inspect output boundaries.

The base, encoder, router and trust head remain dense. Therefore half spatial
coverage is not a claim of half total FLOPs or twice the speed.

## Per-Band Trust Label

Let e = HR - B, and let r be a proposed residual for one channel. Minimizing local
weighted squared error `(t*r - e)^2` gives the least-squares coefficient:

```text
t* = clip(local_mean(mask * r * e) /
          (local_mean(mask * r^2) + 1e-8), 0, 1)
```

The implemented neighborhood is 9x9 HR pixels. If r equals e, t approaches one.
If r points in the opposite direction, the clipped coefficient is zero. If r is
twice the desired error correction, t approaches one-half. A zero proposal yields
zero. This is calculated separately for red, green and blue.

The target-derived coefficient supervises a predictor; it is not applied as an
oracle during testing. A trust predictor can be wrong. T in [0,1] does not guarantee
PSNR improvement, because even a small accepted correction can have the wrong sign.

## Optional Adaptive K

The implemented variant ranks difficulty among active regions and assigns 1..top-k
expert slots. Inactive regions get zero. This is an explicit rank policy, not a
learned optimal computation controller. With top-k=1 it cannot vary active K.

Top-k may be changed after training without changing tensor parameter shapes, but
this changes inference behavior. Treat it as an ablation and choose settings on
validation, not on test. An expert that was poorly trained does not become useful
merely because a larger top-k activates it.

## Check Yourself

1. Can five different experts execute somewhere in one image with top-k=2? Yes.
2. Does a near-zero trust map imply low inference cost? No.
3. Why is the local trust coefficient a training label rather than a usable test oracle?

Next: [Training and losses](04_training_and_losses.md).
