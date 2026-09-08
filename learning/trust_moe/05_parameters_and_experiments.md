# 05. Parameters, Experiments and Results

## Audited Default Capacity

For `TrustMoESR()` with feature width 32, base width 32, two base groups with two
blocks each, five experts and top-k=2:

| Component | Parameters |
|---|---:|
| Base | 76,531 |
| Encoder | 13,024 |
| Transformer router | 17,670 |
| Five experts | 96,815 |
| RGB trust head | 1,347 |
| Total | 205,387 |
| Trainable during residual training | 128,856 |

Each expert has 19,363 parameters. These counts exclude the optional training-only
discriminator and optimizer/EMA storage. Counting parameters is not counting FLOPs.
Different regions may use different experts, so top-k=2 does not mean only two
experts' parameters are used anywhere in an entire image.

This base is smaller than the previously reported 365,667-parameter base. Do not
present those as equal-capacity backbones. Small/medium/large presets are not currently
defined for this model. First isolate base width/depth changes from routing changes.

## Ablation Questions

| Control | Question |
|---|---|
| Bicubic and frozen base | Does extra learning improve reconstruction? |
| Single expert | Are multiple experts needed beyond direct residual learning? |
| All-region Transformer MoE | Does regional bypass reduce time at acceptable quality? |
| Uniform selected regions | Does predicted difficulty choose useful locations? |
| Local router | Is global context worth attention overhead? |
| No trust gate | Does learned acceptance help? |
| Adversarial objective | Does the critic help useful structure rather than just appearance? |
| Adaptive K | Does a rank-based variable expert count improve the trade-off? |

The experiments have equal epochs, not equal FLOPs or equal parameter counts. The
single-expert and local-router controls have different capacity. Report that rather
than treating an ablation as a perfectly controlled capacity comparison.

## Metrics

| Metric | Direction | Interpretation |
|---|---|---|
| L1 | Lower | Average absolute numerical error on valid pixels |
| PSNR | Higher | Logarithmic pixel fidelity at the declared range |
| SSIM | Higher | Local structure/luminance/contrast agreement |
| Edge F1 | Higher | Agreement of detected boundaries under a specified threshold |
| ERGAS | Lower | Scale-aware normalized per-band reconstruction error |
| SAM | Lower | Spectral-vector angle; here only RGB, not a full multispectral claim |
| UIQI | Higher | Correlation, luminance and contrast agreement |
| sCC | Higher | High-pass structural correlation |
| Redegradation L1 | Lower | Approximate consistency with LR, not an exact sensor-physics test |

At data range one, `PSNR = -10*log10(MSE)`. An MSE of 0.0004 corresponds to about
33.98 dB; 0.0003 corresponds to about 35.23 dB. The reported dataset PSNR is a mean
of per-image scores, which is not generally PSNR of the pooled dataset MSE.

A higher Edge F1 with unchanged PSNR is a useful signal, not proof of correct new
objects. Check threshold sensitivity, sCC, images and spectral errors together.
If SSIM improves and PSNR declines, report both rather than saying quality improved
without qualification.

## Selection and Independence

Use validation for checkpoint, coverage and top-k selection. Keep the test pairs
fixed and do not choose the best test run. The notebook saves a test plan and model
copies before testing. Bootstrap paired differences against the same base and
controls. Nearby patches are correlated; tile-cluster intervals and unseen-tile
tests are needed for stronger generalization claims.

At least three training seeds help distinguish a robust result from initialization
luck. They are not three independent datasets. Weak directory labels are not
verified pixel-wise scene classes.

## Efficiency and Novelty

Measure synchronized inference after warm-up using the same GPU, precision, image
size and batch. Report mean, median and p95 latency, throughput, parameter counts,
and the memory measurement scope. Coverage percentages cannot replace timing.

The research hypothesis is useful selective correction at a better quality/time
trade-off, not "we used a Transformer." Close prior work is listed in the
[research protocol](../GeoDiff_TrustMoE_Research_Protocol.md). The current notebook
does not reproduce those external papers, and no first/SOTA claim is established.

## Keep These Files

Keep the executed notebook, configs, dataset manifests, source identity, best/last
checkpoints, per-image and per-class results, routing usage, timing reports and
representative inputs/outputs. Keep negative experiments too. The final ZIP excludes
the complete raw dataset, so retain that separately.

Next: [Supervisor questions](06_supervisor_questions.md).
