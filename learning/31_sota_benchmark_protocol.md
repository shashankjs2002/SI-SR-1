# Recent SR Benchmark Protocol

## Is the proposed comparison correct?

The objective is correct: compare GeoDiff-GAN against recent open-source
super-resolution models trained on the same Sentinel-2 patches.

The original wording needs correction. There is no single "satellite SR SOTA in
2026" because papers use different:

- sensors and ground sampling distances;
- synthetic or real LR formation;
- single-image, multi-image, or continuous-scale inputs;
- train/test geography;
- RGB, multispectral, or panchromatic bands;
- PSNR color spaces and crop borders.

The defensible claim is:

> We compare recent open-source SR architectures under a controlled Sentinel-2
> RGB synthetic 40 m to native 10 m x4 protocol.

## Included papers

| Adapter | Paper | Year | Domain | Official repository |
|---|---|---:|---|---|
| SwinIR | Image Restoration Using Swin Transformer | 2021 | generic SR | `JingyunLiang/SwinIR` |
| HAT-S | Activating More Pixels in Image SR Transformer | 2023 | generic SR | `XPixelGroup/HAT` |
| SRFormer | Permuted Self-Attention for SISR | 2023 | generic SR | `HVision-NKU/SRFormer` |
| DAT | Dual Aggregation Transformer for Image SR | 2023 | generic SR | `zhengchen1999/DAT` |
| OmniSR | Omni Aggregation Networks for Lightweight SR | 2023 | generic SR | `Francis0625/Omni-SR` |
| TTST | Top-k Token Selective Transformer for RSI-SR | 2024 | remote sensing | `XY-boy/TTST` |
| MFG-HMoE | Heterogeneous MoE for Remote Sensing Image SR | 2025 | remote sensing | `Mr-Bamboo/MFG-HMoE` |
| FreMamba/FMSR | Frequency-Assisted Mamba for RSI-SR | 2024 | remote sensing | `XY-boy/FreMamba` |

The first seven are the default benchmark. FreMamba is optional because
`mamba_ssm` and `causal_conv1d` compile against a specific CUDA/PyTorch stack.

## Methods deliberately not presented as trained baselines

- A repository with inference weights but no released training code cannot meet
  the requirement to train on our data.
- Continuous-scale diffusion is a different task and compute regime. It should
  be a separately budgeted experiment rather than silently mixed with
  deterministic x4 backbones.
- A 2026 paper should not be called reproducible solely because a repository
  exists. The required training path must also be public and executable.

## Shared data contract

All models use the existing manifest and `SentinelPatchDataset`.

Training:

- HR comes from the same accepted NPZ patches.
- LR is regenerated with random mild MTF/noise/quantization degradation.
- LR crop is 64 x 64 and target crop is 256 x 256.
- Horizontal/vertical flips and rotations come from the shared dataset loader.

Validation and test:

- Degradation is deterministic with seed 42.
- Full 128 x 128 LR and 512 x 512 HR patches are evaluated.
- Identical records are used for every model.
- Screening uses the first 40 test records to match the existing GeoDiff
  three-variant notebook. Full-test reporting requires rerunning every method,
  including GeoDiff, without a limit.

## Shared optimization

Comparisons use optimizer updates rather than epochs.

```text
L = Charbonnier
  + 0.20 * (1 - SSIM)
  + 0.10 * gradient L1
  + 0.05 * Haar-wavelet L1
```

The optimizer is AdamW with the same initial learning rate, weight decay,
gradient clipping, cosine schedule, validation cadence, and early-stopping rule.

This tests architecture quality under one satellite-specific objective. It does
not reproduce each paper's original training recipe.

## No-PixelShuffle constraint

The official attention/state-space/aggregation backbones are retained. Their
sub-pixel reconstruction heads are replaced with a shared two-stage
resize-convolution head.

MFG-HMoE uses a specialized resize-convolution head that retains:

- heterogeneous 1 x 1 and 3 x 3 experts;
- dual group/expert routing;
- spatial routing features from multilevel aggregation.

Every constructed model is scanned for `torch.nn.PixelShuffle`; training stops
if one remains.

The resulting names must be written as:

- `SwinIR + resize-conv adapter`;
- `HAT-S + resize-conv adapter`;
- and so on.

They are not exact reproductions of published upsampling heads.

## Recommended experiment sequence

1. Run `smoke` for 20 updates to verify every model and output path.
2. Run `screening` for 5,000 updates on all seven models.
3. Select the strongest three using validation PSNR, SSIM, edge F1, and
   re-degradation error.
4. Run `paper` for 50,000 updates only on the shortlisted models.
5. Repeat the final comparison on geographically unseen tiles.
6. Report parameter count, GPU memory, training time, and inference time.

## Interpretation

PSNR alone is insufficient.

- High PSNR with low edge F1 indicates over-smoothing.
- Good texture with poor re-degradation error indicates weak evidence
  consistency or hallucination.
- A small gain on intra-tile test patches does not establish geographic
  generalization.
- GeoDiff-GAN has stochastic sampling and a multi-stage optimizer budget.
  Report both the practical final comparison and a matched-compute ablation.
