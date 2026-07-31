# Paired SOTA High-Frequency Refiner Protocol

## Research question

The defensible research question is not:

> Which super-resolution model is best?

The models use different official crop sizes, training recipes, parameter
budgets, and sometimes different restoration assumptions. A single leaderboard
would mix those differences.

The paired question is:

> For each unchanged open-source x4 SR backbone, does attaching the same
> evidence-constrained high-frequency refiner improve reconstruction on
> Sentinel-2 without degrading LR consistency?

Each paper is therefore its own experiment:

```text
official backbone M  ───────────────> base result
same frozen backbone M + HF refiner ─> improved result
                                      └─ report improved - base
```

Scores from different rows must not be used to rank the backbones. The
cross-backbone conclusion is limited to whether the refiner produces consistent
paired gains across diverse architectures.

## Included official implementations

The executable registry contains more than ten released x4 models. Their
released generator classes and reconstruction heads remain unchanged.

| Key | Paper/model | Year | Domain | Official LR training crop | x4 HR crop |
|---|---|---:|---|---:|---:|
| `swinir` | SwinIR | 2021 | generic SR | 64 x 64 | 256 x 256 |
| `hat` | HAT-S | 2023 | generic SR | 64 x 64 | 256 x 256 |
| `srformer` | SRFormer | 2023 | generic SR | 48 x 48 | 192 x 192 |
| `dat` | DAT | 2023 | generic SR | 64 x 64 | 256 x 256 |
| `omnisr` | OmniSR | 2023 | lightweight SR | 64 x 64 | 256 x 256 |
| `ttst` | TTST | 2024 | remote sensing | 64 x 64 | 256 x 256 |
| `fremamba` | FreMamba/FMSR | 2024 | remote sensing | 64 x 64 | 256 x 256 |
| `swin2mose` | Swin2-MoSE | 2024 | remote sensing | 64 x 64 | 256 x 256 |
| `atd` | ATD | 2024 | generic SR | 64 x 64 | 256 x 256 |
| `mambair` | MambaIR | 2024 | generic restoration | 64 x 64 | 256 x 256 |
| `mfghmoe` | MFG-HMoE | 2025 | remote sensing | 64 x 64 | 256 x 256 |
| `mambairv2` | MambaIRv2 | 2025 | generic restoration | 64 x 64 | 256 x 256 |
| `pft` | PFT-SR | 2025 | generic SR | 64 x 64 | 256 x 256 |
| `sat` | SAT | 2026 | generic SR | 64 x 64 | 256 x 256 |

PFT uses its released progressive focused attention and PixelShuffle head. Its
official source requires `fairscale` and the repository's compiled `smm_cuda`
extension. Replacing that kernel with dense attention would change the method,
so the benchmark fails explicitly when the dependency is absent.

SAT uses the released CVPR 2026 x4 configuration: eight groups, embedding
dimension 228, six heads per group, split sizes 8 and 32, and its original
PixelShuffle reconstruction head.

## Latest methods not in the trainable registry

TexADiff is highly relevant because it is a CVPR 2026 remote-sensing diffusion
method. Its official repository currently marks training-code release as
unfinished. It may be cited and evaluated with released weights, but it cannot
support the required "train base and paired refiner on our data" experiment
until the authors release the training path.

Large foundation restorers such as DiT4SR and VOSR require pretrained diffusion
models, external text/vision encoders, and precomputed latent features. They
belong in a separately budgeted generative case study, not in the deterministic
paired training loop.

## Native-size enforcement

`ModelSpec.native_lr_size` records the released training crop. The benchmark
uses it for all three splits:

- training uses random aligned LR/HR crops;
- validation uses a deterministic aligned center crop;
- test uses the same deterministic rule;
- LR, clean LR, and HR always share the exact crop origin;
- an explicit conflicting `--lr-crop` is rejected.

This means SRFormer is evaluated at 48 x 48 to 192 x 192. Models whose released
training crop is 64 x 64 remain 64 x 64 to 256 x 256. Full 128 x 128 patches are
not silently fed to every architecture.

Different crop sizes are another reason not to compare absolute scores between
model families. Within a pair, the records, crop coordinates, LR degradation,
and target are identical.

## Primary paired experiment

For each model `M`:

1. Train the unchanged official x4 backbone and select its checkpoint using
   validation L1.
2. Freeze that exact checkpoint.
3. Train only the common high-frequency refiner on `HR - M(LR)`.
4. Evaluate `M` and `M + refiner` on identical deterministic samples.
5. Report paired deltas and bootstrap confidence intervals.

The primary experiment freezes the backbone so improvement cannot be explained
by giving the improved branch extra backbone updates. A secondary joint
fine-tuning result is allowed only when accompanied by an equal-compute
continued-training control for the base.

The same refiner architecture, loss weights, optimizer family, update budget,
and early-stopping rule must be used for every compatible backbone. Per-model
hyperparameter tuning would weaken the claim that the module generalizes.

## Required metrics

Report the base value, improved value, and paired delta for each backbone:

| Metric | Desired direction | Interpretation |
|---|---:|---|
| PSNR | higher | pixel fidelity |
| SSIM | higher | structural fidelity |
| LPIPS | lower | perceptual distance |
| DISTS | lower | texture/structure distance |
| edge F1 | higher | edge recovery |
| wavelet L1 | lower | high-frequency reconstruction error |
| LR re-degradation L1 | no worse | evidence consistency |
| inference time and memory | lower | cost of improvement |

Also report the fraction of test patches improved and a paired bootstrap 95%
confidence interval. Across backbones, report only the refiner's win rate and
effect consistency. Do not sort models into a leaderboard.

## Controls needed for publication

- `M` versus frozen `M + refiner`.
- `M` with equal additional training updates but no refiner.
- Refiner without high-pass projection.
- Refiner without LR evidence conditioning.
- Refiner with 0, 1, and 3 consistency projection steps.
- Identical-parameter generic residual head as a capacity control.
- Unseen-tile test split and, if possible, a second sensor or dataset.
- At least three seeds for the final subset of representative backbones.

The useful claim is a reusable refinement principle, not that one more module
beats every published model. It becomes publishable when gains are repeated
across transformer, state-space, lightweight, and remote-sensing backbones while
LR consistency and artifact tests remain controlled.

## Architecture fidelity

`architecture_mode=official` is mandatory for the primary result. The original
generator class and upsampling head are retained, including PixelShuffle where
the paper uses it.

`resize_conv_ablation` changes a backbone and is therefore outside this paired
claim. It may be reported as a separate artifact ablation, never as the
published competitor.

## Execution sequence

1. Probe every source at `native_lr_size` and require an exact x4 output.
2. Run a 20-update smoke test for each official backbone.
3. Train each base independently and retain best/latest checkpoints.
4. Train the frozen paired refiner from the selected base checkpoint.
5. Evaluate base and improved branches in one process on the same batches.
6. Save per-patch metrics, paired deltas, images, frequency plots, and timing.
7. Run final seeds only after artifact and LR-consistency checks pass.

The benchmark registry, native crop enforcement, and SN-HFR runner now cover
steps 1-6. The high-frequency wrapper is evaluated as a paired extension and is
not folded into a cross-model score table.
