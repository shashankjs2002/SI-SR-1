# 14 - Kaggle Execution and Research Workflow

## Learning Objectives

- run the project from a clean Kaggle clone;
- distinguish smoke verification from research training;
- manage data, checkpoints, outputs, and interruptions;
- organize experiments so results remain reproducible.

## 1. Kaggle Workflow

Use [`GeoDiff_GAN_Kaggle.ipynb`](../kaggle/GeoDiff_GAN_Kaggle.ipynb).

The notebook has two independent controls:

| Control | Responsibility |
|---|---|
| `FAST_DEV_RUN` | Tile limit, epoch count, sampling steps, and diagnostic frequency |
| `MODEL_SIZE` | `xs`, `medium`, or `large` architecture capacity |

Dataset and evaluation controls are also independent:

| Control | Responsibility |
|---|---|
| `TEST_SAFE_PREFIXES` | SAFE filename prefixes reserved for test |
| `VAL_SAFE_PREFIXES` | SAFE filename prefixes reserved for validation |
| `UNMATCHED_SAFE_SPLIT` | Split for all other SAFE products, normally `train` |
| `VALIDATION_LIMIT` | Maximum validation batches per validated epoch |
| `EVALUATION_LIMIT` | Maximum patches in each final validation/test report |

Use XS only to verify execution. Medium preserves the research architecture at 21.13M core
parameters and is the practical starting point for 16 GB GPUs. Large uses 81.86M core parameters
and requires a substantially larger compute budget.

```mermaid
flowchart TD
    GH["Clone GitHub repository"] --> INST["Install editable package"]
    DATA["Attach Sentinel-2 SAFE dataset"] --> PREP["Prepare patches and manifest"]
    PREP --> CAP["Optional separate Qwen caption session"]
    CAP --> TRAIN["Train selected stages"]
    INST --> TRAIN
    TRAIN --> DEBUG["Run diagnostics"]
    DEBUG --> EVAL["Evaluate test tiles"]
    EVAL --> EXPORT["Export checkpoints, metrics, figures"]
```

Kaggle settings:

- enable GPU accelerator;
- enable Internet for cloning/model downloads when needed;
- attach input data as a Kaggle Dataset;
- write all generated files under `/kaggle/working`;
- preserve important outputs by creating a notebook version or dataset.

## 2. Clone-and-Install Principle

The notebook installs the repository without replacing Kaggle's CUDA-compatible PyTorch:

```bash
git clone https://github.com/OWNER/geodiff-gan.git
cd geodiff-gan
pip install -e .
```

Replacing PyTorch blindly can break CUDA compatibility. Install only required extras for the current
session.

## 3. Smoke Run versus Research Run

`FAST_DEV_RUN = True`:

- tiny sample count;
- one epoch or few steps per stage;
- verifies code paths and checkpoint transfer;
- does not produce scientific results.

`FAST_DEV_RUN = False`:

- paper-scale data;
- intended epoch schedules;
- frozen SigLIP encoder;
- research metrics and ablations.

Architecture capacity is selected separately by `MODEL_SIZE`.

```mermaid
flowchart LR
    S["Smoke run"] --> Q{"All stages, diagnostics,<br/>and checkpoint paths work?"}
    Q -- "No" --> FIX["Fix before expensive run"]
    Q -- "Yes" --> R["Research run"]
```

Never report smoke-run metrics as model performance.

## 4. Session Separation

Captioning Qwen3-VL-8B in 4-bit and training SR compete for GPU memory. Recommended:

1. data preparation session;
2. captioning session;
3. base/VAE session;
4. diffusion/joint/edit sessions;
5. evaluation session.

Persist outputs between sessions as Kaggle datasets or notebook versions.

## 5. Checkpoint Chain

Before starting a stage, verify the previous checkpoint exists:

```text
runs/base/base_epoch_....pt
runs/vae/vae_epoch_....pt
runs/diffusion/diffusion_epoch_....pt
runs/joint/joint_epoch_....pt
runs/edit/edit_epoch_....pt
```

If the schedule changes, filenames change. Update `init_checkpoint` in the stage overlays instead of
assuming the old path.

## 6. Resource Planning

Memory pressure roughly follows:

- image area;
- activation channels;
- number of retained layers;
- attention context;
- optimizer state;
- discriminator activation;
- batch size.

Recovery order for out-of-memory:

1. confirm no stale models, especially Qwen, remain loaded;
2. reduce batch size;
3. increase gradient accumulation;
4. enable gradient checkpointing;
5. use FP16;
6. reduce diagnostic capture frequency;
7. use compact smoke config only for debugging;
8. change patch/model size only as a documented research variant.

Do not silently train a different architecture after an OOM and compare it as the same model.

## 7. Experiment Directory

Recommended structure:

```text
runs/
  experiment_name/
    resolved_config.yaml
    environment.txt
    manifest_hash.txt
    checkpoints/
    logs/
    diagnostics/
    validation_metrics.json
    test_metrics.json
    figures/
    notes.md
```

Use experiment names that encode the question, not every hyperparameter:

```text
full_seed7
ablate_gate_seed7
ablate_wavelet_d_seed7
fixed_degradation_seed7
```

## 8. Run Ledger

Maintain a table:

| Run | Commit | Data hash | Seed | Change | Status | Best checkpoint | Decision |
|---|---|---|---:|---|---|---|---|
| full_s7 | | | 7 | full model | | | |
| no_gate_s7 | | | 7 | remove gate | | | |

Write the hypothesis and success criterion before launching:

> Removing the gate will increase prompt alignment but worsen mismatched-prompt LR consistency. The
> hypothesis is supported if both changes occur on held-out tiles.

This prevents post-hoc interpretation.

## 9. Failure Recovery

If a Kaggle session ends:

- use `resume` only within the same stage;
- verify optimizer/scaler states were saved;
- do not restart a later stage from an arbitrary "best-looking" output;
- preserve logs and resolved configuration;
- rerun diagnostics after resuming.

If an input dataset path changes, update configuration explicitly and verify manifest identity.

## 10. Minimum Research Execution Order

1. Run unit tests locally or in Kaggle.
2. Run the complete smoke notebook.
3. Prepare and audit tile-split data.
4. Verify CHHATARPUR/test, CHHATARPUR/validation, and unmatched/train counts.
5. Train deterministic baseline.
6. Train full stage chain while monitoring tqdm ETA and per-epoch validation.
7. Run diagnostics on fixed validation cases.
8. Evaluate validation and test independently against baselines.
9. Run priority ablations.
10. Repeat important runs with additional seeds.
11. Generate paper tables, figures, uncertainty maps, and failures.

## Exercises

1. Explain why editable installation is useful in the notebook.
2. List files that must persist between the diffusion and joint sessions.
3. Design a run name and ledger entry for testing coarse LR feature fusion.
4. What is the correct response to OOM during joint training?
5. Write a pre-registered success criterion for the wavelet discriminator.

## Mastery Checklist

- [ ] I can execute the notebook from a clean clone.
- [ ] I distinguish smoke verification from research evidence.
- [ ] I can preserve and resume stage-specific state correctly.
- [ ] I maintain an experiment ledger with hypotheses and decisions.
- [ ] I can recover from resource limits without silently changing the study.

Next: [15 - Paper, Thesis, and Viva Guide](15_paper_thesis_and_viva.md).
