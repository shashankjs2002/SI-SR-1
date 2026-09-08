# GeoDiff-TrustMoE Learning Guide

This guide describes the implementation introduced in commit `ddef690` on
`3x-continued`, not a published model reproduced under another name.
Documentation date: 2026-09-08. No new GPU results are claimed here.

## Reading Order

| Chapter | What you should be able to explain |
|---|---|
| [01. Problem and foundations](01_problem_and_foundations.md) | Why cross-sensor SR is useful and why sharper does not mean correct |
| [02. Architecture and tensor shapes](02_architecture_and_shapes.md) | Trace the actual input through every component |
| [03. Routing, experts and trust](03_routing_experts_and_trust.md) | Distinguish spatial coverage, top-k, mixture weights and trust |
| [04. Training and losses](04_training_and_losses.md) | Explain what learns, why each loss exists, and how resume works |
| [05. Parameters and experiments](05_parameters_and_experiments.md) | Interpret capacity, speed, metrics and ablations fairly |
| [06. Supervisor questions](06_supervisor_questions.md) | Answer short questions without overstating the method |

Read the [research protocol](../GeoDiff_TrustMoE_Research_Protocol.md) for close prior
work, notebook instructions, split limitations and reporting rules. Read the
[real-pair foundations](../33_real_cross_sensor_sr_foundations.md) for more sensor detail.

Throughout this guide, distinguish **implemented behavior**, **design rationale**,
and **research hypothesis**. Code that runs proves neither improved reconstruction
nor publication novelty. The proposed method must beat meaningful controls.

## Source Map

- [Model and router](../../src/geodiff_gan/models/trust_moe.py)
- [Residual-Swin base](../../src/geodiff_gan/models/residual_base.py)
- [Trust and reconstruction losses](../../src/geodiff_gan/training/trust_losses.py)
- [Training and evaluation](../../src/geodiff_gan/experiments/trust_moe.py)
- [Metrics, speed reports and visualizations](../../src/geodiff_gan/experiments/trust_report.py)
- [Training notebook](../../kaggle/GeoDiff_TrustMoE_Transformer_3x.ipynb)
- [Architecture and workflow tests](../../tests/test_trust_moe.py)

Suggested study method: explain each diagram aloud, calculate the tensor dimensions,
find the corresponding function, then answer the chapter's questions without notes.

## Collect Your Own Data

Use the [Earth Engine India dataset guide](../GEE_India_Fixed_Pair_Datasets.md)
for 2,000 / 4,000 / 6,000 real pairs with common validation/test IDs and nested
training subsets. It includes a resumable Colab notebook and explains weak labels,
native grids, geographic blocks, masks and source attribution.
