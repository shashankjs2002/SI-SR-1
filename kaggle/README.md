# Kaggle Usage

Import `GeoDiff_GAN_Kaggle.ipynb` into Kaggle, enable Internet, select a GPU
accelerator, and attach a dataset containing extracted Sentinel-2 L2A `.SAFE`
directories.

The notebook is configured to clone
`https://github.com/shashankjs2002/SI-SR-1.git`.

The notebook:

1. Clones this repository without modifying Kaggle's CUDA PyTorch installation.
2. Finds `.SAFE` products below the selected `/kaggle/input` directory.
3. Extracts cloud-filtered RGB patches into `/kaggle/working`.
4. Quarantines zero-filled edge/corner patches without deleting them.
5. Visualizes five accepted samples with clean and noisy LR inputs.
6. Visualizes the exact samples removed by the edge/corner filter.
7. Optionally captions patches with `Qwen/Qwen3-VL-8B-Instruct`.
8. Writes runtime YAML with calibrated mild degradation and tensor diagnostics.
9. Runs one or all stages using one GPU or `torchrun` on multiple GPUs.
10. Evaluates stochastic predictions and exports checkpoints and metrics.

The notebook defaults to a quick one-epoch development run. Set
`FAST_DEV_RUN = False` before producing research results.

The preparation, filtering, and visualization cells reconstruct their required
paths after a Kaggle kernel restart. Run the clone/install cell again only when
the package is no longer importable or when pulling newer repository code.

A single MGRS tile can exercise the pipeline in `FAST_DEV_RUN`, but all patches
will remain in one tile-level split. Research runs require enough geographically
separated SAFE products to populate train, validation, and test splits.
