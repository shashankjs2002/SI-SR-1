# Kaggle Usage

Import `GeoDiff_GAN_Kaggle.ipynb` into Kaggle, enable Internet, select a GPU
accelerator, and attach a dataset containing extracted Sentinel-2 L2A `.SAFE`
directories.

After publishing the repository, replace `OWNER` in the notebook's
`REPOSITORY_URL` value with your GitHub username or organization.

The notebook:

1. Clones this repository without modifying Kaggle's CUDA PyTorch installation.
2. Finds `.SAFE` products below the selected `/kaggle/input` directory.
3. Extracts cloud-filtered RGB patches into `/kaggle/working`.
4. Optionally captions patches with `Qwen/Qwen3-VL-8B-Instruct`.
5. Writes runtime YAML with absolute Kaggle paths.
6. Runs one or all stages using one GPU or `torchrun` on multiple GPUs.
7. Evaluates stochastic predictions and exports checkpoints and metrics.

The notebook defaults to a quick one-epoch development run. Set
`FAST_DEV_RUN = False` before producing research results.

Run the configuration and clone/install cells again after every Kaggle kernel
restart. The notebook defines its command helper in the configuration cell and
checks that the package is installed before preparing data.

A single MGRS tile can exercise the pipeline in `FAST_DEV_RUN`, but all patches
will remain in one tile-level split. Research runs require enough geographically
separated SAFE products to populate train, validation, and test splits.
