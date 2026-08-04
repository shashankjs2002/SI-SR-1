# Synopsis of First-Year Projects

## Applicant Details

- **Name:** [YOUR FULL NAME]
- **Programme:** [PROGRAM NAME]
- **Institute/University:** [INSTITUTE NAME]
- **Academic year:** 2025-26
- **Project supervisor:** [SUPERVISOR NAME AND DESIGNATION]

## Project Title

**GeoDiff-GAN: Evidence-Constrained Diffusion-GAN Super-Resolution for Sentinel-2 Satellite Imagery**

## 1. Background and Motivation

Satellite imagery is central to GIS applications such as land-cover mapping, agricultural observation, urban analysis, environmental monitoring, and disaster assessment. However, the spatial resolution of freely available imagery can limit the interpretation of small or narrow features. Single-image super-resolution estimates a higher-resolution image from a lower-resolution observation, but the task is ill-posed: information removed by blur and downsampling cannot be uniquely recovered. Conventional reconstruction models often produce smooth results, while unconstrained generative models may create visually plausible but geographically unsupported detail.

My first-year project investigated whether a hybrid diffusion-GAN architecture could improve fine spatial detail while remaining constrained by the observed satellite data. The primary experimental setting uses Sentinel-2 Level-2A surface-reflectance imagery. Native 10 m RGB patches are treated as targets, and synthetic 40 m observations are generated for controlled four-times super-resolution experiments.

The architectural motivation came from three complementary ideas. First, HART separates image representation into a component carrying the broad visual structure and a continuous residual component modeled by lightweight diffusion. This motivated the separation of GeoDiff-GAN into a deterministic, radiometrically conservative super-resolution base and a stochastic residual-detail branch. GeoDiff-GAN does not reproduce HART's autoregressive discrete-token generator; it adapts only the principle of assigning global structure and residual detail to different modeling paths.

Second, *Diffusion-driven GAN Inversion for Multi-Modal Face Image Generation* connects meaningful diffusion features to a GAN through mapping and style-modulation networks. This motivated GeoMapper, which transforms the denoised residual latent and low-resolution satellite features into spatial content and layer-wise decoder modulation. Unlike the source method, GeoDiff-GAN is not a face-generation or pretrained-GAN inversion system: the mapper and residual decoder are trained for evidence-constrained satellite super-resolution.

Third, a wavelet-based discriminator was introduced to inspect generated high-frequency content explicitly. A spatial PatchGAN can judge local realism, but satellite super-resolution also requires attention to directional edges and fine textures. The Haar-wavelet discriminator therefore evaluates the LH, HL, and HH high-frequency sub-bands, complementing the spatial discriminator and wavelet reconstruction loss.

## 2. Project Objectives

1. Develop a reproducible geospatial pipeline for preparing valid Sentinel-2 L2A image patches.
2. Design a four-times satellite-image super-resolution model that combines deterministic reconstruction with generative detail estimation.
3. Constrain generated details using low-resolution evidence, sensor degradation parameters, spatial confidence, and iterative back-projection.
4. Examine whether NIR and SWIR information can improve RGB reconstruction as multispectral conditioning evidence.
5. Evaluate reconstruction fidelity, structural quality, edge recovery, and consistency with the original low-resolution observation.
6. Build visual diagnostics and uncertainty outputs so that model failures and unsupported details can be identified rather than hidden.

## 3. Methodology

### 3.1 Geospatial data preparation

Sentinel-2 L2A SAFE products were processed as raster windows instead of loading complete tiles into memory. RGB targets use bands B04, B03, and B02 at their native 10 m sampling. The multispectral experiments additionally use B08 at 10 m and B11/B12 at 20 m; the 20 m bands are bilinearly aligned to the 10 m grid and are used only as conditioning evidence, not claimed as newly observed 10 m measurements.

Scene Classification Layer masks and raster-validity checks were used to reject clouds, cloud shadows, invalid pixels, saturated areas, and boundary patches with black backgrounds. Accepted 512 x 512 target patches were recorded in a manifest with their product and MGRS information. The workflow supports resumable extraction and avoids regenerating patches from already completed products.

To create paired data, each 10 m target was transformed into a 128 x 128 simulated 40 m observation using randomized blur, four-times downsampling, mild Poisson-Gaussian noise, quantization, and compression. Degradation parameters were stored as model conditions. Random degradations during training expose the model to varying observation conditions, while deterministic degradation is used for repeatable validation and testing.

### 3.2 Proposed architecture

The implemented GeoDiff-GAN contains the following linked modules:

- A deterministic SwinIR-based branch produces a conservative 512 x 512 reconstruction that preserves broad geometry and radiometry.
- A residual variational autoencoder represents the difference between the native target and deterministic base in a compact four-channel latent space.
- A conditional latent diffusion U-Net models plausible residual information using low-resolution features and degradation metadata. Optional text and mode conditions are supported.
- GeoMapper converts the denoised latent into spatial content, decoder modulation parameters, and an evidence-confidence map.
- A resize-convolution residual decoder predicts high-frequency corrections while using multiscale low-resolution feature skips. Resize-convolution replaced PixelShuffle after diagnostics identified periodic lattice artifacts.
- A conditional multiscale PatchGAN assesses local spatial realism during training. A complementary Haar-wavelet discriminator assesses directional LH, HL, and HH high-frequency bands, where missing edges or artificial texture are more directly exposed.
- In reconstruction mode, the residual is high-pass constrained and three sensor back-projection steps reduce disagreement between the generated high-resolution image and the observed low-resolution input.

The model also produces abstention and uncertainty maps. These indicate where generated detail has weak support and are important for responsible use in GIS workflows.

### 3.3 Training and evaluation

Training was organized as a staged curriculum: deterministic base training, residual representation training, conditional diffusion training, joint reconstruction/adversarial fine-tuning, and optional prompt-adapter training. The main losses include Charbonnier reconstruction, SSIM, gradient, wavelet, degradation consistency, evidence calibration, perceptual, and low-weight adversarial objectives. Validation-based early stopping and best/latest checkpoint retention were added to control overfitting and storage use.

Evaluation uses L1 error, PSNR, SSIM, edge F1, LPIPS where available, and re-degradation error. Bicubic interpolation and the deterministic base are reported separately. Visual diagnostics include residual maps, edge and wavelet responses, Fourier spectra, confidence/abstention maps, uncertainty, and the trajectory of sensor back-projection.

## 4. Outcomes Achieved During the First Year

The first-year work produced a functioning end-to-end research system: Sentinel-2 ingestion, quality screening, patch manifests, randomized degradation, RGB and multispectral model variants, staged training, checkpoint recovery, evaluation, and detailed visual debugging.

In a preliminary controlled evaluation on 40 samples, the small improved RGB GeoDiff-GAN achieved PSNR 35.77 dB, SSIM 0.8725, edge F1 0.3319, and re-degradation L1 error 0.000842. Its deterministic base achieved higher pixel-fidelity scores of 36.16 dB PSNR and 0.8829 SSIM, but a lower edge F1 of 0.2483 and higher re-degradation error of 0.002652. The medium multispectral generative variant produced the strongest edge F1 among the four tested GeoDiff-GAN variants at 0.4121, while its deterministic base retained the strongest PSNR/SSIM result.

These results show a clear trade-off rather than an unconditional improvement: the generative branch recovers more edge activity and improves low-resolution consistency, but currently sacrifices some pixel-level fidelity compared with the deterministic base. This finding is scientifically useful because it identifies where further work is required: calibration of generated detail, larger geographically diverse training data, and geographically held-out evaluation.

## 5. Application of GIS Technology and Planned Extension

GIS principles are integral to the completed first-year project rather than being limited to image display. The work uses georeferenced Earth-observation products, spectral-band registration, spatial windowing, MGRS tile identifiers, scene-quality masks, and geographic data partitioning. QGIS was used for raster inspection, georeferenced visualization, layer management, and spatial verification of Sentinel-2 data and model outputs. These controls help prevent spatial leakage and preserve the relationship between every image patch and its source observation.

ArcGIS was not used in the completed experiments. If selected, I plan to use the scholarship's ArcGIS access as a complementary extension to my existing QGIS workflow. Proposed future work includes restoring georeferencing to reconstructed outputs, overlaying results with land-cover and administrative layers, evaluating performance by landscape type, comparing edge behavior around mapped roads and settlements, and creating uncertainty-aware map products. These ArcGIS activities are planned objectives and are not presented as completed outcomes.

## 6. Limitations and Responsible Interpretation

The current experiment estimates native-resolution targets from synthetically degraded Sentinel-2 imagery. It does not prove that every generated fine feature corresponds to a real-world object in an unpaired 40 m observation. Upsampled 20 m NIR/SWIR bands provide contextual evidence but do not become genuine 10 m measurements. Consequently, reconstruction outputs are evaluated with sensor consistency and uncertainty diagnostics, while prompt-driven edit outputs are explicitly treated as synthetic visualizations.

## 7. Conclusion and Planned Extension

The project demonstrates a complete GeoAI workflow for evidence-constrained satellite-image super-resolution and establishes measurable baselines, failure diagnostics, and responsible-use boundaries. The next phase will expand geographically diverse training and test data, perform controlled ablations, compare recent open-source super-resolution architectures under the same data protocol, and analyze results as geospatial layers. The long-term aim is not merely sharper imagery, but a transparent GIS-compatible method that communicates both reconstructed detail and its uncertainty.

## 8. Principal Architectural Inspirations

1. H. Tang et al., *HART: Efficient Visual Generation with Hybrid Autoregressive Transformer*, arXiv:2410.10812, 2024. The project draws on its decomposition of broad image content and continuous residual information; GeoDiff-GAN replaces HART's autoregressive generation with an observation-conditioned deterministic SR base.
2. J. Kim et al., *Diffusion-driven GAN Inversion for Multi-Modal Face Image Generation*, CVPR 2024, arXiv:2405.04356. The project draws on its diffusion-to-GAN mapping and style-modulation concept; GeoMapper adapts this connection to spatially aligned satellite reconstruction.
3. Haar-wavelet adversarial analysis. GeoDiff-GAN applies a training-only discriminator to directional high-frequency sub-bands alongside a conditional multiscale PatchGAN. The contribution claimed is the evidence-constrained integration and its evaluation, not the invention of wavelet discrimination itself.

> Before submission, replace every bracketed field and ensure that the synopsis includes only work actually completed within your first academic year. QGIS is presented as the GIS software already used; ArcGIS is presented only as an optional scholarship-supported extension.
