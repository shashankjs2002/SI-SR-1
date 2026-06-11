# Infographic Generation Record

The ten paper figures were generated with the built-in image-generation tool in
`scientific-educational` mode. Exact technical interpretation is provided by the LaTeX captions;
the images are explanatory artwork rather than quantitative experiment outputs.

## Exact Architecture Poster

`architecture_detailed.png` and `architecture_detailed.svg` are generated deterministically by
`build_architecture_infographic.py`, not by an image model. The poster is derived from the
implemented full configuration and includes:

- the `3 x 128 x 128` LR input and `3 x 512 x 512` base/residual/output tensors;
- every SwinIR base layer and both PixelShuffle stages;
- all four LR evidence grids (`f128`, `f64`, `f32`, and `f16`);
- residual VAE encoder moments, four-channel latent, and reconstruction decoder;
- latent U-Net down, middle, up, skip, attention, and velocity-prediction blocks;
- timestep, degradation, mode, text, null-text, and classifier-free-guidance routes;
- GeoMapper content, evidence-gate, mode-fusion, and four style-head outputs;
- all four modulated decoder stages and LR skip dimensions;
- SR/edit residual filtering and back-projection behavior; and
- training-only multi-scale PatchGAN and Haar-wavelet discriminator dimensions.

## Figure Set

1. `infographic_architecture.png`
   - Pixelated LR satellite tile.
   - Deterministic optical/base branch.
   - Purple latent residual/detail branch.
   - Transparent base and residual layers merged into HR.
   - Training-only spatial and frequency critic instruments.

2. `infographic_task.png`
   - Native 10 m target.
   - Optical blur, 4x sampling, noise, and quantization.
   - Synthetic 40 m input.
   - Estimated 10 m output and error heatmap.

3. `infographic_data.png`
   - SAFE product and satellite.
   - B4/B3/B2 spectral layers and SCL mask.
   - Accepted windows and quarantined cloud/edge patches.
   - Geographic tile separation.
   - HR/LR pairs and optional caption tokens.

4. `infographic_tensors.png`
   - Full-resolution deterministic base film.
   - Multi-scale LR feature maps.
   - Small latent cube.
   - Expanding residual feature volumes.
   - Base and residual overlay.

5. `infographic_geomapper.png`
   - Latent texture cube, LR feature maps, and prompt context.
   - Transparent GeoMapper cutaway.
   - Spatial content canvas, evidence gate mask, and four style controls.
   - Residual decoder producing aligned detail.

6. `infographic_discriminators.png`
   - Real/generated HR comparison.
   - Multi-scale spatial inspection lenses.
   - Haar-style frequency prism and texture inspection.
   - Training signals separated from inference.

7. `infographic_modes.png`
   - One LR source.
   - Evidence-constrained SR output with verification symbol.
   - Prompt-driven edit output with synthetic warning.

8. `infographic_conditioning.png`
   - Null, matched, paraphrased, and mismatched prompts.
   - Frozen text-token representation.
   - Spatial evidence-gate overlay.
   - Conservative SR and stronger edit behaviors.

9. `infographic_evaluation.png`
   - Held-out geographic tiles.
   - Multi-method comparison gallery.
   - Fidelity, perception, edge, consistency, and uncertainty instruments.
   - One-factor ablation and tile-level statistics.

10. `infographic_diagnostics.png`
    - Residual high/low-frequency separation.
    - Clean LR re-degradation comparison.
    - Iterative back-projection.
    - Eight stochastic samples and variance map.
    - Gate, clipping, residual strength, and LR-error diagnostics.

## Shared Prompt Constraints

- Wide 16:9 research-paper infographic.
- Clean white scientific-publication background.
- Realistic overhead satellite imagery.
- Image-led visual explanation rather than conventional flowchart boxes.
- Consistent colors: blue evidence, purple latent/prompt, orange residual, green constrained output,
  red training critics/errors.
- No logos or watermarks.
