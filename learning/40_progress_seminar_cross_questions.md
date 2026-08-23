# Progress Seminar Cross-Question Handbook

## How to answer under pressure

Use this three-level pattern:

1. **Direct answer:** one sentence that answers exactly what was asked.
2. **Mechanism or evidence:** one implementation detail, equation, or result.
3. **Boundary:** one sentence explaining what the answer does not prove.

Example:

> PSNR measures pixel fidelity and higher is better. For images normalized to `[0,1]`,
> it is `-10 log10(MSE)`, so minimizing MSE directly improves PSNR. It does not prove
> that generated edges are real or correctly located.

Do not defend every design choice emotionally. If the evidence is incomplete, say:

> That is a current hypothesis. I have predefined the ablation needed to test it, so I
> will not claim it as a demonstrated contribution until the held-out result is available.

---

# A. Layman and Motivation Questions

## 1. What is satellite image super-resolution?

It is the estimation of a more densely sampled image from a lower-resolution satellite
observation. The model uses the measured LR values plus patterns learned from paired data.
The result is an estimate, not a new sensor measurement.

## 2. Why do we need SR if Sentinel-2 already gives 10 m images?

Use the real Sentinel image whenever it is valid and available for the required location
and date. SR is useful when HR data are cloudy, missing, too old, restricted, or absent
from a long historical record such as Landsat's archive.

## 3. Why not launch or use a better satellite?

New satellites are expensive and cannot retrospectively observe historical dates. SR is
a computational complement for existing archives, not a substitute for better sensors.

## 4. Why not just enlarge the image in QGIS or Photoshop?

Bicubic enlargement increases pixel count using smooth interpolation. It cannot infer
scene-dependent boundaries or textures. Learned SR can estimate them, but also introduces
hallucination risk.

## 5. Is SR creating information from nothing?

It combines LR evidence with learned prior knowledge. The missing subpixel arrangement is
not uniquely observed, so the model estimates a likely solution rather than recovering a
guaranteed hidden truth.

## 6. What practical value does the output have?

It can support visual interpretation, land-cover mapping, agricultural or urban analysis,
change screening, and downstream models when matching HR imagery is unavailable. High-risk
decisions still require independent observations.

## 7. Can the model see a house that Landsat did not record?

It may generate a house-like pattern based on context, but it cannot verify that a specific
house exists. Such detail is a prediction and must not be treated as observed fact.

## 8. Is the output a 10 m Landsat image?

No. It is a 10 m-grid Sentinel-like RGB estimate conditioned on Landsat. Calling it a
physically observed 10 m Landsat product would be misleading.

## 9. Why is the problem ill-posed?

One 30 m pixel corresponds geometrically to nine 10 m samples. Many arrangements of those
nine values can produce a similar 30 m measurement, so there is no unique inverse.

## 10. What does 3x mean?

The linear scale factor is `30 m / 10 m = 3`. Each spatial dimension triples, so the
number of output samples increases by `3 squared = 9`.

## 11. Does 3x mean three times more information?

No. It means three times more samples per dimension. Those samples are estimated and are
not equivalent to nine independently measured observations.

## 12. Why satellite imagery instead of natural photographs?

Satellite images have sensor-specific spectral bands, calibrated reflectance, geographic
projection, cloud masks, large ground footprints, and temporal revisit. Methods designed
only for ordinary photos may produce visually attractive but physically inconsistent output.

## 13. What is your main objective now?

The immediate objective is to improve held-out PSNR and SSIM over the deterministic SwinIR
base using real Landsat-Sentinel pairs. Perceptual sharpness and prompt editing are secondary
until fidelity is established.

## 14. What is the simplest version of your method?

Landsat RGB goes through SwinIR to produce a conservative 10 m-grid base. A conditional
diffusion-decoder pathway proposes only an evidence-gated high-frequency residual that is
added to the base.

## 15. What would count as project success?

The final model must improve PSNR and SSIM over its own base on held-out geography, reduce
error on a meaningful fraction of patches, avoid systematic artifacts, and preserve clear
provenance and uncertainty. A rigorous finding that SwinIR is the safer reconstruction
model would also be scientifically valid.

---

# B. Remote Sensing and Sensor Questions

## 16. What is spatial resolution?

It is the nominal ground sampling distance represented by each image sample. Landsat RGB
is 30 m and Sentinel-2 RGB is 10 m in this study.

## 17. What is spectral resolution?

It describes how the sensor divides reflected energy into wavelength bands. Different
sensors can have bands with similar names but different spectral response functions.

## 18. What is radiometric resolution or radiometry?

Radiometry concerns how measured energy is quantized, calibrated, and converted to values
such as surface reflectance. Sensor calibration, atmospheric correction, illumination,
and processing can shift brightness and contrast.

## 19. What is temporal resolution?

It describes how frequently a sensor revisits a location. In paired training, temporal
difference also means the two images may contain real scene changes.

## 20. Which word beginning with "radio" is relevant here?

The likely term is **radiometric resolution** or **radiometric calibration**. It describes
the sensor's sensitivity and mapping from recorded digital numbers to physical intensity
or reflectance values.

## 21. Why use surface reflectance?

Surface reflectance attempts to remove atmospheric effects and is more comparable across
dates and sensors than raw digital numbers or top-of-atmosphere display images. It does not
eliminate all cross-sensor differences.

## 22. Which Landsat RGB bands are used?

Landsat 8/9 OLI `SR_B4` is red, `SR_B3` is green, and `SR_B2` is blue.

## 23. Which multispectral bands are used?

The six-band input is `[red, green, blue, NIR, SWIR1, SWIR2]`, corresponding to Landsat
`B4, B3, B2, B5, B6, B7`. All are native 30 m OLI reflective bands.

## 24. Which Sentinel bands form the target?

Sentinel-2 L2A `B4, B3, B2` form red, green, and blue at native 10 m sampling.

## 25. Why use NIR and SWIR if the output is RGB?

NIR responds strongly to vegetation, while SWIR helps distinguish moisture, soil, and
built surfaces. These channels can guide spatial interpretation, but they are not visible
colors and therefore should not directly determine base RGB radiometry.

## 26. Why are the improved base and residual inputs different?

The SwinIR base receives calibrated RGB to remain radiometrically conservative. The LR
encoder receives all six bands so NIR/SWIR can guide only the residual pathway.

## 27. Are Landsat and Sentinel images the same image?

No. They are paired observations of the same nominal geographic footprint. Different
instruments, acquisition times, atmospheric correction, spectral responses, and geolocation
mean their pixel values are not identical measurements.

## 28. Why is Sentinel called a reference and not ground truth?

Sentinel is the higher-spatial-resolution observation used for supervised comparison, but
it still has sensor blur, noise, geolocation uncertainty, cloud-mask errors, and temporal
difference. "Reference target" is scientifically more accurate than absolute ground truth.

## 29. Why limit the acquisition gap to three days?

A shorter gap reduces the chance of crop, water, construction, illumination, or disaster
change while retaining enough pairs. It reduces temporal mismatch but cannot eliminate it.

## 30. Why not require the exact same acquisition time?

Landsat and Sentinel have different orbits and schedules, so exact simultaneity is rarely
available. A strict zero-gap rule could leave too little training data.

## 31. What is point-spread function?

The PSF describes how a point on the ground is blurred by the imaging system. Nominal pixel
size alone does not fully describe effective spatial detail.

## 32. What is MTF?

The modulation transfer function is the frequency-domain description of how well the sensor
preserves contrast at different spatial frequencies. High-frequency contrast generally
falls as features approach the sensor resolution limit.

## 33. Why can sensor mismatch reduce PSNR?

PSNR compares aligned numeric values. Bandpass, atmosphere, brightness, or subpixel shifts
increase MSE even when both observations are individually valid.

## 34. Can radiometric harmonization solve spectral mismatch completely?

No. The current robust affine mapping corrects global per-band scale and offset. It cannot
fully model land-cover-dependent nonlinear spectral response, BRDF, atmosphere, or time change.

## 35. Why use train-only calibration?

Calibration fitted from validation or test Sentinel targets would leak evaluation information.
The coefficients are estimated only from training pairs, frozen, and then applied everywhere.

---

# C. Dataset and Preprocessing Questions

## 36. How are Landsat and Sentinel products paired?

Products are matched using acquisition-date distance and geographic overlap. The candidate
with the smallest valid day gap and suitable overlap is selected, with full product IDs
stored for provenance.

## 37. How do you convert Landsat digital numbers?

Collection 2 Level-2 metadata provides multiplicative and additive surface-reflectance
scaling. The standard fallback is approximately `reflectance = DN * 0.0000275 - 0.2`.

## 38. How do you convert Sentinel digital numbers?

The code reads the product quantification value and band-specific BOA offsets from metadata.
This is important for newer processing baselines where simply dividing by 10,000 may be wrong.

## 39. Which spatial grid is the reference?

The Sentinel B4 10 m grid is the HR reference. For every 384-pixel HR window, an exact
corresponding 128-pixel 30 m grid is derived and Landsat is reprojected directly to it.

## 40. Why not upsample Landsat to 10 m during preprocessing?

The native LR tensor should remain 128 by 128. Upsampling first would create interpolated
samples and obscure what is actually measured. Enlargement is used only for display or
baseline comparison.

## 41. Why do 128 by 128 and 384 by 384 look the same size in plots?

Plotting scales each array to the allocated axes. They cover the same ground footprint but
contain different native sample counts.

## 42. What ground footprint does one pair cover?

`128 * 30 m = 3840 m` and `384 * 10 m = 3840 m`, so both cover a nominal 3.84 km square.

## 43. What is stored in each NPZ patch?

At minimum: Landsat LR, Sentinel HR, clean LR, degradation/pair condition, LR and HR validity
masks. Multispectral patches also contain the six-band LR tensor.

## 44. Are LR images stored or generated each epoch?

For real cross-sensor 3x training, LR is the stored real Landsat observation. Synthetic
degradation per epoch belongs to the earlier Sentinel-only experiment, not this paired task.

## 45. Why require 95% valid pixels?

It prevents clouds, borders, and invalid data from dominating training. The remaining mask
still ensures losses and metrics use only valid support.

## 46. Which pixels are rejected?

Landsat fill, dilated cloud, cirrus, cloud, shadow, snow, saturation, and high aerosol when
available; Sentinel no-data and invalid SCL categories; plus invalid scene borders.

## 47. What is quarantine?

Quarantine means retaining rejected patch previews and metadata with explicit rejection
reasons outside the training manifest. It supports debugging and prevents accidental reuse.

## 48. Does 95% valid fraction prove correct registration?

No. It only measures mask coverage. Registration must be audited using stable edges,
correlation shifts, overlays, and manual inspection.

## 49. How is registration audited?

Sentinel is area-downsampled to the Landsat grid and integer LR shifts are tested for normalized
correlation improvement. Frequent preferred nonzero shifts indicate a pairing or reprojection
problem requiring correction before model scaling.

## 50. Why are nearby random patches dangerous for evaluation?

Overlapping or adjacent patches share objects, textures, and spatial context. Random splitting
therefore leaks geography and makes test performance look better than true generalization.

## 51. What split is currently used?

The development suite creates spatial train, validation, and test blocks within each tile with
guard bands. This supplies all splits when few tiles exist. Final claims require complete-tile
or city holdout in addition.

## 52. Why guard bands?

A guard band excludes windows near split boundaries so overlapping patch footprints cannot
appear in two splits.

## 53. Why is the tile, not the patch, the statistical unit?

Patches from one tile are spatially correlated. Treating thousands of overlapping patches as
independent samples produces overconfident statistics. Final comparisons should aggregate per
tile or spatial fold.

## 54. What happens if a new product is added?

Incremental preparation records completed product pairs and processes only new unmatched pairs.
It should not recreate patches for already completed inputs.

## 55. Why save manifests and product IDs?

They make every patch traceable to its sensors, date gap, location grid, validity, and split.
Without provenance, a result cannot be reproduced or audited for leakage.

---

# D. Architecture Questions

## 56. Why combine diffusion and GAN?

Diffusion models a diverse conditional residual distribution and is stable to train; GAN
discriminators can encourage sharper spatial and frequency statistics. In the current
fidelity-first experiment GAN loss is disabled because sharpness was harming PSNR/SSIM.

## 57. Is the discriminator used during inference?

No. PatchGAN and wavelet discriminators influence generator training only. Inference loads the
generator modules stored in the joint checkpoint.

## 58. What does the full joint checkpoint contain?

It stores the entire GeoDiff-GAN generator state: base, VAE, LR encoder, diffusion U-Net,
GeoMapper, and decoder. The inference code builds the complete architecture and loads this state;
it does not load only a small isolated "joint module."

## 59. Why use SwinIR as the base?

Shifted-window attention captures local and neighboring spatial relationships efficiently, and
SwinIR is a strong restoration baseline. It provides conservative structure and an explicit
fallback for unreliable generated detail.

## 60. Why not let diffusion generate the complete HR image?

Full-image generation gives the stochastic model freedom to alter color, low-frequency land
cover, and geometry. Residual generation anchors most of the output to deterministic evidence.

## 61. Why predict a residual?

The base already explains much of the target. Learning only `target - base` focuses model
capacity on missing detail and makes the magnitude and effect of generation measurable.

## 62. Why high-pass filter the SR residual?

It suppresses low-frequency color and illumination changes so the residual primarily contributes
detail. It is a constraint, not proof that every remaining high-frequency feature is correct.

## 63. What is the residual VAE's role?

It compresses the HR residual into a lower-resolution four-channel latent and reconstructs it.
This reduces diffusion cost and creates a structured residual space.

## 64. Is the VAE encoder used at inference?

The HR residual encoder is needed during training to create target latents. At inference there is
no HR target; diffusion generates a residual latent from noise and LR conditioning. That denoised
latent is passed to GeoMapper and the residual decoder, so the residual VAE encoder is not used in
the final inference path.

## 65. What does the LR encoder produce?

It extracts multi-scale features at progressively smaller spatial resolutions. These features
condition diffusion, GeoMapper, and decoder skips so output detail remains spatially related to
the observation.

## 66. What does diffusion learn?

It learns velocity/noise removal in the latent residual distribution conditioned on LR features,
sensor conditions, mode, and optional text. DDIM sampling converts Gaussian noise into a residual
latent at inference.

## 67. Why use velocity prediction?

Velocity parameterization balances clean-signal and noise prediction across timesteps and is
commonly stable with SNR-weighted diffusion training. It does not by itself guarantee better SR.

## 68. What is GeoMapper?

It converts the denoised latent plus LR and optional text conditions into spatial content,
layer-wise style/FiLM parameters, evidence confidence, and edit permission for the decoder.

## 69. What is FiLM?

Feature-wise linear modulation scales and shifts decoder feature channels using conditioning
parameters. It allows the latent/context to control each decoder stage without replacing spatial
LR skip features.

## 70. What is evidence confidence?

It is a learned spatial estimate of where proposed detail is supported. In SR mode it gates the
detail residual and combines with sample uncertainty for abstention.

## 71. Is evidence confidence a calibrated probability?

Not automatically. It is trained with calibration objectives and evaluated against error, but it
must demonstrate useful confidence-error correlation and selective risk before being interpreted
probabilistically.

## 72. What is abstention?

When evidence confidence is low or stochastic disagreement is high, the output is blended back
toward the deterministic base instead of accepting uncertain generated detail.

## 73. Why generate multiple samples?

The one-to-many problem permits multiple plausible residuals. Pixelwise disagreement among
samples provides a practical uncertainty signal, although low variance does not prove correctness.

## 74. What is edit permission?

It is a separate spatial policy for optional prompt-conditioned synthetic editing. It should be
zero or restricted in SR mode and must not be confused with reconstruction confidence.

## 75. Why was PixelShuffle removed?

Previous residuals showed periodic lattice structure consistent with phase imbalance. Resize-
convolution avoids channel-to-phase rearrangement and reduced that architectural risk. Fourier
diagnostics are still required.

## 76. What is the wavelet discriminator?

It applies adversarial supervision to Haar high-frequency subbands so generated edge/texture
statistics can be judged separately from low-frequency appearance. It is currently disabled in
the fidelity-first run and can later be reintroduced as an ablation.

## 77. What does back-projection do?

It degrades the HR estimate through an approximate sensor model, compares it with LR evidence,
upsamples the error, and iteratively corrects the HR output. It is useful only when the degradation
operator represents the real sensor sufficiently well.

## 78. Why is back-projection disabled in the current fidelity experiment?

The current generic degradation operator does not fully model OLI-to-MSI spectral/radiometric
differences. Strong projection could force a Sentinel-like output back toward mismatched Landsat
radiometry and hide weak learned reconstruction.

## 79. How is spatial data conserved?

Through common-grid pairing, the deterministic base, LR feature conditioning, decoder skips,
high-pass residual restriction, evidence gating, masks, and optional re-degradation checks. No
single mechanism guarantees perfect conservation.

## 80. Does the architecture hallucinate?

It can. Diffusion and adversarial components use learned priors. The architecture reduces risk
through residual constraints, evidence gating, uncertainty abstention, validation fallback, and
explicit synthetic labeling, but cannot mathematically eliminate false detail.

---

# E. Training and Optimization Questions

## 81. Why train in stages?

Each subsystem has a different objective. A strong base and residual representation provide stable
targets before diffusion and adversarial interactions are introduced.

## 82. What is trained in the base stage?

Only SwinIR is optimized for deterministic HR reconstruction using pixel and structural losses.

## 83. What is trained in the VAE stage?

The residual VAE plus LR encoder, mapper, and decoder learn residual representation and controlled
reconstruction before diffusion sampling is relied on.

## 84. What is trained in the diffusion stage?

The conditional diffusion U-Net learns the residual latent denoising objective while other output
modules are frozen according to the stage policy.

## 85. What is trained in the improved joint stage?

Only the LR encoder, GeoMapper, and decoder are optimized. Diffusion and the deterministic base
remain frozen; diffusion is still present in the computational path.

## 86. Why freeze diffusion in joint tuning?

The preliminary validation trend suggested joint optimization was destabilizing fidelity. Freezing
diffusion tests whether mapper/decoder adaptation can use the learned latent prior without moving it
toward training-specific texture.

## 87. Why not unfreeze all modules?

More trainable parameters increase overfitting and can destroy the strong base. Unfreezing should
be a controlled later ablation only if frozen-module diagnostics identify underfitting.

## 88. Why only eight joint epochs?

Previous longer joint training showed early validation degradation. A shorter stage plus early
stopping reduces texture overfitting and unnecessary GPU time.

## 89. Why use a lower joint learning rate?

Joint tuning begins from pretrained modules, so large updates can erase useful representations.
The improved run caps joint LR at `5e-6`.

## 90. Why is MSE heavily weighted?

For normalized images, maximizing PSNR is mathematically equivalent to minimizing MSE. The current
supervisor-directed priority is PSNR/SSIM, so distortion losses dominate.

## 91. Does a larger MSE weight guarantee higher test PSNR?

No. It changes optimization pressure, but generalization also depends on data quality, model bias,
registration, learning rate, stage balance, and overfitting.

## 92. Why keep Charbonnier if MSE is used?

Charbonnier supplies robust L1-like reconstruction pressure and is less dominated by a few large
errors. MSE directly targets PSNR; the two provide complementary gradients.

## 93. What is the base-guard loss?

It penalizes samples where the final prediction's error exceeds the frozen base error. It helps
training but cannot guarantee that every unseen test patch improves.

## 94. Why remove GAN and perceptual losses now?

They can improve visual realism while moving pixels away from the paired target. They are disabled
to establish a fidelity operating point first, then can be added one at a time.

## 95. What is early stopping?

Training stops after a defined number of validation checks without sufficient improvement. It
prevents continuing blindly after generalization has plateaued or worsened.

## 96. Which checkpoint is saved?

The system retains the best checkpoint according to the configured validation metric and the latest
checkpoint for resumption. It avoids storing every epoch while preserving recovery and selection.

## 97. Does restarting the notebook restart training?

No, when the same experiment directory and resolved configuration are available. Auto-resume finds
the latest stage checkpoint and continues from the next epoch.

## 98. Why use mixed precision?

FP16/autocast reduces GPU memory and often increases throughput. Numerically sensitive operations
use safe implementations, and nonfinite diagnostics monitor instability.

## 99. Why use gradient accumulation?

Large HR patches limit physical batch size. Accumulating several microbatch gradients approximates
a larger effective batch before each optimizer step.

## 100. What is the difference between an epoch and an iteration?

An epoch is one pass over the training dataset. A batch iteration processes one microbatch. An
optimizer update may occur after several iterations when gradient accumulation is used.

---

# F. Metrics and Result Questions

## 101. What does L1 equal to 0.014 mean?

It means the average absolute error is 0.014 reflectance units per valid RGB sample on the normalized
scale. It does not mean 1.4% accuracy or a 1.4 m location error.

## 102. How is PSNR calculated?

For `[0,1]` images, `PSNR = -10 log10(MSE)`. Higher PSNR means lower squared pixel error.

## 103. What does a 3 dB PSNR increase mean?

It corresponds approximately to halving MSE. It does not mean a 3% visual improvement.

## 104. What is SSIM?

SSIM compares local luminance, contrast, and structure. Its ideal value is 1, but it remains
sensitive to registration and is still a full-reference metric.

## 105. Why use both PSNR and SSIM?

PSNR emphasizes exact pixel values, while SSIM emphasizes local structural relationships. One can
improve while the other worsens, so both are required for a fidelity claim.

## 106. What is edge F1?

It balances precision and recall of predicted edges against target edges within a small tolerance.
High recall with low precision indicates false sharpness; high precision with low recall indicates
over-smoothing.

## 107. What are LPIPS and DISTS?

They are learned perceptual distances intended to reflect feature-level visual similarity. Lower
is better, but they were largely developed on natural-image representations and must be interpreted
cautiously for reflectance imagery.

## 108. What is re-degradation error?

The HR estimate is approximately blurred/downsampled and compared with Landsat. Low error means LR
consistency under that operator, not necessarily correct 10 m detail.

## 109. Why can a model have good LR consistency but poor HR quality?

Many different HR images degrade to similar LR observations. Back-projection can repair low-frequency
consistency without reconstructing the correct roads, fields, or edges.

## 110. Why did rgb_standard beat rgb_fidelity in the preliminary table?

The loss label alone does not guarantee generalization. Optimization balance, stochastic residuals,
limited validation samples, sensor mismatch, and overfitting can outweigh the intended loss pressure.

## 111. Why did multispectral improve edge F1 but lower PSNR?

Extra bands may encourage stronger boundaries while also changing color or producing pixel-inaccurate
detail. The old design let all six bands enter the RGB base, which confounded spectral guidance and
radiometric reconstruction.

## 112. Is the full model currently better than SwinIR?

Not demonstrated consistently. Earlier subsets showed a strong SwinIR base, and the recent full-model
comparison did not include the exact embedded-base delta. The new evaluator reports that delta directly.

## 113. Why not report only the best-looking image?

That is selection bias. The study must report aggregate, per-patch, per-tile, uncertainty, typical,
best, and failure cases using predefined test indices or complete test sets.

## 114. Is 28 test patches enough?

It is enough for pipeline diagnosis, not a strong generalization claim. More geographically independent
tiles, folds, and random seeds are needed.

## 115. What does `fraction_beating_base_psnr` tell you?

It is the proportion of test patches where the final model has higher PSNR than its embedded base.
It prevents a small number of large gains from hiding widespread regression.

## 116. Why select residual scale on validation?

The generative correction can be too strong. Validation selects how much residual to accept without
using test targets. Scale zero is a safe deterministic-base fallback.

## 117. Why require a 0.05 dB gain?

Tiny differences can arise from stochastic or numerical variation. The threshold is a conservative
minimum practical guard, not a universal scientific constant; multiple seeds and confidence intervals
are still needed.

## 118. What if validation chooses scale zero?

It means the residual branch has not demonstrated sufficient fidelity benefit. The correct output for
that operating point is the SwinIR base, and the residual failure should be analyzed rather than hidden.

## 119. What if PSNR improves but the image looks smoother?

Pixel-wise distortion metrics often reward conditional averages. Report the fidelity gain, then assess
edge F1 and perceptual metrics separately. Reconstruction and perceptual modes may require distinct
operating points.

## 120. What if edge F1 improves but PSNR falls?

The model is adding edges, but some may be displaced or false. Since current priority is fidelity, the
tradeoff is not accepted unless a separate perceptual application justifies it.

---

# G. Failure Analysis and Improvement Questions

## 121. What are the main current failure causes?

Cross-sensor radiometric mismatch, possible subpixel registration error, temporal change, diffusion
residual variance, weak confidence calibration, texture overfitting, and the former six-band-to-RGB base
design.

## 122. Why not immediately train the large model?

A larger model cannot correct a flawed pair distribution or objective. It can overfit sensor mismatch
and artifacts more strongly. The targeted small-model experiment should validate the mechanism first.

## 123. When should the medium or large model be used?

After the small model shows stable held-out gains and diagnostics indicate capacity limitation rather
than data or optimization failure. Compare variants at matched data, seeds, and update budgets.

## 124. What is train-only affine harmonization doing?

For each RGB channel it robustly estimates `Sentinel_30m = slope * Landsat_30m + offset` using only
training pairs. The same frozen coefficients transform all later inputs.

## 125. Why area-downsample Sentinel for calibration?

The calibration should compare similar spatial support. Averaging each 3 by 3 Sentinel block produces
a 30 m-grid reference before fitting Landsat-to-Sentinel radiometry.

## 126. Could calibration increase the score unfairly?

It is fair only if fitted on training data and applied unchanged to validation/test. Per-image fitting
to the test Sentinel target would be leakage and is prohibited.

## 127. What if calibration worsens validation bicubic PSNR?

Reject or revise it. Calibration is a hypothesis, not a mandatory improvement. Inspect coefficients,
land-cover dependence, clipping, temporal mismatch, and registration.

## 128. Why preserve raw LR after calibration?

It enables visual and numerical auditing of the adjustment and prevents the transformed input from being
mistaken for the original measurement.

## 129. Why fit calibration separately for each cross-validation fold?

Using coefficients fitted from all folds would allow validation/test target information into training.
Each fold must estimate calibration from its own training records.

## 130. What result would cause you to abandon the generative branch for SR mode?

Persistent nonpositive PSNR/SSIM deltas versus SwinIR across tiles, seeds, scales, and corrected pairs,
especially with artifacts or poor confidence calibration. Diffusion could then remain only for explicitly
synthetic perceptual/edit mode.

---

# H. Novelty and Literature Questions

## 131. Is SwinIR novel?

No. SwinIR is an established restoration architecture used as a deterministic component and baseline.

## 132. Is diffusion novel?

No. Diffusion models are established. The candidate contribution concerns how deterministic,
stochastic, multispectral, policy, and abstention mechanisms are integrated and validated.

## 133. Is combining diffusion and GAN automatically novel?

No. A combination is scientifically meaningful only if it addresses a defined limitation and controlled
ablations demonstrate an advantage over simpler alternatives.

## 134. What motivated the architecture?

The work is motivated by transformer-based high-resolution synthesis such as HART, diffusion-driven GAN
inversion, residual satellite SR, and frequency-aware discrimination. These motivate components; they do
not transfer novelty or guarantee suitability for cross-sensor SR.

## 135. What is your strongest candidate contribution?

The strongest candidate is evidence-controlled residual generation: a conservative RGB base, conditional
latent residual diffusion, separate evidence/edit policies, multispectral guidance isolated from base
radiometry, uncertainty abstention, and validation fallback.

## 136. How will you prove novelty?

Through current literature review plus ablations: remove diffusion, mapper, gating, uncertainty, wavelet
discriminator, harmonization, or dual-stream separation one at a time under the same evaluation protocol.

## 137. Can you claim "first"?

Not without a systematic and current literature search showing no prior equivalent method. The safe
current wording is "we investigate" or "our candidate contribution is."

## 138. Why is a negative result valuable?

It identifies when generative detail does not improve cross-sensor fidelity and prevents unsupported use.
A reproducible diagnosis and ablation can contribute more than a selectively reported visual result.

## 139. How is your method different from ordinary pansharpening?

Pansharpening normally fuses a high-spatial-resolution panchromatic band with lower-resolution
multispectral bands from the same or compatible observation. This study maps a real 30 m multispectral
Landsat observation to a 10 m Sentinel-like RGB reference without a simultaneous high-resolution pan band.

## 140. How is it different from data fusion?

It is a learned cross-sensor reconstruction/fusion problem, but the inference input is Landsat evidence;
the paired Sentinel image is used for training/evaluation, not supplied as an HR input at deployment.

---

# I. Ethics, GIS, and Deployment Questions

## 141. Can the output be used as evidence of a disaster or illegal construction?

Not by itself. Generated detail can be false. Such claims require confirmation from actual imagery,
field data, or independent sources.

## 142. How will users know an output is generated?

Outputs should retain model version, source product IDs, acquisition dates, calibration, uncertainty, and
mode metadata. Edit-mode images explicitly store `synthetic_edit=true`.

## 143. Can uncertainty guarantee safety?

No. A model can be confidently wrong. Uncertainty is an additional diagnostic and must be validated by
confidence-error and uncertainty-error relationships.

## 144. Why use QGIS rather than ArcGIS?

QGIS currently supports the required open geospatial inspection, raster overlays, projections, and
validation. ArcGIS integration is planned work, not a prerequisite for the scientific method unless a
specific scholarship or institutional deliverable requires it.

## 145. What could ArcGIS add later?

Model serving, raster-function integration, catalog management, web maps, dashboards, spatial analysis,
and easier dissemination within organizations already using the Esri ecosystem.

## 146. How would the model be deployed?

A deployment pipeline would ingest a calibrated Landsat product, apply QA masks and the frozen training
calibration, run tiled inference with overlap, blend patches, export georeferenced RGB plus uncertainty,
and retain complete provenance.

## 147. How will full tiles avoid patch seams?

Use overlapping inference windows, tapering/weighted blending, identical preprocessing, and edge padding.
Seam metrics and full-tile visual inspection are required before deployment.

## 148. What happens outside the training geography?

Performance can fall because landscape, atmosphere, architecture, agriculture, and seasons differ. That
is why complete-tile/city holdout and diverse training geography are necessary.

## 149. Can the model handle clouds?

The current reconstruction dataset rejects or masks clouds and shadows. It is not a cloud-removal model.
Applying it to cloudy LR input is out of distribution unless explicitly trained and evaluated for that task.

## 150. What is your final one-minute defense?

> High-resolution imagery is not always available at the required place and time, while Landsat offers a
> long, open 30 m archive. I study whether a 10 m-grid Sentinel-like estimate can be produced without giving
> a generative model unrestricted freedom. The system anchors low-frequency reconstruction in an RGB-only
> SwinIR base, conditions a latent diffusion residual on real Landsat spatial and multispectral evidence,
> gates unsupported detail, and can return toward the base when uncertain. Preliminary experiments exposed
> sensor mismatch and generative regression rather than proving superiority. I therefore added train-only
> radiometric harmonization, separated NIR/SWIR guidance from RGB base reconstruction, shortened and froze
> joint optimization, and predefined exact final-versus-base acceptance metrics. The next experiment will
> determine whether this controlled residual provides measurable held-out benefit; otherwise SwinIR remains
> the scientifically justified reconstruction output.

---

# Rapid-Fire Formula Card

```text
Scale factor = 30 m / 10 m = 3
Output samples per LR sample = 3^2 = 9
Ground footprint = 128*30 m = 384*10 m = 3840 m
MAE = mean(abs(prediction - reference)) over valid pixels
MSE = mean((prediction - reference)^2) over valid pixels
PSNR = -10 log10(MSE), for data range [0,1]
SSIM ideal value = 1
Residual target = Sentinel reference - SwinIR base
Final SR = base + confidence * high_pass(predicted residual)
Calibration = slope * Landsat_RGB + offset, fitted on train only
```

# Statements to Avoid

Do not say:

- "The model recovers the true hidden 10 m pixels."
- "PSNR is accuracy."
- "Sentinel is perfect ground truth."
- "NIR is a high-resolution band in this experiment."
- "The discriminator runs during inference."
- "A low re-degradation error proves correct HR detail."
- "More parameters will definitely improve results."
- "The architecture is novel because no one used this name."

Use instead:

- "The model estimates a Sentinel-like 10 m-grid reference."
- "PSNR measures pixel fidelity relative to a paired reference."
- "Sentinel is the higher-resolution reference observation."
- "NIR/SWIR provide 30 m spectral guidance to the residual pathway."
- "Discriminators are training-only."
- "Re-degradation checks compatibility under an approximate sensor operator."
- "Capacity is tested only after data and optimization failures are controlled."
- "Candidate novelty must be demonstrated through literature review and ablation."
