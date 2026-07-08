Bugs found
Here's a thorough list, ordered by severity:
Critical / silent wrong behavior

caption_qwen.py — wrong AutoModel class. It calls AutoModelForMultimodalLM.from_pretrained(...) but that class does not exist in HuggingFace transformers. Qwen3-VL requires Qwen2_5_VLForConditionalGeneration (or AutoModelForVision2Seq). This will throw an AttributeError at runtime and make the entire captioning step fail.
models/vae.py — the encoder applies 3 downsampling steps (two inside the channel loop and one extra stride-2 conv), giving an 8× spatial reduction, not the 8× reduction the rest of the system expects. For a 512×512 HR image the latent becomes [B, 4, 64, 64], but for a 128×128 LR → 512×512 HR this is correct. However, decode_latent() in system.py computes latent_height = base.shape[-2] // 8 (i.e. 512 // 8 = 64). That's consistent, but it means the VAE must always have exactly 3 downsampling steps. Any change to vae_channels depth without changing this divisor will silently produce shape mismatches.
models/generator.py ResidualSRDecoder.__init__ — the skip_dims tuple is hardcoded as (lr_channels * 2, lr_channels, lr_channels, lr_channels), meaning stage 0 expects lr_channels * 2 = 128 channels from lr_features[1]. But lr_features[1] is f64 from LREncoder, which has channels * 2 = 128 channels — that's coincidentally correct only when lr_channels=64. If lr_channels is changed (e.g. in the smoke config it's 8, giving f64 channels of 16), skip_dims[0] would be 16 but lr_features[1] has channels * 2 = 16… actually these do stay consistent since LREncoder.down1 outputs channels * 2. Let me be more precise: in smoke config lr_channels=8, so f64 = [B, 16, ...] and skip_dims[0] = 8*2 = 16 — consistent. In default lr_channels=64, f64 = [B, 128, ...] and skip_dims[0] = 128 — consistent. The real bug is that source_skips in forward() is [lr_features[1], lr_features[0], lr_features[0], lr_features[0]] but skip_dims is built with [lr_channels*2, lr_channels, lr_channels, lr_channels] — meaning stages 1–3 use f128 (which has lr_channels channels). But f128 actually has lr_channels channels (the stem output), and lr_channels in skip_dims matches. The issue is that skip_dims[0] = lr_channels * 2 assumes f64, while stages 1–3 all reuse f128 repeatedly at arbitrary upscaled sizes. This is architecturally odd (stage 2 is 2× bigger than f128's native resolution) and relies entirely on bilinear interpolation in forward(). Not a crash, but a design smell.
training/trainer.py _discriminator_step — the discriminator is only updated once per gradient accumulation step (if (step + 1) % accumulation == 0), but the generator loss computation using patch_discriminator and wavelet_discriminator happens every single micro-step. This means during accumulation steps 0 through N−1, the generator adversarial loss is computed against a discriminator that hasn't yet been updated on those exact generated samples, making the GAN gradient signal noisy.
training/trainer.py line in _forward_stage — for the joint/edit stage, back_projection_steps=0 is passed to decode_latent(). This means during training the back-projection is deliberately disabled (steps = 0 if self.use_back_projection else 0 effectively). The degradation_consistency loss is computed on the no-projection output, which is fine by design, but it means the model never trains to produce outputs that actually pass back-projection during the forward pass. This inconsistency between training (no projection) and inference (3 projection steps) can cause a distribution shift.

Medium — functional issues

models/blocks.py WindowTransformerBlock — the heads calculation while channels % heads: heads -= 1 starting from min(heads, channels) could in theory reduce heads to 1 for any odd channels value, silently degrading attention quality with no error.
training/trainer.py train() — when self.distributed is True, _synchronize_model_gradients() manually all-reduces gradients with SUM then divides by world_size. But the model is a plain nn.Module, not a DistributedDataParallel wrapper. DDP's own gradient sync is not set up for self.model. This means in distributed mode, gradients are manually summed but DDP's automatic hook is not registered, which could cause gradient desynchronization between ranks for the discriminators (which ARE wrapped in DDP).
data/dataset.py __getitem__ — calls random_degradation which internally calls torch.rand(...) without a seeded generator. This means the degradation parameters are random and non-reproducible across runs, even if the dataset index is fixed. Evaluation code in evaluate.py also uses random_degradation indirectly through the dataset loader, so metrics are not fully reproducible between evaluation runs.
models/system.py sample() — apply_ablation_inputs() is called twice: once before computing base and lr_features, and then again inside decode_latent(). This means when use_text_conditioning=False, context is zeroed out correctly, but when use_degradation_conditioning=False, the degradation tensor passed to scheduler.ddim_sample() is zeroed, but decode_latent() zeroes it again. This double-zeroing is harmless but signals a design flaw where ablation application is not centralized.

Minor / code quality

cli/caption_qwen.py — dtype=torch.float16 is passed as a top-level kwarg to from_pretrained but the correct argument for newer transformers versions is torch_dtype. The dtype kwarg is silently ignored in many versions, meaning the model may load in full float32, consuming ~16 GB instead of ~8 GB on GPU.
data/manifest.py deterministic_split() — uses only the first 8 hex characters of a SHA256 hash (32 bits) for the split bucket. With fewer than ~10,000 tiles this is fine, but there's no enforcement that enough distinct tiles exist for meaningful train/val/test splits. The notebook has a runtime warning for this, but the split logic itself has no minimum-tile guard.
models/degradation.py sensor_degrade() — for add_noise=True, Poisson noise is sampled as torch.poisson((lr.clamp_min(0) * poisson_strength)) but the clamp_min(0) only prevents negative rates; Poisson noise on near-zero pixels can still produce spuriously high values (since it's discrete integer counts), and there's no subsequent clamp before quantization. The final .clamp(0, 1) catches this, but it means quantization is applied to clipped values rather than realistic sensor counts.
training/trainer.py — _configure_stage() for the edit stage does not enable self.model.lr_encoder.requires_grad_(True), but joint does. This means during edit fine-tuning the LR encoder is frozen even though the decoder (which uses its features heavily) is still trained. Potentially intentional as a conservative choice, but undocumented.
diagnostics.py — _contact_sheet() calls ImageFont.load_default(size=18). In Pillow < 10, load_default() does not accept a size argument and will raise a TypeError. The pyproject.toml correctly requires Pillow>=10, but this is a silent breakage for anyone running on older environments (e.g. the Kaggle default image before the requirements-kaggle.txt install completes).
kaggle/GeoDiff_GAN_Kaggle.ipynb cell 2 — run([..., "pip", "install", ..., "transformers>=4.57.0"]) upgrades transformers after the requirements-kaggle.txt install, but requirements-kaggle.txt does not pin transformers at all. This creates a potential version conflict: Kaggle's default transformers may be newer than 4.57 but older than what Qwen3-VL-8B needs, and the upgrade happens in a separate subprocess, so subsequent cells in the same Python kernel won't see the new version unless it's a fresh import.



This is a thorough novelty assessment. Here's my honest analysis.

---

## What the field already has (prior work)

Before evaluating what's new, here's what already exists as of mid-2025 that directly overlaps with this project:

**Latent diffusion for Sentinel-2 SR specifically:** Donike et al. (2025) published "Trustworthy Super-Resolution of Multispectral Sentinel-2 Imagery with Latent Diffusion" in IEEE JSTARS, proposing an adapted latent diffusion approach for Sentinel-2 4× upscaling. This is the closest single competitor — it covers the same sensor, same task, same core technique.

**Diffusion + satellite SR more broadly:** DiffusionSat (Khanna et al., ICLR 2024) allows LR Sentinel-2 image sequences as input with metadata to generate HR outputs, representing an important step toward this task. SatDiffMoE (Luo et al., 2024) further leverages a diffusion-based fusion algorithm conditioned on multiple LR inputs.

**Diffusion-GAN hybrids for SR:** SupResDiffGAN (2025) presents a hybrid GAN-diffusion architecture for super-resolution leveraging latent space representations to reduce diffusion steps while maintaining perceptual quality. AdcSR (CVPR 2025) shows a diffusion-GAN hybrid that reduces inference complexity by ~73% while preserving generation capability.

**Text-guided SR:** PromptSR (submitted to ICLR 2025, later withdrawn) introduced text prompts to image SR to provide degradation priors via a diffusion model. Text-guided Explorable Image Super-resolution (arXiv 2024) proposes zero-shot text-guided SR by modifying the generative process of T2I diffusion models to promote consistency with LR inputs.

**Text editing for remote sensing imagery:** RSEdit (2025) trains on over 60,000 bi-temporal satellite image pairs with rich change descriptions to enable text-guided image editing for remote sensing.

**Wavelet-aware diffusion:** A "Wavelet Diffusion GAN for Image Super-Resolution" was published in 2024, directly combining Haar wavelet decomposition with diffusion GAN for SR.

---

## Novelty assessment, claim by claim

Here's an honest breakdown of each design choice in GeoDiff-GAN and whether it's genuinely novel:

| Design element | Novel? | Reality |
|---|---|---|
| Latent diffusion for Sentinel-2 SR | No | Donike et al. 2025 did exactly this, published in a top EO journal |
| Diffusion-GAN hybrid for SR | No | AdcSR (CVPR 2025), SupResDiffGAN (2025), multi-scale adversarial diffusion (Nature Sci Rep 2025) all cover this |
| SwinIR base branch + diffusion residual | Partial | The residual split is a known strategy (ResShift, DeltaDiff) — SwinIR as the base is new but minor |
| Text/prompt conditioning for RS SR | Partial | PromptSR for natural images, DiffusionSat uses captions/metadata, RSEdit for RS editing — combining these specifically for SR + edit in one model is somewhat new |
| GeoMapper with spatial evidence gate | Potentially novel | FiLM-based spatial conditioning is common, but a learned spatial gate that asks "where in the image does the text actually apply?" for RS is not something I found published |
| Differentiable MTF degradation model | Partial | Degradation-aware SR is standard; MTF-specific modeling for Sentinel-2 has been done (e.g., Donike et al. use similar concepts) |
| Tile-level geographic train/val/test split | Not novel as a concept, but well-implemented | Known good practice; rarely enforced this rigorously |
| Dual SR + edit modes in one model | Potentially novel | The combination of physics-constrained SR and counterfactual text editing in a single model for EO is not something I found directly |
| 5-stage progressive training curriculum | Partial | Staged training is common in SR (ESRGAN lineage); this specific 5-stage schedule is a design choice, not a contribution |
| VLM-generated patch captions (Qwen3-VL) for RS SR training | Potentially novel | Using a VLM to auto-caption LR satellite patches as training signal for prompt-guided SR is genuinely underexplored |

---

## The honest verdict

**Can you publish a paper? Yes, but it requires significant reframing and additions.**

As currently designed, this is not publishable as-is for three reasons:

**1. The primary task is solved.** The Sentinel-2 latent diffusion SR task has a 2025 IEEE JSTARS paper. You cannot publish the same task with similar architecture and call it novel. Reviewers will immediately cite Donike et al. and ask what you add.

**2. No real experimental results exist yet.** The codebase is an implementation — there are no trained models, no PSNR/SSIM/LPIPS numbers, no ablations, no comparison tables. A research paper requires these. Without results you have nothing to submit.

**3. The combination is additive, not multiplicative.** Combining diffusion + GAN + SwinIR + text conditioning + wavelet discriminator is engineering work. Each piece exists. Reviewers at CVPR/ICCV/ECCV or IEEE TGRS will ask: what is the single insight that makes this combination necessary and why does it work?

---

## What could become publishable

There are three genuine angles worth pursuing that the existing literature does not cover well:

**Angle 1 (strongest): The SR + counterfactual edit unification**
No paper currently trains a single model that both (a) physics-constrained SR with LR consistency guarantees and (b) text-guided counterfactual appearance editing with an explicit "what does the text actually change vs. what must be preserved" spatial gate. This is the most novel part of the architecture — the evidence gate + dual mode training. A paper focused entirely on this, with careful ablations of the gate, would be a real contribution.

**Angle 2: VLM-derived scene captions as SR training signal**
Using Qwen3-VL to auto-caption satellite patches and studying whether richer semantic supervision from those captions actually improves SR fidelity (measured by land-cover-specific metrics, not just PSNR) is an interesting empirical question nobody has studied. This could be a short paper with a clear hypothesis.

**Angle 3: Geographic generalization study**
The tile-level splitting is carefully designed to measure geographic generalization. If you collect enough MGRS tiles across diverse land cover types and biomes, and measure how SR quality degrades across distribution shifts, that's a genuine empirical contribution to the EO SR literature.

---

## What you'd need to do to submit

1. Fix the bugs (especially #1 — captioning crashes — and #5 — train/inference inconsistency)
2. Actually train the full model and run baselines (bicubic, SwinIR, Donike et al. if you can reproduce)
3. Pick one of the three angles above as the paper's central claim
4. Run the ablations that support that claim (evidence gate on/off, text conditioning on/off, dual mode vs. separate models)
5. Target IEEE TGRS, IGARSS, or ISPRS Journal for EO-specific SR; or ECCV/ICCV workshops on vision + remote sensing if you want a faster turnaround

The target venue matters because at a general vision venue (CVPR, NeurIPS) the novelty bar relative to natural image SR methods is very high. At an EO-specific venue (TGRS, JSTARS) the bar is more reachable and the community will value the Sentinel-2 specificity more.



This is a thorough novelty assessment. Here's my honest analysis.

---

## What the field already has (prior work)

Before evaluating what's new, here's what already exists as of mid-2025 that directly overlaps with this project:

**Latent diffusion for Sentinel-2 SR specifically:** Donike et al. (2025) published "Trustworthy Super-Resolution of Multispectral Sentinel-2 Imagery with Latent Diffusion" in IEEE JSTARS, proposing an adapted latent diffusion approach for Sentinel-2 4× upscaling. This is the closest single competitor — it covers the same sensor, same task, same core technique.

**Diffusion + satellite SR more broadly:** DiffusionSat (Khanna et al., ICLR 2024) allows LR Sentinel-2 image sequences as input with metadata to generate HR outputs, representing an important step toward this task. SatDiffMoE (Luo et al., 2024) further leverages a diffusion-based fusion algorithm conditioned on multiple LR inputs.

**Diffusion-GAN hybrids for SR:** SupResDiffGAN (2025) presents a hybrid GAN-diffusion architecture for super-resolution leveraging latent space representations to reduce diffusion steps while maintaining perceptual quality. AdcSR (CVPR 2025) shows a diffusion-GAN hybrid that reduces inference complexity by ~73% while preserving generation capability.

**Text-guided SR:** PromptSR (submitted to ICLR 2025, later withdrawn) introduced text prompts to image SR to provide degradation priors via a diffusion model. Text-guided Explorable Image Super-resolution (arXiv 2024) proposes zero-shot text-guided SR by modifying the generative process of T2I diffusion models to promote consistency with LR inputs.

**Text editing for remote sensing imagery:** RSEdit (2025) trains on over 60,000 bi-temporal satellite image pairs with rich change descriptions to enable text-guided image editing for remote sensing.

**Wavelet-aware diffusion:** A "Wavelet Diffusion GAN for Image Super-Resolution" was published in 2024, directly combining Haar wavelet decomposition with diffusion GAN for SR.

---

## Novelty assessment, claim by claim

Here's an honest breakdown of each design choice in GeoDiff-GAN and whether it's genuinely novel:

| Design element | Novel? | Reality |
|---|---|---|
| Latent diffusion for Sentinel-2 SR | No | Donike et al. 2025 did exactly this, published in a top EO journal |
| Diffusion-GAN hybrid for SR | No | AdcSR (CVPR 2025), SupResDiffGAN (2025), multi-scale adversarial diffusion (Nature Sci Rep 2025) all cover this |
| SwinIR base branch + diffusion residual | Partial | The residual split is a known strategy (ResShift, DeltaDiff) — SwinIR as the base is new but minor |
| Text/prompt conditioning for RS SR | Partial | PromptSR for natural images, DiffusionSat uses captions/metadata, RSEdit for RS editing — combining these specifically for SR + edit in one model is somewhat new |
| GeoMapper with spatial evidence gate | Potentially novel | FiLM-based spatial conditioning is common, but a learned spatial gate that asks "where in the image does the text actually apply?" for RS is not something I found published |
| Differentiable MTF degradation model | Partial | Degradation-aware SR is standard; MTF-specific modeling for Sentinel-2 has been done (e.g., Donike et al. use similar concepts) |
| Tile-level geographic train/val/test split | Not novel as a concept, but well-implemented | Known good practice; rarely enforced this rigorously |
| Dual SR + edit modes in one model | Potentially novel | The combination of physics-constrained SR and counterfactual text editing in a single model for EO is not something I found directly |
| 5-stage progressive training curriculum | Partial | Staged training is common in SR (ESRGAN lineage); this specific 5-stage schedule is a design choice, not a contribution |
| VLM-generated patch captions (Qwen3-VL) for RS SR training | Potentially novel | Using a VLM to auto-caption LR satellite patches as training signal for prompt-guided SR is genuinely underexplored |

---

## The honest verdict

**Can you publish a paper? Yes, but it requires significant reframing and additions.**

As currently designed, this is not publishable as-is for three reasons:

**1. The primary task is solved.** The Sentinel-2 latent diffusion SR task has a 2025 IEEE JSTARS paper. You cannot publish the same task with similar architecture and call it novel. Reviewers will immediately cite Donike et al. and ask what you add.

**2. No real experimental results exist yet.** The codebase is an implementation — there are no trained models, no PSNR/SSIM/LPIPS numbers, no ablations, no comparison tables. A research paper requires these. Without results you have nothing to submit.

**3. The combination is additive, not multiplicative.** Combining diffusion + GAN + SwinIR + text conditioning + wavelet discriminator is engineering work. Each piece exists. Reviewers at CVPR/ICCV/ECCV or IEEE TGRS will ask: what is the single insight that makes this combination necessary and why does it work?

---

## What could become publishable

There are three genuine angles worth pursuing that the existing literature does not cover well:

**Angle 1 (strongest): The SR + counterfactual edit unification**
No paper currently trains a single model that both (a) physics-constrained SR with LR consistency guarantees and (b) text-guided counterfactual appearance editing with an explicit "what does the text actually change vs. what must be preserved" spatial gate. This is the most novel part of the architecture — the evidence gate + dual mode training. A paper focused entirely on this, with careful ablations of the gate, would be a real contribution.

**Angle 2: VLM-derived scene captions as SR training signal**
Using Qwen3-VL to auto-caption satellite patches and studying whether richer semantic supervision from those captions actually improves SR fidelity (measured by land-cover-specific metrics, not just PSNR) is an interesting empirical question nobody has studied. This could be a short paper with a clear hypothesis.

**Angle 3: Geographic generalization study**
The tile-level splitting is carefully designed to measure geographic generalization. If you collect enough MGRS tiles across diverse land cover types and biomes, and measure how SR quality degrades across distribution shifts, that's a genuine empirical contribution to the EO SR literature.

---

## What you'd need to do to submit

1. Fix the bugs (especially #1 — captioning crashes — and #5 — train/inference inconsistency)
2. Actually train the full model and run baselines (bicubic, SwinIR, Donike et al. if you can reproduce)
3. Pick one of the three angles above as the paper's central claim
4. Run the ablations that support that claim (evidence gate on/off, text conditioning on/off, dual mode vs. separate models)
5. Target IEEE TGRS, IGARSS, or ISPRS Journal for EO-specific SR; or ECCV/ICCV workshops on vision + remote sensing if you want a faster turnaround

The target venue matters because at a general vision venue (CVPR, NeurIPS) the novelty bar relative to natural image SR methods is very high. At an EO-specific venue (TGRS, JSTARS) the bar is more reachable and the community will value the Sentinel-2 specificity more.


Let me search for the current state of the field to give you targeted advice.Good — I now have a thorough picture of the field as of mid-2026. Let me give you a complete, honest research landscape and the best directions for your paper.

---

## Where the field actually stands right now

The satellite image SR field has become genuinely crowded. A 2025 comprehensive survey covers more than 400 references on remote sensing image SR methods, representing the most up-to-date and well-structured review to date. Existing methods have clear documented limitations: CNNs with restricted receptive fields cause edge blurring, Transformers fail at gigapixel-scale processing, and SSMs have directional biases that miss diagonal features.

The key things that are **already solved or saturated**:
- Single-image Sentinel-2 SR with diffusion (Donike et al. 2025, IEEE JSTARS)
- Multi-temporal SR with diffusion (SatDiffMoE, DiffusionSat, ICLR 2024)
- Diffusion-GAN hybrids for perceptual quality (AdcSR CVPR 2025, SupResDiffGAN 2025)
- Wavelet-conditioned diffusion SR (2024)
- Text-guided SR for natural images (PromptSR, text-guided explorable SR)

The things that are **genuinely open**:

At ESA's SUREDOS24 workshop, one key takeaway was that SR brings great responsibility — ensuring validity and reliability for scientific and commercial use is paramount, and quality control, validation of SR products, and uncertainty quantification are essential for reliable adoption in downstream services.

The EU JRC has published a framework specifically calling out hallucination detection protocols, memorization effects, multi-source verification procedures, and structured uncertainty management as unresolved problems for operational satellite SR.

---

## The real publishable research directions in satellite SR (2025–2026)

Here are the gaps that are legitimately open, ranked by how achievable they are and how much the community cares:

### Direction 1 — Task-aware SR (strongest, most publishable)

The biggest open problem is that SR is evaluated in isolation (PSNR, SSIM) but deployed to serve downstream tasks (crop monitoring, change detection, building extraction, flood mapping). Research on diffusion-based SR for crop type mapping shows that super-resolved predictions preserve crop shapes and boundaries more faithfully, recovering small-scale structures absent in coarse-resolution results — but this downstream evaluation is rarely the primary design criterion.

Nobody has built an SR model that is explicitly optimized to maximize performance on a specific downstream task (e.g. semantic segmentation mIoU or change detection F1) rather than pixel-level metrics. This is a clean, falsifiable, impactful contribution.

**What the paper would look like:** Design a loss function or training objective that jointly optimizes SR quality and a downstream perception task. Show on at least two tasks (e.g. building segmentation + crop classification) that task-aware SR beats PSNR-optimized SR on those tasks, even at similar PSNR levels.

**Target venue:** IEEE TGRS, ISPRS Journal of Photogrammetry, or ECCV/ICCV remote sensing workshop.

---

### Direction 2 — Hallucination detection and SR faithfulness

The Donike et al. 2025 paper delivers an uncertainty product alongside SR results, showing that uncertainty correlates with edges and complex shapes — but this increases computational requirements 10–20× over standard inference. The hallucination problem is identified but not solved:

- When does a diffusion-based SR model invent structures that don't exist (roads, buildings, field boundaries)?
- Can you detect at inference time which pixels are hallucinated vs. faithfully recovered?
- Can you design a model that is by construction conservative where it's uncertain?

Deep ensemble uncertainty quantification for Sentinel-2 SR has been explored at small scale, but systematic hallucination characterization and mitigation is not established.

**What the paper would look like:** A method to identify and spatially localize hallucinated pixels in diffusion SR outputs. You could use cross-sensor validation (if SR of a 10m image invents a building, does a real 1.5m image of the same location confirm it?), or design a self-consistency metric derived from the LR observation. Pair this with a conservative inference mode that blends toward bicubic in high-uncertainty regions.

**Target venue:** IEEE GRSL (Letters), or Nature Scientific Reports remote sensing.

---

### Direction 3 — Geographic generalization and domain shift

A documented challenge is that low-resolution and high-resolution images often come from different sensors, creating a large imaging modality gap that existing methods don't fully address. More specifically: SR models trained on European agricultural landscapes fail on tropical forest, Arctic tundra, or dense urban slums. Nobody has systematically characterized this degradation or proposed a domain-adaptive SR model for EO.

**What the paper would look like:** Collect Sentinel-2 patches across 6–8 biomes or land cover types. Train a single SR model and measure how PSNR/SSIM/downstream task quality varies across biomes. Propose either a domain-adaptive fine-tuning approach (lightweight LoRA-style adapter per land cover type) or a land cover conditioned SR model. Show that conditioning on predicted land cover at inference time improves generalization.

This is achievable without massive compute — a LoRA-style adapter adds very few parameters to an existing model.

**Target venue:** IEEE TGRS or Remote Sensing (MDPI, open access).

---

### Direction 4 — SR for compressed/degraded archival imagery

There is a large archive of historical satellite imagery (Landsat 5, SPOT 1–3, early EO-1) that is 30–60m resolution, heavily compressed, and cloud-affected. Restoring this for change detection over decades is a genuine scientific need. Multi-frame SR is constrained by the limited availability of multi-temporal or multi-sensor imagery and the computational complexity of accurate image registration.

No published method jointly handles: sensor-specific degradation (JPEG compression artifacts + low-bit quantization + MTF blur) + cloud masking + temporal fusion for pre-2000 archives.

**What the paper would look like:** A restoration-then-SR pipeline for Landsat-5 or SPOT-1 imagery, evaluated by comparing against later-acquired higher-resolution coverage of the same area. This has real scientific value for biodiversity, deforestation, and urban growth studies.

---

### Direction 5 — Efficient SR for operational deployment

DeepSpace (ACM SIGCOMM 2025) shows that SR can compress satellite imagery by over two orders of magnitude for space-Earth data transfer using a mixture-of-experts framework on edge hardware. The key open problem is making diffusion-based SR fast enough for operational use — current latent diffusion models take hours per Sentinel-2 tile.

If you can show that a 1–2 step diffusion model (consistency distillation or adversarial diffusion distillation applied to the satellite domain) matches a 20-step model in downstream task accuracy while running 10× faster, that's publishable.

---

## Where your GeoDiff-GAN fits

Given what you've already built, the most efficient path to a paper is **Direction 2 (hallucination/faithfulness)** combined with the dual SR+edit architecture you already have. Here's why:

Your model already:
- Produces stochastic SR samples (you can compute per-pixel variance across samples)
- Has an evidence gate that spatially localizes where text prompts influence the output
- Has differentiable back-projection that measures LR consistency

You are two experiments away from a paper:

1. **Show that your evidence gate predicts hallucination.** Run your SR model, compute pixel-wise variance across 8 samples, and show that high-variance pixels (hallucinated) correlate with low evidence gate activation — meaning the model is uncertain where the LR image gives no signal. If this correlation holds, the gate is a free hallucination detector.

2. **Show that clamping the gate (disabling edit mode, forcing SR mode) reduces hallucination rate** as measured by cross-sensor validation against a held-out VHR reference dataset. This proves that the SR/edit mode distinction actually controls faithfulness.

Those two experiments, with the right framing, are a focused 8-page GRSL paper: "Spatial evidence gating for hallucination-aware satellite image super-resolution."

---

## What you need to do next (concrete steps)

1. Fix the 3 critical bugs (captioning crash, train/inference back-projection mismatch, distributed training sync)
2. Train the full model on at minimum 5–6 geographically separated MGRS tiles covering different land cover types
3. Run the 8-sample stochastic evaluation you already have code for — this gives you variance maps for free
4. Acquire a small VHR reference dataset (WorldStrat is open, WorldView tiles from OpenAerialMap, or Airbus SPOT archive free academic tier) to validate against
5. Run the evidence gate vs. variance correlation analysis
6. Pick one downstream task (semantic segmentation using a pretrained model like SegFormer) and show that your SR output improves it vs. bicubic and SwinIR baseline

That is a publishable paper. The code is mostly already written.