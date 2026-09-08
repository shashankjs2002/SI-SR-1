# 06. Supervisor Questions and Short Answers

## Motivation and Data

**What does the model do?** It estimates a Sentinel-like 10 m RGB reconstruction
from an actual Landsat 30 m RGB observation using a base plus selective corrections.

**Why not use bicubic?** Bicubic interpolates existing samples. A learned model can
estimate cross-sensor structure, but it must demonstrate improvement on held-out pairs.

**Why do LR and HR plots look the same size?** The axes show the same geographic
footprint. Their native arrays differ: 128x128 versus 384x384. Display size is not
sensor resolution.

**Does 3x mean three times more pixels?** No: three times each side, nine times
the pixel count in two dimensions.

**Is Sentinel absolute ground truth?** It is the available cross-sensor reference.
Sensor response, acquisition time and residual registration differences remain.

**Why masks?** To keep invalid or rejected pixels from contributing as if they were
valid supervision. A mask is not a guarantee of perfect correspondence elsewhere.

## Architecture

**Is everything a Transformer?** No. Window attention appears in the base, contextual
attention in the router, and efficient convolutions in the encoder/experts/trust head.

**Is this a published TrustMoE implementation?** No. GeoDiff-TrustMoE is this project's
working name. New code does not by itself establish scientific novelty.

**Where is diffusion?** Not in this new model. The old diffusion implementation is
preserved separately, but the new inference path does not instantiate it.

**Why a base?** It supplies a reconstruction reference and allows a direct audit of
whether the outer branch contributes anything useful.

**Does the base run again at the end?** No. There is one pass, followed by a gated
residual addition and clamping.

**What is an expert?** A small independently parameterized CNN producing an RGB
correction for a selected feature block. It is not automatically a land-cover classifier.

**Are experts assigned urban/forest labels?** No. These meanings must be inferred
carefully from empirical usage and outputs, not assumed from the expert index.

**What is top-k?** The number of experts selected per active block. Coverage controls
how many blocks are active; trust controls their accepted correction afterward.

**Can top-k change after training?** Yes, within the trained expert bank. It is a
changed inference policy requiring validation, not a free performance improvement.

**Why use halos?** To give each expert neighboring context before its output core is
assembled. They add duplicated work and do not completely rule out seams.

**What is the trust map?** Three sigmoid acceptance values at each HR location,
predicted from LR-grid evidence and resized logits. It is not ground-truth confidence.

**What is the difficulty map?** A prediction of local frozen-base log-MSE. Large
values can come from real change or misalignment as well as missing detail.

## Training and Capacity

**What trains first?** The base. Then it is frozen while the outer modules learn
directly in image space. No VAE/diffusion/joint sequence is required.

**How can trust learn without test HR?** Training HR supplies acceptance labels.
At test time the learned predictor receives only observable LR/base information.

**Does a mixed forest/desert batch confuse the experts?** Not inherently. Routing is
per region per image. Averaged gradients are standard; imbalance and collapse need
diagnosis rather than assuming mixed batches are invalid.

**How many parameters?** The default main model has 205,387; 76,531 are in the frozen
base and 128,856 train during residual optimization. The optional discriminator is extra.

**Should we immediately make it large?** First assess base quality and matched
controls. More capacity can help underfitting or worsen cost/overfitting. It is not novelty.

**Is the default base the same size as the earlier one?** No. Compare actual parameter
counts and checkpoint settings, not the shared word "Swin" in their names.

**Does increasing epochs break resume?** Epoch increases are supported. Architecture,
data, loss, batch/crop or parent-base changes require a separate consistent experiment.

## Results and Claims

**Does trust guarantee no PSNR loss?** No. It is a learned gate with a soft local
penalty, not a mathematical guarantee. Zero coverage/scale exactly returns the base.

**If all regions run, is the model still spatially sparse?** No. Top-k expert selection
remains, but there is no spatial bypass at full coverage.

**Is 50% coverage twice as fast?** Not necessarily. The base and shared heads remain
dense; halos, dispatch and attention cost time. Only measured timing can answer.

**Are increasing Edge F1 and stable PSNR enough for a paper?** They are useful evidence.
Novelty, strong comparisons, repeatability, spectral behavior and data independence
also matter. Edge thresholds and selective reporting can otherwise mislead.

**What if one expert is unused?** Report it. Possible causes include collapse, insufficient
training, insufficient diversity or redundant capacity. Do not invent a semantic role.

**Why a separate test set?** To assess a fixed model/policy on pairs not used for
training or validation selection. Repeated tuning from test results weakens that claim.

**What is the proposed contribution?** Selective regional residual computation with
base-error guidance and per-band acceptance, evaluated for fidelity and speed.
Its superiority and uniqueness are hypotheses until evidence supports them.

## One-Minute Explanation

"We first reconstruct the Landsat image with a compact residual-Swin base. A small
Transformer then examines region-level features and predicts where the base needs
correction and which experts should handle those regions. Only selected blocks are
processed by the expert CNNs. A learned RGB trust map controls the added residual.
We compare against the same frozen base, a single expert, all-region processing and
other ablations. Our question is whether this improves spatial detail without a
meaningful fidelity loss while reducing computation. We have implemented the method;
we do not claim that its accuracy, speed or novelty is already established."
