# 36 - Spatial, Spectral, Radiometric, and Temporal Resolution

## Learning objectives

After this chapter, you should be able to explain:

1. the four main resolution dimensions in optical remote sensing;
2. how Landsat 8/9 OLI and Sentinel-2 MSI differ in each dimension;
3. why a Landsat-Sentinel difference image is not pure spatial detail;
4. how the current paired-data pipeline handles each mismatch;
5. which diagnostic plots expose spatial, spectral, radiometric, or temporal problems;
6. what the model can learn incorrectly if these effects are ignored.

## The missing word

The word beginning with "radio" was almost certainly **radiometric**.

The four standard resolution dimensions are:

| Resolution | Basic question | Typical unit |
|---|---|---|
| Spatial | How much ground does one pixel represent? | metres per pixel |
| Spectral | Which wavelength intervals are measured? | bands and nanometres |
| Radiometric | How finely can intensity differences be distinguished? | bits, levels, SNR, accuracy |
| Temporal | How frequently and how closely in time is a place observed? | days |

They are independent. An image can have fine spatial resolution but poor spectral,
radiometric, or temporal suitability.

## One-minute supervisor answer

> Landsat and Sentinel do not differ only in pixel size. Landsat RGB pixels represent
> 30 m ground samples and Sentinel RGB pixels represent 10 m samples, producing the 3x
> spatial task. Their RGB filters also cover different wavelength intervals, so equal
> surfaces need not produce equal red, green, and blue values. Both instruments acquire
> 12-bit measurements, but they use different calibration, atmospheric correction, noise
> characteristics, and product scaling, creating radiometric differences. Finally, the
> scenes can be acquired on different dates, so vegetation, soil moisture, illumination,
> clouds, and human activity may change. Therefore the model target contains spatial,
> spectral, radiometric, temporal, and registration effects, not only missing detail.

## 1. Spatial resolution

### Definition

Spatial resolution describes the ground area represented by one image sample. For this
experiment:

\[
r_L=30\text{ m},\qquad r_S=10\text{ m},\qquad s=\frac{30}{10}=3.
\]

One nominal Landsat pixel covers the same area as a `3 x 3` block of Sentinel pixels:

\[
30^2=900\text{ m}^2,\qquad 9\times10^2=900\text{ m}^2.
\]

A `128 x 128` Landsat patch and a `384 x 384` Sentinel patch therefore cover the same
nominal `3.84 km x 3.84 km` footprint.

### What finer spatial resolution provides

Sentinel can sample smaller roads, field boundaries, roof groups, drainage lines, and
mixed land-cover transitions. It does not guarantee that every 10 m sample is independent
or perfectly sharp. The optical point-spread function, modulation transfer function,
resampling, motion, atmosphere, and registration also determine effective resolution.

### What the current pipeline does

- Uses Sentinel B04 as the `384 x 384` target grid.
- Creates an exactly corresponding `128 x 128` Landsat grid with 30 m spacing.
- Reprojects Landsat RGB using bilinear resampling.
- Reprojects QA masks using nearest-neighbour resampling.
- Expands the Landsat validity mask by 3x for joint HR masking.
- Uses resize-convolution rather than PixelShuffle for 3x upsampling.

### Important distinction

Bicubic interpolation changes array size, not spatial evidence. A bicubic `384 x 384`
Landsat display still contains information measured at approximately 30 m.

## 2. Spectral resolution

### Definition

Spectral resolution describes how the electromagnetic spectrum is divided into bands.
It includes:

- number of bands;
- centre wavelength;
- bandwidth;
- complete spectral response function, not just the centre.

The experiment currently uses only RGB:

| Display channel | Landsat 8/9 OLI | Approximate range | Sentinel-2 MSI | Nominal centre / width |
|---|---|---:|---|---:|
| Blue | B2 | 450-510 nm | B02 | 490 / 65 nm |
| Green | B3 | 530-590 nm | B03 | 560 / 35 nm |
| Red | B4 | 640-670 nm | B04 | 665 / 30 nm |

Landsat wavelength ranges are published by USGS. Sentinel nominal centres and bandwidths
are published in the Sentinel-2 User Handbook. Platform-specific response functions vary
slightly, so these rectangles are explanatory approximations rather than complete sensor
response curves.

### Why nominally matching RGB bands can disagree

A recorded band value is a weighted integration over wavelength:

\[
y_b=\int R(\lambda)S_b(\lambda)E(\lambda)T(\lambda)\,d\lambda+\epsilon,
\]

where:

- \(R(\lambda)\) is surface spectral reflectance;
- \(S_b(\lambda)\) is the sensor band response;
- \(E(\lambda)\) is illumination;
- \(T(\lambda)\) represents atmospheric transmission;
- \(\epsilon\) includes noise and calibration error.

If Landsat and Sentinel use different \(S_b\), equal ground does not imply equal band
values. Vegetation, water, bright soil, and minerals can be especially sensitive because
their reflectance changes rapidly with wavelength.

### What the current pipeline does

The default notebook uses:

```text
BANDPASS_ADJUSTMENT = "none"
```

This preserves each product's supplied reflectance but leaves cross-sensor bandpass bias
in the training target. For Sentinel-2A/B, the pipeline optionally supports:

```text
BANDPASS_ADJUSTMENT = "hls-oli"
```

This applies HLS-style linear coefficients to make Sentinel RGB more OLI-like. The current
coefficient table does not cover Sentinel-2C, so S2C must use `none` unless validated S2C
coefficients are added. Never silently apply S2A/B coefficients to S2C.

### What a spectral scatter plot tells us

For each band, plot enlarged Landsat reflectance \(x\) against Sentinel reflectance \(y\):

- points on \(y=x\): strong agreement;
- shifted cloud: additive radiometric bias;
- different slope: gain or bandpass mismatch;
- curved relation: nonlinear or land-cover-dependent mismatch;
- wide cloud: spatial mixture, temporal change, noise, or misregistration.

This plot diagnoses combined cross-sensor disagreement. It does not isolate spectral
response by itself because the pixels also differ spatially and temporally.

## 3. Radiometric resolution

### Definition

Radiometric resolution is the ability to distinguish small differences in measured
intensity or reflectance. Bit depth gives the number of possible digital levels:

\[
L=2^b.
\]

For 12-bit acquisition:

\[
L=2^{12}=4096\text{ potential levels}.
\]

Both Landsat OLI and Sentinel-2 MSI acquire measurements at 12 bits. This does **not**
make their radiometry identical. Radiometric quality also depends on:

- signal-to-noise ratio;
- saturation behaviour and dynamic range;
- absolute and relative calibration;
- stray light;
- atmospheric correction;
- viewing and solar geometry;
- quantization and product encoding;
- resampling and clipping.

### Instrument bits versus product storage

Do not confuse these quantities:

| Quantity | Meaning |
|---|---|
| 12-bit instrument acquisition | 4096 potential detector intensity codes |
| 16-bit product storage | File container used to deliver processed values |
| Reflectance scale factor | Conversion from stored DN to physical reflectance |
| Radiometric accuracy | Closeness to the calibrated physical value |
| SNR | Signal strength relative to noise variation |

Sentinel documentation states that 12-bit MSI measurements are converted to reflectance
and stored as 16-bit integers. Landsat Collection 2 Level-2 surface reflectance is also
delivered as unsigned 16-bit integers, but the OLI instrument has a 12-bit dynamic range.
The storage type therefore does not mean either instrument has 16-bit physical sensitivity.

### Reflectance conversion used in this project

Landsat Collection 2 Level-2:

\[
\rho_L=DN_L\times M_L+A_L,
\]

with metadata values and USGS fallbacks:

\[
M_L=0.0000275,\qquad A_L=-0.2.
\]

Sentinel-2 L2A:

\[
\rho_S=\frac{DN_S+BOA\_ADD\_OFFSET}{BOA\_QUANTIFICATION\_VALUE}.
\]

The code reads these values from each product's metadata. It then clips prepared model
tensors to `[0,1]`. Clipping is practical for the RGB network but should be reported
because negative atmospheric-correction values and reflectance above one are possible in
source products.

### Why histograms can differ despite correct scaling

Landsat LaSRC and Sentinel Sen2Cor are different atmospheric-correction systems. A
histogram shift can result from calibration, bandpass, atmosphere, sun/view geometry,
BRDF, haze, adjacency effects, resampling, or actual temporal change. Histogram matching
may make images look similar but can remove physically meaningful differences and should
not be applied silently to supervised targets.

### Useful radiometric diagnostics

- signed bias \(mean(S-L)\);
- MAE and RMSE per band;
- robust percentiles `p02`, `p50`, and `p98`;
- saturation and clipping fraction;
- bandwise correlation and regression slope;
- signed difference histograms;
- statistics stratified by land-cover class when labels are available.

## 4. Temporal resolution

### Definition

Temporal resolution describes how frequently a sensor can observe the same location. It
also describes the date separation of a specific supervised pair.

- Each Landsat 8 or 9 satellite has a 16-day repeat cycle; their 8-day offset provides
  combined repeat coverage approximately every 8 days.
- The standard two-satellite Sentinel-2 constellation provides a nominal 5-day revisit at
  the equator.
- Actual usable cloud-free observations are less frequent.

Mission revisit is not the same as pair day gap. The dataset records:

\[
\Delta t=|date_{Sentinel}-date_{Landsat}|.
\]

The default notebook accepts `MAX_DAY_GAP = 3`, although changing the control can admit
larger gaps. Every report must state the actual gap distribution, not only the configured
maximum.

### What can change within a few days

- clouds, shadows, haze, and aerosol;
- soil moisture and water extent;
- crop growth, harvest, or irrigation;
- fire, flood, construction, traffic, or excavation;
- sun angle and shadow length;
- view angle and bidirectional reflectance.

A ten-day pair may be visually similar but cannot be assumed unchanged. Temporal changes
become label noise: the network may be penalized for reproducing the Landsat-time scene or
rewarded for inventing a Sentinel-time change that was not observable in Landsat.

## 5. The cross-sensor residual is a mixture

The pair difference can be written conceptually as:

\[
S-U(L)=D_{spatial}+D_{spectral}+D_{radiometric}+D_{temporal}
+D_{registration}+\epsilon.
\]

Here \(U(L)\) is Landsat placed on the Sentinel grid. This is not an exactly separable
physical equation; it is a reasoning model showing the sources mixed into the target.

GeoDiff-GAN currently learns the complete observed residual. Without controls, it can
mistake:

- spectral colour bias for spatial detail;
- registration double edges for texture;
- temporal land-cover change for a predictable 10 m structure;
- atmospheric haze differences for a decoder residual;
- interpolation patterns for real high frequency.

## 6. How each dimension affects GeoDiff-GAN

| Dimension | Model risk | Current protection | Remaining limitation |
|---|---|---|---|
| Spatial | Hallucinated or shifted edges | common grid, masks, edge plots, LR consistency | reprojection does not guarantee sub-pixel alignment |
| Spectral | Learns colour conversion as "detail" | optional HLS OLI adjustment | no S2C coefficient; RGB filters remain non-identical |
| Radiometric | Learns gain, haze, or calibration bias | metadata scaling, surface reflectance, clipping | different processors and no per-scene BRDF normalization |
| Temporal | Learns scene change as SR | nearest-date pairing and recorded gap | even same-day acquisitions differ in time and geometry |

The degradation vector conditions the model on normalized day gap, platform indicator,
nominal blur, and invalid fraction. It does not yet encode sun zenith, view angle, aerosol,
per-band calibration uncertainty, or a measured registration offset.

## 7. Experiments needed to separate the effects

### Spatial ablation

1. Estimate residual x/y registration using stable edges.
2. Compare no refinement versus sub-pixel coregistration.
3. Report edge F1 and high-frequency spectra, not PSNR alone.
4. Compare 0 and 3 back-projection steps.

### Spectral ablation

1. Restrict to S2A/B pairs.
2. Train `bandpass_adjustment=none` and `hls-oli` with identical splits.
3. Compare per-band bias, slopes, PSNR, SSIM, and edge F1.
4. Do not claim improvement if only colour bias falls while edge fidelity does not.

### Radiometric ablation

1. Report raw metadata-scaled reflectance statistics.
2. Compare optional global train-set normalization, fitted only on training tiles.
3. Never fit normalization on validation or test imagery.
4. Report clipping and saturation fractions.

### Temporal ablation

Train or evaluate gap groups separately:

```text
same day, 1-3 days, 4-7 days, and more than 7 days
```

If performance degrades with gap, temporal mismatch is contributing label noise. If it
does not, verify that geography or scene type is not confounded with the gap groups.

## 8. How to read the notebook diagnostics

### Spatial panel

The `1 Landsat pixel = 3 x 3 Sentinel pixels` diagram explains sample count, not measured
optical sharpness. The edge overlay tests alignment. Repeated coloured double edges imply
registration error.

### Nominal bandpass panel

Overlapping bars show that RGB labels are related, but unequal starts, ends, and response
curves explain cross-sensor colour differences.

### Per-band scatter

Use identity line, correlation, bias, and MAE together. High correlation with nonzero bias
means structure agrees but calibration or bandpass differs.

### Radiometric histograms

A narrow difference around zero suggests agreement. A shifted distribution indicates bias.
Long tails can indicate changed pixels, cloud contamination, saturation, or misregistration.

### Temporal plots

The count chart shows dataset balance. The MAE-by-gap chart tests association, not causality.
Geography, season, cloud, and land cover may also differ between gap groups.

## 9. Common supervisor questions

### Are spatial and spectral resolution the same?

No. Spatial resolution divides the ground into samples; spectral resolution divides
wavelength into bands.

### Does 12-bit mean 12 spectral bands?

No. Twelve-bit is radiometric quantization. Sentinel has 13 spectral bands, while Landsat
OLI has 9 reflective bands including panchromatic and cirrus.

### Do both 12-bit sensors produce the same reflectance?

No. Bit depth counts potential levels. Calibration, SNR, filters, atmosphere, geometry,
processing, and time still differ.

### Is Sentinel the exact ground truth for Landsat?

No. It is a higher-spatial-resolution cross-sensor reference observation. It is not a
simultaneous, spectrally identical measurement of hidden Landsat sub-pixels.

### Why use surface reflectance?

Surface reflectance reduces atmospheric effects and is more physically comparable than raw
DN or top-of-atmosphere radiance, but the two processing chains still differ.

### Why not histogram-match every pair?

Pairwise matching can leak target information into the input and erase real scene-level
radiometric variation. It can artificially improve image metrics without improving SR.

### Why is a ten-day gap dangerous?

Because changes between acquisitions are not inferable from the earlier image. The network
can be trained to hallucinate time change rather than spatial detail.

### What is the strongest statement we can make?

The system learns a statistically useful mapping from real Landsat surface-reflectance RGB
to a nearby-date Sentinel-2 RGB reference under the documented pairing and preprocessing
conditions. It does not recover uniquely true 10 m measurements from 30 m evidence.

## 10. Official references

- [USGS Landsat 8 bands, 12-bit OLI dynamic range, and repeat cycle](https://www.usgs.gov/landsat-missions/landsat-8)
- [USGS Landsat 8/9 wavelength designations](https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites)
- [USGS Collection 2 Level-2 surface-reflectance scaling](https://www.usgs.gov/faqs/how-do-i-use-a-scale-factor-landsat-level-2-science-products)
- [Copernicus Sentinel-2 mission and band spatial resolutions](https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html)
- [Sentinel-2 User Handbook: bandwidth, SNR, and 12-bit radiometry](https://sentinels.copernicus.eu/documents/247904/685211/S2_User_Handbook.pdf/8869acdf-fd84-43ec-ae8c-3e80a436a16c)
- [Copernicus SentiWiki radiometric-resolution explanation](https://sentiwiki.copernicus.eu/web/s2-mission)
- [ESA Sentinel-2 facts and five-day revisit](https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2/Facts_and_figures)
- [NASA HLS v2 user guide for bandpass harmonization](https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf)

## Mastery checklist

- [ ] I can define all four resolution dimensions without mixing them.
- [ ] I can derive the 3x linear and 9x sample-count relationship.
- [ ] I can explain why matching RGB names do not mean identical bandpasses.
- [ ] I can distinguish instrument bit depth from product storage type.
- [ ] I can write both reflectance-scaling equations.
- [ ] I can explain why pair MAE mixes more than spatial detail.
- [ ] I can interpret scatter, histogram, edge, and day-gap plots.
- [ ] I can describe the `none` versus `hls-oli` spectral ablation.
- [ ] I can state why Sentinel is a reference rather than exact ground truth.

Previous: [35 - Training, Debugging, Evaluation, and Supervisor Questions](35_training_debugging_evaluation_viva.md).
