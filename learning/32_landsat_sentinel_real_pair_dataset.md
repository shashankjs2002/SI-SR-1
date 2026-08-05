# 32 - Landsat 30 m to Sentinel-2 10 m Paired Dataset

## Required EarthExplorer files

For each Landsat 8/9 Collection 2 Level-2 product, retain files with the same
product prefix:

```text
*_MTL.txt                 required for metadata audit; standard scaling is the fallback
*_QA_PIXEL.TIF            required
*_QA_RADSAT.TIF           required
*_SR_QA_AEROSOL.TIF       strongly recommended
*_SR_B2.TIF               blue input
*_SR_B3.TIF               green input
*_SR_B4.TIF               red input
*_ANG.txt                 optional; retain for later BRDF work
```

Do not combine individual files from different Landsat product IDs. The
preparer accepts either one directory per product or many complete products in
one flat directory.

The Sentinel input remains complete L2A `.SAFE` products. Complete SAFE
metadata is required because products from processing baseline 04.00 onward
need `BOA_ADD_OFFSET` as well as `BOA_QUANTIFICATION_VALUE`.

## Patch contract

```text
hr:              [3, 384, 384] Sentinel B04/B03/B02 at 10 m
lr:              [3, 128, 128] Landsat SR_B4/SR_B3/SR_B2 at 30 m
clean_lr:        [3, 128, 128] same real Landsat observation
degradation:     [4] sensor condition
valid_mask_hr:   [1, 384, 384]
valid_mask_lr:   [1, 128, 128]
```

The two tensors cover the same 3.84 km square. Landsat is reprojected directly
to an exact 30 m grid derived from the Sentinel 10 m transform. It is not first
upsampled to 10 m and then downsampled.

## Preparation command

```bash
python -m geodiff_gan.cli.prepare_landsat_sentinel \
  --sentinel-input /path/to/sentinel-safe-root \
  --landsat-input /path/to/extracted-landsat-files \
  --output /path/to/landsat-sentinel-3x/patches \
  --manifest /path/to/landsat-sentinel-3x/manifest.jsonl \
  --patch-size 384 \
  --stride 288 \
  --max-day-gap 3 \
  --minimum-overlap-fraction 0.10 \
  --minimum-valid-fraction 0.95 \
  --bandpass-adjustment none \
  --unmatched-split hash
```

Use `--max-pairs 1` for the first development run. Rerunning the same command
skips completed pairs, including pairs that produced zero valid patches.

For S2A/S2B only, `--bandpass-adjustment hls-oli` applies published HLS
coefficients to place the Sentinel target on the Landsat OLI spectral
reference. Do not apply those coefficients to S2C; no S2C coefficients are
hard-coded.

## Quick inspection

```python
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F
import torch

manifest = Path("/path/to/landsat-sentinel-3x/manifest.jsonl")
records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
index = 0
record = records[index]

with np.load(record["patch"]) as data:
    lr = torch.from_numpy(data["lr"]).float()
    hr = torch.from_numpy(data["hr"]).float()
    valid = torch.from_numpy(data["valid_mask_hr"])[0].bool()

lr_up = F.interpolate(
    lr[None], size=hr.shape[-2:], mode="nearest"
)[0]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for axis, image, title in zip(
    axes,
    (lr_up, hr, hr * valid[None]),
    ("Landsat input (display enlarged)", "Sentinel target", "Joint valid area"),
):
    axis.imshow(image.clamp(0, 1).permute(1, 2, 0))
    axis.set_title(title)
    axis.axis("off")
plt.suptitle(
    f"gap={record['day_gap']} day(s) | "
    f"{record['landsat_product']} -> {record['sentinel_product']}"
)
plt.tight_layout()
plt.show()
```

Inspect roads, riverbanks, field boundaries, and coastlines. Reject the pair if
the same feature is shifted consistently between sensors. A high numerical
valid fraction does not prove sub-pixel co-registration or absence of temporal
change.

## First training command

Update `data.manifest` and `training.output_dir` in
`configs/landsat_sentinel_3x_small.yaml`, then train the deterministic base:

```bash
python -m geodiff_gan.cli.train \
  --config configs/landsat_sentinel_3x_small.yaml \
  --defaults configs/default.yaml
```

Do not begin diffusion/GAN training until bicubic and deterministic-base
results have been checked on geographically held-out scene pairs.
