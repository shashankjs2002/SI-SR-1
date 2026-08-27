# Uncertainty Abstention: Numerical Dry Run

Consider one pixel in the satellite image. We will run the exact inference calculation manually.

Assume:

```text
SwinIR base pixel = 0.40
Ground-truth pixel = 0.50

samples = 4
uncertainty_scale = 0.0025
residual_scale = 0.75
```

## Step 1: Generate Four Diffusion Outputs

Diffusion starts from different random noise four times:

```text
Sample 1 = 0.51
Sample 2 = 0.49
Sample 3 = 0.50
Sample 4 = 0.52
```

These values are similar, so diffusion is reasonably certain about this pixel.

## Step 2: Calculate Mean Output

```text
mean = (0.51 + 0.49 + 0.50 + 0.52) / 4
     = 0.505
```

Without uncertainty control, the generated output would be:

```text
Raw generated pixel = 0.505
```

## Step 3: Calculate Uncertainty

The model calculates variance across the samples:

```text
Differences from mean:

0.51 - 0.505 =  0.005
0.49 - 0.505 = -0.015
0.50 - 0.505 = -0.005
0.52 - 0.505 =  0.015
```

Square and average:

```text
variance =
(0.005^2 + 0.015^2 + 0.005^2 + 0.015^2) / 4

= 0.000125
```

Therefore:

```text
uncertainty = 0.000125
```

For RGB images, this is calculated for each channel and then averaged across RGB.

## Step 4: Convert Uncertainty To Agreement

The formula is:

```python
agreement = exp(-uncertainty / uncertainty_scale)
```

Substitute the values:

```text
agreement = exp(-0.000125 / 0.0025)
          = exp(-0.05)
          ~= 0.951
```

Interpretation:

> The four samples agree strongly, so retain approximately 95.1% of the generated correction.

## Step 5: Blend With The Base

The raw generated correction is:

```text
correction = generated_mean - base
           = 0.505 - 0.40
           = 0.105
```

Apply uncertainty agreement:

```text
uncertainty-controlled output
= base + agreement * correction

= 0.40 + 0.951 * 0.105
= 0.4999
```

So uncertainty control produces approximately:

```text
0.500
```

## Step 6: Apply Residual Scale

Now apply:

```text
residual_scale = 0.75
```

```text
final output
= base + 0.75 * (uncertainty output - base)

= 0.40 + 0.75 * (0.4999 - 0.40)
= 0.4749
```

The final pixel is approximately:

```text
0.475
```

Comparison:

| Output | Value | Error from target `0.50` |
|---|---:|---:|
| Base | `0.400` | `0.100` |
| Raw diffusion mean | `0.505` | `0.005` |
| After uncertainty | `0.500` | `0.000` |
| After residual scale `0.75` | `0.475` | `0.025` |

Here, diffusion was reliable, so residual scaling to `0.75` was unnecessarily conservative. But another pixel may behave differently.

## Uncertain Pixel Dry Run

Assume the four diffusion samples disagree:

```text
Sample 1 = 0.44
Sample 2 = 0.56
Sample 3 = 0.38
Sample 4 = 0.62
```

The mean is still perfect:

```text
mean = 0.50
```

But the variance is high:

```text
uncertainty ~= 0.0098
```

Calculate agreement:

```text
agreement = exp(-0.0098 / 0.0025)
          = exp(-3.92)
          ~= 0.020
```

Only 2% of the generated correction is retained:

```text
uncertainty output
= 0.40 + 0.020 * (0.50 - 0.40)

= 0.402
```

Apply residual scale `0.75`:

```text
final
= 0.40 + 0.75 * (0.402 - 0.40)

= 0.4015
```

Although the average generated output was `0.50`, the model returns close to the base because the four samples strongly disagreed.

This is called **abstention**:

> When the generative branch is unstable, trust the deterministic base.

## Three-Region Example

Suppose an image contains three regions:

| Region | Uncertainty | Agreement with scale `0.0025` | Action |
|---|---:|---:|---|
| Large clear road | `0.0001` | `96%` | Retain generated correction |
| Field boundary | `0.0025` | `37%` | Retain limited correction |
| Complex building texture | `0.0100` | `2%` | Return almost completely to base |

Therefore, uncertainty operates spatially:

```text
Stable region    -> use diffusion
Moderate region  -> mix diffusion and base
Unstable region  -> use base
```

## Effect Of Changing Uncertainty Scale

For the same uncertainty:

```text
uncertainty = 0.0025
```

| `uncertainty_scale` | Agreement | Meaning |
|---:|---:|---|
| `0.001` | `8.2%` | Very conservative |
| `0.0025` | `36.8%` | Default |
| `0.005` | `60.7%` | Moderate |
| `0.010` | `77.9%` | Permissive |

So:

```text
Smaller uncertainty scale
-> faster return to base
-> potentially higher PSNR
-> less generated detail

Larger uncertainty scale
-> retain more generated output
-> potentially more detail
-> greater hallucination/error risk
```

The complete inference order is:

```text
Four diffusion samples
        |
        v
Mean generated image
        |
        v
Variance/uncertainty map
        |
        v
Spatial uncertainty abstention
        |
        v
Global residual scaling
        |
        v
Final SR output
```

`uncertainty_scale` decides **where** to trust diffusion.

`residual_scale` decides **how strongly overall** to use the remaining diffusion correction.
