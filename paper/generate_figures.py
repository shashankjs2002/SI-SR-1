from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT = Path(__file__).resolve().parent / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "navy": "#143A66",
    "blue": "#2B6CB0",
    "cyan": "#2C7A7B",
    "purple": "#6B46C1",
    "orange": "#DD6B20",
    "red": "#C53030",
    "green": "#2F855A",
    "pink": "#B83280",
    "gray": "#4A5568",
    "light": "#F7FAFC",
}


def setup(title: str):
    figure, axis = plt.subplots(figsize=(16, 6.5), dpi=150)
    axis.set_xlim(0, 16)
    axis.set_ylim(0, 6.5)
    axis.axis("off")
    figure.patch.set_facecolor("white")
    axis.text(
        8,
        6.12,
        title,
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color=COLORS["navy"],
    )
    return figure, axis


def box(axis, x, y, width, height, title, subtitle="", color="blue", fontsize=12):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor=COLORS["light"],
        edgecolor=COLORS[color],
        linewidth=2,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height * 0.62,
        title,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=COLORS[color],
    )
    if subtitle:
        axis.text(
            x + width / 2,
            y + height * 0.28,
            subtitle,
            ha="center",
            va="center",
            fontsize=max(fontsize - 2, 8),
            color=COLORS["gray"],
        )
    return patch


def arrow(axis, start, end, color="gray", dashed=False, label=None, bend=0.0):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2,
        linestyle="--" if dashed else "-",
        color=COLORS[color],
        connectionstyle=f"arc3,rad={bend}",
    )
    axis.add_patch(patch)
    if label:
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.18)
        axis.text(
            *midpoint,
            label,
            ha="center",
            fontsize=9,
            color=COLORS[color],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
        )


def save(figure, name: str):
    figure.savefig(OUTPUT / name, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def degradation_figure():
    figure, axis = setup("Physics-Informed Synthetic 40 m Observation Model")
    items = [
        (0.3, "Native HR", "B4/B3/B2\n3 x 512 x 512", "navy"),
        (2.8, "MTF / PSF Blur", "Gaussian sigma\nconditioned by theta", "blue"),
        (5.3, "Area Downsample", "scale = 4\n3 x 128 x 128", "cyan"),
        (7.8, "Poisson Noise", "signal-dependent\nphoton model", "purple"),
        (10.3, "Gaussian Noise", "read-noise\nperturbation", "orange"),
        (12.8, "Quantization", "mild calibrated\nlevels", "red"),
    ]
    for x, title, subtitle, color in items:
        box(axis, x, 3.15, 2.05, 1.45, title, subtitle, color)
    for left, right in zip(items[:-1], items[1:]):
        arrow(axis, (left[0] + 2.05, 3.88), (right[0], 3.88))
    box(
        axis,
        5.0,
        0.75,
        6.0,
        1.3,
        "Recorded degradation vector theta",
        "[blur, Gaussian noise, Poisson strength, quantization]",
        "green",
        fontsize=13,
    )
    for x, *_ in items[1:]:
        arrow(axis, (x + 1.0, 3.15), (8.0, 2.05), "green", dashed=True)
    axis.text(
        8,
        0.28,
        "Training samples use randomized theta; evaluation uses deterministic seeds and clean LR for projection.",
        ha="center",
        fontsize=11,
        color=COLORS["gray"],
    )
    save(figure, "degradation_operator.png")


def geomapper_figure():
    figure, axis = setup("GeoMapper: Translating Diffusion Latents into Decoder Controls")
    box(axis, 0.4, 3.6, 2.2, 1.25, "Denoised latent", "4 x 64 x 64", "purple")
    box(axis, 0.4, 1.55, 2.2, 1.25, "LR feature f64", "128 x 64 x 64", "blue")
    box(axis, 3.3, 2.55, 2.4, 1.5, "Concatenate", "132 x 64 x 64", "gray")
    box(axis, 6.3, 2.55, 2.4, 1.5, "4 residual blocks", "128 x 64 x 64", "orange")
    box(axis, 6.3, 0.55, 2.4, 1.15, "Pooled text", "768-D prompt context", "purple")
    outputs = [
        (10.0, 4.35, "Spatial content", "128 x 64 x 64", "orange"),
        (10.0, 2.55, "Evidence gate", "1 x 64 x 64", "green"),
        (10.0, 0.75, "Pooled content", "global descriptor", "cyan"),
        (13.2, 0.75, "4 FiLM styles", "256-D per stage", "pink"),
    ]
    for x, y, title, subtitle, color in outputs:
        box(axis, x, y, 2.35, 1.15, title, subtitle, color)
    arrow(axis, (2.6, 4.2), (3.3, 3.45))
    arrow(axis, (2.6, 2.15), (3.3, 3.05))
    arrow(axis, (5.7, 3.3), (6.3, 3.3))
    arrow(axis, (8.7, 3.35), (10.0, 4.9), "orange")
    arrow(axis, (8.7, 3.1), (10.0, 3.1), "green")
    arrow(axis, (8.7, 2.8), (10.0, 1.3), "cyan")
    arrow(axis, (8.7, 1.1), (10.0, 3.0), "purple", dashed=True, label="prompt map")
    arrow(axis, (8.7, 1.0), (10.0, 1.15), "purple", dashed=True)
    arrow(axis, (12.35, 1.3), (13.2, 1.3), "pink")
    axis.text(
        11.2,
        5.8,
        "Local control",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color=COLORS["orange"],
    )
    axis.text(
        14.35,
        2.3,
        "Layer-wise channel control",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color=COLORS["pink"],
    )
    save(figure, "geomapper_detail.png")


def discriminator_figure():
    figure, axis = setup("Dual Conditional Adversarial Supervision (Training Only)")
    box(axis, 0.35, 3.7, 2.1, 1.2, "Candidate HR", "real or generated", "navy")
    box(axis, 0.35, 1.65, 2.1, 1.2, "Observed LR", "bilinear/area condition", "blue")
    box(axis, 3.25, 3.7, 2.4, 1.2, "RGB + LR concat", "6 channels", "gray")
    box(axis, 3.25, 1.65, 2.4, 1.2, "Haar bands + LR", "LH, HL, HH + condition", "gray")
    box(axis, 6.45, 3.7, 2.5, 1.2, "3-scale PatchGAN", "spectral-normalized convs", "red")
    box(axis, 6.45, 1.65, 2.5, 1.2, "Wavelet PatchGAN", "high-frequency realism", "orange")
    box(axis, 10.0, 3.7, 2.4, 1.2, "Spatial logits", "local realism at 3 scales", "red")
    box(axis, 10.0, 1.65, 2.4, 1.2, "Frequency logits", "edge and texture realism", "orange")
    box(axis, 13.25, 2.6, 2.35, 1.35, "Hinge objectives", "D update + low-weight G loss", "green")
    arrow(axis, (2.45, 4.3), (3.25, 4.3))
    arrow(axis, (2.45, 2.25), (3.25, 2.25))
    arrow(axis, (2.45, 2.25), (3.25, 3.95), "blue", dashed=True)
    arrow(axis, (2.45, 4.0), (3.25, 2.55), "navy", dashed=True)
    arrow(axis, (5.65, 4.3), (6.45, 4.3))
    arrow(axis, (5.65, 2.25), (6.45, 2.25))
    arrow(axis, (8.95, 4.3), (10.0, 4.3))
    arrow(axis, (8.95, 2.25), (10.0, 2.25))
    arrow(axis, (12.4, 4.3), (13.25, 3.55), "green")
    arrow(axis, (12.4, 2.25), (13.25, 2.95), "green")
    axis.text(
        8,
        0.55,
        "Discriminators are absent during inference; they shape the residual decoder only through training gradients.",
        ha="center",
        fontsize=11,
        color=COLORS["gray"],
    )
    save(figure, "dual_discriminator.png")


def conditioning_figure():
    figure, axis = setup("Prompt and Mode Conditioning Policy")
    box(axis, 0.3, 3.8, 2.2, 1.2, "Caption text", "land cover, objects,\nterrain, texture", "purple")
    box(axis, 0.3, 1.65, 2.2, 1.2, "Training policy", "40% null, 20% paraphrase,\n10% mismatch", "gray")
    box(axis, 3.35, 3.8, 2.35, 1.2, "Frozen SigLIP", "token sequence", "blue")
    box(axis, 3.35, 1.65, 2.35, 1.2, "Mode token", "SR = 0, edit = 1", "cyan")
    box(axis, 6.6, 2.75, 2.6, 1.45, "Conditional diffusion", "cross-attention + timestep\n+ degradation + mode", "purple")
    box(axis, 10.1, 3.8, 2.35, 1.2, "SR policy", "weak/null prompt\nhigh-pass residual", "green")
    box(axis, 10.1, 1.65, 2.35, 1.2, "Edit policy", "strong prompt\nfull-band residual", "pink")
    box(axis, 13.3, 3.8, 2.35, 1.2, "Reconstruction", "3x back-projection\nsynthetic_edit=false", "green")
    box(axis, 13.3, 1.65, 2.35, 1.2, "Synthesis", "soft consistency\nsynthetic_edit=true", "pink")
    arrow(axis, (2.5, 4.4), (3.35, 4.4))
    arrow(axis, (2.5, 2.25), (3.35, 2.25))
    arrow(axis, (5.7, 4.4), (6.6, 3.75), "purple")
    arrow(axis, (5.7, 2.25), (6.6, 3.15), "cyan")
    arrow(axis, (9.2, 3.65), (10.1, 4.4), "green")
    arrow(axis, (9.2, 3.25), (10.1, 2.25), "pink")
    arrow(axis, (12.45, 4.4), (13.3, 4.4), "green")
    arrow(axis, (12.45, 2.25), (13.3, 2.25), "pink")
    axis.text(
        8,
        0.55,
        "The prompt is optional in SR mode. Edit-mode outputs are explicitly treated as generated visualizations.",
        ha="center",
        fontsize=11,
        color=COLORS["gray"],
    )
    save(figure, "conditioning_policy.png")


def evaluation_figure():
    figure, axis = setup("Pre-Registered Evaluation and Ablation Protocol")
    box(axis, 0.25, 3.7, 2.25, 1.3, "Held-out geography", "complete MGRS tiles\nand unseen cities", "navy")
    box(axis, 0.25, 1.55, 2.25, 1.3, "8 stochastic samples", "mean prediction +\npixelwise variance", "purple")
    metrics = [
        (3.3, 4.3, "Pixel fidelity", "PSNR, SSIM", "blue"),
        (3.3, 2.65, "Perception", "LPIPS, DISTS", "orange"),
        (3.3, 1.0, "Evidence", "Edge F1, LR error", "green"),
    ]
    for x, y, title, subtitle, color in metrics:
        box(axis, x, y, 2.25, 1.0, title, subtitle, color, fontsize=11)
    box(axis, 6.45, 3.7, 2.45, 1.3, "Fair baselines", "bicubic, SwinIR,\nESRGAN, diffusion/GAN only", "cyan")
    box(axis, 6.45, 1.55, 2.45, 1.3, "One-factor ablations", "mapper, gate, wavelet D,\ndegradation, projection", "red")
    box(axis, 9.85, 3.7, 2.45, 1.3, "Tile-level statistics", "paired differences,\nCI across tiles", "navy")
    box(axis, 9.85, 1.55, 2.45, 1.3, "Prompt evaluation", "alignment, realism,\nLR consistency", "pink")
    box(axis, 13.15, 2.6, 2.55, 1.45, "Scientific claims", "report trade-offs,\nuncertainty and failures", "green")
    arrow(axis, (2.5, 4.35), (3.3, 4.8))
    arrow(axis, (2.5, 4.1), (3.3, 3.15))
    arrow(axis, (2.5, 3.9), (3.3, 1.5))
    arrow(axis, (2.5, 2.2), (3.3, 2.95), "purple", dashed=True)
    arrow(axis, (5.55, 4.8), (6.45, 4.35))
    arrow(axis, (5.55, 3.15), (6.45, 4.05))
    arrow(axis, (5.55, 1.5), (6.45, 2.2))
    arrow(axis, (8.9, 4.35), (9.85, 4.35))
    arrow(axis, (8.9, 2.2), (9.85, 2.2))
    arrow(axis, (12.3, 4.35), (13.15, 3.65), "green")
    arrow(axis, (12.3, 2.2), (13.15, 3.0), "green")
    axis.text(
        8,
        0.3,
        "No component is called beneficial until the corresponding controlled experiment is completed.",
        ha="center",
        fontsize=11,
        color=COLORS["gray"],
    )
    save(figure, "evaluation_protocol.png")


if __name__ == "__main__":
    degradation_figure()
    geomapper_figure()
    discriminator_figure()
    conditioning_figure()
    evaluation_figure()
    print(f"Wrote paper figures to {OUTPUT}")
