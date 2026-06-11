from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

NAVY = "#173B67"
BLUE = "#2B6CB0"
TEAL = "#287D7D"
PURPLE = "#6746C3"
ORANGE = "#D96518"
GREEN = "#2E855B"
RED = "#C53030"
PINK = "#B83280"
GRAY = "#4A5568"
PALE_BLUE = "#EBF4FF"
PALE_GREEN = "#F0FFF4"
PALE_PURPLE = "#FAF5FF"
PALE_ORANGE = "#FFFAF0"
PALE_RED = "#FFF5F5"
WHITE = "#FFFFFF"


def panel(ax, x, y, width, height, title, color, fill, subtitle=None):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.10,rounding_size=1.0",
            facecolor=fill,
            edgecolor=color,
            linewidth=2.6,
            zorder=1,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.8, y + height - 5.6),
            width - 1.6,
            4.7,
            boxstyle="round,pad=0.05,rounding_size=0.55",
            facecolor=color,
            edgecolor=color,
            linewidth=0,
            zorder=2,
        )
    )
    ax.text(
        x + width / 2,
        y + height - 3.25,
        title,
        ha="center",
        va="center",
        color=WHITE,
        fontsize=10.5,
        fontweight="bold",
        zorder=3,
    )
    if subtitle:
        ax.text(
            x + width / 2,
            y + height - 7.0,
            subtitle,
            ha="center",
            va="top",
            color=GRAY,
            fontsize=6.7,
            zorder=3,
        )


def chip(ax, x, y, width, height, title, shape, color, *, fill=WHITE, size=6.7):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.35",
            facecolor=fill,
            edgecolor=color,
            linewidth=1.35,
            zorder=4,
        )
    )
    ax.text(
        x + width * 0.47,
        y + height * 0.67,
        title,
        ha="center",
        va="center",
        fontsize=size,
        fontweight="bold",
        color=color,
        zorder=5,
    )
    ax.text(
        x + width * 0.47,
        y + height * 0.27,
        shape,
        ha="center",
        va="center",
        fontsize=size - 0.5,
        color=GRAY,
        zorder=5,
    )


def tensor(ax, x, y, width, height, depth, color, label, shape, *, alpha=0.80):
    front = Rectangle(
        (x, y),
        width,
        height,
        facecolor=color,
        edgecolor=NAVY,
        linewidth=1.0,
        alpha=alpha,
        zorder=5,
    )
    ax.add_patch(front)
    ax.add_patch(
        Polygon(
            [(x, y + height), (x + depth, y + height + depth), (x + width + depth, y + height + depth), (x + width, y + height)],
            closed=True,
            facecolor=color,
            edgecolor=NAVY,
            linewidth=0.9,
            alpha=alpha * 0.80,
            zorder=4,
        )
    )
    ax.add_patch(
        Polygon(
            [(x + width, y), (x + width + depth, y + depth), (x + width + depth, y + height + depth), (x + width, y + height)],
            closed=True,
            facecolor=color,
            edgecolor=NAVY,
            linewidth=0.9,
            alpha=alpha * 0.65,
            zorder=4,
        )
    )
    ax.text(
        x + width / 2,
        y + height / 2 + 0.8,
        label,
        ha="center",
        va="center",
        fontsize=6.8,
        color=WHITE,
        fontweight="bold",
        zorder=6,
    )
    ax.text(
        x + width / 2,
        y + height / 2 - 1.5,
        shape,
        ha="center",
        va="center",
        fontsize=5.8,
        color=WHITE,
        zorder=6,
    )


def arrow(ax, start, end, color=GRAY, *, dashed=False, width=1.6, bend=0.0, z=8):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=width,
            linestyle="--" if dashed else "-",
            color=color,
            connectionstyle=f"arc3,rad={bend}",
            shrinkA=0,
            shrinkB=0,
            zorder=z,
        )
    )


def routed_arrow(ax, points, color=GRAY, *, dashed=False, width=1.6, z=8):
    """Draw an orthogonal connector with an arrowhead only on its final segment."""
    linestyle = "--" if dashed else "-"
    xs = [point[0] for point in points[:-1]]
    ys = [point[1] for point in points[:-1]]
    ax.plot(
        xs,
        ys,
        color=color,
        linewidth=width,
        linestyle=linestyle,
        solid_capstyle="round",
        zorder=z,
    )
    arrow(
        ax,
        points[-2],
        points[-1],
        color,
        dashed=dashed,
        width=width,
        z=z,
    )


def small_label(ax, x, y, text, color=GRAY, size=5.8, weight="normal", ha="center"):
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va="center",
        fontsize=size,
        color=color,
        fontweight=weight,
        zorder=12,
    )


def build():
    fig, ax = plt.subplots(figsize=(28, 15.5), dpi=220)
    ax.set_xlim(0, 280)
    ax.set_ylim(0, 145)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        140,
        141.2,
        "GeoDiff-GAN Detailed Architecture and Tensor Dimensions",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color=NAVY,
    )
    ax.text(
        140,
        137.2,
        "Full model configuration | Tensor notation: B x C x H x W | 4x Sentinel-2 RGB super-resolution",
        ha="center",
        va="center",
        fontsize=10.5,
        color=GRAY,
    )

    # Legend
    arrow(ax, (10, 132.5), (20, 132.5), NAVY, width=2)
    small_label(ax, 22, 132.5, "inference tensor", NAVY, 6.6, ha="left")
    arrow(ax, (55, 132.5), (65, 132.5), PURPLE, dashed=True, width=2)
    small_label(ax, 67, 132.5, "conditioning / training target", PURPLE, 6.6, ha="left")
    arrow(ax, (121, 132.5), (131, 132.5), RED, dashed=True, width=2)
    small_label(ax, 133, 132.5, "training-only adversarial path", RED, 6.6, ha="left")
    small_label(
        ax,
        272,
        132.5,
        "Solid module borders = active inference modules",
        GREEN,
        6.6,
        "bold",
        "right",
    )

    # Input and conditioning strip.
    panel(ax, 2, 78, 20, 43, "INPUT", BLUE, PALE_BLUE)
    tensor(ax, 6, 94, 11, 14, 2.5, BLUE, "LR RGB", "B x 3 x 128 x 128")
    small_label(ax, 12, 88, "Synthetic 40 m observation", BLUE, 6.5, "bold")
    small_label(ax, 12, 84.5, "range [0,1]", GRAY, 6.0)

    ax.add_patch(
        FancyBboxPatch(
            (78, 123),
            119,
            9.0,
            boxstyle="round,pad=0.10,rounding_size=1.0",
            facecolor=PALE_PURPLE,
            edgecolor=PURPLE,
            linewidth=2.6,
            zorder=1,
        )
    )
    ax.text(
        79.5,
        130.8,
        "CONDITIONING INPUTS",
        ha="left",
        va="center",
        color=PURPLE,
        fontsize=8.0,
        fontweight="bold",
        zorder=6,
    )
    chip(ax, 80, 124, 25, 5.7, "Text tokens (frozen SigLIP)", "B x 64 x 768", PURPLE, size=6.0)
    chip(ax, 108, 124, 23, 5.7, "Degradation theta", "B x 4", ORANGE, size=6.0)
    chip(ax, 134, 124, 21, 5.7, "Timestep t", "B", PURPLE, size=6.0)
    chip(ax, 158, 124, 18, 5.7, "Mode", "B | SR=0, edit=1", TEAL, size=5.8)
    chip(ax, 179, 124, 16, 5.7, "Null text", "B x 64 x 768", GRAY, size=5.8)

    # Base branch.
    panel(ax, 27, 91, 48, 31, "A. DETERMINISTIC SWINIR BASE", NAVY, PALE_BLUE)
    base_y = 96
    base_steps = [
        ("Conv 3x3", "3 -> 60\n60 x 128 x 128"),
        ("6x Swin blocks", "Win=8 | heads=6\nMLP 60->120->60"),
        ("Body Conv + skip", "60 -> 60\n60 x 128 x 128"),
        ("Conv + PS x2", "60 -> 240 -> 60\n60 x 256 x 256"),
        ("Conv + PS x2", "60 -> 240 -> 60\n60 x 512 x 512"),
        ("RGB + bicubic", "60 -> 3 + LR up\n3 x 512 x 512"),
    ]
    x_positions = [29, 36.6, 45.3, 53.7, 61.6, 68.5]
    widths = [6.3, 7.5, 7.2, 6.7, 6.7, 5.2]
    for i, ((title, shape), x, width) in enumerate(zip(base_steps, x_positions, widths)):
        chip(ax, x, base_y, width, 16, title, shape, NAVY, size=5.2)
        if i < len(base_steps) - 1:
            arrow(ax, (x + width, base_y + 8), (x_positions[i + 1], base_y + 8), NAVY, width=1.1)
    small_label(ax, 50.5, 114.7, "Alternating shifted-window attention; spatial size stays 128 until PixelShuffle", NAVY, 5.9, "bold")

    # LR encoder.
    panel(ax, 27, 51, 48, 35, "B. LR EVIDENCE ENCODER", TEAL, PALE_GREEN)
    lr_steps = [
        ("Stem", "Conv 3->64 + ResBlock\nf128: B x 64 x 128 x 128"),
        ("Down 1", "Conv4 s2 64->128 + Res\nf64: B x 128 x 64 x 64"),
        ("Down 2", "Conv4 s2 128->256 + Res\nf32: B x 256 x 32 x 32"),
        ("Down 3", "Conv4 s2 256->256 + Res\nf16: B x 256 x 16 x 16"),
    ]
    for i, (title, shape) in enumerate(lr_steps):
        y = 74.2 - i * 6.2
        chip(ax, 30, y, 42, 5.4, title, shape, TEAL, size=5.6)
        if i < 3:
            arrow(ax, (51, y), (51, y - 0.8), TEAL, width=1.0)
    small_label(ax, 51, 53.2, "f64 and f128 feed the generator; f32/f16 are currently diagnostic/ablation features", RED, 5.4, "bold")

    # Diffusion UNet.
    panel(ax, 80, 48, 76, 74, "C. CONDITIONAL LATENT DIFFUSION U-NET", PURPLE, PALE_PURPLE)
    chip(ax, 83, 109.5, 21, 5.5, "Noisy latent z_t", "B x 4 x 64 x 64", PURPLE, size=5.9)
    chip(ax, 107, 109.5, 25, 5.5, "Concat with f64", "B x (4+128) x 64 x 64", TEAL, size=5.9)
    chip(ax, 135, 109.5, 17, 5.5, "Input Conv", "132 -> 128", PURPLE, size=5.9)
    arrow(ax, (104, 112.25), (107, 112.25), PURPLE, width=1.2)
    arrow(ax, (132, 112.25), (135, 112.25), PURPLE, width=1.2)

    # U-shape.
    down = [
        (84, 90, 13, 9, "Down 0", "2x CRes\n128 x 64 x 64", False),
        (100, 80, 13, 9, "Down 1", "2x CRes + XAttn\n256 x 32 x 32", True),
        (116, 70, 13, 9, "Down 2", "2x CRes + XAttn\n384 x 16 x 16", True),
        (132, 60, 13, 9, "Down 3", "2x CRes\n512 x 8 x 8", False),
    ]
    for i, (x, y, w, h, title, shape, _) in enumerate(down):
        chip(ax, x, y, w, h, title, shape, PURPLE, size=5.5)
        if i < len(down) - 1:
            arrow(ax, (x + w, y + h / 2), (down[i + 1][0], down[i + 1][1] + h / 2), PURPLE, width=1.2)
            small_label(ax, (x + w + down[i + 1][0]) / 2, y + h / 2 - 2.2, "Conv4 s2", GRAY, 4.6)

    chip(ax, 132, 49, 13, 9, "Middle", "2x CRes + XAttn\n512 x 8 x 8", PURPLE, fill="#EFE5FF", size=5.5)
    arrow(ax, (138.5, 60), (138.5, 58), PURPLE, width=1.2)

    up = [
        (116, 60, 13, 9, "Up 2", "PS2 + skip + XAttn\n384 x 16 x 16"),
        (100, 70, 13, 9, "Up 1", "PS2 + skip + XAttn\n256 x 32 x 32"),
        (84, 80, 13, 9, "Up 0", "PS2 + skip\n128 x 64 x 64"),
    ]
    arrow(ax, (132, 53.5), (129, 64.5), PURPLE, width=1.2, bend=-0.1)
    for i, (x, y, w, h, title, shape) in enumerate(up):
        chip(ax, x, y, w, h, title, shape, ORANGE, fill=PALE_ORANGE, size=5.35)
        if i < len(up) - 1:
            arrow(ax, (x, y + h / 2), (up[i + 1][0] + w, up[i + 1][1] + h / 2), ORANGE, width=1.2)

    # Skip connections.
    for d, u in zip(down[:3], reversed(up)):
        arrow(
            ax,
            (d[0] + d[2] / 2, d[1] + d[3]),
            (u[0] + u[2] / 2, u[1] + u[3]),
            BLUE,
            dashed=True,
            width=1.0,
            bend=-0.20,
            z=6,
        )
    chip(ax, 101, 91, 17, 5.7, "Output head", "GN + SiLU + Conv\nv: B x 4 x 64 x 64", PURPLE, size=5.2)
    arrow(ax, (97, 84.5), (101, 93.8), ORANGE, width=1.2, bend=-0.12)
    routed_arrow(ax, [(143.5, 109.5), (143.5, 100.0), (90.5, 100.0), (90.5, 99.0)], PURPLE, width=1.2)
    small_label(ax, 119, 49.8, "DDIM sampling: 1000-step training schedule, typically 20 inference steps", PURPLE, 5.6, "bold")

    # Condition block details inside diffusion.
    chip(ax, 105, 101.5, 46, 6.0, "Global condition sum", "time: 128->512 | theta: 4->512 | mode embedding: 512", PURPLE, fill=WHITE, size=5.2)
    arrow(ax, (143, 124), (143, 107.5), PURPLE, dashed=True, width=1.0)
    small_label(ax, 149, 84.0, "Text XAttn at 32, 16, middle, 16, 32", PURPLE, 4.9, "bold", "right")
    arrow(ax, (90, 124), (108, 89), PURPLE, dashed=True, width=1.0, bend=0.15)

    # GeoMapper.
    panel(ax, 160, 63, 35, 59, "D. GEOMAPPER", ORANGE, PALE_ORANGE)
    geomap_rows = [
        ("Input concat", "z:4 + f64:128\nB x 132 x 64 x 64"),
        ("Conv + 4 ResBlocks", "132 -> 128\nB x 128 x 64 x 64"),
        ("Text context map", "mean tokens 768 -> 128\nexpand to 64 x 64"),
        ("Evidence gate", "concat 256 -> 128 -> 1\nSigmoid | B x 1 x 64 x 64"),
        ("Mode-scaled fusion", "context scale: SR 0.25 | edit 1.0\ncontent: B x 128 x 64 x 64"),
        ("4 style heads", "pool 128 + text 768 = 896\n896 -> 256 -> 256 each"),
    ]
    for i, (title, shape) in enumerate(geomap_rows):
        y = 108 - i * 7.2
        chip(ax, 163, y, 29, 6.1, title, shape, ORANGE if i < 3 else (GREEN if i < 5 else PINK), size=5.25)
        if i < len(geomap_rows) - 1:
            arrow(ax, (177.5, y), (177.5, y - 1.0), ORANGE, width=0.9)
    small_label(ax, 177.5, 66.0, "Outputs: content 128x64x64 | gate 1x64x64 | four Bx256 styles", ORANGE, 5.4, "bold")

    # Decoder.
    panel(ax, 200, 52, 43, 70, "E. RESIDUAL GAN DECODER", ORANGE, PALE_ORANGE)
    decoder_rows = [
        ("Input + Stage 0", "Conv 128->128 + f64 128->128\nFiLM style 0 | B x 128 x 64 x 64"),
        ("Stage 1", "Conv 128->384 + PixelShuffle2\n+ f128 64->96 | FiLM 1\nB x 96 x 128 x 128"),
        ("Stage 2", "Conv 96->256 + PixelShuffle2\n+ upsampled f128 64->64 | FiLM 2\nB x 64 x 256 x 256"),
        ("Stage 3", "Conv 64->192 + PixelShuffle2\n+ upsampled f128 64->48 | FiLM 3\nB x 48 x 512 x 512"),
        ("Residual RGB head", "Conv 48->48 | LReLU | Conv 48->3 | tanh\nraw residual: B x 3 x 512 x 512"),
    ]
    for i, (title, shape) in enumerate(decoder_rows):
        y = 107.2 - i * 11.8
        chip(ax, 203, y, 37, 10.4, title, shape, ORANGE, size=5.3)
        if i < len(decoder_rows) - 1:
            arrow(ax, (221.5, y), (221.5, y - 1.4), ORANGE, width=1.0)
    small_label(ax, 221.5, 55.0, "Each FiLM block: LayerNorm -> style gamma/beta -> Conv -> LayerNorm -> style -> Conv + residual", PINK, 5.1, "bold")

    # Output controller.
    panel(ax, 248, 52, 30, 70, "F. OUTPUT CONTROLLER", GREEN, PALE_GREEN)
    chip(ax, 251, 108, 24, 8.5, "Base input", "B x 3 x 512 x 512", NAVY, size=5.8)
    chip(ax, 251, 96, 24, 8.5, "Raw residual", "B x 3 x 512 x 512", ORANGE, size=5.8)
    chip(ax, 251, 79, 24, 13, "SR mode", "5x5 high-pass residual\nadd base + clip\n3x back-project | eta=0.5", GREEN, size=5.7)
    chip(ax, 251, 62, 24, 13, "Edit mode", "full-band residual\nadd base + clip\n1x back-project | eta=0.15", PINK, fill=PALE_PURPLE, size=5.7)
    chip(ax, 251, 54, 24, 5.5, "HR output", "B x 3 x 512 x 512", GREEN, size=5.5)
    arrow(ax, (263, 96), (263, 92), GREEN, width=1.0)
    arrow(ax, (257, 79), (257, 75), GREEN, width=1.0)
    arrow(ax, (269, 62), (269, 59.5), PINK, width=1.0)
    small_label(ax, 263, 52.5, "metadata: edit => synthetic_edit=true", RED, 5.1, "bold")

    # Training VAE.
    panel(ax, 2, 3, 153, 42, "G. RESIDUAL VAE (TRAINING TARGET AND STAGE-2 RECONSTRUCTION)", TEAL, PALE_GREEN)
    chip(ax, 5, 31, 21, 8, "Target residual", "HR - frozen base\nB x 3 x 512 x 512", TEAL, size=5.7)
    vae_encoder = [
        ("Stem", "3->64\n64 x 512"),
        ("Down", "64->128\n128 x 256"),
        ("Down", "128->256\n256 x 128"),
        ("Down", "256->256\n256 x 64"),
        ("Moments", "256->8\nmean/logvar 4+4"),
        ("z0", "B x 4 x 64 x 64"),
    ]
    enc_x = [30, 47, 64, 81, 98, 117]
    enc_w = [14, 14, 14, 14, 16, 12]
    for i, ((title, shape), x, w) in enumerate(zip(vae_encoder, enc_x, enc_w)):
        chip(ax, x, 31, w, 8, title, shape, TEAL, size=5.2)
        if i < len(vae_encoder) - 1:
            arrow(ax, (x + w, 35), (enc_x[i + 1], 35), TEAL, width=1.0)
    small_label(ax, 79, 41, "Encoder", TEAL, 6.0, "bold")

    vae_decoder = [
        ("From z", "4->256\n256 x 64"),
        ("PS2", "256 x 128"),
        ("PS2", "128 x 256"),
        ("PS2", "64 x 512"),
        ("RGB", "3 x 512"),
    ]
    dec_x = [117, 99, 82, 65, 48]
    dec_w = [12, 14, 14, 14, 14]
    for i, ((title, shape), x, w) in enumerate(zip(vae_decoder, dec_x, dec_w)):
        chip(ax, x, 17, w, 8, title, shape, BLUE, fill=PALE_BLUE, size=5.2)
        if i < len(vae_decoder) - 1:
            arrow(ax, (x, 21), (dec_x[i + 1] + dec_w[i + 1], 21), BLUE, width=1.0)
    small_label(ax, 86, 27, "VAE decoder reconstructs the RGB residual during Stage 2; final inference uses GeoMapper + GAN decoder", BLUE, 5.4, "bold")
    arrow(ax, (123, 31), (118, 25), BLUE, width=1.0)
    arrow(ax, (123, 39), (106, 48), PURPLE, dashed=True, width=1.4, bend=0.15)
    small_label(ax, 117, 44, "z0 trains diffusion", PURPLE, 5.2, "bold")

    # Discriminators.
    panel(ax, 160, 3, 118, 42, "H. TRAINING-ONLY CONDITIONAL DISCRIMINATORS", RED, PALE_RED)
    chip(ax, 163, 31, 26, 8, "Real or generated HR", "B x 3 x 512 x 512", NAVY, size=5.7)
    chip(ax, 163, 18, 26, 8, "Observed LR condition", "B x 3 x 128 x 128", BLUE, size=5.7)
    chip(ax, 194, 29, 38, 10, "3-scale PatchGAN", "concat HR + resized LR = 6 channels\nspectral Conv4: 64 -> 128 -> 256 -> 256 -> 1\nlogit grids: 62x62 | 30x30 | 14x14", RED, size=5.2)
    chip(ax, 194, 15, 38, 10, "Haar wavelet discriminator", "HR -> LH/HL/HH = 9 x 256 x 256\nLR repeated = 9 x 256 x 256 | concat=18\nPatchGAN logit grid: B x 1 x 30 x 30", ORANGE, fill=PALE_ORANGE, size=5.2)
    chip(ax, 239, 22, 36, 12, "Hinge objectives", "D: max(0,1-real)+max(0,1+fake)\nG: -mean(fake)\nadversarial generator weight = 0.01", GREEN, fill=PALE_GREEN, size=5.4)
    arrow(ax, (189, 35), (194, 34), RED, width=1.1)
    arrow(ax, (189, 22), (194, 20), ORANGE, width=1.1)
    arrow(ax, (232, 34), (239, 29), GREEN, width=1.1)
    arrow(ax, (232, 20), (239, 26), GREEN, width=1.1)
    arrow(ax, (176, 26), (202, 29), BLUE, dashed=True, width=1.0)
    arrow(ax, (176, 26), (202, 25), BLUE, dashed=True, width=1.0)
    small_label(ax, 219, 7, "Removed entirely at inference; only their gradients shape the decoder during joint/edit training", RED, 5.6, "bold")

    # Main graph connections. Long paths use dedicated buses outside module panels.
    arrow(ax, (22, 105), (27, 105), NAVY, width=2.2)
    arrow(ax, (22, 96), (27, 72), TEAL, width=2.2)
    routed_arrow(ax, [(75, 68), (77.5, 68), (77.5, 117.0), (119.5, 117.0), (119.5, 115.0)], TEAL, width=1.6)
    small_label(ax, 79.0, 95, "f64 to diffusion", TEAL, 5.4, "bold", "left")
    routed_arrow(ax, [(75, 67), (77.0, 67), (77.0, 46.7), (158.0, 46.7), (158.0, 111), (160, 111)], TEAL, width=1.5)
    small_label(ax, 113, 46.7, "f64 spatial evidence bus", TEAL, 5.2, "bold")
    routed_arrow(ax, [(75, 78), (76.0, 78), (76.0, 47.2), (198.0, 47.2), (198.0, 103), (200, 103)], BLUE, width=1.4)
    small_label(ax, 171, 47.2, "f128 multi-resolution skip bus", BLUE, 5.2, "bold")
    routed_arrow(ax, [(118, 93.8), (121, 93.8), (121, 99.2), (157.5, 99.2), (157.5, 112), (160, 112)], PURPLE, width=2.0)
    small_label(ax, 132, 99.2, "denoised latent B x 4 x 64 x 64", PURPLE, 5.4, "bold")
    arrow(ax, (195, 91), (200, 105), ORANGE, width=2.0)
    small_label(ax, 197, 99, "content + styles + gate", ORANGE, 5.2, "bold")
    arrow(ax, (243, 96), (248, 96), ORANGE, width=2.0)
    small_label(ax, 245.5, 98.2, "raw residual", ORANGE, 5.0, "bold")
    routed_arrow(ax, [(75, 104), (78.5, 104), (78.5, 45.8), (246.0, 45.8), (246.0, 112), (248, 112)], NAVY, width=1.8)
    small_label(ax, 215, 45.8, "radiometrically conservative base HR", NAVY, 5.4, "bold")

    # Conditioning connections use vertical ports and two external buses.
    routed_arrow(ax, [(92, 124), (92, 121.8), (81.5, 121.8), (81.5, 95)], PURPLE, dashed=True, width=1.0)
    small_label(ax, 82.5, 120.2, "text cross-attention", PURPLE, 4.8, "bold", "left")
    arrow(ax, (120, 124), (120, 107.5), ORANGE, dashed=True, width=1.0)
    arrow(ax, (144, 124), (143, 107.5), PURPLE, dashed=True, width=1.0)
    arrow(ax, (166, 124), (151, 107.5), TEAL, dashed=True, width=1.0)
    routed_arrow(ax, [(92, 124), (92, 122.4), (157.0, 122.4), (157.0, 104.8), (163, 104.8)], PURPLE, dashed=True, width=1.0)
    small_label(ax, 149, 122.4, "pooled text", PURPLE, 4.8, "bold")
    routed_arrow(ax, [(166, 124), (193.5, 124), (193.5, 88.2), (192, 88.2)], TEAL, dashed=True, width=1.0)
    small_label(ax, 194.5, 116, "mode gate", TEAL, 4.8, "bold", "left")
    routed_arrow(ax, [(120, 124), (120, 122.5), (245.0, 122.5), (245.0, 92), (251, 92)], ORANGE, dashed=True, width=1.0)
    small_label(ax, 218, 122.5, "degradation operator parameters", ORANGE, 4.8, "bold")
    routed_arrow(ax, [(187, 124), (187, 121.7), (154.5, 121.7), (154.5, 116)], GRAY, dashed=True, width=0.9)
    small_label(ax, 181, 120.4, "null text for CFG", GRAY, 4.7, "bold")

    # Training-only connections to discriminators.
    arrow(ax, (240, 58), (217, 45), RED, dashed=True, width=1.2)
    routed_arrow(ax, [(12, 78), (1.0, 78), (1.0, 1.4), (158.0, 1.4), (158.0, 22), (163, 22)], BLUE, dashed=True, width=1.0)
    small_label(ax, 112, 1.4, "observed LR condition (training only)", BLUE, 5.0, "bold")
    small_label(ax, 270, 47, "red paths are absent at inference", RED, 5.3, "bold")

    fig.savefig(OUTPUT / "architecture_detailed.png", bbox_inches="tight", facecolor="white", pad_inches=0.12)
    fig.savefig(OUTPUT / "architecture_detailed.svg", bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)
    print("Wrote architecture_detailed.png and architecture_detailed.svg")


if __name__ == "__main__":
    build()
