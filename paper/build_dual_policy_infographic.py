from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "images"
OUTPUT.mkdir(parents=True, exist_ok=True)
PAPER_OUTPUT = ROOT / "paper" / "figures"
PAPER_OUTPUT.mkdir(parents=True, exist_ok=True)

NAVY = "#163A63"
BLUE = "#2563A6"
TEAL = "#147D78"
PURPLE = "#6B46C1"
ORANGE = "#C75B12"
GREEN = "#247A4D"
RED = "#B83232"
GRAY = "#4A5568"
LIGHT = "#F7FAFC"
WHITE = "#FFFFFF"


def box(ax, x, y, w, h, title, lines, color, fill=WHITE, title_size=10):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.25,rounding_size=0.35",
            facecolor=fill,
            edgecolor=color,
            linewidth=2.2,
        )
    )
    ax.text(
        x + w / 2,
        y + h - 0.75,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=color,
    )
    ax.text(
        x + w / 2,
        y + h / 2 - 0.35,
        "\n".join(lines),
        ha="center",
        va="center",
        fontsize=8.2,
        color=GRAY,
        linespacing=1.35,
    )


def arrow(ax, start, end, color=NAVY, dashed=False, bend=0.0, width=1.8):
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
        )
    )


def routed_arrow(ax, points, color=NAVY, dashed=False, width=1.8):
    linestyle = "--" if dashed else "-"
    xs = [point[0] for point in points[:-1]]
    ys = [point[1] for point in points[:-1]]
    ax.plot(xs, ys, color=color, linewidth=width, linestyle=linestyle)
    arrow(
        ax,
        points[-2],
        points[-1],
        color=color,
        dashed=dashed,
        width=width,
    )


def label(ax, x, y, text, color=GRAY, size=7.5, weight="normal"):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=color,
        fontweight=weight,
    )


def build() -> None:
    fig, ax = plt.subplots(figsize=(20, 12), dpi=220)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 68)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    ax.text(
        50,
        65.8,
        "GeoDiff-GAN: Dual-Policy Evidence-Constrained Architecture",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=NAVY,
    )
    ax.text(
        50,
        63.2,
        "Novel mechanism: reconstruction confidence is separated from prompt edit authority",
        ha="center",
        va="center",
        fontsize=11,
        color=GRAY,
    )

    box(
        ax,
        2,
        43,
        13,
        12,
        "LR OBSERVATION",
        ["RGB Sentinel-2", "B x 3 x 128 x 128", "synthetic 40 m"],
        BLUE,
        "#EBF4FF",
    )
    box(
        ax,
        19,
        49,
        17,
        10,
        "SwinIR Base",
        ["6 residual Swin blocks", "pixel shuffle 4x", "B x 3 x 512 x 512"],
        NAVY,
        "#EBF4FF",
    )
    box(
        ax,
        19,
        36,
        17,
        11,
        "LR Evidence Encoder",
        [
            "f128: B x 64 x 128 x 128",
            "f64: B x 128 x 64 x 64",
            "f32 / f16 diagnostics",
        ],
        TEAL,
        "#E8FFFB",
    )
    arrow(ax, (15, 50), (19, 54), BLUE)
    arrow(ax, (15, 47), (19, 41.5), BLUE)

    box(
        ax,
        40,
        43,
        18,
        16,
        "Conditional Latent Diffusion",
        [
            "noise zT -> denoised z0",
            "B x 4 x 64 x 64",
            "v-prediction U-Net",
            "LR + degradation + mode",
            "+ optional text tokens",
        ],
        PURPLE,
        "#FAF5FF",
    )
    arrow(ax, (36, 41.5), (40, 48), TEAL)
    label(ax, 38, 45.5, "f64", TEAL, 7, "bold")

    box(
        ax,
        40,
        31,
        18,
        8,
        "Conditioning",
        ["SigLIP tokens B x T x 768", "degradation B x 4", "mode: SR=0 / edit=1"],
        ORANGE,
        "#FFFAF0",
    )
    arrow(ax, (49, 39), (49, 43), ORANGE, dashed=True)

    box(
        ax,
        62,
        42,
        18,
        17,
        "Dual-Policy GeoMapper",
        [
            "content: B x 128 x 64 x 64",
            "4 FiLM styles: B x 256",
            "",
            "evidence confidence ce",
            "B x 1 x 64 x 64",
            "edit permission cp",
            "B x 1 x 64 x 64",
        ],
        GREEN,
        "#F0FFF4",
    )
    arrow(ax, (58, 51), (62, 51), PURPLE)
    routed_arrow(
        ax,
        [(36, 40), (38, 40), (38, 30.5), (60.5, 30.5), (60.5, 46), (62, 46)],
        TEAL,
    )
    label(ax, 50, 31.4, "direct f64 evidence path", TEAL, 7, "bold")
    arrow(ax, (58, 35), (66, 42), ORANGE, dashed=True)

    box(
        ax,
        84,
        47,
        14,
        12,
        "Shared Decoder Trunk",
        ["64 -> 128 -> 256 -> 512", "FiLM-modulated blocks", "LR skip connections"],
        NAVY,
        LIGHT,
    )
    arrow(ax, (80, 52), (84, 53), GREEN)
    routed_arrow(
        ax,
        [(36, 37.5), (37, 37.5), (37, 29.2), (82, 29.2), (82, 49), (84, 49)],
        TEAL,
        dashed=True,
    )
    label(ax, 71, 30.1, "multi-resolution LR skips", TEAL, 7, "bold")

    box(
        ax,
        80,
        33,
        18,
        10,
        "Two Residual Heads",
        [
            "detail rd: B x 3 x 512 x 512",
            "edit re: B x 3 x 512 x 512",
            "independent RGB outputs",
        ],
        ORANGE,
        "#FFFAF0",
    )
    arrow(ax, (91, 47), (91, 43), NAVY)

    ax.axhline(28.5, xmin=0.02, xmax=0.98, color="#CBD5E0", linewidth=1.4)
    label(ax, 50, 29.6, "POLICY-CONTROLLED OUTPUT COMPOSITION", NAVY, 9, "bold")

    box(
        ax,
        2,
        12,
        29,
        13,
        "SR MODE: EVIDENCE-CONSTRAINED",
        [
            "rSR = HighPass(rd) x ce",
            "x0 = base + rSR",
            "3 sensor back-projection steps",
            "prompt edit head is disabled",
        ],
        BLUE,
        "#EBF4FF",
    )
    box(
        ax,
        36,
        12,
        29,
        13,
        "EDIT MODE: SYNTHETIC CHANGE",
        [
            "redit = HighPass(rd) x ce + re x cp",
            "x0 = base + redit",
            "1 soft consistency step",
            "metadata: synthetic_edit=true",
        ],
        ORANGE,
        "#FFFAF0",
    )
    box(
        ax,
        70,
        12,
        28,
        13,
        "UNCERTAINTY ABSTENTION",
        [
            "8 stochastic SR samples",
            "cfinal = ce x exp(-variance / su)",
            "x = base + agreement(xmean - base)",
            "low support returns toward base",
        ],
        RED,
        "#FFF5F5",
    )
    routed_arrow(
        ax,
        [(86, 33), (86, 27.8), (16.5, 27.8), (16.5, 25)],
        BLUE,
    )
    routed_arrow(
        ax,
        [(92, 33), (92, 26.8), (50.5, 26.8), (50.5, 25)],
        ORANGE,
    )
    label(ax, 31, 27.9, "detail head + evidence confidence", BLUE, 6.7, "bold")
    label(ax, 70, 26.9, "detail head + edit head + both policies", ORANGE, 6.7, "bold")

    box(
        ax,
        4,
        2,
        92,
        7,
        "TRAINING SIGNALS THAT MAKE THE POLICIES MEANINGFUL",
        [
            "Evidence calibration: ce targets local ungated reconstruction accuracy   |   "
            "Edit supervision: null/matched prompts suppress cp; mismatched prompts receive sparse coverage   |   "
            "Localization: raw edit energy is penalized outside cp"
        ],
        GREEN,
        "#F0FFF4",
        title_size=9.5,
    )

    label(
        ax,
        50,
        0.7,
        "The discriminator is training-only. SR remains a reconstruction estimate; edit output is explicitly synthetic.",
        RED,
        8,
        "bold",
    )

    fig.tight_layout(pad=0.6)
    fig.savefig(OUTPUT / "geodiff-dual-policy-architecture.png", bbox_inches="tight")
    fig.savefig(OUTPUT / "geodiff-dual-policy-architecture.svg", bbox_inches="tight")
    fig.savefig(
        PAPER_OUTPUT / "geodiff-dual-policy-architecture.png",
        bbox_inches="tight",
    )
    fig.savefig(
        PAPER_OUTPUT / "geodiff-dual-policy-architecture.svg",
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    build()
