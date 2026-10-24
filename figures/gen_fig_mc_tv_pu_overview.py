"""Generate the result-free MC-TV-PU method overview as vector PDF."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch


OUTPUT_DIR = Path(__file__).resolve().parent
FONT = font_manager.FontProperties(family="DejaVu Sans")
FONT_BOLD = font_manager.FontProperties(family="DejaVu Sans", weight="bold")

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
SKY = "#56B4E9"
GRAY = "#5B6573"
LIGHT_GRAY = "#F2F4F7"
INK = "#1F2933"


def box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    edge: str = GRAY,
    face: str = "white",
    linestyle: str = "-",
    fontsize: float = 9.0,
    bold: bool = False,
    rounding: float = 0.012,
    linewidth: float = 1.3,
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={rounding}",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=face,
        linestyle=linestyle,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        color=INK,
        fontsize=fontsize,
        fontproperties=FONT_BOLD if bold else FONT,
        linespacing=1.25,
        zorder=3,
    )
    return patch


def arrow(
    ax,
    start,
    end,
    *,
    color=GRAY,
    style="-|>",
    linewidth=1.4,
    linestyle="-",
    mutation_scale=12,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        shrinkA=2,
        shrinkB=2,
        zorder=4,
    )
    ax.add_patch(patch)
    return patch


def panel_label(ax, x: float, label: str, heading: str):
    ax.text(
        x,
        0.955,
        label,
        ha="left",
        va="center",
        color="white",
        fontsize=10.0,
        fontproperties=FONT_BOLD,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": INK, "edgecolor": INK},
    )
    ax.text(
        x + 0.035,
        0.955,
        heading,
        ha="left",
        va="center",
        color=INK,
        fontsize=8.3,
        fontproperties=FONT_BOLD,
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "stix",
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Panel boundaries are deliberately subtle and do not act as decorative backgrounds.
    for x, width in ((0.015, 0.30), (0.345, 0.31), (0.685, 0.30)):
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.19),
                width,
                0.735,
                boxstyle="round,pad=0.008,rounding_size=0.012",
                edgecolor="#CBD2D9",
                facecolor="white",
                linewidth=0.9,
                zorder=0,
            )
        )

    panel_label(ax, 0.025, "A", "Compatible PU set")
    panel_label(ax, 0.355, "B", "Certified selection")
    panel_label(ax, 0.695, "C", "Source cross-fit")

    # A: ambiguity-set geometry and the exact difference from the matched rectangle.
    box(
        ax,
        0.045,
        0.79,
        0.24,
        0.085,
        "Query $Q$  |  anchor $p^0$\nPrior $\\pi$  |  radius $\\tau$",
        edge=BLUE,
        face="#EAF5FB",
        fontsize=7.5,
    )
    arrow(ax, (0.165, 0.785), (0.165, 0.735), color=BLUE)
    ax.add_patch(
        FancyBboxPatch(
            (0.045, 0.37),
            0.24,
            0.34,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            linewidth=1.6,
            edgecolor=ORANGE,
            facecolor="#FFF8E5",
            linestyle="--",
            zorder=1,
        )
    )
    ax.text(
        0.165,
        0.665,
        "Matched rectangle  $\\mathcal{A}_{\\mathrm{rect}}$\n"
        "same $\\bar{\\pi}_{\\mathrm{comp}}$; coupling removed",
        ha="center",
        va="center",
        color=INK,
        fontsize=7.4,
        fontproperties=FONT,
        linespacing=1.3,
        zorder=3,
    )
    ellipse = Ellipse(
        (0.165, 0.49),
        0.175,
        0.17,
        edgecolor=GREEN,
        facecolor="#E8F5EF",
        linewidth=1.7,
        zorder=2,
    )
    ax.add_patch(ellipse)
    ax.text(
        0.165,
        0.49,
        "Compatible  $\\mathcal{A}_{\\mathrm{comp}}$\n"
        "$\\mathrm{TV}(p,p^0) \\leq \\tau$\n"
        "$\\pi p_i \\leq Q_i$",
        ha="center",
        va="center",
        color=INK,
        fontsize=7.5,
        fontproperties=FONT_BOLD,
        linespacing=1.25,
        zorder=3,
    )
    box(
        ax,
        0.067,
        0.245,
        0.196,
        0.075,
        "Tighter directed bounds\ncan change the selected ranker",
        edge=VERMILLION,
        face="#FCEFEA",
        fontsize=7.2,
    )

    # Inter-panel arrow.
    arrow(ax, (0.315, 0.56), (0.345, 0.56), color=INK, linewidth=1.8)

    # B: frozen candidates to certified set-valued selection.
    box(
        ax,
        0.385,
        0.80,
        0.23,
        0.075,
        "15 frozen rankers\nverified scaffold test-fold mapping",
        edge=BLUE,
        face="#EAF5FB",
        fontsize=7.5,
        bold=True,
    )
    arrow(ax, (0.50, 0.795), (0.50, 0.76), color=BLUE)
    box(
        ax,
        0.385,
        0.68,
        0.23,
        0.075,
        "Exact midrank-loss difference\n$s=L_h-L_g$",
        edge=GRAY,
        face=LIGHT_GRAY,
        fontsize=7.4,
    )
    arrow(
        ax,
        (0.50, 0.675),
        (0.50, 0.645),
        color=GRAY,
        linewidth=0.9,
        mutation_scale=7,
    )
    box(
        ax,
        0.375,
        0.49,
        0.25,
        0.15,
        "Rational MC-TV-PU oracle\n12 $\\tau$ values | certified gap $\\leq 10^{-3}$",
        edge=GREEN,
        face="#E8F5EF",
        fontsize=7.4,
        bold=True,
    )
    arrow(ax, (0.50, 0.485), (0.50, 0.46), color=GREEN)
    box(
        ax,
        0.385,
        0.35,
        0.23,
        0.105,
        "Drug-macro loss difference\none shared comparator $g$",
        edge=ORANGE,
        face="#FFF8E5",
        fontsize=7.4,
    )
    arrow(ax, (0.50, 0.345), (0.50, 0.30), color=ORANGE)
    box(
        ax,
        0.377,
        0.205,
        0.246,
        0.105,
        "Set-valued selection\npossibly optimal\nor certifiably unique",
        edge=VERMILLION,
        face="#FCEFEA",
        fontsize=6.8,
        bold=True,
    )
    arrow(ax, (0.655, 0.56), (0.685, 0.56), color=INK, linewidth=1.8)

    # C: source-crossfit keeps held-out source folds out of all selector anchors.
    box(
        ax,
        0.725,
        0.80,
        0.22,
        0.075,
        "Global PMID/source\n5 fixed folds",
        edge=BLUE,
        face="#EAF5FB",
        fontsize=7.5,
        bold=True,
    )
    arrow(ax, (0.835, 0.795), (0.77, 0.715), color=BLUE)
    arrow(ax, (0.835, 0.795), (0.9125, 0.715), color=BLUE)
    box(
        ax,
        0.705,
        0.61,
        0.13,
        0.10,
        "Exclude fold $f$\nbuild anchors",
        edge=GREEN,
        face="#E8F5EF",
        fontsize=7.1,
    )
    box(
        ax,
        0.86,
        0.61,
        0.105,
        0.10,
        "Fold $f$\nheld out",
        edge=VERMILLION,
        face="#FCEFEA",
        fontsize=7.1,
        linestyle="--",
    )
    arrow(ax, (0.77, 0.605), (0.77, 0.54), color=GREEN)
    box(
        ax,
        0.705,
        0.45,
        0.13,
        0.085,
        "Independent $S_f$\nno cross-fold pooling",
        edge=GREEN,
        face="white",
        fontsize=7.1,
        bold=True,
    )
    arrow(ax, (0.77, 0.445), (0.82, 0.43), color=GREEN)
    arrow(
        ax,
        (0.9125, 0.605),
        (0.88, 0.43),
        color=VERMILLION,
        linestyle="--",
    )
    box(
        ax,
        0.71,
        0.31,
        0.25,
        0.115,
        "Worst member of $S_f$ on $H_{df}$\nAP primary | AUC/Recall secondary",
        edge=ORANGE,
        face="#FFF8E5",
        fontsize=6.8,
        bold=True,
    )
    arrow(ax, (0.835, 0.305), (0.835, 0.27), color=ORANGE)
    ax.text(
        0.835,
        0.235,
        "20,000 drug bootstraps\n48 simultaneous AP lower bounds",
        ha="center",
        va="center",
        color=GRAY,
        fontsize=7.2,
        fontproperties=FONT,
    )

    # Claim boundary: this is the scientific guarantee boundary, not decoration.
    box(
        ax,
        0.02,
        0.035,
        0.96,
        0.105,
        "Scope: finite RNA query | verified target-drug test-fold mapping | source-isolated anchors\n"
        "No guarantee: end-to-end PMID isolation | zeroes are true negatives | "
        "population coverage | unseen scaffolds",
        edge=INK,
        face="#F7F8FA",
        fontsize=7.0,
        bold=True,
        rounding=0.008,
        linewidth=1.2,
    )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(OUTPUT_DIR / "fig_mc_tv_pu_overview.pdf", format="pdf")
    fig.savefig(OUTPUT_DIR / "fig_mc_tv_pu_overview.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
