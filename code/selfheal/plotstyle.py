"""Shared publication-quality plotting style and helpers for the result figures."""

from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Consistent, restrained palette: greys for greedy, cool tones for the strong baselines,
# warm tones for the quantum-inspired pair so the synergy comparison reads at a glance.
PALETTE = {
    "greedy":      "#9aa0a6",
    "greedy-conn": "#7e9bc4",
    "GA":          "#5aa469",
    "SA":          "#9b72c7",
    "QIEA-noAI":   "#e8a13a",
    "QIEA+AI":     "#c0392b",
}
LABELS = {
    "greedy": "greedy", "greedy-conn": "greedy\n(conn.)", "GA": "GA",
    "SA": "SA", "QIEA-noAI": "QIEA\n(no AI)", "QIEA+AI": "QIEA\n+AI",
}


def apply():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "legend.frameon": False,
        "figure.dpi": 220,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    })


def clean(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", color="0.85", linewidth=0.7, linestyle="-", zorder=0)
        ax.set_axisbelow(True)


def bars(ax, methods, means, stds, fmt="{:.2f}"):
    """Draw a labelled bar group with error bars using the shared palette."""
    colors = [PALETTE.get(m, "#888") for m in methods]
    xs = range(len(methods))
    bb = ax.bar(xs, means, yerr=stds, color=colors, edgecolor="black", linewidth=0.6,
                capsize=3, error_kw={"elinewidth": 0.9, "capthick": 0.9}, zorder=3)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([LABELS.get(m, m) for m in methods])
    for b, v, s in zip(bb, means, stds):
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height() + s),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8.5)
    return bb


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "n.s."


def sig_bracket(ax, i, j, y, p, h=0.025):
    """Draw a significance bracket between bar indices i and j at height y."""
    ax.plot([i, i, j, j], [y, y + h, y + h, y], lw=0.9, c="0.2")
    ax.text((i + j) / 2, y + h, stars(p), ha="center", va="bottom", fontsize=9.5)
