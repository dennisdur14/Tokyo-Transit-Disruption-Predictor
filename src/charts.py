"""Small matplotlib helpers so every chart in the project shares one quiet, readable style."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834"]  # categorical slots 1-2 (checked for colour-blind separation)

DPI = 200
BAR_MAX_PX = 24 * DPI / 96  # bars never thicker than ~24 screen px
RADIUS_PX = 4 * DPI / 96  # 4 px rounded data end


def figure(n_panels: int = 1, width: float = 8.0, height: float = 3.4):
    fig, axes = plt.subplots(1, n_panels, figsize=(width, height), dpi=DPI, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=7.5, length=0)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    return fig, axes


def titles(fig, title: str, subtitle: str | None = None) -> None:
    """Title + subtitle, left-aligned with the leftmost row label (call after drawing bars)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    lefts = [t.get_window_extent(renderer).x0 for ax in fig.axes for t in ax.get_yticklabels() if t.get_text()]
    x = fig.transFigure.inverted().transform((min(lefts), 0))[0] if lefts else 0.012
    fig.text(x, 0.975, title, ha="left", va="top", fontsize=10.5, color=TEXT, weight="semibold")
    if subtitle:
        fig.text(x, 0.905, subtitle, ha="left", va="top", fontsize=8, color=TEXT_SECONDARY)


def hbars(ax, labels: list[str], values: list[float], colors: list[str], fmt: str = "{:,.0f}",
          panel_title: str | None = None, value_range: tuple[float, float] | None = None) -> None:
    """Horizontal bars, first label at the top, value printed at each bar's tip.

    Pass the same value_range to several panels so their axes are comparable."""
    values = np.asarray(values, dtype=float)
    y = np.arange(len(labels))[::-1]
    lo, hi = value_range or (values.min(), values.max())
    lo, hi = min(0.0, lo), max(0.0, hi)
    span = (hi - lo) or 1.0
    ax.set_xlim(lo - (0.16 * span if lo < 0 else 0), hi + 0.16 * span)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_yticks(y, labels, color=TEXT)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    if panel_title:
        ax.set_title(panel_title, loc="left", fontsize=8.5, color=TEXT, pad=6)

    ax.figure.canvas.draw()  # need the final axes size to convert pixels to data units
    bbox = ax.get_window_extent()
    px_x = bbox.width / np.diff(ax.get_xlim())[0]
    px_y = bbox.height / np.diff(ax.get_ylim())[0]
    height = min(0.62, BAR_MAX_PX / px_y)
    rx, ry = RADIUS_PX / px_x, RADIUS_PX / px_y
    for yi, v, c in zip(y, values, colors):
        if v == 0:
            continue
        x0, w = min(0.0, v), abs(v)
        r = min(rx, w / 2)
        ax.add_patch(FancyBboxPatch((x0, yi - height / 2), w, height, boxstyle=f"round,pad=0,rounding_size={r}",
                                    mutation_aspect=ry / rx, linewidth=0, facecolor=c))
        # square off the end that sits on the baseline
        ax.add_patch(Rectangle((0 if v > 0 else -r, yi - height / 2), r, height, linewidth=0, facecolor=c))
        pad = 0.012 * span
        ax.text(v + (pad if v > 0 else -pad), yi, fmt.format(v), va="center", ha="left" if v > 0 else "right",
                fontsize=7.5, color=TEXT_SECONDARY)


def save(fig, path) -> None:
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
