from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final

import numpy as np
from matplotlib import collections
from matplotlib import colormaps as cm
from matplotlib import colors, patches
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.path import Path as PlotPath

import mckit_meshes.plot.read_plotm_file as rpf
from mckit_meshes.plot.brief_ticks_around_one_ticker import BriefTicksAroundOneTicker

SetupAxesStrategyType = Callable[[plt.Axes], None]


def default_setup_axes_strategy(
    basis: np.ndarray,
    extent: np.ndarray,
    origin: np.ndarray,
) -> SetupAxesStrategyType:
    """Create axis configuring method based on plotm page parameters.

    Parameters
    ----------
    basis
        plotm page basis
    extent
        ... extent
    origin
        ... origin

    Returns
    -------
    Callable[[plt.Axes], None]
        method to setup axis scale and labels


    """

    def _call(axes: plt.Axes) -> None:
        axes.set_aspect("equal")
        if basis is rpf.XZ:
            axes.set_xlabel("X, cm")
            axes.set_ylabel("Z, cm")
            axes.set_xlim(origin[0] - extent[0], origin[0] + extent[0])
            axes.set_ylim(origin[2] - extent[1], origin[2] + extent[1])
        elif basis is rpf.YZ:
            axes.set_xlabel("Y, cm")
            axes.set_ylabel("Z, cm")
            axes.set_xlim(origin[1] - extent[0], origin[1] + extent[0])
            axes.set_ylim(origin[2] - extent[1], origin[2] + extent[1])
        elif basis is rpf.XY:
            axes.set_xlabel("X, cm")
            axes.set_ylabel("Y, cm")
            axes.set_xlim(origin[0] - extent[0], origin[0] + extent[0])
            axes.set_ylim(origin[1] - extent[1], origin[1] + extent[1])
        else:
            raise ValueError(f"Basis {basis} is not supported")

    return _call


def plot_ps_page(
    axes: plt.Axes,
    page: rpf.Page,
    *,
    setup_axes_strategy: SetupAxesStrategyType | None = None,
) -> None:
    coll = collections.LineCollection(
        page.lines,
        colors="xkcd:blue green",
        facecolors="xkcd:pale green",
        linestyles="solid",
        linewidths=0.5,
    )
    coll.set_rasterized(True)
    axes.add_collection(coll)
    if setup_axes_strategy:
        setup_axes_strategy(axes)


_INDICES: Final = [
    [0, 0],
    [0, 1],
    [1, 1],
    [1, 0],
    [0, 0],
]

_CODES: Final = [
    PlotPath.MOVETO,
    PlotPath.LINETO,
    PlotPath.LINETO,
    PlotPath.LINETO,
    PlotPath.CLOSEPOLY,
]


def rectangle_plotter(axs: plt.Axes) -> Callable[[Any], None]:
    def _call(points: np.ndarray):
        assert points.shape == (
            2,
            2,
        ), "`points` should contain 2x2 array with coordinates of corners"

        vertices = []

        for ix, iy in _INDICES:
            bottom_left, top_right = points[ix], points[iy]
            x, y = bottom_left[0], top_right[1]
            vertices.append((x, y))

        path = PlotPath(vertices, _CODES)
        patch = patches.PathPatch(path, facecolor="orange", lw=0.2, alpha=0.2)
        axs.add_patch(patch)

    return _call


def mids(x):
    return 0.5 * (x[1:] + x[:-1])


def plot_2d_distribution(
    x,
    y,
    data,
    fig,
    ax,
    *,
    color_bar_title=r"$\frac{1} {cm^{2} \cdot s}$",
    max_log_power=None,
    min_max_log_ratio=1e-4,
    transform=None,
    levels=None,
):
    if max_log_power is None:
        max_log_power = int(np.log10(data.max()))
    vmax = 10.0**max_log_power
    vmin = data.min()
    min_log_power = int(np.log10(vmin)) + 1
    vmin = max(min_max_log_ratio * vmax, 10.0**min_log_power)
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap("hot")
    pcm = ax.pcolormesh(
        x,
        y,
        data,
        norm=norm,
        cmap=cmap,
        # antialiased=True,
        # shading="gouraud",
        shading="flat",
        # transform=transform,
    )
    color_bar = fig.colorbar(pcm, ax=ax, shrink=0.8)
    color_bar.ax.set_title(color_bar_title, pad=20, fontsize=12)
    tick_formatter = BriefTicksAroundOneTicker()
    color_bar.ax.yaxis.set_major_formatter(tick_formatter)
    color_bar.outline.set_edgecolor("white")
    if levels is not None:
        _colors = "k"  # cm.get("Wistia_r")(norm(np.array(levels)))
        contours = ax.contour(
            mids(x),
            mids(y),
            data,
            norm=norm,
            levels=levels,
            colors=_colors,
            linewidths=1.0,
            # alpha=0.5,
        )
        levels = contours.levels
        contour_labeled_levels = levels
        fmt = ticker.LogFormatterMathtext()
        fmt.create_dummy_axis()
        ax.clabel(
            contours,
            contour_labeled_levels,
            inline=1.0,
            # fmt="%.1g",
            fmt=fmt,
            # colors="k",
            fontsize=9,
        )
