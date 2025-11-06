from __future__ import annotations

from typing import Any, Final

import warnings

from collections.abc import Callable

import matplotlib
import numpy as np

from matplotlib import collections, colors, patches
from matplotlib import pyplot as plt
from matplotlib.path import Path as PlotPath

from triniti_ne.mcnp import read_plotm_file as rpf
from triniti_ne.ploting import BriefTicksAroundOneTicker

SetupAxesStrategyType = Callable[[plt.Axes], None]


def plot_ps_page(
    axes: plt.Axes,
    page: rpf.Page,
    setup_axes_strategy: bool | SetupAxesStrategyType | None = None,
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
        if isinstance(setup_axes_strategy, Callable):
            setup_axes_strategy(axes)
        else:
            warnings.warn("Usage of bool instead of setup_axes_strategy is deprecated")
            basis, origin, extent = page.basis, page.origin, page.extent
            default_setup_access_strategy(axes, basis, extent, origin)


def default_setup_access_strategy(
    axes: plt.Axes,
    basis: np.ndarray,
    extent: np.ndarray,
    origin: np.ndarray,
) -> None:
    """Use this for setup axes strategy, when there was True parameter in the old code.

        plot_ps_page(
            axes,
            page,
            lambda ax:
        )
    Parameters
    ----------
    axes
    basis
    extent
    origin

    Returns:
    -------
    """
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


def plot_2d_distribution(
    x: np.ndarray,
    y: np.ndarray,
    data: np.ndarray,
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    color_bar_title=r"$\frac{n} {cm^{2} \cdot s}$",
    max_log_power=None,
    min_max_log_ratio=1e-4,
    transform=None,
):
    if max_log_power is None:
        max_log_power = int(np.log10(data.max()))
    vmax = data.max()  # 10.0**max_log_power
    vmin = data.min()  # max(min_max_log_ratio * vmax, 10.0**min_log_power)
    # min_log_power = int(np.log10(vmin)) + 1
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.cm.get_cmap("hot")
    pcm = ax.pcolormesh(
        x,
        y,
        data,
        norm=norm,
        cmap=cmap,
        antialiased=True,
        shading="gouraud",
        transform=transform,
    )
    color_bar = fig.colorbar(pcm, ax=ax, shrink=0.8)
    color_bar.ax.set_title(color_bar_title, fontsize=8)
    tick_formatter = BriefTicksAroundOneTicker()
    color_bar.ax.yaxis.set_major_formatter(tick_formatter)
    color_bar.outline.set_edgecolor("white")
    # contours_no = max(max_log_power - min_log_power, 1)
    # levels = generate_logarithmic_contour_levels(min_log_power, max_log_power, contours_no)
    contours = ax.contour(
        x,
        y,
        data,
        norm=norm,
        levels=1,  # levels,
        # colors="xkcd:steel blue",  # cm.winter(norm(np.array(levels))),
        colors="white",  # cm.winter(norm(np.array(levels))),
        linewidths=1.0,
        alpha=0.5,
    )
    # index_10_in_13 = np.searchsorted(levels, 4e13)
    # # Thicken the 10^13 contour.
    # zc = contours.collections[index_10_in_13]
    # plt.setp(zc, linewidth=3)
    levels = contours.levels
    print(f"levels:{levels}")
    contour_labeled_levels = levels
    for ii in contour_labeled_levels:
        assert ii in levels, f"{ii} is not levels"
    ax.clabel(
        contours,
        contour_labeled_levels,
        inline=1.0,
        fmt=BriefTicksAroundOneTicker(),
        colors="xkcd:steel blue",  # colors=cm.winter(norm(np.array(contour_labeled_levels))),
        fontsize=9,
    )

    # ax.clabel(contours, levels[1::2],  # label every second level
    #            inline=1,
    #            fmt='%1.1g',
    #            colors='white',
    #            fontsize=12)

    # plt.flag()
    # make a colorbar for the contour lines
    # CB = fig.colorbar(contours, shrink=0.8, extend='both')

    # # We can still add a colorbar for the image, too.
    # locator = ticker.LogLocator(base=10)
    # CBI = fig.colorbar(CF, orientation='vertical', shrink=0.8, ticks=locator, format=CustomTicker())

    # CBI = fig.colorbar(CF, orientation='vertical', shrink=0.8, ticks=generate_logarithmic_contour_levels(10, 14, 3), format="%.0g")

    # This makes the original colorbar look a bit out of place,
    # so let's improve its position.

    # l, b, w, h = ax.get_position().bounds
    # ll, bb, ww, hh = CB.ax.get_position().bounds
    # CB.ax.set_position([l+w-ww*0.1, b + 0.1*h, ww, h*0.8])
