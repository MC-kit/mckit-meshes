"""Plotting and ps-files loading utils."""

from __future__ import annotations

from warnings import warn

try:
    import matplotlib as mpl

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warn("matplotlib is not installed, mckit_meshes plotting is disabled", stacklevel=0)

from .check_coordinate_plane import BASES, XY, XZ, YZ, X, Y, Z
from .read_plotm_file import Page, split_input, transform_page
from .read_plotm_file import load_all_pages as load_plotm_file
from .read_plotm_file import scan_pages as read_pages

__all__ = [
    "BASES",
    "XY",
    "XZ",
    "YZ",
    "Page",
    "X",
    "Y",
    "Z",
    "load_plotm_file",
    "read_pages",
    "split_input",
    "transform_page",
]

if MATPLOTLIB_AVAILABLE:
    from ._plot import (
        default_setup_axes_strategy,
        plot_2d_distribution,
        plot_ps_page,
        rectangle_plotter,
    )
    from .brief_ticks_around_one_ticker import BriefTicksAroundOneTicker

    __all__ += [
        "BriefTicksAroundOneTicker",
        "default_setup_axes_strategy",
        "mpl",
        "plot_2d_distribution",
        "plot_ps_page",
        "rectangle_plotter",
    ]
