"""Plotting utils."""

from __future__ import annotations

from warnings import warn

try:
    import matplotlib as mpl

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warn("matplotlib is not installed, mckit_meshes plotting is disabled", stacklevel=0)

if MATPLOTLIB_AVAILABLE:
    from .brief_ticks_around_one_ticker import BriefTicksAroundOneTicker
    from .read_plotm_file import (
        Page,
        load as load_plotm_file,
        load_pages,
        read as read_plotm_file,
        X,
        XY,
        XZ,
        Y,
        YZ,
        Z,
        BASES,
    )
    from .plot import (
        plot_ps_page,
        plot_2d_distribution,
        rectangle_plotter,
    )

    __all__ = [
        "BASES",
        "XY",
        "XZ",
        "YZ",
        "BriefTicksAroundOneTicker",
        "Page",
        "X",
        "Y",
        "Z",
        "load_pages",
        "load_plotm_file",
        "mpl",
        "plot_2d_distribution",
        "plot_ps_page",
        "read_plotm_file",
        "rectangle_plotter",
    ]
