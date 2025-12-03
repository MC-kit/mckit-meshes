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
    from .plot import (
        default_setup_axes_strategy,
        plot_ps_page,
        plot_2d_distribution,
        rectangle_plotter,
    )
    from .read_plotm_file import (
        Page,
        load_all_pages as load_plotm_file,
        split_input,
        scan_pages as read_pages,
        transform_page,
        X,
        XY,
        XZ,
        Y,
        YZ,
        Z,
        BASES,
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
        "default_setup_axes_strategy",
        "load_plotm_file",
        "mpl",
        "plot_2d_distribution",
        "plot_ps_page",
        "read_pages",
        "rectangle_plotter",
        "split_input",
        "transform_page",
    ]
