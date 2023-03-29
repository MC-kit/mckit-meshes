from __future__ import annotations

from matplotlib.ticker import LogFormatterSciNotation
from numpy import isnan


class BriefTicksAroundOneTicker(LogFormatterSciNotation):
    def __call__(self, x, pos=None):
        if isnan(x):
            return "NaN"
        elif x not in [0.1, 1, 10]:
            return LogFormatterSciNotation.__call__(self, x, pos=None)
        else:
            return f"{x:g}"
