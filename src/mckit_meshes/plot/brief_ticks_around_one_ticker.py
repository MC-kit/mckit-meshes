from __future__ import annotations

from typing import ClassVar

from matplotlib.ticker import LogFormatterSciNotation
from numpy import isnan


class BriefTicksAroundOneTicker(LogFormatterSciNotation):
    """Format float number ticks in sci-notation excluding 0.1, 1, 10 values."""

    _excluded: ClassVar[set[float]] = {0.1, 1.0, 10.0}

    def __call__(self, x: float, pos: int | None = None) -> str:
        r"""Format `x` in sci-notation excluding 0.1, 1, 10 values.

        Parameters
        ----------
        x
            tick value
        pos, optional
            position for `x`

        Returns
        -------
            formatted value

        Examples
        --------
        >>> bt = BriefTicksAroundOneTicker()
        >>> bt(1e-3)
        '$\\mathdefault{10^{-3}}$'
        >>> bt(0.1)
        0.1
        >>> bt(1.0)
        1
        >>> bt(10.0)
        10
        >>> bt(1e3)
        '$\\mathdefault{10^{3}}$'
        >>> bt(float("NaN"))
        'NaN'
        """
        if isnan(x):
            return "NaN"
        if x in BriefTicksAroundOneTicker._excluded:
            return f"{x:g}"
        return LogFormatterSciNotation.__call__(self, x, pos=pos)


if __name__ == "__main__":
    import xdoctest

    xdoctest.doctest_module(command="all", verbose=3)
