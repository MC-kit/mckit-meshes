from __future__ import annotations

from typing import TYPE_CHECKING, Final

from enum import IntEnum

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


X, Y, Z = np.eye(3, dtype=float)
XY = np.vstack((X, Y))
XZ = np.vstack((X, Z))
YZ = np.vstack((Y, Z))
YX = np.vstack((Y, X))
ZX = np.vstack((Z, X))
ZY = np.vstack((Z, Y))
BASES = [XY, XZ, YZ, YX, ZX, ZY]


class CoordinateNumber(IntEnum):
    """Coordinate to select from basis."""

    x = 0
    y = 1
    z = 2


_PLANE_RTOL: Final[float] = 1e-9
_PLANE_ATOL: Final[float] = 1e-9


def is_coordinate_plane(
    basis: NDArray,
    coordinate: CoordinateNumber,
    rtol: float = _PLANE_RTOL,
    atol: float = _PLANE_ATOL,
):
    """Check, if the basis defines a plane perpendicular to one of the axes.

    Parameters
    ----------
    basis
        two vectors defining plotm projection plane.
    coordinate
        x, y or z axis
    rtol
        relative tolerance for :meth:`np.allclose`
    atol
        absolute tolerance for :meth:`np.allclose`

    Returns
    -------
    True - if all values for the coordinate in basis are close to zero

    Examples
    --------
    >>> is_coordinate_plane(XY, CoordinateNumber.z)
    True
    """
    return np.allclose(basis[:, coordinate], 0.0, rtol=rtol, atol=atol)


def is_x_plane(basis: NDArray, *, rtol: float = _PLANE_RTOL, atol: float = _PLANE_ATOL):
    """Check, if the basis defines a plane perpendicular to X-axes.

    Parameters
    ----------
    basis
        two vectors defining plotm projection plane.
    rtol
        relative tolerance for :meth:`np.close`
    atol
        absolute tolerance for :meth:`np.allclose`

    Returns
    -------
    True - if all values for x-coordinate in basis are close to zero

    Examples
    --------
    >>> is_coordinate_plane(YZ, CoordinateNumber.x)
    True
    """
    return is_coordinate_plane(basis, CoordinateNumber.x, rtol=rtol, atol=atol)


def is_y_plane(basis: NDArray, rtol: float = _PLANE_RTOL, atol: float = _PLANE_ATOL):
    """Check, if the basis defines a plane perpendicular to Y-axes."""
    return is_coordinate_plane(basis, CoordinateNumber.y, rtol=rtol, atol=atol)


def is_z_plane(basis: NDArray, rtol: float = _PLANE_RTOL, atol: float = _PLANE_ATOL):
    """Check, if the basis defines a plane perpendicular to Z-axes."""
    return is_coordinate_plane(basis, CoordinateNumber.z, rtol=rtol, atol=atol)
