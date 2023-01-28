"""Common mesh geometry specification classes and functions.

## Relative or absolute coordinates

    There are variations when coordinates are presented as relative to origin
    or absolute. This depends on is output for specification, or input/output
    to Weight of Meshtal files and is it cartesian or cylinder mesh.

    Cartesian:

    |       |  wwinp   | meshtal  |
    | ===== | =======  | ======== |
    |  spec | relative | absolute (but origin is extracted to separate item) |
    | ----- | -------  | -------- |
    |  file | relative | absolute |

    Cylinder:

    |       |  wwinp   | meshtal  |
    | ===== | =======  | ======== |
    |  spec | relative | relative |
    | ----- | -------  | -------- |
    |  file | relative | relative |

    The new callers are to use local_coordinates converter to avoid difficulties.
    For the old callers we will use ZERO_ORIGIN for Geometry Specification being
    used in FMesh.
"""
from __future__ import annotations

from typing import Final, Iterable, List, Optional, Sequence, TextIO, Tuple, Union, cast

import abc

from dataclasses import dataclass, field

import mckit_meshes.utils as ut
import mckit_meshes.utils.io
import numpy as np
import numpy.linalg as linalg

from mckit_meshes.utils.cartesian_product import cartesian_product

_2PI: Final[float] = 2.0 * np.pi
_1_TO_2PI: Final[float] = 1 / _2PI
__DEG_2_RAD: Final[float] = np.pi / 180.0
CARTESIAN_BASIS: Final[np.ndarray] = np.eye(3, dtype=np.double)
(NX, NY, NZ) = CARTESIAN_BASIS

DEFAULT_AXIS: Final[np.ndarray] = NZ
DEFAULT_VEC: Final[np.ndarray] = NX


ZERO_ORIGIN: Final[np.ndarray] = np.zeros((3,), dtype=np.double)


def as_float_array(array) -> np.ndarray:
    """Convert any sequence of numbers to numpy array of floats.

    Args:
        array: Anything that can be converted to numpy ndarray.

    Returns:
        np.ndarray:  either original or conversion.
    """
    return np.asarray(array, dtype=float)


@dataclass(eq=False)
class AbstractGeometrySpecData:
    """Data mixin for :py:class:`AbstractGeometrySpec`.

    Provides reusable data fields.
    """

    ibins: np.ndarray
    jbins: np.ndarray
    kbins: np.ndarray
    origin: np.ndarray = field(default_factory=lambda: ZERO_ORIGIN.copy())

    def __post_init__(self) -> None:
        """Force a caller provided data as numpy arrays.

        Raises:
            TypeError: if any of the fields is not a numpy array.
        """
        for b in self.bins:
            if not isinstance(b, np.ndarray):
                raise TypeError(f"Expected numpy array, actual {b[0]}...{b[-1]}")

    def __hash__(self) -> int:
        return hash(self.bins)

    def __eq__(self, other: "AbstractGeometrySpecData") -> bool:
        if not isinstance(other, AbstractGeometrySpecData):
            return False
        a, b = self.bins, other.bins
        return len(a) == len(b) and arrays_equal(zip(a, b))

    @property
    def bins(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Pack the fields to tuple.

        Returns:
            tuple of origin and bins.
        """
        return self.origin, self.ibins, self.jbins, self.kbins


class AbstractGeometrySpec(AbstractGeometrySpecData, abc.ABC):
    """Common base for rectilinear and cylinder mesh specifications."""

    @property
    @abc.abstractmethod
    def cylinder(self) -> bool:
        """Is this an instance of a cylinder mesh specification?"""
        ...

    @abc.abstractmethod
    def local_coordinates(self, points: np.ndarray) -> np.ndarray:
        """Convert points coordinates to local system.

        Args:
            points: ... with global coordinates
        """
        ...

    @abc.abstractmethod
    def get_mean_square_distance_weights(self, point) -> float:
        """Estimate weights as a voxel mean square distance from the point.

        Args:
            point: ... from where to compute distance
        """
        ...

    @abc.abstractmethod
    def calc_cell_centers(self) -> np.ndarray:
        """Calculate cell (voxel) centers."""
        ...

    @abc.abstractmethod
    def print_geom(self, io: TextIO, indent: str) -> None:
        """Print geometry specification.

        Args:
            io: stream to print to
            indent: indent to insert before lines
        """
        ...

    # Generic methods

    @property
    def bins_shape(self) -> Tuple[int, int, int]:
        """Shape of data corresponding to spatial bins.

        Returns:
            Tuple with the data shape.
        """
        return (self.ibins.size - 1), (self.jbins.size - 1), (self.kbins.size - 1)

    @property
    def bins_size(self) -> int:
        """Size of data corresponding to spatial bins.

        Returns:
            int: number of voxels
        """
        return (self.ibins.size - 1) * (self.jbins.size - 1) * (self.kbins.size - 1)

    @property
    def boundaries(self) -> np.ndarray:
        """Corners or min, max values of bins."""
        return np.vstack(
            (
                self.ibins[[0, -1]],
                self.jbins[[0, -1]],
                self.kbins[[0, -1]],
            ),
        )

    @property
    def boundaries_shape(self) -> Tuple[int, int, int]:
        """Bins (boundaries) shape."""
        return self.ibins.size, self.jbins.size, self.kbins.size

    def surrounds_point(self, x: float, y: float, z: float, local: bool = True) -> bool:
        """Check if the point (x,y,z) is within the volume of mesh.

        By default, assumes that the point is given in local coordinates.
        """
        if not local:
            x, y, z = self.local_coordinates(np.array([x, y, z], dtype=float))
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.boundaries
        return cast(bool, (xmin < x < xmax) and (ymin < y < ymax) and (zmin < z < zmax))

    def select_indexes(
        self, *, i_values=None, j_values=None, k_values=None
    ) -> Tuple[
        Union[int, slice, np.ndarray],
        Union[int, slice, np.ndarray],
        Union[int, slice, np.ndarray],
    ]:
        """Select indices for data corresponding to given spatial values.

        Args:
            i_values: indices along i (X or R) dimension
            j_values: ... along j (Y or Z)
            k_values: ... along k (Z or Theta)

        Returns:
            see :func:`select_indexes()`
        """
        return (
            select_indexes(self.ibins, i_values),
            select_indexes(self.jbins, j_values),
            select_indexes(self.kbins, k_values),
        )

    def print(self, io: TextIO, columns: int = 6) -> None:
        indent = " " * 8
        self.print_geom(io, indent)
        print(indent, "origin=", " ".join(format_floats(self.origin)), sep="", file=io)
        _print_bins(indent, "i", self.ibins, io, columns=columns)
        _print_bins(indent, "j", self.jbins, io, columns=columns)
        _print_bins(indent, "k", self.kbins, io, columns=columns)


class CartesianGeometrySpec(AbstractGeometrySpec):

    # TODO dvp: add transformation

    @property
    def cylinder(self) -> bool:
        return False

    @property
    def x(self) -> np.ndarray:
        return self.ibins

    @property
    def y(self) -> np.ndarray:
        return self.jbins

    @property
    def z(self) -> np.ndarray:
        return self.kbins

    def local_coordinates(self, points: np.ndarray) -> np.ndarray:
        assert points.shape[-1] == 3, "Expected cartesian point array or single point"
        if self.origin is not ZERO_ORIGIN:
            return cast(
                np.ndarray, points - ZERO_ORIGIN
            )  # TODO dvp: recall what is this subtraction for?
        else:
            return points

    def print_geom(self, io: TextIO, indent: str) -> None:
        pass  # Defaults will do for cartesian mesh

    def get_mean_square_distance_weights(self, point):
        ni, nj, nk = self.bins_shape

        def calc_sum(bins):
            bins_square = np.square(bins)
            bins_mult = bins[:-1] * bins[1:]
            bins_square = bins_square[:-1] + bins_square[1:] + bins_mult
            return bins_square

        x_square, y_square, z_square = [
            calc_sum(x - px) for x, px in zip((self.ibins, self.jbins, self.kbins), point)
        ]
        w = np.zeros((ni, nj, nk), dtype=float)
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    w[i, j, k] = x_square[i] + y_square[j] + z_square[k]
        w = (1.0 / 3.0) * w

        w = w * (1024.0 / np.max(w))

        return w

    def calc_cell_centers(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented method calc_cell_centers"
        )


@dataclass(eq=False)
class CylinderGeometrySpec(AbstractGeometrySpec):
    """Cylinder spec.

    Attributes:
        axs: cylinder axis
        vec: vector to measure angle (theta) from
    """

    axs: np.ndarray = field(default_factory=lambda: DEFAULT_AXIS.copy())
    vec: np.ndarray = field(default_factory=lambda: DEFAULT_VEC.copy())

    def __post_init__(self):
        super().__post_init__()

        if self.axs is not DEFAULT_AXIS:
            if not isinstance(self.axs, np.ndarray):
                raise TypeError(f"Expected axs as numpy array, actual {self.axs}")

        if self.vec is not DEFAULT_VEC:
            if not isinstance(self.vec, np.ndarray):
                raise TypeError(f"Expected vec as numpy array, actual {self.vec}")

        if not (self.theta[0] == 0.0 and self.theta[-1] == 1.0):
            raise ValueError("Theta is expected in rotations only")

    @property
    def bins(self):
        return super().bins + (self.axs, self.vec)

    @property
    def cylinder(self) -> bool:
        return True

    @property
    def r(self) -> np.ndarray:
        return self.ibins

    @property
    def z(self) -> np.ndarray:
        return self.jbins

    @property
    def theta(self) -> np.ndarray:
        return self.kbins

    def local_coordinates(self, points: np.ndarray) -> np.ndarray:
        assert points.shape[-1] == 3, "Expected cartesian point array or single point"
        assert np.array_equal(self.axs, DEFAULT_AXIS) and (
            np.array_equal(self.vec, DEFAULT_VEC) or self.vec[1] == 0.0  # vec is in xz plane
        ), "Tilted cylinder meshes are not implemented yet"
        # TODO dvp: implement tilted cylinder meshes
        # ez = self.axs / np.linalg.norm(self.axs)
        # ey = np.cross(ez, self.vec)
        # ey /= np.linalg.norm(ey)
        # ex = np.cross(ey, ez)
        # ex /= np.linalg.norm(ex)
        local_points: np.ndarray = points - self.origin
        local_points[..., :] = (
            np.sqrt(local_points[..., 0] ** 2 + local_points[..., 1] ** 2),  # r
            local_points[..., 2],  # z, just copy
            np.arctan2(local_points[..., 1], local_points[..., 0])
            * _1_TO_2PI,  # theta in rotations
        )
        return local_points

    # TODO dvp: add opposite method global_coordinates

    def print_geom(self, io: TextIO, indent: str) -> None:
        print(indent, "geom=cyl", sep="", file=io)
        print(
            indent,
            "axs=",
            " ".join(format_floats(self.axs)),
            "\n",
            indent,
            "vec=",
            " ".join(format_floats(self.vec)),
            sep="",
            file=io,
        )

    # noinspection SpellCheckingInspection
    def get_mean_square_distance_weights(self, point: np.ndarray) -> np.ndarray:
        ni, nj, nk = self.bins_shape
        assert self.vec is not None
        # Define synonyms for cylinder coordinates
        r = self.ibins  # radius
        phi = self.kbins
        assert phi[-1] == 1.0
        phi = phi * _2PI
        z = self.jbins
        px, py, pz = (
            point - self.origin
        )  # TODO dvp: apply local_coordinates instead of the following
        l1_square = px**2 + py**2
        l1 = np.sqrt(l1_square)  # distance to origin from point projection on z=0 plane
        assert 0.0 < l1
        # Terms of integration of L^2 in cylindrical coordinates
        # r^2
        gamma = np.arcsin(py / l1)
        r_square = np.square(r)
        r_square = 0.5 * (r_square[1:] + r_square[:-1])
        r_sum = r[1:] + r[:-1]
        r_mult = r[1:] * r[:-1]
        dphi = phi[1:] - phi[:-1]
        dsins = np.sin(phi - gamma)
        dsins = dsins[1:] - dsins[:-1]
        dsins = dsins / dphi
        z_minus_pz = z - pz
        z_minus_pz_square = np.square(z_minus_pz)
        z_sum = (1.0 / 3.0) * (
            z_minus_pz_square[1:] + z_minus_pz_square[:-1] + z_minus_pz[1:] * z_minus_pz[:-1]
        )
        w = np.zeros((ni, nj, nk), dtype=float)

        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    a = r_square[i]
                    b = (-4.0 / 3.0) * l1 * (r_sum[i] - r_mult[i] / r_sum[i]) * dsins[k]
                    d = z_sum[j]
                    w[i, j, k] = a + b + d
        w = w + l1_square
        w = w * (1024.0 / np.max(w))

        return w

    def calc_cell_centers(self) -> np.ndarray:
        _x0, _y0, _z0 = self.origin
        r_mids = (self.ibins[1:] + self.ibins[:-1]) * 0.5
        z_mids = (self.jbins[1:] + self.jbins[:-1]) * 0.5
        t_mids = (self.kbins[1:] + self.kbins[:-1]) * 0.5
        if self.kbins[-1] == 1.0:
            t_mids = t_mids * _2PI
        v2 = np.cross(self.axs, self.vec)
        v1 = np.cross(v2, self.axs)
        v2 /= linalg.norm(v2)
        v1 /= linalg.norm(v1)
        axs = self.axs / linalg.norm(self.axs)
        axs_z = np.dot(axs, NZ)

        def _aggregator(elements):
            r, z, fi = elements
            x, y = r * (v1 * np.cos(fi) + v2 * np.sin(fi))[0:2]
            x += _x0
            y += _y0
            z = axs_z * z + _z0
            return np.array([x, y, z], dtype=float)

        cell_centers: np.ndarray = cartesian_product(r_mids, z_mids, t_mids, aggregator=_aggregator)

        return cell_centers

    # def __hash__(self):
    #     return hash((super().__hash__(), self.axs, self.vec))
    #
    # def __eq__(self, other):
    #     if not isinstance(other, CylinderGeometrySpec):
    #         return False
    #     return (
    #         super().__eq__(other)
    #         and np.array_equal(self.axs, other.axs)
    #         and np.array_equal(self.vec, other.vec)
    #     )

    def adjust_axs_vec_for_mcnp(self) -> "CylinderGeometrySpec":
        """Set `axs` and `vec` attributes to the values, which MCNP considers orthogonal.

        Assumptions
        -----------

        Cylinder mesh is not tilted:
            - `self.vec` is in PY=0 plane
            - `self.axs` is vertical

        Returns
        -------
        gs:
            new CylinderGeometrySpec with adjusted `axs` and `vec` attributes.
        """
        # TODO dvp: fix for arbitrary axs and vec
        axs = self.origin + DEFAULT_AXIS * self.z[-1]
        vec = self.origin + DEFAULT_VEC * self.r[-1]
        return CylinderGeometrySpec(
            self.r, self.z, self.theta, origin=self.origin, axs=axs, vec=vec
        )


def _print_bins(indent, prefix, _ibins, io, columns: int = 6) -> None:
    intervals, coarse_mesh = compute_intervals_and_coarse_bins(_ibins)
    coarse_mesh = coarse_mesh[1:]  # drop the first value - it's presented with origin
    print(indent, f"{prefix}mesh=", sep="", end="", file=io)
    second_indent = indent + " " * 5
    mckit_meshes.utils.io.print_n(
        (f"{x:.6g}" for x in coarse_mesh),
        io=io,
        indent=second_indent,
        max_columns=columns,
    )
    print(indent, f"{prefix}ints=", sep="", end="", file=io)
    mckit_meshes.utils.io.print_n(intervals, io=io, indent=second_indent, max_columns=columns)


def select_indexes(
    a: np.ndarray, x: Optional[Union[float, List[float], np.ndarray]]
) -> Union[int, slice, np.ndarray]:
    """Find indexes for a mesh bin, corresponding given coordinates.

    Assumes that `a` is sorted.

    Examples:

        >>> r = np.arange(5)
        >>> r
        array([0, 1, 2, 3, 4])

        For x is None return slice over all `a` indexes.

        >>> select_indexes(r, None)
        slice(0, 5, None)

        For non specified x, if input array represents just one bin,
        then return index 0 to squeeze results.
        >>> select_indexes(np.array([10,20]), None)
        0

        For x = 1.5, we have 1 < 1.5 < 2, so the bin index is to be 1
        >>> select_indexes(r, 1.5)
        1

        For x = 0, it's the first bin, and index is to be 0
        >>> select_indexes(r, 0)
        0

        For coordinates below r[0] return -1.
        >>> select_indexes(r, -1)
        -1

        For coordinates above  r[-1] return a.size-1.
        >>> select_indexes(r, 5)
        4

        And for array of coordinates
        >>> select_indexes(r, np.array([1.5, 0, -1, 5]))
        array([ 1,  0, -1,  4])

    Args:
        a:  bin boundaries
        x: one or more coordinates along `a`-boundaries

    Returns:
        out: index or indices for each given coordinate
    """
    assert 1 < a.size, "Parameter a doesn't represent binning"

    if x is None:
        return slice(0, a.size) if 2 < a.size else 0  # squeeze if there's only one bin

    i: np.ndarray = a.searchsorted(x) - 1

    if np.isscalar(i):
        if i < 0:
            if x == a[0]:
                return 0
    else:
        neg = i < 0
        if np.any(neg):
            eq_to_min = a[0] == x
            i[np.logical_and(neg, eq_to_min)] = 0

    return i


def format_floats(floats: Iterable[float], _format="{:.6g}") -> Iterable[str]:
    yield from map(_format.format, floats)


def compute_intervals_and_coarse_bins(
    arr: Sequence[float], tolerance: float = 1.0e-4
) -> Tuple[List[int], Sequence[float]]:
    """Compute fine intervals and coarse binning.

    Examples:

    Find equidistant bins and report as intervals
    >>> arr = np.array([1, 2, 3, 4], dtype=float)
    >>> arr
    array([1., 2., 3., 4.])
    >>> intervals, coarse = compute_intervals_and_coarse_bins(arr)
    >>> intervals
    [3]
    >>> coarse
    [1.0, 4.0]

    A bins with two interval values.
    >>> arr = np.array([1, 2, 3, 6, 8, 10], dtype=float)
    >>> intervals, coarse = compute_intervals_and_coarse_bins(arr)
    >>> intervals
    [2, 1, 2]
    >>> coarse
    [1.0, 3.0, 6.0, 10.0]

    On zero (or negative tolerance) just use 1 intervals and return original array.
    >>> intervals, coarse = compute_intervals_and_coarse_bins(arr, tolerance=0.0)
    >>> intervals
    [1, 1, 1, 1, 1]
    >>> coarse is arr
    True

    Args:
        arr: actual bins
        tolerance: precision to distinguish intervals with

    Returns:
        Tuple: numbers of fine intervals between coarse bins, coarse binning
    """
    if tolerance <= 0.0:
        return [1] * (len(arr) - 1), arr
    fine_intervals = []
    coarse_bins = [arr[0]]
    d_old = arr[1] - arr[0]
    count = 0
    for i in range(1, len(arr)):
        d = arr[i] - arr[i - 1]
        if abs(d - d_old) < tolerance:
            count += 1
        else:
            d_old = d
            fine_intervals.append(count)
            coarse_bins.append(arr[i - 1])
            count = 1
    fine_intervals.append(count)
    coarse_bins.append(arr[-1])
    return fine_intervals, coarse_bins


def arrays_equal(arrays: Iterable[Tuple[np.ndarray, np.ndarray]]) -> bool:
    for a, b in arrays:
        if not (a is b or np.array_equal(a, b)):
            return False
    return True
