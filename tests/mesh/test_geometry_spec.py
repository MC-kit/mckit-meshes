from __future__ import annotations

import numpy as np

from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal

import pytest

from mckit_meshes.mesh.geometry_spec import (
    DEFAULT_AXIS,
    DEFAULT_VEC,
    CartesianGeometrySpec,
    CylinderGeometrySpec,
    select_indexes,
)
from mckit_meshes.utils.testing import a


def test_cartesian_constructor():
    cartesian = CartesianGeometrySpec(ibins=a(1, 2, 3), jbins=a(4, 5, 6), kbins=a(7, 8, 9))
    assert not cartesian.cylinder
    assert isinstance(cartesian.x, np.ndarray)
    assert np.array_equal(a(1, 2, 3), cartesian.x)
    assert cartesian.y.dtype == float
    assert not hasattr(cartesian, "r")
    assert np.array_equal(a(7, 8, 9), cartesian.z)
    assert cartesian.bins_size == 8


def test_cylinder_constructor():
    cylinder = CylinderGeometrySpec(a(1, 2, 3), a(4, 5, 6), a(0, 0.5, 1), origin=a(1, 0, 0))
    assert cylinder.cylinder
    assert isinstance(cylinder.r, np.ndarray)
    assert np.array_equal(a(1, 2, 3), cylinder.r)
    assert isinstance(cylinder.origin, np.ndarray)
    assert np.array_equal(a(1, 0, 0), cylinder.origin)
    assert cylinder.axs.dtype == float
    assert not hasattr(cylinder, "x")
    assert_array_equal(a(4, 5, 6), cylinder.z)
    assert cylinder.bins_size == 8


def test_cylinder_constructor_with_wrong_theta():
    with pytest.raises(ValueError, match="Theta is expected in rotations only"):
        CylinderGeometrySpec(a(1, 2, 3), a(4, 5, 6), a(7, 8, 9), origin=a(1, 0, 0))


def test_cartesian_local_coordinates():
    cartesian = CartesianGeometrySpec(*np.arange(1, 10).reshape(3, 3))
    points = a(1, 2, 3, 4, 5, 6).reshape(2, 3)
    actual = cartesian.local_coordinates(points)
    assert_array_almost_equal(points, actual)


@pytest.mark.parametrize(
    "points,origin,expected",
    [
        (
            a(2, 2, 3),
            a(0, 0, 0),
            a(np.sqrt(8), 3, 45 / 360),  # x == y => 45 degrees
        ),
        (
            a(np.cos(30 * np.pi / 180), np.sin(30 * np.pi / 180), 6),
            a(0, 0, 0),
            a(1, 6, 30 / 360),  # 30 degrees
        ),
    ],
)
def test_cylinder_local_coordinates(points, origin, expected):
    cylinder = CylinderGeometrySpec(a(1, 2, 3), a(4, 5, 6), a(0, 0.5, 1), origin=origin)
    actual = cylinder.local_coordinates(points)
    assert_array_almost_equal(expected, actual)


def test_boundaries():
    gc = CartesianGeometrySpec(a(1, 2, 3), a(4, 5, 6), a(7, 8, 9))
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = gc.boundaries
    assert xmin == 1.0
    assert xmax == 3.0
    assert ymin == 4.0
    assert ymax == 6.0
    assert zmin == 7.0
    assert zmax == 9.0

    gc = CylinderGeometrySpec(a(1, 2, 3), a(4, 5, 6), a(0, 0.5, 1), origin=a(0, 0, 0))
    (rmin, rmax), (zmin, zmax), (tmin, tmax) = gc.boundaries
    assert rmin == 1.0
    assert rmax == 3.0
    assert zmin == 4.0
    assert zmax == 6.0
    assert tmin == 0.0
    assert tmax == 1.0


@pytest.mark.parametrize(
    "point,expected",
    [
        (a(2, 5, 8), True),
        (a(1, 5, 8), False),
    ],
)
def test_surrounds_point_cartesian(point, expected):
    gc = CartesianGeometrySpec(a(1, 2, 3), a(4, 5, 6), a(7, 8, 9))
    assert gc.surrounds_point(*point) == expected


@pytest.mark.parametrize(
    "point,expected",
    [
        (a(2, 2, 0), True),
        (a(2, 2, 2), False),
    ],
)
def test_surrounds_point_cylinder(point, expected):
    gc = CylinderGeometrySpec(a(1, 2, 3), a(4, 5, 6), a(0, 0.5, 1), origin=a(0, 0, -5))
    local_point = gc.local_coordinates(point)
    assert gc.surrounds_point(*local_point) == expected


def test_eq():
    gc1 = CartesianGeometrySpec(a(1, 2, 3), a(4, 5, 6), a(7, 8, 9))
    gc2 = CartesianGeometrySpec(a(1, 2, 3), a(4, 5, 6), a(7, 8, 9))
    gc3 = CartesianGeometrySpec(a(1, 2, 3), a(4, 5, 6), a(7, 8, 2))
    assert gc1 == gc2
    assert gc1 != gc3


def test_cylinder_mesh_trivial_constructor():
    origin = np.array([0.0, 0.0, -15.0])
    r = np.array([0.0, 1.0])
    z = np.array([0.0, 1.0])
    t = np.array([0.0, 1.0])
    axs = np.array([0.0, 0.0, 2000.0])
    vec = np.array([1680.4, -645.06, -1500.0])
    gc = CylinderGeometrySpec(r, z, t, origin=origin, axs=axs, vec=vec)
    cc = gc.calc_cell_centers()
    assert_almost_equal(cc, np.array([[[[-0.4667888, 0.1791876, -14.5]]]], dtype=float))


def test_adjust_axs_vec_for_mcnp():
    origin = np.array([0.0, 0.0, -15.0])
    r = np.array([0.0, 30.0])
    z = np.array([0.0, 40.0])
    t = np.array([0.0, 1.0])
    axs = DEFAULT_AXIS
    vec = DEFAULT_VEC
    gc = CylinderGeometrySpec(r, z, t, origin=origin, axs=axs, vec=vec)
    cc = gc.adjust_axs_vec_for_mcnp()
    assert_almost_equal(cc.axs, origin + DEFAULT_AXIS * z[-1])
    assert_almost_equal(cc.vec, origin + DEFAULT_VEC * r[-1])


@pytest.mark.parametrize(
    "inp,value,expected",
    [
        (a(0, 5, 10), -0.1, -1),
        (a(0, 5, 10), 0, 0),
        (a(0, 5, 10), 0.1, 0),
        (a(0, 5, 10), 2.5, 0),
        (a(0, 5, 10), 5, 0),
        (a(0, 5, 10), 5.5, 1),
        (a(0, 5, 10), 10, 1),
        (a(0, 5, 10), 100, 2),
    ],
)
def test_select_indices(inp, value, expected):
    actual = select_indexes(inp, value)
    assert expected == actual


@pytest.mark.parametrize(
    "inp,values,expected",
    [
        (a(0, 5, 10), [-1, 0, 2], [-1, 0, 0]),
        (a(0, 5, 10), [-1, 0, 6], [-1, 0, 1]),
        (a(0, 5, 10), [-1, 0, 10], [-1, 0, 1]),
        (a(0, 5, 10), [-1, 0, 11], [-1, 0, 2]),
        (a(0, 5, 10), [1, 2, 3], [0, 0, 0]),
    ],
)
def test_select_indices_with_arrays(inp, values, expected):
    actual = select_indexes(inp, values)
    assert np.array_equal(
        expected,
        actual,
    ), f"for {inp} and {values}, we expect {expected}, actual {actual}"
