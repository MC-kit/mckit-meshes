from __future__ import annotations

from typing import Callable

import io

from functools import reduce
from itertools import product
from operator import mul

import numpy as np

from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal

import pytest

from mckit_meshes import wgtmesh
from mckit_meshes.utils.testing import a
from mckit_meshes.wgtmesh import WgtMesh, make_geometry_spec, parse_coordinates

DEFAULT_ORIGIN = np.zeros((3,), dtype=float)


@pytest.fixture(scope="module")
def weights_ijk() -> Callable[[float], WgtMesh]:
    """Weights vary along ijk dimensions.

    Single energy bin.

    Returns:
        Callable to build weights with given offset.
    """
    ebins = np.array([0.0, 20], dtype=float)
    ibins = np.linspace(0, 10, 3, endpoint=True, dtype=float)
    jbins = np.linspace(0, 20, 3, endpoint=True, dtype=float)
    kbins = np.linspace(0, 30, 3, endpoint=True, dtype=float)

    def build_weights(start_value: float) -> np.ndarray:
        isz, jsz, ksz = (x.size - 1 for x in [ibins, jbins, kbins])
        wgt_shape = (1, isz, jsz, ksz)
        result = np.zeros(wgt_shape, dtype=float)
        for ii, ji, ki in product(range(isz), range(jsz), range(ksz)):
            result[0, ii, ji, ki] = reduce(
                mul, [0, ii, ji, ki], 1.0
            )  # TODO dvp: too high difference in far voxels
        return result + start_value

    def call(start_value: float):
        return WgtMesh(
            make_geometry_spec(origin=DEFAULT_ORIGIN, ibins=ibins, jbins=jbins, kbins=kbins),
            [ebins],
            [build_weights(start_value)],
        )

    return call


@pytest.fixture(scope="module")
def weights_eijk() -> Callable[[float], WgtMesh]:
    """Weights vary along all dimensions.

    Returns:
        Callable to build weights with given offset.
    """
    ebins = np.linspace(0.0, 20, 5, endpoint=True, dtype=float)
    ibins = np.linspace(0, 10, 3, endpoint=True, dtype=float)
    jbins = np.linspace(0, 20, 3, endpoint=True, dtype=float)
    kbins = np.linspace(0, 30, 3, endpoint=True, dtype=float)

    def build_weights(start_value: float) -> np.ndarray:
        esz, isz, jsz, ksz = (x.size - 1 for x in [ebins, ibins, jbins, kbins])
        wgt_shape = (esz, isz, jsz, ksz)
        result = np.zeros(wgt_shape, dtype=float)
        for ei, ii, ji, ki in product(range(esz), range(isz), range(jsz), range(ksz)):
            result[ei, ii, ji, ki] = reduce(
                mul, [ei, ii, ji, ki], 1.0
            )  # TODO dvp: too high difference in far voxels
        return result + start_value

    def call(start_value: float):
        return WgtMesh(
            make_geometry_spec(origin=DEFAULT_ORIGIN, ibins=ibins, jbins=jbins, kbins=kbins),
            [ebins],
            [build_weights(start_value)],
        )

    return call


@pytest.fixture()
def wwinp(data) -> WgtMesh:
    filename = data / "wwinp"
    with open(filename) as stream:
        return WgtMesh.read(stream)


def test_trivial_constructor():
    origin = np.array([0.0, 0.0, -17.0])
    ebins = np.array([0.0, 15.0])
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    z = np.array([0.0, 1.0])
    w = np.array([[[[0.5]]]])
    m = WgtMesh(make_geometry_spec(origin, x, y, z), [ebins], [w])
    assert w == m._weights


def test_read_write(tmpdir, wwinp):
    assert len(wwinp.energies) == 2
    assert 16 == wwinp.energies[0].size
    assert 2 == wwinp.energies[1].size
    new_filename = tmpdir.join("wwinp-new")
    with open(new_filename, "w") as nf:
        wwinp.write(nf)
    with open(new_filename) as nf:
        m2 = WgtMesh.read(nf)
    assert wwinp == m2


@pytest.mark.parametrize(
    ["text", "expected"],
    [
        ("0.0  1.0 10.0 1.0 1.0 100.0 1.0", a(0.0, 10.0, 100.0)),
        ("0.0  2.0 10.0 1.0 1.0 100.0 1.0", a(0.0, 5.0, 10.0, 100.0)),
        ("0.0  3.0 10.0 1.0 1.0 100.0 1.0", a(0.0, 3.333333, 6.666667, 10.0, 100.0)),
    ],
)
def test_mesh_coordinate_parsing(text, expected):
    actual = parse_coordinates(text.split())
    assert_array_almost_equal(actual, expected, err_msg="Failed to parse coordinates " + text)


def test_constructor_from_lists():
    wgm = WgtMesh(
        make_geometry_spec(DEFAULT_ORIGIN, [0, 10], [0, 20], [0, 30]),
        [[0, 20], [0, 20]],
        [[[[[1]]]], [[[[10]]]]],
    )
    for b in [
        wgm.origin,
        wgm.ibins,
        wgm.jbins,
        wgm.kbins,
        wgm.neutron_weights,
        wgm.photon_weights,
    ]:
        assert isinstance(b, np.ndarray), f"{b} is not np.ndarray"
        assert b.dtype == float, f"{b} is not an array of floats"
    for b in [
        wgm.energies,
        wgm.weights,
    ]:
        assert isinstance(b, list), f"{b} is not a list"
        assert len(b) == 2, f"{b} should have two parts for neutrons and photons"
    assert np.array_equal(wgm.origin, np.zeros((3,), dtype=float))
    assert np.array_equal(wgm.ibins, np.array([0, 10], dtype=float))
    assert np.array_equal(wgm.jbins, np.array([0, 20], dtype=float))
    assert np.array_equal(wgm.kbins, np.array([0, 30], dtype=float))
    assert np.array_equal(wgm.energies, np.array([[0, 20], [0, 20]], dtype=float))
    assert np.array_equal(wgm.weights, np.array([[[[[1]]]], [[[[10]]]]], dtype=float))


def test_add_happy_path():
    am = WgtMesh(
        make_geometry_spec(DEFAULT_ORIGIN, [0, 10], [0, 20], [0, 30]),
        [[0, 20], [0, 20]],
        [[[[[1]]]], [[[[10]]]]],
    )
    bm = WgtMesh(
        make_geometry_spec(DEFAULT_ORIGIN, [0, 10], [0, 20], [0, 30]),
        [[0, 20], [0, 20]],
        [[[[[2]]]], [[[[20]]]]],
    )
    cm = am + bm
    assert np.array_equal(cm.weights, np.array([[[[[3]]]], [[[[30]]]]], dtype=float))


def test_add_bad_path():
    am = WgtMesh(
        make_geometry_spec(DEFAULT_ORIGIN, [0, 10], [0, 20], [0, 30]),
        [[0, 20], [0, 20]],
        [[[[[1]]]], [[[[10]]]]],
    )
    bm = WgtMesh(
        make_geometry_spec(DEFAULT_ORIGIN, [0, 10], [0, 20], [0, 50]),  # <-- kbins differ
        [[0, 20], [0, 20]],
        [[[[[2]]]], [[[[20]]]]],
    )
    with pytest.raises(AssertionError):
        am + bm


@pytest.mark.parametrize(
    ["nps", "weights", "expected"],
    [
        (1, a(0, 1), (a(0, 1, dtype=int), a(0, 1))),
        (1, a(0, 2), (a(0, 1, dtype=int), a(0, 0.5))),
        (1, a(1, 2), (a(1, 1, dtype=int), a(1, 0.5))),
    ],
)
def test_prepare_probabilities_and_nps(nps: int, weights: np.ndarray, expected: np.ndarray):
    actual = wgtmesh.prepare_probabilities_and_nps(nps, weights)
    for _a, _b in zip(actual, expected):
        assert_array_equal(_a, _b)


def test_merge(weights_eijk) -> None:
    am = weights_eijk(0.0)

    actual = WgtMesh.merge((am, 1.0), (am, 1.0))
    assert actual.wm == am
    assert actual.nps == 2

    bm = weights_eijk(100.0)

    actual = WgtMesh.merge((am, 2), (bm, 2))
    assert actual.wm.weights[0][-1, -1, -1, -1] == 4 / (2 / 103 + 2 / 3)
    assert 4 == actual.nps

    actual = WgtMesh.merge((am, 10), (bm, 2))
    assert_almost_equal(12 / (2 / 103 + 10 / 3), actual.wm.weights[0][-1, -1, -1, -1])
    assert 12 == actual.nps


def my_assert_array_equal(actual, expected):
    for i, (ai, ei) in enumerate(zip(actual, expected)):
        try:
            ai, ei = _a, _b = map(float, [ai, ei])  # noqa: PLW2901
        except ValueError:
            pass
        assert ai == ei, f"{i} - items are not equal: {ai} != {ei}"


def test_print_mcnp_generator_spec(data, wwinp):
    ios = io.StringIO()
    wwinp.print_mcnp_generator_spec(ios)
    actual = ios.getvalue()
    spec_filename = data / "wwinp-spec.txt"
    with open(spec_filename) as stream:
        expected = stream.read()
    my_assert_array_equal(actual.lower().split(), expected.lower().split())


def test_print_meshtal_spec(data, wwinp):
    ios = io.StringIO()
    wwinp.print_meshtal_spec(ios)
    actual = ios.getvalue()
    spec_filename = data / "meshtal-spec.txt"
    with open(spec_filename) as stream:
        expected = stream.read()
    my_assert_array_equal(actual.lower().split(), expected.lower().split())


def test_reciprocal(weights_eijk) -> None:
    am = weights_eijk(2.0)
    actual = am.reciprocal()
    assert_array_equal(actual.weights[0], 1.0 / am.weights[0])

    am = weights_eijk(0.0)
    actual = am.reciprocal()
    indices = actual.weights[0] != 0
    assert_array_equal(actual.weights[0][indices], 1.0 / am.weights[0][indices])
    assert_array_equal(am.weights[0] != 0, indices)


def test_normalize(weights_ijk) -> None:
    am: WgtMesh = weights_ijk(2.0)
    gs = am._geometry_spec
    normalization_point = np.average(gs.boundaries, axis=1)
    x, y, z = normalization_point
    ix, iy, iz = am._geometry_spec.select_indexes(i_values=x, j_values=y, k_values=z)
    am_value = am.weights[0][0, ix, iy, iz]
    assert am_value != 1.0
    actual = am.normalize(normalization_point)
    actual_value = actual.weights[0][0, ix, iy, iz]
    assert actual_value == 1.0


def test_drop_lower_energies(wwinp):
    initial_size = wwinp.energies[0].size
    actual = wwinp.drop_lower_energies(1.0)
    assert wwinp.energies[0].size == initial_size
    assert actual.energies[0].size == 5
    assert actual.energies[0][0] == 0.0
    assert actual.energies[0][1] == 1.0
    assert actual.weights[0].shape[0] == 4
    actual = wwinp.drop_lower_energies(0.5)
    assert actual.energies[0].size == 5
    assert actual.energies[0][0] == 0.0
    assert actual.energies[0][1] == 1.0
    actual = wwinp.drop_lower_energies(0.1)
    assert actual.energies[0].size == 6
    assert actual.energies[0][0] == 0.0
    assert actual.energies[0][1] == 0.1
    actual = wwinp.drop_lower_energies(1.0e-8)
    assert actual is wwinp
    actual = wwinp.drop_lower_energies(20.0)
    assert actual.energies[0].size == 2
    assert actual.weights[0].shape[0] == 1
    assert actual.energies[0][0] == 0.0
    assert actual.energies[0][1] == 20.0
