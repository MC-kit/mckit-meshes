import os

from collections import namedtuple
from copy import copy
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from mckit_meshes.fmesh import FMesh, iter_meshtal, m_2_npz, merge_tallies, read_meshtal
from mckit_meshes.mesh.geometry_spec import CartesianGeometrySpec
from mckit_meshes.utils.resource import path_resolver
from mckit_meshes.utils.testing import a
from numpy.testing import assert_almost_equal, assert_array_equal


@pytest.fixture
def simple_bins():
    def call(
        *,
        name=14,
        kind=1,
        xbins=a(0, 1),
        ybins=a(2, 3),
        zbins=a(4, 5),
        ebins=a(0, 6, 7),
    ):
        return name, kind, xbins, ybins, zbins, ebins

    return call


def test_trivial_constructor(simple_bins):
    name, kind, xbins, ybins, zbins, ebins = simple_bins()
    data = [[[[5.0]]], [[[10.0]]]]
    errors = [[[[0.2]]], [[[0.1]]]]
    m = FMesh(
        name,
        kind,
        CartesianGeometrySpec(xbins, ybins, zbins),
        a(*ebins),
        np.asarray(data),
        np.asarray(errors),
    )
    assert name == m.name


def test_read_mesh_tall(tmp_path, simple_bins):
    name, kind, xbins, ybins, zbins, ebins = simple_bins()
    data = [[[[8]]], [[[10.0]]]]
    errors = [[[[0.2]]], [[[0.1]]]]
    m = FMesh(
        name,
        kind,
        CartesianGeometrySpec(xbins, ybins, zbins),
        a(*ebins),
        np.asarray(data),
        np.asarray(errors),
    )
    assert name == m.name
    tfp = tmp_path / "test_read_mesh_tall-1.m"
    with tfp.open("wt") as fid:
        fid.write("timestamp\n")
        fid.write("problem title\n")
        fid.write(
            "Number of histories used for normalizing tallies =      39594841.00\n\n"
        )
        m.save_2_mcnp_mesh(fid)
    actual = read_meshtal(tfp.open())
    assert len(actual) == 1
    actual = actual[0]
    assert actual == m
    # now use already opened file
    mesh_file_info = namedtuple("MFI", "nps")
    with tfp.open() as inp:
        actual = read_meshtal(inp, mesh_file_info=mesh_file_info)
        assert len(actual) == 1
        actual = actual[0]
        assert actual == m
        assert mesh_file_info.nps == 39594841


def test_merge_tallies(simple_bins):
    name, kind, xbins, ybins, zbins, ebins = simple_bins(name=18)
    data = np.array([[[[10.0]]], [[[10.0]]]])
    errors = np.array([[[[0.1]]], [[[0.1]]]])
    m1 = FMesh(
        name, kind, CartesianGeometrySpec(xbins, ybins, zbins), ebins, data, errors
    )
    m2 = copy(m1)
    # noinspection PyTypeChecker
    actual = merge_tallies(3, 1, (m1, 1.0), (m2, 2.0))
    expected_data = data * 3.0
    expected_errors = (
        np.sqrt((data * errors) ** 2 + (2 * data * errors) ** 2) / expected_data
    )
    expected = FMesh(
        3,
        1,
        CartesianGeometrySpec(xbins, ybins, zbins),
        ebins,
        expected_data,
        expected_errors,
    )
    assert actual == expected


def test_save2vtk(simple_bins, tmp_path):
    name, kind, xbins, ybins, zbins, ebins = simple_bins(name=22)
    data = np.array([[[[11.0]]], [[[11.0]]]])
    errors = np.array([[[[0.1]]], [[[0.1]]]])
    m = FMesh(
        name, kind, CartesianGeometrySpec(xbins, ybins, zbins), ebins, data, errors
    )
    tfn = tmp_path / "test_fmesh_.m"
    m.save2vtk(str(tfn))


def test_m_2_npz(tmp_path, simple_bins):
    name, kind, xbins, ybins, zbins, ebins = simple_bins(name=14)
    data = np.array([[[[7.0]]], [[[10.0]]]])
    errors = np.array([[[[0.1]]], [[[0.05]]]])
    m1 = FMesh(
        name, kind, CartesianGeometrySpec(xbins, ybins, zbins), ebins, data, errors
    )
    m2 = copy(m1)
    m2.name = 2
    tfn = str(tmp_path / "fmesh.m")
    with open(tfn, "w") as fid:
        fid.write("timestamp\n")
        fid.write("problem title\n")
        fid.write(
            "Number of histories used for normalizing tallies =      39594841.00\n\n"
        )
        m1.save_2_mcnp_mesh(fid)
        m2.save_2_mcnp_mesh(fid)
    # now use already opened file
    m_2_npz(Path(tfn).open(), lambda name_: name_ == 14)
    npz = "14.npz"
    assert os.path.exists(npz)
    assert not os.path.exists("2.npz")
    actual = FMesh.load_npz(npz)
    assert actual == m1
    os.remove(npz)
    prefix = str(tmp_path / "out/")
    m_2_npz(Path(tfn).open(), prefix=prefix)
    npz_index = {prefix + "14.npz": m1, prefix + "2.npz": m2}
    for npz in npz_index.keys():
        assert os.path.exists(npz)
        actual = FMesh.load_npz(npz)
        assert actual == npz_index[npz]
        os.remove(npz)


def test_m_2_npz_with_comment(tmp_path, simple_bins):
    name, kind, xbins, ybins, zbins, ebins = simple_bins(name=14)
    data = np.array([[[[7.0]]], [[[10.0]]]])
    errors = np.array([[[[0.1]]], [[[0.05]]]])
    comment = "test"
    m1 = FMesh(
        name,
        kind,
        CartesianGeometrySpec(xbins, ybins, zbins),
        ebins,
        data,
        errors,
        comment=comment,
    )
    m2 = copy(m1)
    m2.name = 2
    tfn = str(tmp_path / "fmesh.m")
    with open(tfn, "w") as fid:
        fid.write("timestamp\n")
        fid.write("problem title\n")
        fid.write(
            "Number of histories used for normalizing tallies =      39594841.00\n\n"
        )
        m1.save_2_mcnp_mesh(fid)
        m2.save_2_mcnp_mesh(fid)
    # now use already opened file
    m_2_npz(Path(tfn).open(), lambda name_: name_ == 14)
    npz = "14.npz"
    assert os.path.exists(npz)
    assert not os.path.exists("2.npz")
    actual = FMesh.load_npz(npz)
    assert actual == m1
    os.remove(npz)
    prefix = str(tmp_path / "out/")
    m_2_npz(Path(tfn).open(), prefix=prefix)
    npz_index = {prefix + "14.npz": m1, prefix + "2.npz": m2}
    for npz in npz_index.keys():
        assert os.path.exists(npz)
        actual = FMesh.load_npz(npz)
        assert actual == npz_index[npz]
        os.remove(npz)


def test_total_by_energy(simple_bins):
    name, kind, xbins, ybins, zbins, ebins = simple_bins(
        ebins=np.array([0.0, 6.0, 7.0, 8.0])
    )
    data = np.array([[[[10.0]]], [[[20.0]]], [[[30.0]]]])
    errors = np.array([[[[0.1]]], [[[0.2]]], [[[0.3]]]])
    m = FMesh(
        name, kind, CartesianGeometrySpec(xbins, ybins, zbins), ebins, data, errors
    )
    desired = FMesh(
        0,
        kind,
        CartesianGeometrySpec(xbins, ybins, zbins),
        np.array([0.0, 8.0]),
        np.array([[[[60.0]]]]),
        np.array(
            [[[[np.sqrt((10 * 0.1) ** 2 + (20 * 0.2) ** 2 + (30 * 0.3) ** 2) / 60.0]]]]
        ),
    )
    actual = m.total_by_energy(new_name=0)
    assert actual == desired


def test_get_totals(simple_bins):
    name, kind, xbins, ybins, zbins, ebins = simple_bins(
        name=34, ebins=np.array([0.0, 6.0, 7.0, 8.0])
    )
    data = np.array([[[[10.0]]], [[[20.0]]], [[[30.0]]]])
    errors = np.array([[[[0.1]]], [[[0.2]]], [[[0.3]]]])
    totals = np.asarray([[[60.0]]])
    totals_err = np.asarray(
        [[[np.sqrt((10 * 0.1) ** 2 + (20 * 0.2) ** 2 + (30 * 0.3) ** 2) / 60.0]]]
    )
    m = FMesh(
        name,
        kind,
        CartesianGeometrySpec(xbins, ybins, zbins),
        ebins,
        data,
        errors,
        totals,
        totals_err,
    )
    actual_totals, actual_totals_err = m.get_totals()
    desired_totals, desired_totals_err = totals.item(), totals_err.item()
    assert_array_equal(actual_totals, desired_totals)
    assert_array_equal(actual_totals_err, desired_totals_err)


@pytest.mark.skip
@pytest.mark.parametrize(
    "x, y, z, expected_total, expected_rel_error, msg",
    [
        (0.5, None, None, 60.0, 0.16499, "# 1: slice over x=0.5"),
        (1.5, None, None, 120.0, 0.16499, "# 2: slice over x=1.5"),
        (1.5, 2.5, None, 120.0, 0.16499, "# 3: slice over x=0.5, y=2.5"),
        (
            np.array([0.5, 1.5]),
            None,
            None,
            np.array([60.0, 120.0]),
            np.array([0.16499, 0.16499]),
            "# 4: slice over x=[0.5, 1.5]",
        ),
        (
            np.array([0.0, 1.0]),
            None,
            None,
            60.0,
            0.16499,
            "# 5: slice over x=[0.0, 1.0]",
        ),
        (
            np.array([0.0, 2.0]),
            None,
            None,
            np.array([60.0, 120.0]),
            np.array([0.16499, 0.16499]),
            "# 6: slice over x=[0.0, 2.0]",
        ),
    ],
)
def test_get_totals_slice(tmpdir, x, y, z, expected_total, expected_rel_error, msg):
    tf = tmpdir.join("gts.m")
    tf.write(_TEXT)
    tfn = str(tf)
    meshes = read_meshtal(Path(tfn).open())
    assert meshes, msg
    assert 1 == len(meshes), msg
    m = meshes[0]
    assert 14 == m.name
    actual_total, actual_rel_error = m.get_totals(x=x, y=y, z=z)
    assert_array_equal(expected_total, actual_total)
    assert_array_equal(expected_rel_error, actual_rel_error)


_TEXT = """timestamp
problem title
Number of histories used for normalizing tallies =      39594841.00

Mesh Tally Number   14
 This is a photon mesh tally.

 Tally bin boundaries:
X direction: 0 1 2
Y direction: 2 3
Z direction: 4 5
Energy bin boundaries: 0 6 7 8

 Energy          X         Y         Z      Result   Rel Error
  6.000e+00    0.500     2.500     4.500 1.00000e+01 1.00000e-01
  6.000e+00    1.500     2.500     4.500 2.00000e+01 1.00000e-01
  7.000e+00    0.500     2.500     4.500 2.00000e+01 2.00000e-01
  7.000e+00    1.500     2.500     4.500 4.00000e+01 2.00000e-01
  8.000e+00    0.500     2.500     4.500 3.00000e+01 3.00000e-01
  8.000e+00    1.500     2.500     4.500 6.00000e+01 3.00000e-01
   Total       0.500     2.500     4.500 6.00000e+01 0.16499e+00
   Total       1.500     2.500     4.500 1.20000e+02 0.16499e+00
"""


@pytest.mark.parametrize(
    "new_x, new_y, new_z, expected_data, expected_err, expected_total, expected_rel_error",
    [
        (
            # new binning with common edges with the old one
            np.array([0.5, 1.0, 1.5]),
            np.array([2.0, 3.0]),
            np.array([4.0, 5.0]),
            np.array(
                [
                    [[[10.0]], [[20.0]]],
                    [[[20.0]], [[40.0]]],
                    [[[30.0]], [[60.0]]],
                ]
            ),
            np.array(
                [
                    [[[0.1]], [[0.1]]],
                    [[[0.2]], [[0.2]]],
                    [[[0.3]], [[0.3]]],
                ]
            ),
            np.array(
                [
                    [[60.0]],
                    [[120.0]],
                ]
            ),
            np.array(
                [
                    [[0.16499]],
                    [[0.16499]],
                ]
            ),
        ),
        (
            # new binning with one bin crossing edge of the old binning
            np.array([0.5, 1.5]),
            np.array([2.0, 3.0]),
            np.array([4.0, 5.0]),
            np.array(
                [
                    [[[15.0]]],
                    [[[30.0]]],
                    [[[45.0]]],
                ]
            ),
            np.array(
                [
                    [[[0.1]]],
                    [[[0.2]]],
                    [[[0.3]]],
                ]
            ),
            np.array(
                [
                    [[90.0]],
                ]
            ),
            np.array(
                [
                    [[0.16499]],
                ]
            ),
        ),
    ],
)
def test_rebin(
    tmpdir,
    new_x,
    new_y,
    new_z,
    expected_data,
    expected_err,
    expected_total,
    expected_rel_error,
):
    tf = tmpdir.join("gts.m")
    tf.write(_TEXT)
    tfn = str(tf)
    meshes = read_meshtal(Path(tfn).open())
    assert meshes
    assert 1 == len(meshes)
    m = meshes[0]
    new_mesh = m.rebin(new_x, new_y, new_z, new_name=20)
    assert 20 == new_mesh.name
    assert_array_equal(new_x, new_mesh.ibins)
    assert_array_equal(new_y, new_mesh.jbins)
    assert_array_equal(new_z, new_mesh.kbins)
    assert_array_equal(expected_data, new_mesh.data)
    assert_array_equal(expected_err, new_mesh.errors)
    assert_array_equal(expected_total, new_mesh.totals)
    assert_almost_equal(expected_rel_error, new_mesh.totals_err)


@pytest.mark.parametrize(
    "msg, emin, emax, xmin, xmax, ymin, ymax, zmin, zmax, expected_mesh",
    [
        (
            "# select the first layer by x",
            None,
            None,  # e
            0.1,
            0.9,  # x
            None,
            None,  # y
            None,
            None,  # z
            FMesh(
                20,
                2,
                CartesianGeometrySpec(a(0, 1), a(2, 3), a(4, 5)),
                a(0, 6, 7, 8),  # name, kind, x, y, z bins
                a(10, 20, 30).reshape((3, 1, 1, 1)),  # data
                a(0.1, 0.2, 0.3).reshape((3, 1, 1, 1)),  # errors
                a(60).reshape((1, 1, 1)),  # totals
                a(0.16499).reshape((1, 1, 1)),  # total errors
            ),
        ),
    ],
)
def test_shrink(
    tmp_path, msg, emin, emax, xmin, xmax, ymin, ymax, zmin, zmax, expected_mesh
):
    text = """timestamp
problem title
Number of histories used for normalizing tallies =      39594841.00

Mesh Tally Number   14
 This is a photon mesh tally.

 Tally bin boundaries:
X direction: 0 1 2
Y direction: 2 3
Z direction: 4 5
Energy bin boundaries: 0 6 7 8

 Energy          X         Y         Z      Result   Rel Error
  6.000e+00    0.500     2.500     4.500 1.00000e+01 1.00000e-01
  6.000e+00    1.500     2.500     4.500 2.00000e+01 1.00000e-01
  7.000e+00    0.500     2.500     4.500 2.00000e+01 2.00000e-01
  7.000e+00    1.500     2.500     4.500 4.00000e+01 2.00000e-01
  8.000e+00    0.500     2.500     4.500 3.00000e+01 3.00000e-01
  8.000e+00    1.500     2.500     4.500 6.00000e+01 3.00000e-01
   Total       0.500     2.500     4.500 6.00000e+01 0.16499e+00
   Total       1.500     2.500     4.500 1.20000e+02 0.16499e+00
"""
    tf = tmp_path / "gts.m"
    tf.write_text(text)
    meshes = read_meshtal(tf.open())
    assert meshes
    assert 1 == len(meshes)
    m = meshes[0]
    new_mesh = m.shrink(emin, emax, xmin, xmax, ymin, ymax, zmin, zmax, new_name=20)
    assert expected_mesh == new_mesh


def test_repr(simple_bins):
    name, kind, xbins, ybins, zbins, ebins = simple_bins()
    data = np.asarray([[[[5.0]]], [[[10.0]]]], dtype=float)
    errors = np.asarray([[[[0.2]]], [[[0.1]]]], dtype=float)
    m = FMesh(
        name, kind, CartesianGeometrySpec(xbins, ybins, zbins), a(*ebins), data, errors
    )
    assert "Fmesh(14, 1, 0.0..1.0, 2.0..3.0, 4.0..5.0, 0.0..7.0)" == repr(m)


def test_reading_mfile_with_negative(data):
    data_path = data / "with_negatives.m"

    with data_path.open() as fid:
        m1, m2 = iter_meshtal(fid)
        assert (
            m1.name == 1355114 and m2.name == 1355214
        ), "reads files with negative values OK"
        assert (
            m1.data[0, 0, 0, 0] == 0.0 and m1.errors[0, 0, 0, 0] == 0.0
        ), "Should convert entries with negative values to zeroes"
