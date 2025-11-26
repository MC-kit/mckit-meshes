from __future__ import annotations

from typing import TYPE_CHECKING

import datetime as dt

import numpy as np

from numpy.testing import assert_almost_equal, assert_array_almost_equal


import pytest

from  mckit_meshes.plot import MATPLOTLIB_AVAILABLE

from mckit_meshes.utils.testing import a

if TYPE_CHECKING:
    from matplotlib.path import Path

if MATPLOTLIB_AVAILABLE:
    import mckit_meshes.plot.read_plotm_file as rpf
else:
    pytest.mark.skip(
        reason="optional package matplotlib is not installed, run `uv pip install -e \".[plot]\"'"
    )


@pytest.fixture(scope="module")
def test_file(data: Path) -> Path:
    inp = data / "plotm-1.ps"
    assert inp.exists(), f"File {inp} should be available."
    return inp


@pytest.mark.parametrize(
    "path,pages",
    [
        ("ng-2.3.5.ps", 1),
        ("plotm-1.ps", 2),
    ],
)
def test_reads_two_pages_from_contour_file(data: Path, path: str, pages: int) -> None:
    path = data / path
    with path.open() as fid:
        total_pages = 0
        for _ in rpf.read(fid):
            total_pages += 1
        assert total_pages == pages


def test_first_page_from_contour_file_has_3_lines(test_file: Path) -> None:
    with test_file.open() as fid:
        page = next(rpf.read(fid))
        lines = page.lines
        assert lines.size == 12
        assert len(lines) == 3
        (from_x, from_y), (to_x, to_y) = lines[0]
        assert_almost_equal(from_x, -1.28)
        assert_almost_equal(from_y, 0.0)
        assert_almost_equal(to_x, 1.28)
        assert_almost_equal(to_y, 0.0)


def test_first_page_from_contour_file_meta_info_is_complete(test_file: Path) -> None:
    with test_file.open() as fid:
        iterator = enumerate(rpf.read(fid))
        i, page = next(iterator)
        assert i == 0
        expected_date = dt.datetime(2018, 8, 24, 19, 4, 46, tzinfo=dt.UTC)
        assert expected_date == page.date
        assert page.title == "hfs-reflectometer"
        expected_probid = dt.datetime(2018, 8, 24, 18, 20, 19, tzinfo=dt.UTC)
        assert expected_probid == page.probid
        expected_basis = rpf.YZ
        assert_array_almost_equal(expected_basis, page.basis)
        expected_origin = np.array([0, 0, 0], dtype=float)
        assert_array_almost_equal(expected_origin, page.origin)
        expected_extent = np.array([100, 100], dtype=float)
        assert_array_almost_equal(expected_extent, page.extent)
        i, page = next(iterator)
        assert i == 1
        expected_origin = np.array([591.49, 0.00, 495.00], dtype=float)
        assert_array_almost_equal(expected_origin, page.origin)
        with pytest.raises(StopIteration):
            next(iterator)


@pytest.mark.parametrize(
    "msg,lines,expected,x,y,scale",
    [
        (
            "# bottom -> origin ",
            a(1875, 188, 1875, 1125).reshape(1, 2, 2),
            a(0, -100, 0, 0).reshape(1, 2, 2),
            0.0,
            0.0,
            a(100, 100),
        ),
    ],
)
def test_convert_to_real_coordinates(
    msg: str, lines: np.ndarray, expected: np.ndarray, x: float, y: float, scale: np.ndarray
) -> None:
    page = rpf.Page(
        lines=lines,
        basis=rpf.XZ,
        origin=a(x, 0.0, y),
        extent=scale,
    )
    actual = page.lines
    norm = np.abs(actual).max()
    assert_array_almost_equal(expected / norm, actual / norm, decimal=3), msg
