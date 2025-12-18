from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from mckit_meshes.plot import MATPLOTLIB_AVAILABLE
from mckit_meshes.utils.testing import a

if TYPE_CHECKING:
    from matplotlib.path import Path

if MATPLOTLIB_AVAILABLE:
    import mckit_meshes.plot.read_plotm_file as rpf
else:
    pytest.skip(
        reason='optional package matplotlib is not installed, run `uv pip install -e ".[plot]"\'',
        allow_module_level=True,
    )


def test_is_x_plane():
    assert rpf.is_x_plane(rpf.YZ)
    assert not rpf.is_x_plane(rpf.XY)


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
        for _ in rpf.scan_pages(fid):
            total_pages += 1
        assert total_pages == pages


@pytest.mark.parametrize(
    "path,lines",
    [
        ("plot/cube.ps", 6),
        ("plot/cube-wv.ps", 6),
    ],
)
def test_reads_simple_ps_files(data: Path, path: str, lines: int) -> None:
    _path = data / path
    page = rpf.load_all_pages(_path)[0]
    assert page.lines.shape[0] == lines, f"PS-file {path} should contain {lines} lines."


def test_first_page_from_contour_file_has_3_lines(test_file: Path) -> None:
    with test_file.open() as fid:
        page = next(rpf.scan_pages(fid))
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
        iterator = enumerate(rpf.scan_pages(fid))
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


_DEFAULT_PROBID = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
_DEFAULT_DATE = _DEFAULT_PROBID + dt.timedelta(minutes=5)


@pytest.mark.skipif(
    not MATPLOTLIB_AVAILABLE,
    reason='optional package matplotlib is not installed, run `uv pip install -e ".[plot]"\'',
)
@pytest.mark.parametrize(
    "msg,lines,basis,origin,expected,scale",
    [
        (
            "# bottom -> origin ",
            a(1875, 188, 1875, 1125).reshape(1, 2, 2),
            rpf.XZ,
            a(0, 0, 0),
            a(0, -100, 0, 0).reshape(1, 2, 2),
            a(100, 100),
        ),
        (
            "# PX=50 ",
            a(1875, 188, 1875, 1125).reshape(1, 2, 2),
            rpf.YZ,
            a(50.0, 0.0, 0.0),  # <-- x = 50
            a(0, -100, 0, 0).reshape(1, 2, 2),
            a(100, 100),
        ),
        (
            "# PY=50 ",
            a(1875, 188, 1875, 1125).reshape(1, 2, 2),
            rpf.XZ,
            a(0.0, 50.0, 0.0),  # <-- y = 50
            a(0, -100, 0, 0).reshape(1, 2, 2),
            a(100, 100),
        ),
        (
            "# PZ=50 ",
            a(1875, 188, 1875, 1125).reshape(1, 2, 2),
            rpf.XY,
            a(0.0, 0.0, 50.0),  # <-- z = 50
            a(0, -100, 0, 0).reshape(1, 2, 2),
            a(100, 100),
        ),
        (
            "# rotated around Z-axis ",
            a(1875, 188, 1875, 1125).reshape(1, 2, 2),
            np.array(
                [
                    [-0.980785, -0.19509, 0.0],
                    [-0.19509, 0.980785, 0.0],
                ]
            ),
            a(0.0, 0.0, 50.0),  # <-- z = 50
            a(0, -100, 0, 0).reshape(1, 2, 2),
            a(100, 100),
        ),
    ],
)
def test_convert_to_real_coordinates_with_arbitrary_basis_and_origin(
    msg: str,
    lines: np.ndarray,
    basis: np.ndarray,
    origin: np.ndarray,
    expected: np.ndarray,
    scale: np.ndarray,
) -> None:
    page = rpf.Page(
        lines=lines,
        basis=basis,
        origin=origin,
        extent=scale,
        date=_DEFAULT_DATE,
        title="xxx",
        probid=_DEFAULT_PROBID,
    )
    actual = page.lines
    norm = np.abs(actual).max()
    assert_array_almost_equal(expected / norm, actual / norm, decimal=3), msg
