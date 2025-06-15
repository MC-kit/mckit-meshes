from __future__ import annotations

import sys

import numpy as np

from numpy.testing import assert_array_equal

import pytest

from mckit_meshes.utils.rebin import (
    interpolate,
    rebin_1d,
    rebin_nd,
    rebin_spec_composer,
    shrink_1d,
    shrink_nd,
    trim_spec_composer,
)
from mckit_meshes.utils.testing import a

skip_windows = pytest.mark.skipif((sys.platform == "win32"), reason="Fails on windows")


# noinspection PyTypeChecker
@pytest.mark.parametrize(
    "msg,x_new,x,y,axis,expected_y",
    [
        (
            "# one point",
            a(0.7),
            a(0, 1),
            a(0, 2),
            None,
            a(1.4),
        ),
        (
            "# two point",
            a(0.2, 0.7),
            a(0, 1),
            a(0, 2),
            None,
            a(0.4, 1.4),
        ),
        (
            "# with points on edges",
            a(0, 0.5, 1.0),
            a(0, 1),
            a(0, 2),
            None,
            a(0, 1.0, 2.0),
        ),
        (
            "# 2-D over rows",
            a(0.0, 0.25, 0.5, 0.75, 1.0),
            a(0, 1),
            a(0, 1, 2, 3).reshape((2, 2)),
            0,
            np.array(
                [
                    [0.0, 1.0],
                    [0.5, 1.5],
                    [1.0, 2.0],
                    [1.5, 2.5],
                    [2.0, 3.0],
                ],
                dtype=float,
            ),
        ),
        (
            "# 2-D over max_columns",
            np.linspace(0.0, 1.0, 5),  # x_new
            a(0, 1),  # x
            np.linspace(0.0, 3.0, 4).reshape((2, 2)),  # y [[0, 1], [2, 3]]
            1,  # axis
            np.array([[0.0, 0.25, 0.5, 0.75, 1.0], [2.0, 2.25, 2.5, 2.75, 3.0]]),
        ),
    ],
)
def test_interpolate(msg, x_new, x, y, axis, expected_y):
    assert x.ndim == 1, msg
    assert x_new.ndim == 1
    if axis is not None:
        assert x.size == y.shape[axis]
        new_shape = list(y.shape)
        new_shape[axis] = x_new.size
        assert tuple(new_shape) == expected_y.shape
    actual_y = interpolate(x_new, x, y, axis)
    assert_array_equal(expected_y, actual_y)


@pytest.mark.parametrize(
    "msg,array,bins,left,right,axis,expected_bins,expected_data",
    [
        (
            "# shrink to the left bin",
            a(0, 1),
            a(0, 1, 2),
            0.0,
            1.0,
            None,
            a(0, 1),
            a(0),
        ),
        (
            "# shrink to the right bin",
            a(0, 1),
            a(0, 1, 2),
            1.0,
            2.0,
            None,
            a(1, 2),
            a(1),
        ),
        (
            "# shrink to the middle bin with exact edges",
            a(0, 1, 2),
            a(0, 1, 2, 3),
            1.0,
            2.0,
            None,
            a(1, 2),
            a(1),
        ),
        (
            "# shrink to the middle bin with not exact edges",
            a(0, 1, 2),
            a(0, 1, 2, 3),
            1.0 + 0.01,
            2.0 - 0.01,
            None,
            a(1, 2),
            a(1),
        ),
        (
            "# 2D array shrink to the middle bin with not exact edges over axis 0",
            np.array(
                [
                    [0.0, 1, 2],
                    [1.0, 2, 3],
                    [3.0, 4, 5],
                ],
            ),
            a(0, 1, 2, 3),
            1.0 + 0.01,
            2.0 - 0.01,
            0,
            a(1, 2),
            np.array(
                [
                    [1, 2, 3],
                ],
            ),
        ),
        (
            "# 2D array shrink to the middle bin with not exact edges over axis 1",
            np.array(
                [
                    [0.0, 1, 2],
                    [1.0, 2, 3],
                    [3.0, 4, 5],
                ],
            ),
            a(0, 1, 2, 3),
            1.0 + 0.01,
            2.0 - 0.01,
            1,
            a(1, 2),
            np.array(
                [
                    [1],
                    [2],
                    [4],
                ],
            ),
        ),
    ],
)
def test_shrink_1d(msg, array, bins, left, right, axis, expected_bins, expected_data):
    actual_bins, actual_data = shrink_1d(array, bins, left, right, axis)
    assert_array_equal(expected_bins, actual_bins, err_msg=msg)
    assert_array_equal(expected_data, actual_data, err_msg=msg)


# noinspection PyTypeChecker
@pytest.mark.parametrize(
    "msg,array,trim_spec,expected_bins,expected_data",
    [
        (
            "# 2D array shrink to the middle bin with not exact edges",
            np.array(
                [
                    [0.0, 1, 2],
                    [1.0, 2, 3],
                    [3.0, 4, 5],
                ],
            ),
            trim_spec_composer(
                [a(0, 1, 2, 3), a(0, 1, 2, 3)],
                [1.0 + 0.01, 1.0 + 0.01],
                [
                    2.0 - 0.01,
                    2.0 - 0.01,
                ],
            ),
            [np.array([1.0, 2]), np.array([1.0, 2])],
            np.array([[2.0]]),
        ),
        (
            "# 2D array shrink to the top left two bins with not exact edges",
            np.array(
                [
                    [0.0, 1, 2],
                    [1.0, 2, 3],
                    [3.0, 4, 5],
                ],
            ),
            trim_spec_composer(
                [a(0, 1, 2, 3), a(0, 1, 2, 3)],
                [0, 0],
                [
                    1.0 - 0.01,
                    2.0 - 0.01,
                ],
            ),
            [np.array([0.0, 1]), np.array([0.0, 1.0, 2.0])],
            np.array([[0.0, 1.0]]),
        ),
    ],
)
def test_shrink_nd(msg, array, trim_spec, expected_bins, expected_data):
    actual_bins, actual_data = shrink_nd(array, trim_spec)
    assert len(expected_bins) == len(actual_bins), msg
    for eb, ab in zip(expected_bins, actual_bins, strict=False):
        assert_array_equal(eb, ab)
    assert_array_equal(expected_data, actual_data)


@pytest.mark.parametrize(
    "rebinned_data,data,bins,new_bins,axis,grouped",
    [
        (
            # rebin to more frequent bins
            np.array([0.0, 0.5, 0.5]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 0.5, 1.5, 2.0]),
            0,
            True,
        ),
        (
            # rebin to less frequent bins
            np.array([0.5, 0.5]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.5, 1.5, 2.0]),
            np.array([0.0, 1.0, 2.0]),
            0,
            True,
        ),
        (
            np.array([0.5, 0.5]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 1, 3, 4.0]),
            np.array([0.0, 2.0, 4.0]),
            0,
            True,
        ),
        (
            # rebin to more frequent bins, not grouped
            np.array([0.0, 0.5, 1]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 0.5, 1.5, 2.0]),
            0,
            False,
        ),
        (
            np.array([0.5, 0.5]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.5, 1.5, 2.0]),
            np.array([0.0, 1.0, 2.0]),
            0,
            False,
        ),
        (
            np.array([0.5, 0.5]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 1, 3, 4.0]),
            np.array([0.0, 2.0, 4.0]),
            0,
            False,
        ),
        (
            np.array([0.5, 0.5]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 1, 3, 4.0]) * 11,
            np.array([0.0, 2.0, 4.0]) * 11,  # to have all the bin widths not equal ones
            0,
            False,
        ),
    ],
)
def test_rebin_1d(rebinned_data, data, bins, new_bins, axis, grouped):
    actual_rebinned_data = rebin_1d(data, bins, new_bins, axis, grouped=grouped)
    assert_array_equal(rebinned_data, actual_rebinned_data)


# noinspection PyTypeChecker
@pytest.mark.parametrize(
    "data,rebin_spec,rebinned_data",
    [
        (
            # rebin to more frequent bins
            np.array([[0.0, 1.0], [1.0, 2.0]]),
            rebin_spec_composer(
                [np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])],
                [
                    np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
                    np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
                ],
                grouped_flags=True,
            ),
            np.array(
                [
                    [0.0, 0.0, 0.25, 0.25],
                    [0.0, 0.0, 0.25, 0.25],
                    [0.25, 0.25, 0.5, 0.5],
                    [0.25, 0.25, 0.5, 0.5],
                ],
            ),
        ),
        (
            # rebin to less frequent bins (reverse from above)
            np.array(
                [
                    [0.0, 0.0, 0.25, 0.25],
                    [0.0, 0.0, 0.25, 0.25],
                    [0.25, 0.25, 0.5, 0.5],
                    [0.25, 0.25, 0.5, 0.5],
                ],
            ),
            rebin_spec_composer(
                [
                    np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
                    np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
                ],
                [np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])],
                grouped_flags=True,
            ),
            np.array([[0, 1], [1, 2]], dtype=float),
        ),
        (
            # rebin to inner more frequent bins
            np.array([[0, 1], [1, 2]], dtype=float),
            rebin_spec_composer(
                [np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])],
                [
                    np.array([0.25, 0.5, 1.0, 1.5, 1.75]),
                    np.array([0.25, 0.5, 1.0, 1.5, 1.75]),
                ],
                grouped_flags=True,
            ),
            np.array(
                [
                    [0, 0, 1 / 8, 1 / 16],
                    [0, 0, 1 / 4, 1 / 8],
                    [1 / 8.0, 1 / 4, 2 / 4, 2 / 8],
                    [1 / 16, 1 / 8, 2 / 8, 2 / 16],
                ],
                dtype=float,
            ),
        ),
        (
            # rebin to inner more frequent bins, not grouped
            np.array([[0, 1], [1, 2]], dtype=float),
            rebin_spec_composer(
                [np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])],
                [
                    np.array([0.25, 0.5, 1.0, 1.5, 1.75]),
                    np.array([0.25, 0.5, 1.0, 1.5, 1.75]),
                ],
            ),
            np.array([[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 2, 2], [1, 1, 2, 2]], dtype=float),
        ),
        (
            # rebin by the second axis
            np.array([[0, 1], [1, 2]], dtype=float),
            rebin_spec_composer(
                [np.array([0.0, 1.0, 2.0])],
                [np.array([0.25, 0.5, 1.0, 1.5, 1.75])],
                axes=[1],
            ),
            np.array([[0, 0, 1, 1], [1, 1, 2, 2]], dtype=float),
        ),
        (
            # rebin by the second axis, grouped
            np.array([[0, 1], [1, 2]], dtype=float),
            rebin_spec_composer(
                [np.array([0.0, 1.0, 2.0])],
                [np.array([0.25, 0.5, 1.0, 1.5, 1.75])],
                axes=[1],
                grouped_flags=True,
            ),
            np.array([[0, 0, 1 / 2, 1 / 4], [1 / 4, 1 / 2, 2 / 2, 2 / 4]], dtype=float),
        ),
    ],
)
def test_rebin_nd(data, rebin_spec, rebinned_data):
    actual_rebinned_data = rebin_nd(data, rebin_spec)
    assert_array_equal(rebinned_data, actual_rebinned_data)


# noinspection PyTypeChecker
@skip_windows
@pytest.mark.parametrize(
    "data,rebin_spec,rebinned_data",
    [
        (
            # rebin by the second axis, grouped
            np.array([[0, 1], [1, 2]], dtype=float),
            [
                *rebin_spec_composer(
                    [np.array([0.0, 1.0, 2.0])],
                    [np.array([0.25, 0.5, 1.0, 1.5, 1.75])],
                    axes=[1],
                    grouped_flags=True,
                ),
            ],  # check also passing the spec as list
            np.array([[0, 0, 1 / 2, 1 / 4], [1 / 4, 1 / 2, 2 / 2, 2 / 4]], dtype=float),
        ),
    ],
)
def test_rebin_nd_in_external_process(data, rebin_spec, rebinned_data):
    actual_rebinned_data = rebin_nd(data, rebin_spec, external_process_threshold=1)
    assert_array_equal(rebinned_data, actual_rebinned_data)
