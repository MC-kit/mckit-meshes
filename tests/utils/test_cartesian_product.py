import numpy as np

from numpy.testing import assert_array_equal

import pytest

from mckit_meshes.utils.cartesian_product import cartesian_product
from mckit_meshes.utils.testing import a


@pytest.mark.parametrize(
    "arrays,expected",
    [
        (([1, 2], [3, 4]), [[[1, 3], [1, 4]], [[2, 3], [2, 4]]]),
        (
            ([1, 2], [3, 4], [5, 6]),
            [
                [[[1, 3, 5], [1, 3, 6]], [[1, 4, 5], [1, 4, 6]]],
                [[[2, 3, 5], [2, 3, 6]], [[2, 4, 5], [2, 4, 6]]],
            ],
        ),
    ],
)
def test_cartesian_product_with_inner_vector(arrays, expected):
    actual = cartesian_product(*[np.array(x) for x in arrays], aggregator=lambda x: a(*x))
    assert_array_equal(actual, np.array(expected))


@pytest.mark.parametrize(
    "arrays,expected",
    [
        (([1, 2], [3, 4]), [[4, 5], [5, 6]]),
        (
            ([1, 2], [3, 4], [5, 6]),
            [
                [[9, 10], [10, 11]],
                [[10, 11], [11, 12]],
            ],
        ),
    ],
)
def test_cartesian_product_with_sum(arrays, expected):
    actual = cartesian_product(*[np.array(x) for x in arrays], aggregator=sum)
    assert_array_equal(actual, np.array(expected))
