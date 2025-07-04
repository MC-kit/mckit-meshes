"""Apply function to cartesian product of arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from itertools import product

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike, NDArray


# noinspection PyUnresolvedReferences
def cartesian_product(
    *arrays: ArrayLike,
    aggregator: Callable = lambda x: np.array(x),
    **kw: Any,  # noqa: ANN401
) -> NDArray:
    """Computes transformations of cartesian product of all the elements in arrays.

    Args:
        arrays:  The arrays to product.
        aggregator: Callable to handle an item from product iterator.
            The first parameter of the callable is tuple of current product item.
            May return scalar or numpy ndarray.
        kw: keyword arguments to pass to aggregator

    Examples:
        >>> a = [1, 2, 3]
        >>> b = [4, 5, 6]
        >>> cartesian_product(a, b, aggregator=lambda x: x[0] * x[1])
        array([[ 4,  5,  6],
               [ 8, 10, 12],
               [12, 15, 18]])

        >>> cartesian_product(a, b)
        array([[[1, 4],
                [1, 5],
                [1, 6]],
               [[2, 4],
                [2, 5],
                [2, 6]],
               [[3, 4],
                [3, 5],
                [3, 6]]])

    Returns:
        ret: Numpy array with dimension of arrays and
            additional dimensions for their cartesian product.
    """
    res = np.stack([aggregator(x, **kw) for x in product(*arrays)])
    shape = tuple(map(len, arrays))
    if len(res.shape) > 1:  # the aggregation result is vector
        shape = shape + res.shape[1:]
    return cast("NDArray", res.reshape(shape))
