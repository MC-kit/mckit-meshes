from typing import Callable, Iterable, Sized

from itertools import product

import numpy as np


def convert_tuple_to_array(elements: Iterable, **kw) -> np.ndarray:
    if "dtype" in kw:
        dtype = kw["dtype"]
    else:
        dtype = np.result_type(*elements)
    return np.array(elements, dtype=dtype)


def cartesian_product(*arrays: Sized, aggregator: Callable, **kw) -> np.ndarray:
    """
    Computes transformations of cartesian product of all the elements in arrays.

    Parameters
    ----------

    arrays: iterable of Sized
        The arrays to product.

    aggregator:
        Callable to handle an item from product iterator.
        May return scalar or numpy ndarray.

    Returns
    -------
    ret:
        Array with dimension of arrays and one more dimension
        for their cartesian product.
    """
    res = np.stack([aggregator(x, **kw) for x in product(*arrays)])
    shape = tuple(map(len, arrays))
    if 1 < len(res.shape):
        shape = shape + res.shape[1:]
    return res.reshape(shape)
