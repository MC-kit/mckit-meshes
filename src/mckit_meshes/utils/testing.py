from typing import cast

import numpy as np

__all__ = ["a"]


def a(*args, dtype=float) -> np.ndarray:
    """Shorten typing in parametrized tests.

    Equivalent to np.ndarray([*args], dtype='dtype')`.

    Args:
        args:  sequence of numbers (any type) will be converted to the specified type.
        dtype: A type for the output array

    Returns:
        np.ndarray:  The ndarray with the given numbers and type.

    Examples:

    >>> a(1, 2, 3)
    array([1., 2., 3.])
    >>> a(1, 2, 3, dtype=np.int32)
    array([1, 2, 3], dtype=int32)
    >>> a(0, 1, 2, 3).reshape(2,2)
    array([[0., 1.],
           [2., 3.]])
    """
    return cast(np.ndarray, np.fromiter(args, dtype=dtype))


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
