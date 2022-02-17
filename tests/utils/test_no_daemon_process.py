import platform
import sys

from contextlib import closing
from functools import partial

import mckit_meshes.utils.no_daemon_process as ndp
import pytest

pytest_mark = pytest.mark.skipif((sys.platform == "win32"), reason="Fails on windows")


def foo(x, depth=0):
    if depth == 0:
        return x
    else:
        with closing(ndp.Pool()) as p:
            return p.map(partial(foo, depth=depth - 1), range(x + 1))


@pytest.mark.skipif(
    platform.system() != "Linux", reason="Some issues with pickle, if OS is not Linux."
)
def test_no_daemon_pool():
    actual = foo(10, depth=2)
    expected = [
        [0],
        [0, 1],
        [0, 1, 2],
        [0, 1, 2, 3],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5, 6],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ]
    assert actual == expected
