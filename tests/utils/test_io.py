from __future__ import annotations

from typing import TYPE_CHECKING, Any

from io import StringIO

import pytest

from mckit_meshes.utils import get_override_strategy, print_cols, print_n

if TYPE_CHECKING:
    from collections.abc import Iterable


@pytest.mark.parametrize(
    "seq, columns, expected, expected_last_col,  msg",
    [
        (range(5), 3, "0 1 2\n3 4", 2, "5 by 3"),
        (range(5), 5, "0 1 2 3 4\n", 0, "5 by 5"),  # inconsistent \n
        (range(5), 6, "0 1 2 3 4", 5, "5 by 6"),
    ],
)
def test_print_cols(
    seq: Iterable[Any], columns: int, expected: str, expected_last_col: int, msg: str
) -> None:
    buf = StringIO()
    last_col = print_cols(seq=seq, fid=buf, max_columns=columns)
    assert last_col == expected_last_col, "Last col: " + msg
    actual = buf.getvalue()
    assert actual == expected, "Text: " + msg


@pytest.mark.parametrize(
    "seq, indent, columns, expected,  msg",
    [
        (range(5), "    ", 3, "0 1 2\n    3 4\n", "5 by 3"),
        (range(5), "    ", 5, "0 1 2 3 4\n", "5 by 5"),  # inconsistent \n
        (range(5), "    ", 6, "0 1 2 3 4\n", "5 by 6"),
        ([], "", 2, "", "nothing to print"),
    ],
)
def test_print_n(seq: Iterable[Any], indent: str, columns: int, expected: str, msg: str) -> None:
    buf = StringIO()
    print_n(seq, buf, indent, columns)
    actual = buf.getvalue()
    assert actual == expected, "Text: " + msg


@pytest.mark.parametrize("exists", [True, False])
@pytest.mark.parametrize("override", [True, False])
def test_get_override_strategy(tmp_path, exists, override):
    out = tmp_path / "test_get_override_strategy.txt"
    if exists:
        out.touch()
        strategy = get_override_strategy(override=override)
        if override:
            actual = strategy(out)
            assert actual.exists()
        else:
            with pytest.raises(FileExistsError, match=str(out)):
                strategy(out)
    else:
        strategy = get_override_strategy(override=override)
        actual = strategy(out)
        assert not actual.exists()
