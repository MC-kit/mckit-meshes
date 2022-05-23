"""Output utilities."""
from __future__ import annotations

import typing as t

from typing import Any, Iterable, TextIO

import sys

from pathlib import Path


def print_cols(
    seq: Iterable[Any], fid: TextIO = sys.stdout, max_columns: int = 6, fmt: str = "{}"
) -> int:
    """Print sequence in columns.

    Args:
        seq: sequence to print
        fid: output
        max_columns: max columns in a line
        fmt: format string

    Returns:
        int: the number of the last column printed on the last row

    """
    column = 0
    for s in seq:
        print(fmt.format(s), file=fid, end="")
        column += 1
        if column == max_columns:
            print(file=fid)
            column = 0
        else:
            print(" ", file=fid, end="")
    return column


def ignore_existing_file_strategy(_: str | Path) -> None:
    pass


def raise_error_when_file_exists_strategy(path: str | Path) -> None:
    path = Path(path)
    if path.exists():
        errmsg = f"""\
Cannot override existing file \"{path}\".
Please remove the file or specify --override option"""
        # raise click.UsageError(errmsg)
        raise FileExistsError(errmsg)


def check_if_path_exists(override: bool) -> t.Callable[[str | Path], None]:
    return (
        ignore_existing_file_strategy
        if override
        else raise_error_when_file_exists_strategy
    )
