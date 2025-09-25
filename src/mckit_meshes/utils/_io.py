"""Output utilities."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import sys

from pathlib import Path

from eliot import start_action

__LOG = logging.getLogger("mckit_meshes.io")

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from _typeshed import SupportsWrite


def ignore_existing_file_strategy(_: str | Path) -> None:
    """Do nothing if file exists."""


def raise_error_when_file_exists_strategy(path: str | Path) -> None:
    """Strategy to use when file exists.

    Args:
        path: path to check

    Raises:
        FileExistsError: if `path` exits.
    """
    path = Path(path)
    if path.exists():
        errmsg = f"""\
Cannot override existing file \"{path}\".
Please remove the file or specify --override option"""
        raise FileExistsError(errmsg)


def check_if_path_exists(*, override: bool) -> Callable[[str | Path], None]:
    """Select strategy to handle existing files, depending on option `override`.

    Args:
        override: if True ignore the case if file exists, otherwise rise Error

    Returns:
        The selected strategy.
    """
    return ignore_existing_file_strategy if override else raise_error_when_file_exists_strategy


def print_cols(
    seq: Iterable[Any],
    fid: SupportsWrite[str] = sys.stdout,
    max_columns: int = 6,
    fmt: str = "{}",
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
    i = 0
    for s in seq:
        if i > 0:
            print(" ", file=fid, end="")
        print(fmt.format(s), file=fid, end="")
        i += 1
        if i >= max_columns:
            print(file=fid)
            i = 0

    return i


def print_n(
    words: Iterable,
    io: SupportsWrite[str] = sys.stdout,
    indent: str = "",
    max_columns: int = 5,
) -> None:
    """Print sequence in columns with indentation starting from the second row.

    If anything was printed, add a newline.

    Args:
        words: sequence ot items to print
        io: where to print
        indent: indent to apply starting the second row
        max_columns: max number of columns in row
    """
    column = 0
    for w in words:
        if column == 0:
            to_print = f"{w}"
            column = 1
        elif column % max_columns == 0:
            to_print = f"\n{indent}{w}"
            column = 1
        else:
            to_print = f" {w}"
            column += 1
        print(to_print, end="", file=io)
    if column > 0:
        print(file=io)


def revise_files(ext: str, *files: Path) -> tuple[Path, ...]:
    """Find files by extension, if files are not specified.

    Log warning if files are neither specified nor found.

    Parameters
    ----------
    ext
        Extension to search for.

    Returns:
    -------
        Specified, if available, otherwise found.
    """
    if not files:
        with start_action(action_type="look for meshtally files") as logger:
            cwd = Path.cwd()
            files = tuple(cwd.glob(f"*.{ext}"))
            if not files:
                cwd = cwd.absolute()
                logger.log(message_type="WARNING", reason=f"No .{ext}-files found", directory=cwd)
                __LOG.warning("nothing to do: no .%s-files in %s", ext, cwd)
    return files
