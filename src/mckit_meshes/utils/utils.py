from typing import Any, Callable, Iterable, TextIO

import sys


def foreach(
    predicate: Callable[[Any], Any],
    iterable: Iterable[Any],
) -> None:
    """
    Applies `predicate` to all the items in `iterable`.
    """
    for i in iterable:
        predicate(i)


def print_n(words: Iterable, io: TextIO = None, indent=None, columns: int = 5) -> None:
    if io is None:
        io = sys.stdout
    if indent is None:
        indent = ""
    i = 0
    for w in words:
        if columns <= i:
            print("\n", indent, sep="", end="", file=io)
            i = 0
        if 0 < i:
            print(end=" ", file=io)
        print(w, end="", file=io)
        i = i + 1
    if 0 < i:
        print(file=io)
