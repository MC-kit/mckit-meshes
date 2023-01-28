"""Shared utilities."""
from __future__ import annotations

from typing import Any, Callable, Iterable


def foreach(
    predicate: Callable[[Any], Any],
    iterable: Iterable[Any],
) -> None:
    """Applies `predicate` to all the items in `iterable`.

    Args:
        predicate: function to apply to an item
        iterable: sequence of items
    """
    for i in iterable:
        predicate(i)
