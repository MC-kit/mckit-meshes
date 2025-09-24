from __future__ import annotations

from typing import TYPE_CHECKING

from pathlib import Path
from rich.console import Console

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

_DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def data() -> Path:
    """Compute the path to test data.

    Returns:
        Path to test data.
    """
    return _DATA


@pytest.fixture
def cd_tmpdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Temporarily change to temp directory.

    Returns:
    --------
    Path: to temporary directory
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


class MemoryDestination(MemoryLogger):
    """Eliot memory logger."""

    def __call__(self, message):
        self.write(message)


@pytest.fixture
def eliot_file_trace() -> Callable[[Path | str], _GeneratorContextManager[None]]:
    """Eliot trace to file."""

    @contextmanager
    def _wrap(path: Path | str) -> Generator[None]:
        if isinstance(path, str):
            path = Path(path)
        with path.open("w") as fid:
            pth = FileDestination(fid)
            add_destinations(pth)
            try:
                yield
            finally:
                remove_destination(pth)

    return _wrap


@pytest.fixture
def eliot_mem_trace() -> Generator[MemoryDestination]:
    """Eliot trace to memory logger."""
    mem = MemoryDestination()
    add_destinations(mem)
    try:
        yield mem
    finally:
        remove_destination(mem)


@pytest.fixture
def cyclopts_runner(cd_tmpdir):
    def _wrapper(app, args, **kwargs):
        console = Console()
        with console.capture() as capture:
            result = app(args, console=console, **kwargs)
        return result, capture.get(), cd_tmpdir

    return _wrapper
