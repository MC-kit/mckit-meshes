from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from attr import dataclass
import pytest

from click.testing import CliRunner
from eliot import FileDestination, MemoryLogger, add_destinations, remove_destination
from loguru import logger
from rich.console import Console

if TYPE_CHECKING:
    # noinspection PyCompatibility
    from collections.abc import Generator

    from _pytest.logging import LogCaptureFixture


@pytest.fixture
def caplog(caplog: LogCaptureFixture) -> Generator[LogCaptureFixture, None, None]:
    """Fixture to capture loguru logging.

    Emitting logs from loguru's logger.log means that they will not show up in
    caplog which only works with Python's standard logging. This adds the same
    LogCaptureHandler being used by caplog to hook into loguru.

    Args:
        caplog (LogCaptureFixture): caplog fixture

    See Also:
        https://github.com/mcarans/pytest-loguru/blob/main/src/pytest_loguru/plugin.py
        https://florian-dahlitz.de/articles/logging-made-easy-with-loguru

    Yields:
        LogCaptureFixture
    """
    handler_id = logger.add(caplog.handler, format="{message} {extra}")
    yield caplog
    logger.remove(handler_id)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def cyclopts_runner(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def _wrapper(app, args, **kwargs):
        console = Console()
        with console.capture() as capture:
            result = app(args, console=console, **kwargs)
        return result, capture.get()

    return _wrapper


class MemoryDestination(MemoryLogger):
    """Eliot memory logger."""

    def __call__(self, message):
        self.write(message)


@pytest.fixture
def eliot_trace():
    # noinspection PyUnreachableCode
    @contextmanager
    def _wrap(
        *, path: Path | None = None, use_memory_logger: bool = False
    ) -> Generator[MemoryLogger, None, None]:
        if path:
            fid = path.open("w")
            pth = FileDestination(fid)
        else:
            fid = None
            pth = None
        mem = MemoryDestination() if use_memory_logger else None
        try:
            add_destinations(*(i for i in (pth, mem) if i is not None))
            yield mem
        finally:
            for i in (mem, pth):
                if i is not None:
                    remove_destination(i)
            if fid:
                fid.close()

    return _wrap
