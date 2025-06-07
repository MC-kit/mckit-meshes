from __future__ import annotations

from typing import TYPE_CHECKING


import pytest

from click.testing import CliRunner

from loguru import logger

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
