from __future__ import annotations

import logging
import loguru

from mckit_meshes.cli.logging_setup import init_logger


def test_std_logging(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    init_logger("test.log", quiet=False, verbose=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    assert logger.isEnabledFor(logging.INFO)
    logger.info("test info")
    logger.log(logging.INFO - 3, "test non-standard level")
    log_text = (tmp_path / "test.log").read_text()
    assert "test non-standard level" in log_text


def test_quiet() -> None:
    logger = loguru.logger
    init_logger(None, quiet=True, verbose=False)
    logger.info("xxx")
    logger.warning("yyy")
    # caplog create handler, which is not controlled in init_logger
    # so, it logs everything regardless quiet or verbose settings
    # Anyway, the scheme with verbosity is too primitive.
    # Use external configuration instead.


def test_logging_without_stderr():
    logger = loguru.logger
    logger.remove()
    init_logger(None, quiet=True, verbose=False, stderr_format=None)
    logger.info("xxx")
    logger.warning("yyy")
    # caplog create handler, which is not controlled in init_logger
    # so, it logs everything regardless quiet or verbose settings
    # Anyway, the scheme with verbosity is too primitive.
    # Use external configuration instead.
