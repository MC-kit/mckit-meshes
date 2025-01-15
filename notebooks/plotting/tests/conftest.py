from __future__ import annotations

import os

from pathlib import Path
from collections.abc import Generator

import pytest

HERE = Path(__file__).parent


@pytest.fixture(scope="session")
def data() -> Path:
    """Path to the directory with the tests data."""
    return HERE / "data"


@pytest.fixture
def cd(tmp_path: Path) -> Generator[Path, None, None]:
    """Switch to temp dir for a test run.

    Yields:
        Path: to temp directory
    """
    old_dir = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        os.chdir(old_dir)
