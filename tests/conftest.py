from __future__ import annotations

from typing import Generator

import os

from pathlib import Path

import pytest

_DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def data() -> Path:
    """Compute the path to test data.

    Returns:
        Path to test data.
    """
    return _DATA


@pytest.fixture()
def cd_tmpdir(tmp_path) -> Generator[Path, None, None]:
    """Temporarily change to temp directory.

    Yields:
        Path: to temporary directory
    """
    old_dir = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        os.chdir(old_dir)
