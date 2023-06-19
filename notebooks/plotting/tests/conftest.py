from __future__ import annotations

import os

from pathlib import Path

import pytest

from mckit_meshes.utils.resource import path_resolver


@pytest.fixture(scope="session")
def data() -> Path:
    return path_resolver("tests")("data")


@pytest.fixture()
def cd_tmpdir(tmpdir):
    old_dir = tmpdir.chdir()
    try:
        yield
    finally:
        os.chdir(old_dir)
