from __future__ import annotations

import pytest


@pytest.fixture
def weights_in_cylinder_geometry(data):
    return data / "wwinp"
