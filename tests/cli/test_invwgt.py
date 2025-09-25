from __future__ import annotations

import pytest

from mckit_meshes.__main__ import app as mckit_meshes


@pytest.fixture
def weights_in_cylinder_geometry(data):
    return data / "wwinp"


def test_help(cyclopts_runner):
    out = cyclopts_runner(mckit_meshes, ["invwgt", "--help"])
    assert "Usage: " in out


# def test_invwgt(source, cyclopts_runner):
#     cyclopts_runner(mckit_meshes, ["invwgt", str(source)])
