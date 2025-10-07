from __future__ import annotations

from mckit_meshes.__main__ import app as mckit_meshes


def test_help(cyclopts_runner):
    out = cyclopts_runner(mckit_meshes, ["mesh2wgt", "--help"])
    assert "Usage: " in out
