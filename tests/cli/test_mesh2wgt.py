from __future__ import annotations

from pathlib import Path

from mckit_meshes.__main__ import app as mckit_meshes
from mckit_meshes.wgtmesh import WgtMesh


def test_help(cyclopts_runner):
    """Test help output."""
    out = cyclopts_runner(mckit_meshes, ["mesh2wgt", "--help"])
    assert "Usage: " in out


def test_happy_path(cyclopts_runner, data, eliot_file_trace):
    """Check if the simpliest cartesian mesh can be converted to WgtMesh."""
    with eliot_file_trace("test.log"):
        _input = data / "2035224.m"
        args = ["mesh2wgt", str(_input), "--mesh", "2035124"]
        cyclopts_runner(mckit_meshes, args)
        wgt_path = Path.cwd() / "2035224.wwinp"
        assert wgt_path.exists(), f"should create {wgt_path}"
        with wgt_path.open() as fid:
            wgt_mesh = WgtMesh.read(fid)
            assert not wgt_mesh.is_cylinder
