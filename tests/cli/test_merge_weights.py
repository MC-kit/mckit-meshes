from __future__ import annotations

import shutil
import textwrap

from pathlib import Path

from mckit_meshes.__main__ import app as mckit_meshes
from mckit_meshes.wgtmesh import WgtMesh


def test_help(cyclopts_runner):
    out = cyclopts_runner(mckit_meshes, ["merge-weights", "--help"])
    assert "Usage: " in out


def test_happy_path(cyclopts_runner, weights_in_cylinder_geometry, eliot_file_trace):
    """Check if the simpliest cartesian mesh can be converted to WgtMesh."""
    with eliot_file_trace("test.log"):
        _input1 = "wwinp1"
        _input2 = "wwinp2"
        shutil.copy(weights_in_cylinder_geometry, _input1)
        shutil.copy(weights_in_cylinder_geometry, _input2)
        merge_spec = Path("merge-spec.txt")
        merge_spec.write_text(
            textwrap.dedent(f"""
            # This is an oversimplified test
            {_input1} 1000
            {_input2} 2000
            """)
        )
        out = Path("wwinp.merged")
        args = [
            "merge-weights",
            "--out",
            str(out),
            "--merge-spec",
            str(merge_spec),
            _input1,
            _input2,
        ]
        cyclopts_runner(mckit_meshes, args)
        assert out.exists(), f"should create {out}"
        with out.open() as fid:
            wgt_mesh = WgtMesh.read(fid)
            assert wgt_mesh.is_cylinder
