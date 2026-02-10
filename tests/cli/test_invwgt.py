from __future__ import annotations

from pathlib import Path

from mckit_meshes.__main__ import app as mckit_meshes


def test_help(cyclopts_runner):
    out = cyclopts_runner(mckit_meshes, ["invwgt", "--help"])
    assert "Usage: " in out


def test_with_wwinp(cyclopts_runner, data, eliot_file_trace):
    with eliot_file_trace("test.log"):
        out = cyclopts_runner(mckit_meshes, ["invwgt", str(data / "wwinp")])
        assert out == ""


def test_with_wwinp_outt_override(cyclopts_runner, data, eliot_file_trace):
    with eliot_file_trace("test.log"):
        output = Path("inv-wwinp")
        output.touch()
        assert output.stat().st_size == 0
        stdout = cyclopts_runner(
            mckit_meshes, ["invwgt", "--out", str(output), "--override", str(data / "wwinp")]
        )
        assert stdout == ""
        assert output.stat().st_size > 0
