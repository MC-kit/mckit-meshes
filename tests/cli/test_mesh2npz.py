from __future__ import annotations

import shutil

from pathlib import Path

from cyclopts import ValidationError
import pytest

from mckit_meshes.__main__ import app
from mckit_meshes.fmesh import FMesh


@pytest.fixture
def source(data):
    return (data / "1.m").absolute()


def test_help(cyclopts_runner):
    _, out, _ = cyclopts_runner(app, args=["mesh2npz", "--help"])
    assert "Usage: " in out


def test_with_prefix(tmp_path, cyclopts_runner, data):
    prefix = tmp_path / "npz"
    args = ["mesh2npz", "--prefix", str(prefix), str(data / "1.m")]
    _, _, tmp = cyclopts_runner(app, args)
    output_path = Path(tmp, prefix, "1004.npz")
    assert output_path.exists()


def test_multiple_files(tmp_path, cyclopts_runner, data):
    original = data / "1.m"
    input1 = tmp_path / "1.m"
    shutil.copy(original, input1)
    input2 = tmp_path / "2.m"
    shutil.copy(original, input2)
    prefix = tmp_path / "some_npz"
    args = ["mesh2npz", "--prefix", str(prefix), str(input1), str(input2)]
    cyclopts_runner(app, args)
    for i in [1, 2]:
        assert (
            prefix / f"{i}"
        ).exists(), """When multiple mesh files are specified the tallies should be distributed
             to different directories named as the mesh files"""
        output_path = prefix / f"{i}" / "1004.npz"
        assert output_path.exists()


def test_without_prefix(cyclopts_runner, source):
    args = ["mesh2npz", str(source)]
    _, _, tmp = cyclopts_runner(app, args)
    output_path = tmp / "npz" / "1004.npz"
    assert output_path.exists()


def test_existing_mesh_tally_file_and_not_specified_mesh_tally(cyclopts_runner, source):
    t = Path.cwd()
    shutil.copy(source, t)
    input_path = t / "1.m"
    assert input_path.exists()
    output_path = t / "npz/1004.npz"
    assert not output_path.exists()
    args = ["mesh2npz"]
    cyclopts_runner(app, args)
    assert output_path.exists(), (
        "Failed to process meshtal file in current directory with empty command line"
    )


def test_no_mesh_tally_file_and_not_specified_mesh_tally(cyclopts_runner, eliot_mem_trace):
    assert not list(Path.cwd().glob("*.m")), "There shouldn't be any .m files in current directory"
    args = ["mesh2npz"]
    cyclopts_runner(app, args)
    eliot_mem_trace.check_message("message_type", "WARNING")


def test_not_existing_mesh_tally_file(cyclopts_runner):
    args = ["mesh2npz", "not-existing.m"]
    with pytest.raises(ValidationError, match="does not exist"):
        cyclopts_runner(app, args, exit_on_error=False)


def test_failure_on_existing_output_file_when_override_is_not_set(
    tmp_path, cyclopts_runner, source
):
    prefix = tmp_path / "npz"
    output_path = prefix / "1004.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.touch(exist_ok=True)
    assert output_path.exists()
    args = ["mesh2npz", "--prefix", str(prefix), str(source)]
    with pytest.raises(FileExistsError, match="Cannot override"):
        cyclopts_runner(app, args)


def test_long_mesh_number(cyclopts_runner, data, eliot_file_trace):
    """Check if mesh number representation in npz file is long enough to handle large numbers."""
    with eliot_file_trace("test.log"):
        prefix = Path.cwd()
        _input = data / "2035224.m"
        args = ["mesh2npz", "--prefix", str(prefix), str(_input)]
        cyclopts_runner(app, args)
        prefix = Path(prefix)
        npz_path = prefix / "2035224.npz"
        assert npz_path.exists(), f"should create {npz_path}"
        mesh = FMesh.load_npz(npz_path)
        assert mesh.name == 2035224, (
            "Should correctly save and load the 2035224 mesh id, which requires 32 bit"
        )
