import shutil

from pathlib import Path

import pytest

from mckit_meshes.cli.runner import mckit_meshes
from mckit_meshes.fmesh import FMesh

# @pytest.fixture()
# def source(data):
#     return data / "1.m"


def test_help(runner):
    result = runner.invoke(
        mckit_meshes, args=["npz2vtk", "--help"], catch_exceptions=False
    )
    assert result.exit_code == 0
    assert "Usage: " in result.output


def test_with_prefix(tmp_path, runner, data):
    prefix = tmp_path
    result = runner.invoke(
        mckit_meshes,
        args=["npz2vtk", "--prefix", prefix, str(data / "1004.npz")],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    output_path = prefix / "1004.vtr"
    assert output_path.exists()


def test_multiple_files(tmp_path, runner, data):
    inputs = []
    for i in [1004, 2004]:
        original = data / f"{i}.npz"
        shutil.copy(original, tmp_path)
        inputs.append(str(tmp_path / f"{i}.npz"))
    prefix = tmp_path / "some_vtk"
    result = runner.invoke(
        mckit_meshes,
        args=["npz2vtk", "-p", str(prefix), *inputs],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    # prefix = Path(prefix)
    for i in [1004, 2004]:
        assert (
            prefix / f"{i}.vtr"
        ).exists(), """When multiple npz files are specified the vtk files should be created for every one."""


# def test_without_prefix(cd_tmpdir, runner, source):
#     result = runner.invoke(
#         mckit_meshes, args=["npz2vtk", str(source)], catch_exceptions=False
#     )
#     assert result.exit_code == 0
#     cwd = Path.cwd()
#     output_path = cwd / "npz" / "1004.npz"
#     assert output_path.exists()
#
#
# def test_existing_mesh_tally_file_and_not_specified_mesh_tally(
#     cd_tmpdir, runner, source
# ):
#     t = Path.cwd()
#     shutil.copy(source, t)
#     input_path = t / "1.m"
#     assert input_path.exists()
#     output_path = t / "npz/1004.npz"
#     assert not output_path.exists()
#     result = runner.invoke(mckit_meshes, args=["npz2vtk"], catch_exceptions=False)
#     assert result.exit_code == 0
#     assert (
#         output_path.exists()
#     ), "Failed to process meshtal file in current directory with empty command line"
#
#
# def test_no_mesh_tally_file_and_not_specified_mesh_tally(cd_tmpdir, runner):
#     assert not list(
#         f for f in Path.cwd().glob("*.m")
#     ), "There shouldn't be any .m files in current directory"
#     result = runner.invoke(mckit_meshes, args=["npz2vtk"], catch_exceptions=False)
#     assert result.exit_code == 0, "Should be noop, when nothing to do"
#     assert (
#         "WARNING" in result.output and "nothing to do" in result.output
#     ), "Should warn, when nothing to do"
#
#
# def test_not_existing_mesh_tally_file(runner):
#     result = runner.invoke(
#         mckit_meshes, args=["npz2vtk", "not-existing.m"], catch_exceptions=False
#     )
#     assert result.exit_code > 0
#     assert "does not exist" in result.output
#
#
# def test_failure_on_existing_output_file_when_override_is_not_set(
#     tmp_path, runner, source
# ):
#     prefix = tmp_path / "npz"
#     output_path = prefix / "1004.npz"
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     output_path.touch(exist_ok=True)
#     assert output_path.exists()
#     result = runner.invoke(
#         mckit_meshes,
#         args=["npz2vtk", "-p", prefix, str(source)],
#         catch_exceptions=True,
#     )
#     assert result.exit_code != 0
#     errmsg = f"""\
# Cannot override existing file \"{output_path}\".
# Please remove the file or specify --override option"""
#     assert errmsg in result.output
#
#
# def test_long_mesh_number(cd_tmpdir, runner, data):
#     """Check if mesh number representation in npz file is long enough to handle large numbers."""
#     prefix = Path.cwd()
#     _input = data / "2035224.m"
#     result = runner.invoke(
#         mckit_meshes,
#         args=["npz2vtk", "-p", prefix, str(_input)],
#         catch_exceptions=False,
#     )
#     assert result.exit_code == 0, "should successfully process {}".format(_input)
#     prefix = Path(prefix)
#     npz_path = prefix / "2035224.npz"
#     assert npz_path.exists(), "should create {}".format(npz_path)
#     mesh = FMesh.load_npz(npz_path)
#     assert (
#         mesh.name == 2035224
#     ), "Should correctly save and load the 2035224 mesh id, which requires 32 bit"
