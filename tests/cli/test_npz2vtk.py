"""Tests for npz2vtk CLI module."""

from __future__ import annotations

import shutil

from pathlib import Path

from cyclopts import ValidationError
import pytest

from mckit_meshes.__main__ import app as mckit_meshes


@pytest.fixture
def source(data):
    return data / "1004.npz"


def test_help(cyclopts_runner):
    out = cyclopts_runner(mckit_meshes, ["npz2vtk", "--help"])
    assert "Usage: " in out


def test_with_prefix(cyclopts_runner, source):
    prefix = "some-dir"
    cyclopts_runner(mckit_meshes, ["npz2vtk", "--prefix", prefix, str(source)])
    output_path = Path(prefix, "1004.vtr")
    assert output_path.exists()


def test_multiple_files(cyclopts_runner, data):
    tmp_path = Path.cwd()
    inputs = []
    for i in [1004, 2004]:
        original = data / f"{i}.npz"
        shutil.copy(original, tmp_path)
        inputs.append(str(tmp_path / f"{i}.npz"))
    prefix = tmp_path / "some_vtk"
    cyclopts_runner(mckit_meshes, ["npz2vtk", "-p", str(prefix), *inputs])

    for i in [1004, 2004]:
        assert (prefix / f"{i}.vtr").exists(), (
            "When multiple npz files are specified the vtr file should be created for every one."
        )


def test_without_prefix(cyclopts_runner, source):
    cyclopts_runner(mckit_meshes, ["npz2vtk", str(source)])
    output_path = Path("1004.vtr")
    assert output_path.exists()


def test_no_npz_files_and_not_specified_npz(cyclopts_runner, eliot_mem_trace):
    assert not list(
        Path.cwd().glob("*.npz"),
    ), "There shouldn't be any .npz files in current directory"
    cyclopts_runner(mckit_meshes, ["npz2vtk"])
    eliot_mem_trace.check_message("message_type", "WARNING")


def test_not_existing_input_file(cyclopts_runner):
    with pytest.raises(ValidationError, match="does not exist"):
        cyclopts_runner(mckit_meshes, ["npz2vtk", "not-existing.npz"], exit_on_error=False)


def test_glob_inputs(cyclopts_runner, data):
    tmp = Path.cwd()
    for i in [1004, 2004]:
        original = data / f"{i}.npz"
        shutil.copy(original, tmp)
    prefix = tmp / "some_vtk"
    cyclopts_runner(mckit_meshes, ["npz2vtk", "-p", str(prefix)])

    for i in [1004, 2004]:
        assert (prefix / f"{i}.vtr").exists(), (
            "When multiple npz files are specified the vtr file should be created for every one."
        )


def test_absent_npz_files(cyclopts_runner, eliot_mem_trace):
    cyclopts_runner(mckit_meshes, ["npz2vtk"])
    eliot_mem_trace.check_message("message_type", "WARNING")
