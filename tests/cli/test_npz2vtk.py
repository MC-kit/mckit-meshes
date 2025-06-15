"""Tests for npz2vtk CLI module."""

from __future__ import annotations

import shutil

from pathlib import Path

import pytest

from mckit_meshes.cli.runner import mckit_meshes


@pytest.fixture
def source(data):
    return data / "1004.npz"


def test_help(runner):
    result = runner.invoke(mckit_meshes, args=["npz2vtk", "--help"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "Usage: " in result.output


def test_with_prefix(tmp_path, runner, source):
    prefix = tmp_path
    result = runner.invoke(
        mckit_meshes,
        args=["npz2vtk", "--prefix", prefix, str(source)],
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
    for i in [1004, 2004]:
        assert (prefix / f"{i}.vtr").exists(), (
            "When multiple npz files are specified the vtr file should be created for every one."
        )


def test_without_prefix(cd_tmpdir, runner, source):  # noqa: ARG001
    result = runner.invoke(mckit_meshes, args=["npz2vtk", str(source)], catch_exceptions=False)
    assert result.exit_code == 0
    cwd = Path.cwd()
    output_path = cwd / "1004.vtr"
    assert output_path.exists()


def test_no_npz_files_and_not_specified_npz(cd_tmpdir, runner):  # noqa: ARG001
    assert not list(
        Path.cwd().glob("*.npz"),
    ), "There shouldn't be any .npz files in current directory"
    result = runner.invoke(mckit_meshes, args=["npz2vtk"], catch_exceptions=False)
    assert result.exit_code == 0, "Should be noop, when nothing to do"
    assert "WARNING" in result.output, "Should warn"
    assert "nothing to do" in result.output, "when nothing to do"


def test_not_existing_input_file(runner):
    result = runner.invoke(
        mckit_meshes,
        args=["npz2vtk", "not-existing.npz"],
        catch_exceptions=False,
    )
    assert result.exit_code > 0
    assert "does not exist" in result.output


def test_glob_inputs(tmp_path, runner, data, monkeypatch):
    for i in [1004, 2004]:
        original = data / f"{i}.npz"
        shutil.copy(original, tmp_path)
    prefix = tmp_path / "some_vtk"
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        mckit_meshes,
        args=["npz2vtk", "-p", str(prefix)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    for i in [1004, 2004]:
        assert (prefix / f"{i}.vtr").exists(), (
            "When multiple npz files are specified the vtr file should be created for every one."
        )


def test_absent_npz_files(runner, caplog, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        mckit_meshes,
        args=["npz2vtk"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert "nothing to do" in caplog.text
