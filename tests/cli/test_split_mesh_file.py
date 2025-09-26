from __future__ import annotations

import shutil
import sys

from importlib.resources import files
from pathlib import Path
from time import sleep
from typing import Any

import pytest

from cyclopts.exceptions import MissingArgumentError, ValidationError
from mckit_meshes.__main__ import app
from mckit_meshes import m_file_iterator


@pytest.fixture
def source():
    res = files("tests").joinpath("data/2.m")
    assert res.exists()
    return res


def test_when_there_is_no_args(cyclopts_runner):
    with pytest.raises(MissingArgumentError, match="meshtally-file"):
        cyclopts_runner(app, ["split"], exit_on_error=False)


def test_not_existing_mesh_tally_file(cyclopts_runner):
    with pytest.raises(ValidationError, match="does not exist"):
        cyclopts_runner(app, ["split", "not-existing.m"], exit_on_error=False)


def test_when_only_mesh_is_specified(source, cyclopts_runner, eliot_file_trace, eliot_mem_trace):
    with eliot_file_trace("test.log"):
        cyclopts_runner(app, ["split", str(source)])
        eliot_mem_trace.validate()
    tmp = Path.cwd()
    for i in [1004, 2004]:
        output_path = tmp / f"{i}.m"
        assert output_path.exists()
        assert_content_is_correct(output_path, i)


def assert_content_is_correct(output_path, mesh_no):
    with output_path.open() as fid:
        it = m_file_iterator(fid)
        header = next(it)
        assert len(header) == 3, "The mesh file header is 3 lines long"
        assert header[0].startswith("mcnp")
        mesh = next(it)
        first_line = mesh[0]
        actual_mesh_no = int(first_line.split()[3])
        assert mesh_no == actual_mesh_no


def test_with_prefix(source, cyclopts_runner, eliot_file_trace, eliot_mem_trace):
    with eliot_file_trace("test.log"):
        prefix = "out"
        cyclopts_runner(app, ["split", "--prefix", prefix, str(source)])
        eliot_mem_trace.validate()
    for i in [1004, 2004]:
        output_path = Path(prefix, f"{i}.m")
        assert output_path.exists()


def test_when_output_file_exist_and_override_is_not_specified(
    source, cyclopts_runner, eliot_file_trace, eliot_mem_trace
):
    with eliot_file_trace("test.log"):
        output_path = Path("1004.m")
        output_path.touch()
        with pytest.raises(FileExistsError, match="--override"):
            cyclopts_runner(app, ["split", str(source)])
        eliot_mem_trace.validate()


@pytest.mark.skipif(sys.platform == "windows", reason="Doesn't work on Windows")
def test_when_output_file_exist_and_override_is_specified(
    source, cyclopts_runner, eliot_file_trace
):
    with eliot_file_trace("test.log"):
        output_path = Path("1004.m")
        output_path.touch()
        prev_out_time = output_path.stat().st_mtime_ns
        sleep(0.01)
        cyclopts_runner(app, ["split", "--override", str(source)])
        out_time = output_path.stat().st_mtime_ns
        assert prev_out_time < out_time, (
            "Overrides existing file if '--override' option is specified"
        )
