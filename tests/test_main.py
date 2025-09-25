from __future__ import annotations

from cyclopts import MissingArgumentError, ValidationError
import pytest


from mckit_meshes.__main__ import __version__, app


def test_version(cyclopts_runner):
    out = cyclopts_runner(app, ["--version"])
    assert __version__ in out


def test_help(cyclopts_runner):
    out = cyclopts_runner(app, ["--help"])

    assert "Usage: " in out


def test_when_there_is_no_command(cyclopts_runner):
    out = cyclopts_runner(app, [], exit_on_error=False)

    assert "Usage: " in out


def test_when_there_is_no_args(cyclopts_runner):
    with pytest.raises(MissingArgumentError, match="meshtally-file"):
        cyclopts_runner(app, ["split"], exit_on_error=False)


def test_not_existing_mesh_tally_file(cyclopts_runner):
    with pytest.raises(ValidationError, match="does not exist"):
        cyclopts_runner(app, ["split", "not-existing.m"], exit_on_error=False)
