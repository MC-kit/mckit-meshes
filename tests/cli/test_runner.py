from __future__ import annotations

from mckit_meshes import __version__
from mckit_meshes.cli.runner import mckit_meshes


def run_version(runner):
    command = mckit_meshes
    version = __version__
    result = runner.invoke(command, args=["--version"], catch_exceptions=False)
    assert result.exit_code == 0
    assert version in result.output


def test_help(runner):
    result = runner.invoke(mckit_meshes, args=["--help"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "Usage: " in result.output
