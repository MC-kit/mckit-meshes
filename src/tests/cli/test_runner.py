from mckit_meshes import __version__
from mckit_meshes.cli.runner import mckit_meshes

from .utils import run_version


def test_version(runner):
    run_version(runner, mckit_meshes, __version__)


def test_help(runner):
    result = runner.invoke(mckit_meshes, args=["--help"], catch_exceptions=False)
    assert result.exit_code == 0
    # print(result.output)
    assert "Usage: " in result.output
