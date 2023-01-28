"""The CLI application module.

The module applies :meth:`click` API to organize CLI interface for McKit-meshes package.
"""
from __future__ import annotations

import typing as t

import datetime

from pathlib import Path

import click
import mckit_meshes.version as meta

from mckit_meshes.cli.commands import do_mesh2npz, do_npz2vtk

# from mckit_meshes.cli.commands.common import get_default_output_directory
from mckit_meshes.cli.logging import init_logger, logger

# from mckit_meshes.utils import MCNP_ENCODING

NAME = "mckit_meshes"
VERSION = meta.__version__
# LOG_FILE_RETENTION = 3
# NO_LEVEL_BELOW = 30


@click.group("mckit-meshes", help=meta.__summary__)
@click.pass_context
@click.option("--override/--no-override", is_flag=True, default=False)
@click.option("--verbose/--no-verbose", is_flag=True, default=False, help="Log everything")
@click.option(
    "--quiet/--no-quiet",
    is_flag=True,
    default=False,
    help="Log only WARNINGS and above",
)
@click.option("--logfile", default=None, help="File to log to")
@click.version_option(VERSION, prog_name=NAME)
def mckit_meshes(
    ctx: click.Context, verbose: bool, quiet: bool, logfile: str, override: bool
) -> None:
    """McKit-meshes command line interface."""
    init_logger(logfile, quiet, verbose)
    # TODO dvp: add customized logger configuring from a configuration toml-file.

    obj = ctx.ensure_object(dict)
    obj["OVERRIDE"] = override


@mckit_meshes.command()
@click.pass_context
@click.option(
    "--prefix",
    "-p",
    default="npz",
    help="""A prefix to prepend output files (default: "npz/"),
output files are also prepended with MESH_TALLY file base name,
if there are more than 1 input file""",
)
@click.argument(
    "mesh_tallies",
    metavar="[<meshtally_file>...]",
    type=click.Path(exists=True),
    nargs=-1,
    required=False,
)
def mesh2npz(ctx: click.Context, prefix: str | Path, mesh_tallies: t.List[t.Any]) -> None:
    """Converts mesh files to npz files."""
    do_mesh2npz(prefix, mesh_tallies, ctx.obj["OVERRIDE"])
    #


@mckit_meshes.command()
@click.pass_context
@click.option(
    "--prefix",
    "-p",
    default=".",
    help="""A prefix to prepend output files (default: "./"),
output files are also prepended with MESH_TALLY file base name,
if there are more than 1 input file""",
)
@click.argument(
    "npz_files",
    metavar="[<npz_file>...]",
    type=click.Path(exists=True),
    nargs=-1,
    required=False,
)
def npz2vtk(ctx: click.Context, prefix: str | Path, npz_files: t.List[t.Any]) -> None:
    """Converts npz files to VTK files."""
    do_npz2vtk(prefix, npz_files, ctx.obj["OVERRIDE"])
    # Don't remove these comments: this makes flake8 happy on absent arguments in the docstring.
    #


if __name__ == "__main__":
    ct = datetime.datetime.now()
    mckit_meshes(obj={})
    logger.success(f"Elapsed time: {datetime.datetime.now() - ct}")
