from typing import List

import sys

from contextlib import contextmanager
from pathlib import Path

import click
import mckit_meshes.version as meta

from mckit_meshes.cli.commands import do_mesh2npz

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
@click.option(
    "--verbose/--no-verbose", is_flag=True, default=False, help="Log everything"
)
@click.option(
    "--quiet/--no-quiet",
    is_flag=True,
    default=False,
    help="Log only WARNINGS and above",
)
@click.option("--logfile", default=None, help="File to log to")
@click.version_option(VERSION, prog_name=NAME)
def mckit_meshes(ctx, verbose: bool, quiet: bool, logfile: str, override: bool) -> None:
    # """McKit-meshes command line interface."""

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
def mesh2npz(ctx, prefix, mesh_tallies):
    """Converts mesh files to npz files."""
    return do_mesh2npz(prefix, mesh_tallies, ctx.obj["OVERRIDE"])


if __name__ == "__main__":
    mckit_meshes(obj={})
