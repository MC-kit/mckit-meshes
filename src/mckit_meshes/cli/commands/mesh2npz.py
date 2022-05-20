# noinspection PyPep8
"""
    Convert MCNP meshtal file to a number of npz files, one for each meshtal.

    Usage:

        mesh2npz ] [-s TALLIES] [-p PREFIX] MESH_TALLY...
        mesh2npz --version | -h | --help

    Options:
        -h, --help  print this message and exit
        --version print the version and exit
        -s, --select TALLIES            - comma separated list of tallies to extract from the mesh file
        -p, --prefix PREFIX             - prefix to prepend output files (default: "npz/"),
                                          output files are also prepended with MESH_TALLY file base name
        --override                      - override existing output files, default(false)
        --update                        - override existing output files, which are older than the source file

    Arguments:
        MESH_TALLY... - files to load mesh tallies from (default: all the .m files in current folder)


    Features:
        Fails if, an output file exist and neither --override, nor --update options is specified in command line
        Uses standard mckit_meshes module logging (see logging_cfg docs).
        Default log file mesh_2_npz.log.
"""
from __future__ import annotations

import typing as t

import logging

from pathlib import Path

import click
import mckit_meshes.fmesh as fmesh

__LOG = logging.getLogger(__name__)


def revise_mesh_tallies(mesh_tallies) -> t.List[Path]:
    if mesh_tallies:
        return list(map(Path, mesh_tallies))

    cwd = Path.cwd()
    rv = list(cwd.glob("*.m"))
    if not rv:
        errmsg = "No .m-files found in directory '{}', nothing to do.".format(
            cwd.absolute()
        )
        __LOG.warning(errmsg)
    return rv


def mesh2npz(
    prefix: str | Path, mesh_tallies: t.Iterable[str | Path], override: bool = False
) -> None:
    """Convert MCNP meshtal file to a number of npz files, one for each mesh tally."""
    mesh_tallies = revise_mesh_tallies(mesh_tallies)
    single_input = len(mesh_tallies) == 1
    prefix = Path(prefix)
    for m in mesh_tallies:
        m = Path(m)
        if single_input:
            p = prefix
        else:
            p = prefix / m.stem
        __LOG.info("Processing {}".format(m))
        __LOG.debug("Saving tallies with prefix {}".format(prefix))
        p.mkdir(parents=True, exist_ok=True)
        try:
            with m.open() as stream:
                fmesh.m_2_npz(stream, prefix=p, override=override)
        except FileExistsError as x:
            file_name = str(x)
            errmsg = (
                f'Cannot override existing file "{file_name}".\n'
                "Please remove the file or specify --override option"
            )
            # __LOG.error(errmsg)
            raise click.BadParameter(errmsg)
