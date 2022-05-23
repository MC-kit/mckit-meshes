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

import mckit_meshes.fmesh as fmesh

from ...utils.io import check_if_path_exists

__LOG = logging.getLogger(__name__)


def revise_npz_files(npz_files) -> t.List[Path]:
    if npz_files:
        return list(map(Path, npz_files))

    cwd = Path.cwd()
    rv = list(cwd.glob("*.npz"))
    if not rv:
        errmsg = f"No .npz-files found in directory '{cwd.absolute()}', nothing to do."
        __LOG.warning(errmsg)
    return rv


def npz2vtk(
    prefix: str | Path, npz_files: t.Iterable[str | Path], override: bool = False
) -> None:
    """Convert MCNP meshtal file to a number of npz files, one for each mesh tally."""
    npz_files = revise_npz_files(npz_files)
    prefix = Path(prefix)
    file_exists_strategy = check_if_path_exists(override)
    for npz in npz_files:
        npz = Path(npz)
        __LOG.info("Processing {}".format(npz))
        __LOG.debug("Saving VTK file with prefix {}".format(prefix))
        prefix.mkdir(parents=True, exist_ok=True)
        mesh = fmesh.FMesh.load_npz(npz)
        vtk_file_stem = f"{prefix / str(mesh.name)}"
        vtk_file_name = (
            vtk_file_stem + ".vtr"
        )  # TODO dvp: revise this when it comes to saving structured mesh
        file_exists_strategy(vtk_file_name)
        vtk = mesh.save2vtk(vtk_file_stem)
        __LOG.info("Saved VTK to {}", vtk)
