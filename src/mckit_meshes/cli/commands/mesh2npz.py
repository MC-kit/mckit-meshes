# noinspection PyPep8
"""Convert MCNP meshtal file to a number of npz files, one for each meshtal."""
from __future__ import annotations

import typing as t

import logging

from pathlib import Path

import mckit_meshes.fmesh as fmesh

from ...utils.io import check_if_path_exists

__LOG = logging.getLogger(__name__)


def revise_mesh_tallies(mesh_tallies) -> t.List[Path]:
    if mesh_tallies:
        return list(map(Path, mesh_tallies))

    cwd = Path.cwd()
    rv = list(cwd.glob("*.m"))
    if not rv:
        errmsg = "No .m-files found in directory '{}', nothing to do.".format(cwd.absolute())
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
        with m.open() as stream:
            fmesh.m_2_npz(
                stream,
                prefix=p,
                check_existing_file_strategy=check_if_path_exists(override),
            )
