# noinspection PyPep8
"""Convert MCNP meshtal file to a number of npz files, one for each meshtal."""
from __future__ import annotations

import typing as t

import logging

from pathlib import Path

from mckit_meshes import fmesh
from mckit_meshes.utils.io import check_if_path_exists

__LOG = logging.getLogger(__name__)


def revise_mesh_tallies(mesh_tallies) -> list[Path]:
    if mesh_tallies:
        return list(map(Path, mesh_tallies))

    cwd = Path.cwd()
    rv = list(cwd.glob("*.m"))
    if not rv:
        errmsg = f"No .m-files found in directory '{cwd.absolute()}', nothing to do."
        __LOG.warning(errmsg)
    return rv


def mesh2npz(
    prefix: str | Path,
    mesh_tallies: t.Iterable[str | Path],
    override: bool = False,
) -> None:
    """Convert MCNP meshtal file to a number of npz files, one for each mesh tally."""
    mesh_tallies = revise_mesh_tallies(mesh_tallies)
    single_input = len(mesh_tallies) == 1
    prefix = Path(prefix)
    for m in mesh_tallies:
        _m = Path(m)
        p = prefix if single_input else prefix / _m.stem
        __LOG.info(f"Processing {_m}")
        __LOG.debug(f"Saving tallies with prefix {prefix}")
        p.mkdir(parents=True, exist_ok=True)
        with _m.open() as stream:
            fmesh.m_2_npz(
                stream,
                prefix=p,
                check_existing_file_strategy=check_if_path_exists(override),
            )
