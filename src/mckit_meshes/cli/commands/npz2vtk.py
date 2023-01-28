"""Convert npz files to VTK vtr-files."""

from __future__ import annotations

import typing as t

import logging

from pathlib import Path

import mckit_meshes.fmesh as fmesh

from ...utils.io import check_if_path_exists

__LOG = logging.getLogger(__name__)


def revise_npz_files(npz_files: t.Optional[t.Iterable[t.Any]]) -> t.List[Path]:
    """Use specified list of file to process, if any, or find them in current directory.

    Args:
        npz_files: npz files to process passed from command line, optional.

    Returns:
        List of the specified or found files as Path objects.
    """
    if npz_files:
        return list(map(Path, npz_files))

    cwd = Path.cwd()
    rv = list(cwd.glob("*.npz"))
    if not rv:
        errmsg = f"No .npz-files found in directory '{cwd.absolute()}', nothing to do."
        __LOG.warning(errmsg)
    return rv


def npz2vtk(prefix: str | Path, npz_files: t.Iterable[str | Path], override: bool = False) -> None:
    """Convert MCNP meshtal file to a number of npz files, one for each mesh tally.

    Args:
        prefix: output directory
        npz_files: files to process, optional
        override: define behaviour when output file, exists, default - rise FileExistsError.
    """
    npz_files = revise_npz_files(npz_files)
    prefix = Path(prefix)
    file_exists_strategy = check_if_path_exists(override)
    for npz in npz_files:
        npz = Path(npz)
        __LOG.info("Processing {}", npz)
        __LOG.debug("Saving VTK file with prefix {}", prefix)
        prefix.mkdir(parents=True, exist_ok=True)
        mesh = fmesh.FMesh.load_npz(npz)
        vtk_file_stem = f"{prefix / str(mesh.name)}"
        vtk_file_name = (
            vtk_file_stem + ".vtr"
        )  # TODO dvp: revise this when it comes to saving structured mesh
        file_exists_strategy(vtk_file_name)
        vtk = mesh.save2vtk(vtk_file_stem)
        __LOG.info("Saved VTK to {}", vtk)
