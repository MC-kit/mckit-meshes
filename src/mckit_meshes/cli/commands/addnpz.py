"""Convert npz files to VTK vtr-files."""

from __future__ import annotations

import typing as t

import logging

from pathlib import Path

import numpy as np

from mckit_meshes import fmesh
from mckit_meshes.utils.io import check_if_path_exists

__LOG = logging.getLogger(__name__)


def add(
    out: str | Path,
    comment: str,
    number: int,
    npz_files: t.Iterable[str | Path],
    override: bool = False,
) -> None:
    """Convert MCNP meshtal file to a number of npz files, one for each mesh tally.

    Applicable to meshes with the same geometry.

    Note:
        We assume that statistics is the same for all the meshes, preferably they
        are produced in the same run (like neutron and photon heat).

    Args:
        out: output file
        comment: comment for a new fmesh npz file, if not provided, then the comment from the first mesh is used
        number: ... to assign to resulting mesh
        npz_files: files to process, optional
        override: define behaviour when output file, exists, default - rise FileExistsError.
    """
    npz_files = [Path(f) for f in npz_files]

    if not npz_files:
        __LOG.warning("No files specified to process")
        return

    out = Path(out) if out else Path("+".join(x.stem for x in npz_files) + ".npz")

    out.parent.mkdir(parents=True, exist_ok=True)
    file_exists_strategy = check_if_path_exists(override)

    _sum: fmesh.FMesh | None = None
    data = None
    errors = None
    totals = None
    tot_errors = None
    for npz in npz_files:
        __LOG.info("Adding {}", npz)
        mesh = fmesh.FMesh.load_npz(npz)
        if _sum is None:
            _sum = mesh
            data = _sum.data
            errors = np.power(_sum.errors * _sum.data, 2)
            totals = _sum.totals
            if totals is not None:
                tot_errors = np.power(_sum.totals_err * _sum.totals, 2)
        else:
            if not _sum.is_equal_by_geometry(mesh):
                msg = f"Mesh {npz} is not compatible by geometry with previous ones."
                raise ValueError(msg)
            data += mesh.data
            errors += np.power(mesh.data * mesh.errors, 2)
            if totals is not None:
                totals += mesh.totals
                tot_errors += np.power(mesh.totals * mesh.totals_err, 2)

    back_to_relative_values(data, errors)

    if totals is not None:
        back_to_relative_values(totals, tot_errors)

    new_mesh = fmesh.FMesh(
        number,
        _sum.kind,
        _sum.geometry_spec,
        _sum.e,
        data,
        errors,
        totals,
        tot_errors,
        comment if comment else _sum.comment,
    )

    new_mesh.save_2_npz(
        out,
        check_existing_file_strategy=file_exists_strategy,
    )

    __LOG.info("Sum is saved to {}", out)


def back_to_relative_values(data, errors) -> None:
    bad_idx = data <= 0.0
    idx = np.logical_not(bad_idx)
    errors[idx] = np.sqrt(errors[idx]) / data[idx]
    errors[bad_idx] = 1.0
    errors[errors > 1.0] = 1.0
