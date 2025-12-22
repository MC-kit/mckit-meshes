"""Convert npz files to VTK vtr-files."""

from __future__ import annotations

import logging

from pathlib import Path

import numpy as np

from eliot import log_message, start_action

from mckit_meshes import fmesh
from mckit_meshes.utils import get_override_strategy

__LOG = logging.getLogger(__name__)


def add(
    *npz_files: Path,
    out: Path | None = None,
    comment: str | None = None,
    number: int = 1,
    override: bool = False,
) -> None:
    """Convert MCNP meshtal file to a number of npz files, one for each mesh tally.

    Applicable to meshes with the same geometry.

    Note
    ----
    We assume that statistics is the same for all the meshes, preferably they
    are produced in the same run (like neutron and photon heat).

    Parameters
    ----------
    out
        output file
    comment
        comment for a new fmesh npz file,
        if not provided, then the comment from the first mesh is used
    number
        ... to assign to resulting mesh
    npz_files
        files to process, optional
    override
        define behaviour when output file, exists, default - rise FileExistsError.
    """
    if not npz_files:
        __LOG.warning("No files specified to process")
        return

    if out is None:
        out = Path("+".join(x.stem for x in npz_files)).with_suffix(".npz")

    out.parent.mkdir(parents=True, exist_ok=True)
    file_exists_strategy = get_override_strategy(override=override)

    with start_action(action_type="creating sum of meshes", out=out):
        _sum: fmesh.FMesh | None = None
        data = None
        errors = None
        totals = None
        tot_errors = None
        for npz in npz_files:
            with start_action(action_type="adding mesh", mesh=npz):
                mesh = fmesh.FMesh.load_npz(npz)
                if _sum is None:
                    _sum = mesh
                    data = _sum.data
                    errors = np.power(_sum.errors * _sum.data, 2)
                    totals = _sum.totals
                    if totals is not None and _sum.totals_err is not None:
                        tot_errors = np.power(_sum.totals_err * _sum.totals, 2)
                else:
                    if not _sum.is_equal_by_geometry(mesh):
                        msg = f"Mesh {npz} is not compatible by geometry with previous ones."
                        raise ValueError(msg)
                    data += mesh.data
                    errors += np.power(mesh.data * mesh.errors, 2)
                    if totals is not None and mesh.totals is not None:
                        totals += mesh.totals
                        tot_errors += np.power(mesh.totals * mesh.totals_err, 2)

        _back_to_relative_values(data, errors)

        if totals is not None:
            _back_to_relative_values(totals, tot_errors)

        if _sum is None or data is None or errors is None:
            msg = "No npz-files found."
            raise ValueError(msg)

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

        with start_action(action_type="save mesh") as logger:
            new_mesh.save_2_npz(
                out,
                check_existing_file_strategy=file_exists_strategy,
            )
            logger.add_success_fields(out=out.absolute)

        __LOG.info("Sum is saved to %s", out)


def _back_to_relative_values(data, errors) -> None:
    bad_idx = data < 0.0
    if np.any(bad_idx):
        log_message(
            message_type="WARNING", reason="found negative data", count=np.count_nonzero(bad_idx)
        )
        __LOG.warning("found negative data")
        data[bad_idx] = 0.0
        errors[bad_idx] = 0.0
    bad_idx = data <= 0.0
    idx = np.logical_not(bad_idx)
    errors[idx] = np.sqrt(errors[idx]) / data[idx]
    errors[errors > 1.0] = 1.0
