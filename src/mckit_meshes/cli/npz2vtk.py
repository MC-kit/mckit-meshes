"""Convert npz files to VTK vtr-files."""

from __future__ import annotations


from pathlib import Path

from eliot import start_action

from mckit_meshes import fmesh
from mckit_meshes.utils import get_override_strategy, revise_files


def npz2vtk(*npz_files: Path, prefix: str | Path, override: bool = False) -> None:
    """Convert MCNP meshtal file to a number of npz files, one for each mesh tally.

    Parameters
    ----------
        prefix
            output directory
        npz_files
            files to process, optional
        override
            define behaviour when output file, exists, default - rise FileExistsError.
    """
    npz_files = revise_files("npz", *npz_files)
    prefix = Path(prefix)
    file_exists_strategy = get_override_strategy(override=override)
    for npz in npz_files:
        with start_action(action_type="processing", _npz_file=npz) as logger:
            prefix.mkdir(parents=True, exist_ok=True)
            mesh = fmesh.FMesh.load_npz(npz)
            vtk_file_stem = f"{prefix / npz.stem}"
            vtk_file_name = (
                vtk_file_stem + ".vtr"
            )  # TODO dvp: revise this when it comes to saving structured mesh
            file_exists_strategy(Path(vtk_file_name))
            vtk = mesh.save2vtk(vtk_file_stem)
            logger.add_success_fields(saved_to=vtk)
