# noinspection PyPep8
"""Convert MCNP meshtal file to a number of npz files, one for each meshtal."""

from __future__ import annotations


from pathlib import Path

from eliot import start_action

from mckit_meshes import fmesh
from mckit_meshes.utils import check_if_path_exists


def revise_mesh_tallies(*mesh_tallies: Path) -> tuple[Path, ...]:
    if mesh_tallies:
        return mesh_tallies
    with start_action(action_type="look for meshtally files") as logger:
        cwd = Path.cwd()
        mesh_tallies = tuple(cwd.glob("*.m"))
        if not mesh_tallies:
            logger.log(message_type="WARNING", reason="No .m-files found", directory=cwd.absolute())
    return mesh_tallies


def mesh2npz(
    *mesh_tallies: Path,
    prefix: Path,
    override: bool = False,
) -> None:
    """Convert MCNP meshtal file to a number of npz files, one for each mesh tally."""
    mesh_tallies = revise_mesh_tallies(*mesh_tallies)
    single_input = len(mesh_tallies) == 1
    prefix = Path(prefix)
    for m in mesh_tallies:
        with start_action(action_type="processing .m-file", meshtally_file=m):
            p = prefix if single_input else prefix / m.stem
            p.mkdir(parents=True, exist_ok=True)
            with m.open() as stream:
                fmesh.m_2_npz(
                    stream,
                    prefix=p,
                    check_existing_file_strategy=check_if_path_exists(override=override),
                )
