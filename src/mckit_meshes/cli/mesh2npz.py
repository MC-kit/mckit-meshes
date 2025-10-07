# noinspection PyPep8
"""Convert MCNP meshtal file to a number of npz files, one for each meshtal."""

from __future__ import annotations


from pathlib import Path

from eliot import start_action

from mckit_meshes import fmesh
from mckit_meshes.utils import get_override_strategy, revise_files


def mesh2npz(
    *mesh_tallies: Path,
    prefix: Path,
    override: bool = False,
) -> None:
    """Convert MCNP meshtal file to a number of npz files, one for each mesh tally."""
    mesh_tallies = revise_files("m", *mesh_tallies)
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
                    check_existing_file_strategy=get_override_strategy(override=override),
                )
