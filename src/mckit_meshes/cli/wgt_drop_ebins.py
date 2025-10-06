from __future__ import annotations

from typing import TYPE_CHECKING

from mckit_meshes.wgtmesh import WgtMesh
from mckit_meshes.utils import get_override_strategy

if TYPE_CHECKING:
    from pathlib import Path


def load(path: Path) -> WgtMesh:
    with path.open("r") as fid:
        return WgtMesh.read(fid)


def save(mesh: WgtMesh, path: Path) -> None:
    with path.open("wt") as stream:
        mesh.write(stream)


def wgt_drop_ebins(*, override: bool, output, min_energy: float, part: int, wgtfile: Path) -> None:
    """Drop bins with upper boundary below the specified min_energy.

    Use this to drop the too ambitious bins generated with ADVANTG at lower energies.
    """
    save(
        load(wgtfile).drop_lower_energies(min_energy, part),
        get_override_strategy(override=override)(output),
    )
