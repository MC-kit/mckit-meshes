from __future__ import annotations

from typing import TYPE_CHECKING


import numpy as np

from mckit_meshes.wgtmesh import WgtMesh

if TYPE_CHECKING:
    from pathlib import Path

    from mckit_meshes.wgtmesh import Point


def load(path: Path) -> WgtMesh:
    with path.open("r") as fid:
        return WgtMesh.read(fid)


def invert(mesh: WgtMesh, normalization_point: Point) -> WgtMesh:
    local_normalisation_point = mesh.geometry_spec.local_coordinates(normalization_point)
    return mesh.invert(local_normalisation_point)


def check_output_exists(path: Path, *, override: bool) -> Path:
    if not override and path.exists():
        raise FileExistsError(path, ' - consider to use "--override" option.')
    return path


def save(mesh: WgtMesh, path: Path) -> None:
    with path.open("wt") as stream:
        mesh.write(stream)


def invwgt(path: Path, *, normalisation_point: str, override: bool = False) -> None:
    """Inverts MCNP weight window file: all values became reciprocals (w[...] = 1/w[...]).

    Use this for anti-forward weight estimations.

    Features:
        - Zero values remain zeros.
        - After all normalises the resulting weights, so at given point the weight is 1.0.

    Multiple energy bins are not implemented yet.
    """
    _normalisation_point = np.fromstring(normalisation_point, sep=",", dtype=float)
    save(
        invert(load(path), _normalisation_point),
        check_output_exists(path.with_suffix(".inv-wwinp"), override=override),
    )
