from __future__ import annotations

from typing import TYPE_CHECKING

from pathlib import Path

import numpy as np

from eliot import start_action

from mckit_meshes.wgtmesh import WgtMesh

if TYPE_CHECKING:
    from mckit_meshes.wgtmesh import Point


def load(path: Path) -> WgtMesh:
    with start_action(action_type="load weights", input=path), path.open("r") as fid:
        return WgtMesh.read(fid)


def invert(mesh: WgtMesh, normalization_point: Point, normalization_value=1.0) -> WgtMesh:
    with start_action(
        action_type="invert weights",
        normalization_point=normalization_point,
        normalization_value=normalization_value,
    ):
        local_normalisation_point = mesh.geometry_spec.local_coordinates(normalization_point)
        return mesh.invert(local_normalisation_point, normalization_value)


def check_output_exists(path: Path, *, override: bool) -> Path:
    if not override and path.exists():
        raise FileExistsError(path, ' - consider to use "--override" option.')
    return path


def save(mesh: WgtMesh, path: Path) -> None:
    with path.open("wt") as stream:
        mesh.write(stream)


def invwgt(
    path: Path,
    *,
    normalization_point: str,
    normalization_value: float = 1.0,
    override: bool = False,
    out: Path | None = None,
) -> None:
    """Invert MCNP weight window file: all values became reciprocals (w[...] = 1/w[...]).

    Use this for anti-forward weight estimations.

    Features:
        - Zero values remain zeros.
        - After all normalises the resulting weights, so at given point the weight is 1.0.

    Multiple energy bins are not implemented yet.
    """
    with start_action(action_type="Inverting weights") as eliot:
        _normalization_point = np.fromstring(normalization_point, sep=",", dtype=float)

        if out is None:
            out = (Path.cwd() / path.name).with_suffix(".inv-wwinp")

        save(
            invert(load(path), _normalization_point, normalization_value),
            check_output_exists(out, override=override),
        )
        eliot.add_success_fields(output=out)
