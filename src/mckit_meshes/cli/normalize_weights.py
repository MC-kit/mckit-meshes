"""Normalize weights: the weight value at given point and given energy bin is to be of the given value."""

from __future__ import annotations


from eliot import start_action
import numpy as np

from mckit_meshes.wgtmesh import WgtMesh
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mckit_meshes.wgtmesh import Point
    from pathlib import Path


def do_normalize_weights(
    path: Path,
    *,
    normalization_point: Point,
    normalization_value: float = 1 / 3,
    energy_bin: int = -1,
) -> WgtMesh:
    with path.open("r") as fid:
        wgtmesh = WgtMesh.read(fid)
    local_normalisation_point = wgtmesh.geometry_spec.local_coordinates(normalization_point)
    return wgtmesh.normalize(local_normalisation_point, normalization_value, energy_bin)


def normalize_weights(
    override: bool,
    normalization_point,
    normalization_value,
    energy_bin,
    weight_file: Path,
):
    """Normalize weights file."""
    if not override and out.exists():
        raise FileExistsError(out, " consider --override option")
    with start_action(action_type="normalize weights", weight_file=weight_file) as logger:
        assert weight_file.exists(), f"Path {weight_file} is not found"
        wgtmesh = do_normalize_weights(
            weight_file,
            normalization_point=np.fromstring(normalization_point, sep=","),
            normalization_value=normalization_value,
            energy_bin=energy_bin,
        )
        out = weight_file.with_suffix(".normalized")
        with out.open("wt") as stream:
            wgtmesh.write(stream)
        logger.add_success_fields(output=out)
