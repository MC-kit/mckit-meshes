"""Normalize weights.

The weight value at given point and given energy bin is to be of the given value.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from eliot import start_action

from mckit_meshes.utils import get_override_strategy
from mckit_meshes.wgtmesh import WgtMesh

if TYPE_CHECKING:
    from pathlib import Path

    from mckit_meshes.wgtmesh import Point


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
    weight_file: Path,
    *,
    out: Path | None,
    normalization_point,
    normalization_value,
    energy_bin,
    override: bool,
):
    """Normalize weights file."""
    with start_action(action_type="normalize weights", weight_file=weight_file) as logger:
        if out is None:
            out = weight_file.with_suffix(".normalized")
        out = get_override_strategy(override=override)(out)
        assert weight_file.exists(), f"Path {weight_file} is not found"
        wgtmesh = do_normalize_weights(
            weight_file,
            normalization_point=np.fromstring(normalization_point, sep=","),
            normalization_value=normalization_value,
            energy_bin=energy_bin,
        )
        with out.open("wt") as stream:
            wgtmesh.write(stream)
        logger.add_success_fields(output=out)
