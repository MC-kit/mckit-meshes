"""Normalize weights: the weight value at given point and given energy bin is to be of the given value."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from mckit_meshes.wgtmesh import Point, WgtMesh


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


@click.command(options_metavar="options...")
@click.option(
    "--override/--no-override",
    is_flag=True,
    default=False,
    help="override existing file (default:False)",
)
@click.option(
    "--normalization-point",
    metavar="NORMALIZATION_POINT",
    default="610, 0, 57",
    help='Point where to set weight to normalized value"'
    '(default:"610, 0, 57" - ITER magnetic axis position in standard scenario)',
)
@click.option(
    "--normalization-value",
    metavar="NORMALIZATION_VALUE",
    type=click.FLOAT,
    default=1 / 3,
    help="The value to set at normalization point and energy bin",
)
@click.option(
    "--energy-bin",
    metavar="ENERGY_BIN",
    type=click.INT,
    default=-1,
    help="Energy bin at which to normalize (default: -1, i.e. the last bin )",
)
@click.argument(
    "weight-file",
    metavar="WEIGHT_FILE",
    type=click.Path(exists=True),
    nargs=1,
    required=True,
)
def normalize_weights(
    override: bool,
    normalization_point,
    normalization_value,
    energy_bin,
    weight_file: click.Path,
):
    """Normalize weights file."""
    path = Path(str(weight_file))
    assert path.exists(), f"Path {path} is not found"
    wgtmesh = do_normalize_weights(
        path,
        normalization_point=np.fromstring(normalization_point, sep=","),
        normalization_value=normalization_value,
        energy_bin=energy_bin,
    )
    out = path.with_suffix(".normalized")
    if override or not out.exists():
        with out.open("wt") as stream:
            wgtmesh.write(stream)
    else:
        raise FileExistsError(out, " consider --override option")


if __name__ == "__main__":
    normalize_weights()
