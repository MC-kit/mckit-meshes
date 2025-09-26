from __future__ import annotations

from typing import TYPE_CHECKING

from pathlib import Path

import click
import numpy as np

from mckit_meshes.fmesh import read_meshtal
from mckit_meshes.wgtmesh import WgtMesh, make_geometry_spec

if TYPE_CHECKING:
    from mckit_meshes.fmesh import FMesh


def convert_mesh_to_weights(
    path: Path, *, mesh_no: int | None = None, beta: int = 5, soft: float | None = None
) -> WgtMesh:
    def get_weights(_mesh: FMesh):
        _weights = _mesh.data
        if soft is not None:
            assert 0.0 < soft < 1.0
            _weights = np.power(_weights, soft)
        # See:
        # A. J. van Wijk, G. Van den Eynde, and J. E. Hoogenboom,
        # “An easy to implement global variance reduction procedure for MCNP,”
        # Annals of Nuclear Energy, vol. 38, no. 11, pp. 2496-2503, Nov. 2011,
        # doi: 10.1016/j.anucene.2011.07.037.
        norm_factor = 2.0 / (beta + 1) / _weights.max()
        return _weights * norm_factor

    with path.open("r") as fid:
        meshes = read_meshtal(fid)
        if mesh_no is None:
            if len(meshes) != 1:
                raise ValueError(
                    f"Meshtal file {path} contains more than one mesh."
                    " Please specify the mesh number using option --mesh."
                )
        else:
            meshes = list(filter(lambda m: m.name == mesh_no, meshes))
            lm = len(meshes)
            if lm == 0:
                raise ValueError(f"Mesh {mesh_no} is not found in {path}")
            assert lm == 1
        mesh: FMesh = meshes[0]
        energies = [mesh.e]
        weights = [get_weights(mesh)]

    gs = make_geometry_spec(
        origin=mesh.origin,
        ibins=mesh.ibins,
        jbins=mesh.jbins,
        kbins=mesh.kbins,
        axs=mesh.axis,
        vec=mesh.vec,
    ).adjust_axs_vec_for_mcnp()
    wgtmesh = WgtMesh(
        geometry_spec=gs,
        energies=energies,
        weights=weights,
    )
    return wgtmesh


def mesh2wgt( mesh_file: Path, *, mesh: int | None = None, beta: int = 5, soft: float | None = None, override: bool= False):
    """Converts mesh tally file to weight mesh file.

    This can be used for GVR weights computing.
    """
    path = Path(str(mesh_file))
    assert path.exists(), f"Path {path} is not found"
    wgtmesh = convert_mesh_to_weights(path, mesh_no=mesh, beta=beta, soft=soft)
    out = path.with_suffix(".wwinp")
    if override or not out.exists():
        with out.open("wt") as stream:
            wgtmesh.write(stream)
    else:
        raise FileExistsError(out, " consider --override option")
