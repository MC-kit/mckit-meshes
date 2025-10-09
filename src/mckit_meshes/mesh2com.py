from __future__ import annotations

from typing import TYPE_CHECKING, TextIO

import sys

if TYPE_CHECKING:
    from numpy.typing import NDArray


def mesh2com(xbins: NDArray, ybins: NDArray, zbins: NDArray, out: TextIO = sys.stdout) -> None:
    """Print content for mcnp COM-file to plot crossections.

    The bins are usually from mesh spec.
    The crossections go through the voxels centers and with normals in x, y and z directions.

    Parameters
    ----------
    xbins
        Bins along X-axis
    ybins
        ... Y-axis
    zbins
        ... Z-axis
    out, optional
        output stream, by default sys.stdout
    """
    print("label 0 0", file=out)
    print("file all", file=out)
    current_basis = current_extent = current_origin = None
    basis = "0 1 0 0 0 1"
    extent = ybins[-1] - ybins[0], zbins[-1] - zbins[0]
    extent = max(*extent)
    extent = extent, extent
    origin_y, origin_z = 0.5 * (ybins[-1] + ybins[0]), 0.5 * (zbins[-1] + zbins[0])
    xmids = 0.5 * (xbins[:-1] + xbins[1:])
    for x in xmids:
        command = ""
        if not current_basis or current_basis != basis:
            command = "basis " + basis
            current_basis = basis
        if not current_extent or current_extent != extent:
            command += " extent {} {}".format(*extent)
            current_extent = extent
        origin = x, origin_y, origin_z
        if not current_origin or current_origin != origin:
            command += " origin {} {} {}".format(*origin)
            current_origin = origin
        if command:
            print(command, file=out)
    basis = "1 0 0 0 0 1"
    current_extent = None
    extent = xbins[-1] - xbins[0], zbins[-1] - zbins[0]
    extent = max(*extent)
    extent = extent, extent
    origin_x, origin_z = 0.5 * (xbins[-1] + xbins[0]), 0.5 * (zbins[-1] + zbins[0])
    ymids = 0.5 * (ybins[:-1] + ybins[1:])
    for y in ymids:
        command = ""
        if not current_basis or current_basis != basis:
            command = "basis " + basis
            current_basis = basis
        if not current_extent or current_extent != extent:
            command += " extent {} {}".format(*extent)
            current_extent = extent
        origin = origin_x, y, origin_z
        if not current_origin or current_origin != origin:
            command += " origin {} {} {}".format(*origin)
            current_origin = origin
        if command:
            print(command, file=out)
    basis = "1 0 0 0 1 0"
    current_extent = None
    extent = xbins[-1] - xbins[0], ybins[-1] - ybins[0]
    extent = max(*extent)
    extent = extent, extent
    origin_x, origin_y = 0.5 * (xbins[-1] + xbins[0]), 0.5 * (ybins[-1] + ybins[0])
    zmids = 0.5 * (zbins[:-1] + zbins[1:])
    for z in zmids:
        command = ""
        if not current_basis or current_basis != basis:
            command = "basis " + basis
            current_basis = basis
        if not current_extent or current_extent != extent:
            command += " extent {} {}".format(*extent)
            current_extent = extent
        origin = origin_x, origin_y, z
        if not current_origin or current_origin != origin:
            command += " origin {} {} {}".format(*origin)
            current_origin = origin
        if command:
            print(command, file=out)
    print("end", file=out)
