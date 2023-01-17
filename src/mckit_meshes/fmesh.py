"""Classes and functions for operations with fmesh tallies."""

# TODO dvp: redesign this class as xarray data structure.
#           multidimensional array with coordinates is more appropriate for this class.

from __future__ import annotations

from typing import Callable, Generator, Iterable, List, Optional, TextIO, Tuple, Union

import logging

from pathlib import Path

import mckit_meshes.mesh.geometry_spec as gc
import mckit_meshes.utils.no_daemon_process as ndp
import mckit_meshes.utils.rebin as rebin
import numpy as np

from mckit_meshes.particle_kind import ParticleKind as Kind
from mckit_meshes.utils.io import raise_error_when_file_exists_strategy
from numpy.typing import ArrayLike
from pyevtk.hl import gridToVTK
from toolz.itertoolz import concatv

__LOG = logging.getLogger(__name__)


def _expand_args(args):
    """Adapter to invoke rebin.

    Args:
        args: what to rebin and how

    Returns:
        Rebinned result
    """
    return rebin.rebin_nd(*args)


class FMesh:
    """Fmesh tally.

    Class to load MCNP FMESH output file, process and write in various formats.
    Processing includes normalization, merging and conversion to weights mesh.

    Attrs:
        NPZ_MARK: Signature to be stored in the of meta entry in a npz file.
            This is used to check that the file is for FMesh object.
        NPZ_FORMAT: Identifies version of format of data stored in npz file.
    """

    NPZ_MARK = np.int16(5445)
    NPZ_FORMAT = np.int16(4)

    class X(RuntimeError):
        """FMesh class specific exception."""

    def __init__(
        self,
        name: int,
        kind: int | Kind,
        geometry_spec: gc.AbstractGeometrySpec,
        ebins: np.ndarray,
        data: np.ndarray,
        errors: np.ndarray,
        totals: Optional[np.ndarray] = None,
        totals_err: Optional[np.ndarray] = None,
        comment: Optional[str] = None,
    ) -> None:
        """Construct FMesh instance object.

        Args:
            name: FMESH tally number
            kind: neutron, photon
            geometry_spec: mesh geometry specification
            ebins: Energy bin boundaries.
            data: Data values at centers of mesh cells.
                  Shape (Ne-1)x(Nx-1)x(Ny-1)x(Nz-1), where Ne, Nx, Ny and Nz - the number
                  of corresponding bin boundaries.
            errors:
                Relative errors of corresponding data values.
                Shape (Ne-1)x(Nx-1)x(Ny-1)x(Nz-1), where Ne, Nx, Ny and Nz - the number
                of corresponding bin boundaries.
            totals: Can be provided with 'Total' data from mesh file,
                    if there are more than 1 energy bin, optional.
            totals_err: Can be provided with data from mesh file,
                    if there are more than 1 energy bin, optional.
            comment: Comment from a meshtal file (content of FC card in MCNP model).
        """
        self.name = int(name)
        self.kind = Kind(kind)

        self._geometry_spec: Union[
            gc.CartesianGeometrySpec, gc.CylinderGeometrySpec, gc.AbstractGeometrySpec
        ] = geometry_spec
        self.bins = {}
        self.bins["X"] = self._x = geometry_spec.ibins
        self.bins["Y"] = self._y = geometry_spec.jbins
        self.bins["Z"] = self._z = geometry_spec.kbins
        self.bins["E"] = self._e = gc.as_float_array(ebins)
        self.data = gc.as_float_array(data)
        self.errors = gc.as_float_array(errors)
        if 2 < self._e.size:
            if totals is None:
                if totals_err is not None:
                    raise ValueError("totals are omitted but totals_err are provided")
                self._totals = np.sum(self.data, axis=0)
                non_zero = self._totals > 0.0
                self._totals_err = np.zeros_like(self._totals)
                self._totals_err[non_zero] = (
                    np.sqrt(np.sum((self.errors * self.data) ** 2, axis=0))[non_zero]
                    / self._totals[non_zero]
                )
            else:
                if totals_err is None:
                    raise ValueError("totals are provided but totals_err are not")
                self._totals = np.asarray(totals, dtype=float)
                self._totals_err = np.asarray(totals_err, dtype=float)
        else:
            self._totals = None
            self._totals_err = None
        self._comment = comment
        self.check_attributes()

    @property
    def e(self) -> np.ndarray:
        """Energy bins."""
        return self._e

    @property
    def has_multiple_energy_bins(self) -> bool:
        """Check if there's more than 1 energy bin.

        If True, then totals and totals err should present.

        Returns:
            True if there are more than one energy bins.
        """
        return 2 < self.e.size

    @property
    def ibins(self) -> np.ndarray:
        """Synonym to geometry bins x or R."""
        return self._geometry_spec.ibins

    @property
    def jbins(self) -> np.ndarray:
        """Synonym to geometry jbins (y or Z).

        Returns:
            jbins from the geometry spec
        """
        return self._geometry_spec.jbins

    @property
    def kbins(self) -> np.ndarray:
        """Synonym to geometry kbins (z or Theta).

        Returns:
            kbins from the geometry spec
        """
        return self._geometry_spec.kbins

    @property
    def totals(self) -> np.ndarray | None:
        """Total values over energy."""
        return self._totals

    @property
    def totals_err(self) -> np.ndarray | None:
        """Relative errors of total values over energy."""
        return self._totals_err

    @property
    def comment(self) -> str | None:
        """Comment from FC card for this mesh tally."""
        return self._comment

    @property
    def origin(self) -> np.ndarray:
        """Get origin for cylinder mesh."""
        return self._geometry_spec.origin

    @property
    def axis(self) -> np.ndarray:
        """Get axis of a cylinder mesh."""
        return self._geometry_spec.axs

    @property
    def vec(self) -> np.ndarray:
        """Get Theta reference direction for cylinder mesth."""
        return self._geometry_spec.vec

    @property
    def is_cylinder(self) -> bool:
        """Is this mesh cylinder?

        Note:
            MCNP uses `origin` on mesh tally specification, both rectilinear and cylinder,
            but outputs origin only for cylinder mesh.

        Returns:
            True if this is a cylinder mesh.
        """
        return self._geometry_spec.cylinder

    @property
    def total_precision(self) -> float:
        """"""
        if self.has_multiple_energy_bins:
            return self.totals_err[
                -1
            ]  # TODO dvp: assumes max energy bin is most representative, check usage
        else:
            return self.errors[0, 0, 0, 0]

    def check_attributes(self) -> None:
        """Check consistency of attributes."""
        assert 2 <= self._e.size
        assert self.data.shape == self.errors.shape
        assert self.data.shape == (self.e.size - 1,) + self._geometry_spec.bins_shape
        assert (
            self._totals is None
            or isinstance(self._totals, np.ndarray)
            and isinstance(self._totals_err, np.ndarray)
            and self._totals.shape == self._totals_err.shape
            and self._totals.shape == self._geometry_spec.bins_shape
        )

    def is_equal_by_mesh(self, other: "FMesh") -> bool:
        """Args:

          other: "FMesh":

        Returns:
        """
        return (
            self.kind == other.kind
            and self._geometry_spec == other._geometry_spec
            and np.array_equal(self.e, other.e)
        )

    def has_better_precision_than(self, other: "FMesh") -> bool:
        """Compare precision achieved for the meshes.

        Args:
            other: mesh

        Returns:
            True if this mesh is more precise than other one.
        """
        assert self.is_equal_by_mesh(other), "Incompatible meshes for precision comparison."
        return self.total_precision < other.total_precision

    def surrounds_point(self, x: float, y: float, z: float, local: bool = True) -> bool:
        """Check if a point x,y,z is within the mesh spatial grid.

        Args:
            x: point's coordinate x
            y: ... y
            z: ... z
            local: if True the point coordinates are local for the mesh

        Returns:
            True if point is within the mesh's grid.
        """
        return self._geometry_spec.surrounds_point(x, y, z, local)

    def get_spectrum(
        self, x: ArrayLike, y: ArrayLike, z: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """Gets energy spectrum at the specified point.

        Args:
            x: X, Y and Z coordinate of the point where energy spectrum is
                required. If point is located outside the mesh, zeros are returned.
            y: ...
            z: ...

        Returns:
            ebins, data, err
                Energy bin boundaries, group energy spectrum and relative errors.
        """
        key_index = {0: "X", 1: "Y", 2: "Z"}
        values = [x, y, z]
        result_data = self.data
        result_error = self.errors
        for i, value in reversed(list(enumerate(values))):
            key = key_index[i]
            index = np.searchsorted(self.bins[key], value) - 1
            if index < 0 or index >= self.bins[key].size - 1:
                result_data *= 0
                result_error *= 0
                index = 0
            result_data = result_data.take(index, axis=i + 1)
            result_error = result_error.take(index, axis=i + 1)
        return self.e, result_data, result_error

    def select_indexes(
        self, *, x: ArrayLike = None, y: ArrayLike = None, z: ArrayLike = None
    ) -> Tuple[
        Union[int, slice, np.ndarray],
        Union[int, slice, np.ndarray],
        Union[int, slice, np.ndarray],
    ]:
        """Select indexes in spatial bins corresponding to given coordinates.

        If coordinate is not specified, then return all the points along this coordinate.

        Args:
            x:  (Default value = None)
            y:  (Default value = None)
            z:  (Default value = None)

        Returns:
            tuple of indexes along the coordinates
        """
        return self._geometry_spec.select_indexes(i_values=x, j_values=y, k_values=z)

    def get_totals(
        self, *, x: ArrayLike = None, y: ArrayLike = None, z: ArrayLike = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get total values for specified grid points.

        If a coordinate is not specified, than all the points along this coordinate.

        Args:
            x:  (Default value = None)
            y:  (Default value = None)
            z:  (Default value = None)

        Returns:
            totals, total_err for the specified coordinates
        """
        if self._totals is None:
            return None
        found_x, found_y, found_z = self.select_indexes(x=x, y=y, z=z)
        totals, rel_error = (
            self._totals[found_x, found_y, found_z],
            self._totals_err[found_x, found_y, found_z],
        )
        return totals, rel_error

    def save_2_npz(
        self,
        filename: Path,
        check_existing_file_strategy=raise_error_when_file_exists_strategy,
    ) -> None:
        """Writes this object to numpy npz file.

        Args:
            filename: Filename to which the object is saved. If file is a
                file-object, then the filename is unchanged. If file is a string,
                a .npz extension will be appended to the file name if it does not
                already have one. By default, the name of file is the tally name.
            check_existing_file_strategy: what to do if an output file already exists
        """
        if filename.suffix != ".npz":
            filename = filename.with_suffix(".npz")

        check_existing_file_strategy(filename)

        kwd = {
            "meta": np.array(
                [FMesh.NPZ_MARK, FMesh.NPZ_FORMAT, self.name, self.kind],
                dtype=np.uint32,
            ),
            "E": self.e,
            "X": self.ibins,
            "Y": self.jbins,
            "Z": self.kbins,
            "data": self.data,
            "errors": self.errors,
            "totals": self.totals,
            "totals_err": self.totals_err,
        }
        if self.comment:
            kwd["comment"] = np.array(self.comment)
        if self.is_cylinder:
            kwd["origin"] = np.array(self._geometry_spec.origin)
            kwd["axis"] = np.array(self._geometry_spec.axs)

        filename.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(filename), **kwd)

    @classmethod
    def load_npz(cls, _file: Union[str, Path]) -> "FMesh":
        """Loads Fmesh object from the binary file.

        Args:
          _file: npz-file to load from.

        Returns:
            The loaded FMesh object.
        """
        if isinstance(_file, Path):
            _file = str(_file)
        with np.load(_file) as data:
            meta = data["meta"]
            mark = meta[0]
            assert mark == FMesh.NPZ_MARK, f"Incompatible file format {_file}"
            version = meta[1]
            name, kind = meta[2:4]
            if 1 <= version:
                e = data["E"]
                x = data["X"]
                y = data["Y"]
                z = data["Z"]
                d = data["data"]
                r = data["errors"]
                if 2 < e.size:
                    try:
                        totals = data["totals"]
                        totals_err = data["totals_err"]
                    except KeyError:
                        totals = None
                        totals_err = None
                else:
                    totals = None
                    totals_err = None
                comment = None
                origin = None
                axis = None
                if 2 <= version:
                    if "comment" in data:
                        comment = data["comment"]
                        comment = comment.item()
                        assert comment
                    if 3 <= version:
                        if "origin" in data:
                            assert "axis" in data
                            origin = data["origin"]
                            axis = data["axis"]
                            assert origin.size == 3
                            assert axis.size == 3
                        if 4 <= version:
                            pass
                        else:
                            kind = int(kind) + 1
                if origin is None:
                    geometry_spec = gc.CartesianGeometrySpec(x, y, z)
                else:
                    geometry_spec = gc.CylinderGeometrySpec(x, y, z, origin=origin, axs=axis)
                return cls(
                    name,
                    kind,
                    geometry_spec,
                    e,
                    d,
                    r,
                    totals,
                    totals_err,
                    comment=comment,
                )
            else:
                raise FMesh.X("Invalid version for FMesh file %d" % version)

    def save2vtk(self, filename: str = None, data_name: str = None) -> str:
        """Saves this fmesh data to vtk file.

        Data is saved for every energy bin and, if there are multiple energy bins,
        for total values (sum across energy axis).

        Args:
            filename: Name of file to which this object is stored. A .vtk extension will
                be appended. By default, the name of file is the tally name.
            data_name: Name of data which will appear in vtk file. If None, tally name
                and type will be used.

        Returns:
            Full path to saved VTK file.
        """
        assert not self.is_cylinder, "Not implemented for cylinder geometry"
        # TODO dvp: implement for cylinder geometry (see iwwgvr project for example).

        if filename is None:
            filename = str(self.name)
        if data_name is None:
            data_name = str(self.name) + " " + self.kind.name

        cell_data = {}
        for i, e in enumerate(self.e[1:]):
            key = data_name + " E={0:.4e}".format(e)
            cell_data[key] = self.data[i, :, :, :]
        if self.has_multiple_energy_bins:
            name = data_name + " total"
            cell_data[name] = np.sum(self.data, axis=0)
        return gridToVTK(filename, self.ibins, self.jbins, self.kbins, cellData=cell_data)

    # noinspection PyUnresolvedReferences
    def save_2_mcnp_mesh(self, stream: TextIO) -> None:
        """Saves the mesh in a file in a format similar to mcnp mesh tally textual representation.

        Args:
            stream: stream to store the mesh.
        """

        def format_comment(a: "FMesh") -> str:
            return "\n" + a.comment if a.comment else ""

        header = f"""
 Mesh Tally Number   {self.name}{format_comment(self)}
 This is a {self.kind.name} mesh tally.

 Tally bin boundaries:{self.format_cylinder_origin_and_axis_label()}
"""[
            1:-1
        ]
        e = self.e[1:]
        x = 0.5 * (self.ibins[1:] + self.ibins[:-1])
        y = 0.5 * (self.jbins[1:] + self.jbins[:-1])
        z = 0.5 * (self.kbins[1:] + self.kbins[:-1])
        print(header, file=stream)
        print(
            f"{'R' if self.is_cylinder else 'X'} direction:",
            file=stream,
            end="",
        )
        for f in np.nditer(self.ibins):
            print("% g" % f, file=stream, end="")
        print(file=stream)
        print(
            f"{('Z' if self.is_cylinder else 'Y')} direction:",
            file=stream,
            end="",
        )
        for f in np.nditer(self.jbins):
            print(" %g" % f, file=stream, end="")
        print(file=stream)
        print(
            "{} direction:".format("Theta" if self.is_cylinder else "Z"),
            file=stream,
            end="",
        )
        for f in np.nditer(self.kbins):
            print(" %g" % f, file=stream, end="")
        print(file=stream)
        print("Energy bin boundaries:", file=stream, end="")
        for f in np.nditer(self.e):
            print(" %g" % f, file=stream, end="")
        print("\n", file=stream)
        if self.is_cylinder:
            print(
                "   Energy         R         Z         Th    Result     Rel Error",
                file=stream,
            )
        else:
            print(
                "   Energy         X         Y         Z     Result     Rel Error",
                file=stream,
            )

        for ie in range(e.size):
            for ix in range(x.size):
                for iy in range(y.size):
                    for iz in range(z.size):
                        value = self.data[ie, ix, iy, iz]
                        err = self.errors[ie, ix, iy, iz]
                        row = " %10.3e%10.3f%10.3f%10.3f %11.5e %11.5e" % (
                            e[ie],
                            x[ix],
                            y[iy],
                            z[iz],
                            value,
                            err,
                        )
                        print(row, file=stream)

        for ix in range(x.size):
            for iy in range(y.size):
                for iz in range(z.size):
                    if self._totals:
                        value = self._totals[ix, iy, iz]
                        err = self._totals_err[ix, iy, iz]
                    else:
                        portion = self.data[:, ix, iy, iz]
                        value = np.sum(portion)
                        err = portion * self.errors[:, ix, iy, iz]
                        err = np.sqrt(np.sum(err * err)) / value
                    row = "%11s%10.3f%10.3f%10.3f %11.5e %11.5e" % (
                        "   Total   ",
                        x[ix],
                        y[iy],
                        z[iz],
                        value,
                        err,
                    )
                    print(row, file=stream, end="")

            print("\n", file=stream)

    def total_by_energy(self, new_name: int = 0) -> "FMesh":
        """Integrate over energy bins.

        Args:
            new_name: name for new `FMesh`  (Default value = 0)

        Returns:
            The new FMesh object with only one energy bin.
        """
        e = np.array([self.e[0], self.e[-1]])
        data = self.totals[np.newaxis, ...]
        errors = self.totals_err[np.newaxis, ...]
        return FMesh(new_name, self.kind, self._geometry_spec, e, data, errors)

    def shrink(
        self,
        emin=None,
        emax=None,
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
        zmin=None,
        zmax=None,
        new_name=-1,
    ) -> "FMesh":
        """Select subset of e-voxels within given geometry and energy limits.

        Args:
            emin:  limits for new bins
            emax:  (Default value = None)
            xmin:  (Default value = None)
            xmax:  (Default value = None)
            ymin:  (Default value = None)
            ymax:  (Default value = None)
            zmin:  (Default value = None)
            zmax:  (Default value = None)
            new_name: name for mesh to be created, default -1.

        Returns:
            A new FMesh with reduced bins.
        """
        trim_spec = list(
            rebin.trim_spec_composer(
                [self.e, self.ibins, self.jbins, self.kbins],
                [emin, xmin, ymin, zmin],
                [emax, xmax, ymax, zmax],
            )
        )
        new_bins_list, new_data = rebin.shrink_nd(self.data, iter(trim_spec), assume_sorted=True)
        _, new_errors = rebin.shrink_nd(self.errors, iter(trim_spec), assume_sorted=True)

        assert all(np.array_equal(a, b) for a, b in zip(new_bins_list, _))

        new_ebins, new_xbins, new_ybins, new_zbins = new_bins_list
        if self.totals is None:
            new_totals = None
            new_totals_err = None
        else:
            totals_trim_spec = list(
                rebin.trim_spec_composer(
                    [self.ibins, self.jbins, self.kbins],
                    [xmin, ymin, zmin],
                    [xmax, ymax, zmax],
                )
            )
            _, new_totals = rebin.shrink_nd(self.totals, iter(totals_trim_spec), assume_sorted=True)
            _, new_totals_err = rebin.shrink_nd(
                self.totals_err, iter(totals_trim_spec), assume_sorted=True
            )

        return FMesh(
            new_name,
            self.kind,
            gc.CartesianGeometrySpec(new_xbins, new_ybins, new_zbins),
            new_ebins,
            new_data,
            new_errors,
            new_totals,
            new_totals_err,
        )

    def rebin(
        self,
        new_x: np.ndarray,
        new_y: np.ndarray,
        new_z: np.ndarray,
        new_name=-1,
        extra_process_threshold: int = 1000000,
    ) -> "FMesh":
        """Extract data for a new spatial grid.

        Args:
            new_x: A new binning over X axis.
            new_y: A new binning over Y axis.
            new_z: A new binning over Z axis.
            new_name: A name for the rebinned mesh to be created. (Default value = -1)
            extra_process_threshold:  At which size of data use multiple Python processes

        Returns:
            New FMesh object with the rebinned data.
        """
        assert not self.is_cylinder, "Not implemented for cylinder meshes"

        if self.data.size < extra_process_threshold:
            return self.rebin_single(new_x, new_y, new_z, new_name)

        # To avoid huge memory allocations, iterate over energy with external processes
        pool = ndp.Pool(processes=4)
        data_rebin_spec = list(
            rebin.rebin_spec_composer(
                [self.ibins, self.jbins, self.kbins],
                [new_x, new_y, new_z],
                axes=[0, 1, 2],
            )
        )

        def iter_over_e(data):
            for i in range(self.e.size - 1):
                yield data[i], data_rebin_spec, True

        new_data = np.stack(
            pool.map(_expand_args, iter_over_e(self.data)), axis=0
        )  # : ignore[PD013]
        t = self.data * self.errors
        new_errors = np.stack(pool.map(_expand_args, iter_over_e(t)), axis=0)  # : ignore[PD013]
        new_errors /= new_data
        if self.totals is None:
            new_totals = None
            new_totals_err = None
        else:
            new_totals = rebin.rebin_nd(self.totals, data_rebin_spec, assume_sorted=True)
            t = self.totals * self.totals_err
            new_totals_err = rebin.rebin_nd(t, data_rebin_spec, assume_sorted=True)
            new_totals_err /= new_totals

        return FMesh(
            new_name,
            self.kind,
            gc.CartesianGeometrySpec(new_x, new_y, new_z),
            self.e,
            new_data,
            new_errors,
            new_totals,
            new_totals_err,
        )

    def rebin_single(
        self,
        new_x: np.ndarray,
        new_y: np.ndarray,
        new_z: np.ndarray,
        new_name: int = -1,
    ) -> "FMesh":
        """Create FMesh object corresponding to this one by fluxes, but over new mesh.

        Ags:
            new_x: A new binning over X axis.
            new_y: A new binning over Y axis.
            new_z: A new binning over Z axis.
            new_name: name for the rebinned mesh to be created.

        Returns:
            New FMesh object with the rebinned data.
        """
        assert not self.is_cylinder, "Not implemented for cylinder meshes"

        data_rebin_spec = list(
            rebin.rebin_spec_composer(
                [self.ibins, self.jbins, self.kbins],
                [new_x, new_y, new_z],
                axes=[1, 2, 3],
            )
        )
        new_data = rebin.rebin_nd(self.data, iter(data_rebin_spec), assume_sorted=True)
        t = self.data * self.errors
        new_errors = rebin.rebin_nd(t, iter(data_rebin_spec), assume_sorted=True)
        new_errors /= new_data
        if self.totals is None:
            new_totals = None
            new_totals_err = None
        else:
            totals_rebin_spec = list(
                rebin.rebin_spec_composer(
                    [self.ibins, self.jbins, self.kbins],
                    [new_x, new_y, new_z],
                    axes=[0, 1, 2],
                )
            )
            new_totals = rebin.rebin_nd(self.totals, iter(totals_rebin_spec), assume_sorted=True)
            t = self.totals * self.totals_err
            new_totals_err = rebin.rebin_nd(t, iter(totals_rebin_spec), assume_sorted=True)
            new_totals_err /= new_totals

        return FMesh(
            new_name,
            self.kind,
            gc.CartesianGeometrySpec(new_x, new_y, new_z),
            self.e,
            new_data,
            new_errors,
            new_totals,
            new_totals_err,
        )

    def format_cylinder_origin_and_axis_label(self) -> str:
        """Format the first string for cylinder mesh."""
        if self.is_cylinder:
            return (
                f"\n  Cylinder origin at {' '.join(self._geometry_spec.origin)}, "
                f"axis in {' '.join(self._geometry_spec.axs)} direction\n"
            )
        return ""

    def __eq__(self, other) -> bool:
        if not isinstance(other, FMesh):
            return False
        res = (
            self.name == other.name
            and self.is_equal_by_mesh(other)
            and np.array_equal(self.data, other.data)
            and np.array_equal(self.errors, other.errors)
            and self.comment == other.comment
        )
        if res and self._totals:
            res = np.all(np.isclose(self.totals, other.totals)) and np.all(
                np.isclose(self.totals_err, other.totals_err)
            )
        return res

    def __hash__(self) -> int:
        return hash(
            (
                self.name,
                self.kind,
                self._geometry_spec,
                self.e,
                self.data,
                self.errors,
                self.comment,
            )
        )

    def __repr__(self) -> str:
        msg = (
            "Fmesh({name}, {kind}, {xmin}..{xmax}, {ymin}..{ymax}, {zmin}..{zmax}, {emin}..{emax})"
        )
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self._geometry_spec.boundaries
        return msg.format(
            name=self.name,
            kind=self.kind,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            zmin=zmin,
            zmax=zmax,
            emin=self.e[0],
            emax=self.e[-1],
        )


# noinspection PyTypeChecker,PyProtectedMember
def merge_tallies(
    name: int, kind: int, *tally_weight: Tuple[FMesh, float], comment: str = None
) -> FMesh:
    """Makes superposition of tallies with specific weights.

    Args:
        name: Name of new fmesh tally.
        kind: Type of new fmesh tally. It can be -1 (or any arbitrary integer).
        tally_weight: List of tally-weight pairs (tuples). tally is FMesh instance. weight
                    is float.
        comment: A comment to assign to the new mesh tally

    Returns:
        The merged FMesh.
    """
    result_data = None
    errors = None
    geometry_spec = None
    ebins = None
    for t, w in tally_weight:  # type: FMesh, float
        if result_data is None:
            result_data = t.data * w
            errors = (t.errors * t.data * w) ** 2
            geometry_spec = t._geometry_spec
            ebins = t.e
        else:
            result_data += t.data * w
            errors += (t.errors * t.data * w) ** 2
            assert geometry_spec == t._geometry_spec
            assert np.array_equal(
                ebins.size, t.e.size
            )  # allow merging neutron and photon heating meshes
    nonzero_idx = np.logical_and(result_data > 0.0, errors > 0.0)
    result_error = np.zeros_like(result_data)
    result_error[nonzero_idx] = np.sqrt(errors[nonzero_idx]) / result_data[nonzero_idx]
    return FMesh(
        name,
        kind,
        geometry_spec,
        ebins,
        result_data,
        result_error,
        comment=comment,
    )


def read_meshtal(stream: TextIO, select=None, mesh_file_info=None) -> List[FMesh]:
    """Reads fmesh tallies from a stream.

    Args:
        stream: The text stream to read.
        select: Selects the meshes actually to process (Default value = None)
        mesh_file_info: object to collect information from m-file header (Default value = None)

    Returns:
        The list of individual fmesh tally.
    """
    next(stream)  # TODO dvp check if we need to store problem time stamp
    next(stream)  # TODO dvp check if we need to store problem title
    line = next(stream)
    nps = int(float((line.strip().split("=")[1])))
    if mesh_file_info is not None:
        mesh_file_info.nps = nps
    return list(iter_meshtal(stream, select))


def _iterate_bins(stream, _n, _with_ebins):
    """Parse line with mesh values.

    Args:
        stream: stream of strings
        _n: number of items
        _with_ebins: are ebins specified

    Yields:
        pairs value - error
    """
    value_start, value_end = (41, 53) if _with_ebins else (32, 44)
    for _i in range(_n):
        _line = next(stream)
        _value = float(_line[value_start:value_end])
        _error = float(_line[value_end:])
        if _value < 0.0:
            _value = _error = 0.0
        yield _value
        yield _error


# noinspection PyTypeChecker
def iter_meshtal(
    fid: TextIO,
    name_select: Callable[[int], bool] = None,
    tally_select: Callable[[FMesh], bool] = None,
) -> Generator[FMesh, None, None]:
    """Iterates fmesh tallies from fid.

    Args:
        fid: A stream to read meshes from.
        name_select: A function returning True,
            if tally name is acceptable, otherwise skips tally reading and parsing
        tally_select: A function returning True,
            if total tally content is acceptable

    Yields:
        Mesh tallies filtered.
    """
    try:
        while True:
            # Skip first two comment lines ms version and model title
            # noinspection PyUnresolvedReferences
            name = int(_find_words_after(fid, "Mesh", "Tally", "Number")[0])
            if not name_select or name_select(name):
                comment = fid.readline().strip()
                if comment.startswith("This is a"):
                    kind = comment.split()[3]
                    comment = None
                else:
                    # noinspection PyUnresolvedReferences
                    kind = _find_words_after(fid, "This", "is", "a")[0]
                if comment:
                    comment = fix_mesh_comment(name, comment)
                kind = Kind[kind]

                # TODO dvp read "dose function modified" here

                _find_words_after(fid, "Tally", "bin", "boundaries:")

                line = next(fid).lstrip()
                if line.startswith("Cylinder"):
                    # retrieve cylinder origin and axis
                    part1, part2 = line.split(",")
                    origin = np.fromiter(part1.split()[3:6], dtype=float)
                    axis = np.fromiter(part2.split()[2:5], dtype=float)
                    ibins = np.array(
                        [
                            float(w)
                            for w in _find_words_after(concatv([line], fid), "R", "direction:")
                        ]
                    )

                    jbins = np.array([float(w) for w in _find_words_after(fid, "Z", "direction:")])

                    kbins = np.array(
                        [
                            float(w)
                            for w in _find_words_after(fid, "Theta", "direction", "(revolutions):")
                        ]
                    )

                    geometry_spec = gc.CylinderGeometrySpec(
                        ibins, jbins, kbins, origin=origin, axs=axis
                    )

                    ebins = np.array(
                        [float(w) for w in _find_words_after(fid, "Energy", "bin", "boundaries:")]
                    )
                    with_ebins = check_ebins(
                        fid, ["Energy", "R", "Z", "Th", "Result", "Rel", "Error"]
                    )
                else:
                    xbins = np.array(
                        [
                            float(w)
                            for w in _find_words_after(concatv([line], fid), "X", "direction:")
                        ]
                    )

                    ybins = np.array([float(w) for w in _find_words_after(fid, "Y", "direction:")])

                    zbins = np.array([float(w) for w in _find_words_after(fid, "Z", "direction:")])

                    geometry_spec = gc.CartesianGeometrySpec(xbins, ybins, zbins)

                    ebins = np.array(
                        [float(w) for w in _find_words_after(fid, "Energy", "bin", "boundaries:")]
                    )
                    with_ebins = check_ebins(
                        fid, ["Energy", "X", "Y", "Z", "Result", "Rel", "Error"]
                    )

                spatial_bins_size = geometry_spec.bins_size
                bins_size = spatial_bins_size * (ebins.size - 1)

                data_items = np.fromiter(_iterate_bins(fid, bins_size, with_ebins), dtype=float)
                data_items = data_items.reshape(bins_size, 2)
                shape = (ebins.size - 1,) + geometry_spec.bins_shape
                data, error = data_items[:, 0].reshape(shape), data_items[:, 1].reshape(shape)

                def _iterate_totals(stream, totals_number):
                    """Reading totals.

                    Args:
                        stream: sequence or stream of strings
                        totals_number: number of items to read

                    Yields:
                        total values and errors
                    """
                    for _i in range(totals_number):
                        _line = next(stream).split()
                        # TODO dvp: check for negative values in an MCNP meshtal file
                        assert _line[0] == "Total"
                        for w in _line[4:]:
                            yield float(w)

                if ebins.size > 2:  # Totals are not output if there's only one bin in energy domain
                    totals_items = np.fromiter(_iterate_totals(fid, spatial_bins_size), dtype=float)
                    totals_items = totals_items.reshape(spatial_bins_size, 2)
                    shape = geometry_spec.bins_shape
                    totals = totals_items[:, 0].reshape(shape)
                    totals_err = totals_items[:, 1].reshape(shape)
                else:
                    totals = None
                    totals_err = None
                res = FMesh(
                    name,
                    kind,
                    geometry_spec,
                    ebins,
                    data,
                    error,
                    totals,
                    totals_err,
                    comment=comment,
                )
                if not tally_select or tally_select(res):
                    yield res
                else:
                    __LOG.debug("Skipping mesh tally %s", name)
    except EOFError:
        pass


def check_ebins(fid: Iterable[str], keys: List[str]) -> bool:
    """Check if energy bins present in a mesh tally output values.

    If next nonempty line starts with a word keys[0] (i.e. "Energy"), then the energy bins present.
    Also check that the remaining keys correspond to the nonempty line.

    Args:
        fid: text rows to scan, including prepending empty rows
        keys: sequence of words to check
        fid: Iterable[str]:
        keys: List[str]:

    Returns:
        True if energy bins are present, False otherwise.

    Raises:
        ValueError: if keys don't correspond to the nonempty line.
    """
    title_line = _next_not_empty_line(fid)
    if title_line is None:
        raise ValueError(f"Cannot find titles {keys[1:]}")
    if title_line[0] == keys[0]:
        assert keys[1:] == title_line[1:]
        with_ebins = True
    else:
        if keys[1:] != title_line:
            raise ValueError(f"Unexpected values title {title_line}")
        with_ebins = False
    return with_ebins


def _next_not_empty_line(f: Iterable[str]) -> Optional[List[str]]:
    """Skip empty lines from a string sequence.

    Args:
        f: sequence or stream of strings

    Returns:
        The first not empty line.
    """
    for line in f:
        words = line.split()
        if 0 < len(words):
            return words
    return None


def _find_words_after(f: TextIO, *keywords: str) -> list[str]:
    """Searches for words that follow keywords.

    The line from file f is read. Then it is split into words (by spaces).
    If its first words are the same as keywords, then remaining words (up to
    newline character) are returned. Otherwise, new line is read.

    Args:
        f: File in which words are searched.
        keywords: List of keywords after which right words are. The order is important.

    Returns:
        The list of words that follow keywords.
    """
    for line in f:
        words: list[str] = line.split()
        i = 0  # : ignore[SIM113]
        for w, kw in zip(words, keywords):
            if w != kw:
                break
            i += 1
        if i >= len(keywords):
            return words[i:]
    raise EOFError()


def m_2_npz(
    stream: TextIO,
    prefix: Path,
    *,
    name_select=lambda _: True,
    tally_select=lambda _: True,
    suffix: str = "",
    mesh_file_info=None,
    check_existing_file_strategy=raise_error_when_file_exists_strategy,
) -> int:
    """Splits the tallies from the mesh file into separate npz files.

    Args:
        stream: File with MCNP mesh tallies
        prefix: Prefix for separate mesh files names
        name_select:
            Filter fmesh by names (Default value = lambda _: True)
        tally_select: function(FMesh)->bool
            Filter fmesh by content. (Default value = lambda _: True)
        suffix:
            Prefix for separate mesh files names
        mesh_file_info: structure to store meshtal file header info: nps.
        check_existing_file_strategy:
            what to do if an output file already exists
        stream: TextIO:
        prefix: Path:
        suffix: str:  (Default value = "")

    Returns:
        Total number of files created
    """
    next(stream)  # TODO dvp check if we need to store problem time stamp
    next(stream)  # TODO dvp check if we need to store problem title
    line = next(stream)
    nps = int(float((line.strip().split("=")[1])))
    if mesh_file_info is not None:
        mesh_file_info.nps = nps
    total = 0  # : ignore[SIM113]
    for t in iter_meshtal(stream, name_select=name_select, tally_select=tally_select):
        t.save_2_npz(prefix / (str(t.name) + suffix), check_existing_file_strategy)
        total += 1

    return total


def fix_mesh_comment(mesh_no: int, comment: str) -> str:
    """Remove digits from an FMESH comment.

    MCNP error: prints digits in front of comment when the tally
    number takes more than 3 digits.

    Args:
        mesh_no: mesh tally number
        comment: ... comment

    Returns:
        corrected comment
    """
    str_mesh_no = f"{mesh_no}"
    chars_to_remove = len(str_mesh_no) - 3
    if chars_to_remove > 0:
        comment = comment[chars_to_remove:]
    return comment.strip()


def meshes_to_vtk(
    *meshes: FMesh,
    out_dir: Path = None,
    get_mesh_description_strategy: Callable[[FMesh], str],
) -> None:
    """Export FMesh objects to VTK files.

    Args:
        meshes: one or more meshes to output
        out_dir: path to output directory
        get_mesh_description_strategy: strategy to create a mesh description
    """
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    for mesh in meshes:
        particle = mesh.kind.short
        function = get_mesh_description_strategy(mesh)
        data_name = f"{particle}-{function}"
        file_name = f"{data_name}-{mesh.name}"
        if out_dir:
            file_name = str(out_dir / file_name)
        mesh.save2vtk(file_name, data_name)
