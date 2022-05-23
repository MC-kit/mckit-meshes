"""Classes and functions for operations with fmesh tallies.

Classes
-------
FMesh - class for storing individual fmesh tally data.

Functions
---------
read_meshtal(file) - reads all tallies from MCNP meshtal file.
merge_tallies(*tally_weight) - merges tallies with specific weights.

Examples
--------
Suppose we have meshtal file - MCNP output for fmesh tallies - sample.m.
It contains several tallies with numbers 14, 24, and 34. First, we have to
read it:

    ```python
        tally_list = list(read_meshtal('sample.m'))
    ```

x and z are coordinates of mesh cell centers along x and z axis. data is
2-dimensional array of fmesh tally values. err - relative errors.

In order to save fmesh data to vtk format, method save2vtk should be used.
Because it is actually regular grid the result is .vtr file.

    ```python
        tally_list[0].save2vtk(filename='sample', data_name='heating neutron')
    ```

Dependencies
------------

pyevkt
    https://github.com/pyscience-projects/pyevtk
    https://bitbucket.org/pauloh/pyevtk
"""

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
from pyevtk.hl import gridToVTK
from toolz.itertoolz import concatv

__LOG = logging.getLogger(__name__)


def expand_args(args):
    return rebin.rebin_nd(*args)


class FMesh:
    """Fmesh tally object.

    Attributes:

    name: The number of tally.

    kind: Kind of tally: neutron, photon, electron or generic.
    """

    NPZ_MARK = np.int16(5445)
    """
    'Signature' to be stored in the first entry of meta entry in an npz file to check that the file is for FMesh object
    """

    NPZ_FORMAT = np.int16(4)
    """
    Identifies version of format of data stored in npz file
    """

    class X(RuntimeError):
        pass

    def __init__(
        self,
        name: int,
        kind: int | Kind,
        geometry_spec: gc.AbstractGeometrySpec,
        ebins: np.ndarray,
        data: np.ndarray,
        errors: np.ndarray,
        totals: np.ndarray = None,
        totals_err: np.ndarray = None,
        comment: str = None,
    ) -> None:
        """Construct FMesh instance object.

        Args:
            name: 'name' attribute value
            kind: 'kind' attribute value
            geometry_spec: mesh geometry specification
            ebins: Energy bin boundaries.

            data: Data values at centers of mesh cells.
                Shape (Ne-1)x(Nx-1)x(Ny-1)x(Nz-1), where Ne, Nx, Ny and Nz - the number
                of corresponding bin boundaries.

            errors:
                Relative errors of corresponding data values.
                Shape (Ne-1)x(Nx-1)x(Ny-1)x(Nz-1), where Ne, Nx, Ny and Nz - the number
                of corresponding bin boundaries.
            totals: Can be provided with 'Total' data from mesh file if there are more than 1 energy bin, optional.
            totals_err: Can be provided with data from mesh file if there are more than 1 energy bin, optional.
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
        assert 2 <= self._e.size
        self.data = gc.as_float_array(data)
        self.errors = gc.as_float_array(errors)
        assert self.data.shape == self.errors.shape
        assert self.data.shape == (self.e.size - 1,) + self._geometry_spec.bins_shape
        self._totals = totals
        self._totals_err = totals_err
        if 2 < self._e.size:
            if totals is None or totals_err is None:
                assert (
                    totals is None and totals_err is None
                ), "Both totals and totals_err are to be provided or omitted"
                totals = np.sum(self.data, axis=0)
                non_zero = totals > 0.0
                totals_err = np.zeros_like(totals)
                totals_err[non_zero] = (
                    np.sqrt(np.sum((self.errors * self.data) ** 2, axis=0))[non_zero]
                    / totals[non_zero]
                )
            else:
                assert (
                    totals is not None and totals_err is not None
                ), "Both totals and totals_err are to be provided or omitted"
                totals = np.asarray(totals, dtype=float)
                totals_err = np.asarray(totals_err, dtype=float)
        else:
            assert totals is None
            assert totals_err is None
        self._totals = totals
        self._totals_err = totals_err
        assert self._totals is None or self._totals.shape == self._totals_err.shape
        assert (
            self._totals is None or self._totals.shape == self._geometry_spec.bins_shape
        )
        self._comment = comment

    # @property
    # def x(self):
    #     return self._x
    #
    # @property
    # def y(self):
    #     return self._y
    #
    # @property
    # def z(self):
    #     return self._z

    @property
    def e(self) -> np.ndarray:
        """Energy bins."""
        return self._e

    @property
    def has_multiple_energy_bins(self) -> bool:
        """Check if there's more than 1 energy bin.

        If True, then totals and totals err should present.
        """
        return 2 < self.e.size

    @property
    def ibins(self) -> np.ndarray:
        """Synonym to geometry ibins (x or R)."""
        return self._geometry_spec.ibins

    @property
    def jbins(self) -> np.ndarray:
        """Synonym to geometry jbins (y or Z)."""
        return self._geometry_spec.jbins

    @property
    def kbins(self) -> np.ndarray:
        """Synonym to geometry kbins (z or Theta)."""
        return self._geometry_spec.kbins

    @property
    def totals(self) -> np.ndarray:
        """Total values over energy."""
        return self._totals

    @property
    def totals_err(self) -> np.ndarray:
        """Relative errors of total values over energy."""
        return self._totals_err

    @property
    def comment(self) -> str:
        """Comment from FC card for this mesh tally."""
        return self._comment

    @property
    def origin(self):
        assert self.is_cylinder, "Only valid for cylinder mesh."
        return self._geometry_spec.origin

    @property
    def axis(self) -> np.ndarray:
        assert self.is_cylinder, "Only valid for cylinder mesh."
        return self._geometry_spec.axs

    @property
    def vec(self) -> np.ndarray:
        assert self.is_cylinder, "Only valid for cylinder mesh."
        return self._geometry_spec.vec

    @property
    def is_cylinder(self) -> bool:
        """Is this mesh cylinder?

        Note:
            MCNP uses `origin` on mesh tally specification, both rectilinear and cylinder,
            but outputs origin only for cylinder mesh.
        """
        return self._geometry_spec.cylinder

    @property
    def total_precision(self) -> float:
        if self.has_multiple_energy_bins:
            return self.totals_err[
                -1
            ]  # TODO dvp: assumes max energy bin is most representative, check usage
        else:
            return self.errors[0, 0, 0, 0]

    def is_equal_by_mesh(self, other: "FMesh") -> bool:
        return (
            self.kind == other.kind
            and self._geometry_spec == other._geometry_spec
            and np.array_equal(self.e, other.e)
        )

    def has_better_precision_than(self, other) -> bool:
        assert self.is_equal_by_mesh(
            other
        ), "Incompatible meshes for precision comparison."
        return self.total_precision < other.total_precision

    def __eq__(self, other: "FMesh") -> bool:
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
        # if res:
        #     res = self.is_cylinder == other.is_cylinder
        #     if res and self.is_cylinder:
        #         res = not self.origin(
        #             np.all(np.isclose(self.origin, other.origin))
        #             and np.all(np.isclose(self.axis, other.axis))
        #         )
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
        msg = "Fmesh({name}, {kind}, {xmin}..{xmax}, {ymin}..{ymax}, {zmin}..{zmax}, {emin}..{emax})"
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

    def surrounds_point(self, x: float, y: float, z: float, local: bool = True) -> bool:
        return self._geometry_spec.surrounds_point(x, y, z, local)

    # def get_slice(
    #     self, free: str = "XY", **kwargs
    # ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #     """Gets slice of fmesh tally.
    #
    #     kwargs specify names of fixed variables and their values. If the key
    #     corresponding to the one of fixed variables is present, then its value
    #     is ignored. If the point is outside the mesh then zeros are returned.
    #
    #     Parameters
    #     ----------
    #     free : str
    #         Names of free parameters (those, which corresponds to slice axis).
    #     kwargs : dict
    #         Key-value pairs of fixed parameters. Possible keys:
    #         E - energy value of slice. If 'total' - data is summed over all
    #         energy bins; default: 'total';
    #         X - position of slice in x direction; default: first bin;
    #         Y - position of slice in y direction; default: first bin;
    #         Z - position of slice in z direction; default: first bin;
    #
    #     Returns
    #     -------
    #     x_centers, y_centers : numpy.ndarray, ndim=1
    #         Coordinates of cell centers along free variables.
    #     result, error : numpy.ndarray, ndim=2
    #         An array of data values in the specified section (in phase space).
    #     """
    #     key_index = {0: "E", 1: "X", 2: "Y", 3: "Z"}
    #     free = free.upper()
    #     free_keys = [free[0], free[1]]
    #     slice_index = []
    #     sum_axis = []
    #     for i in range(4):
    #         key = key_index[i]
    #         if key in free_keys:
    #             index = np.arange(self.bins[key].size - 1)
    #         elif key in kwargs and isinstance(kwargs[key], (int, float)):
    #             index = np.searchsorted(self.bins[key], kwargs[key]) - 1
    #         elif key == "E":
    #             index = np.arange(self.bins[key].size - 1)
    #             sum_axis.append(i)
    #         else:
    #             index = 0
    #         slice_index.append(index)
    #
    #     result_data = self.data
    #     result_error = self.errors
    #     for i, index in reversed(list(enumerate(slice_index))):
    #         if not isinstance(index, np.ndarray) and (
    #             index < 0 or index >= self.bins[key_index[i]].size - 1
    #         ):
    #             result_data *= 0
    #             result_error *= 0
    #             index = 0
    #         result_data = result_data.take(index, axis=i)
    #         result_error = result_error.take(index, axis=i)
    #
    #     if sum_axis:
    #         abs_err_square = (result_data * result_error) ** 2
    #         abs_tot_err = np.sqrt(np.sum(abs_err_square, axis=tuple(sum_axis)))
    #         result_data = np.sum(result_data, axis=tuple(sum_axis))
    #         result_error = np.nan_to_num(abs_tot_err / result_data)
    #
    #     xaxs = (self.bins[free_keys[0]][1:] + self.bins[free_keys[0]][:-1]) / 2.0
    #     yaxs = (self.bins[free_keys[1]][1:] + self.bins[free_keys[1]][:-1]) / 2.0
    #
    #     return xaxs, yaxs, result_data, result_error

    def get_spectrum(self, x, y, z):
        """Gets energy spectrum at the specified point.


        Args:
            x, y, z : double
                X, Y and Z coordinate of the point where energy spectrum is
                required. If point is located outside the mesh, zeros are returned.

        Returns:
            ebins, spec, err : numpy.ndarray[double]
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

    def select_indexes(self, *, x=None, y=None, z=None):
        return self._geometry_spec.select_indexes(i_values=x, j_values=y, k_values=z)

    def get_totals(self, *, x=None, y=None, z=None):
        if self._totals is None:
            return None, None
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
        """Writes this object to numpy npz file_.

        Args:
            filename:
                Filename to which the object is saved. If file_ is a
                file-object, then the filename is unchanged. If file_ is a string,
                a .npz extension will be appended to the file_ name if it does not
                already have one. By default, the name of file_ is the tally name.
            check_existing_file_strategy: what to do if an output file already exists
        """
        if not filename.suffix == ".npz":
            filename = filename.with_suffix(".npz")

        check_existing_file_strategy(filename)

        kwd = dict(
            meta=np.array(
                [FMesh.NPZ_MARK, FMesh.NPZ_FORMAT, self.name, self.kind],
                dtype=np.uint32,
            ),
            E=self.e,
            X=self.ibins,
            Y=self.jbins,
            Z=self.kbins,
            data=self.data,
            errors=self.errors,
            totals=self.totals,
            totals_err=self.totals_err,
        )
        if self.comment:
            kwd["comment"] = np.array(self.comment)
        if self.is_cylinder:
            kwd["origin"] = np.array(self._geometry_spec.origin)
            kwd["axis"] = np.array(self._geometry_spec.axs)

        filename.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(filename), **kwd)

    @classmethod
    def load_npz(cls, file_: Union[str, Path]) -> "FMesh":
        """

        Loads Fmesh object from the binary file_.

        Parameters
        ----------
        file_ : file or str
            File or filename from which the object will be loaded.
        """
        if isinstance(file_, Path):
            file_ = str(file_)
        with np.load(file_) as data:
            meta = data["meta"]
            mark = meta[0]
            assert mark == FMesh.NPZ_MARK, "Incompatible file format %s" % file_
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
                    geometry_spec = gc.CylinderGeometrySpec(
                        x, y, z, origin=origin, axs=axis
                    )
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
            filename :
                Name of file to which this object is stored. A .vtk extension will
                be appended. By default, the name of file is the tally name.
            data_name :
                Name of data which will appear in vtk file. If None, tally name
                and type will be used.
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
        return gridToVTK(
            filename, self.ibins, self.jbins, self.kbins, cellData=cell_data
        )

    # noinspection PyUnresolvedReferences
    def save_2_mcnp_mesh(self, stream: TextIO) -> None:
        """
        Saves the mesh in a file in a format similar to mcnp mesh tally textual representation.

        Args:
            stream :  stream to store the mesh.
        """

        def format_comment(a):
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
            "{} direction:".format("Z" if self.is_cylinder else "Y"),
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
        """Select subset of e-voxels within given geometry limits.

        Args:
            emin:
            emax:
            xmin:
            xmax:
            ymin:
            ymax:
            zmin:
            zmax:
            new_name: name for mesh to be created, default -1.

        Returns:

        """
        trim_spec = [
            f
            for f in rebin.trim_spec_composer(
                [self.e, self.ibins, self.jbins, self.kbins],
                [emin, xmin, ymin, zmin],
                [emax, xmax, ymax, zmax],
            )
        ]
        new_bins_list, new_data = rebin.shrink_nd(
            self.data, iter(trim_spec), assume_sorted=True
        )
        _, new_errors = rebin.shrink_nd(
            self.errors, iter(trim_spec), assume_sorted=True
        )

        assert all(np.array_equal(a, b) for a, b in zip(new_bins_list, _))

        new_ebins, new_xbins, new_ybins, new_zbins = new_bins_list
        if self.totals is None:
            new_totals = None
            new_totals_err = None
        else:
            totals_trim_spec = [
                f
                for f in rebin.trim_spec_composer(
                    [self.ibins, self.jbins, self.kbins],
                    [xmin, ymin, zmin],
                    [xmax, ymax, zmax],
                )
            ]
            _, new_totals = rebin.shrink_nd(
                self.totals, iter(totals_trim_spec), assume_sorted=True
            )
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
            new_x:
                A new binning over X axis.
            new_y:
                A new binning over Y axis.
            new_z:
                A new binning over Z axis.
            new_name: optional
                A name for the rebinned mesh to be created.
            extra_process_threshold: optional
                At which size of data use multiple Python processes

        Returns:
            mesh:
                New FMesh object with the rebinned data.
        """
        assert not self.is_cylinder, "Not implemented for cylinder meshes"

        if self.data.size < extra_process_threshold:
            return self.rebin_single(new_x, new_y, new_z, new_name)

        # To avoid huge memory allocations, iterate over energy with external processes
        pool = ndp.Pool(processes=4)
        data_rebin_spec = [
            i
            for i in rebin.rebin_spec_composer(
                [self.ibins, self.jbins, self.kbins],
                [new_x, new_y, new_z],
                axes=[0, 1, 2],
            )
        ]

        def iter_over_e(data):
            for i in range(self.e.size - 1):
                yield data[i], data_rebin_spec, True

        new_data = np.stack(pool.map(expand_args, iter_over_e(self.data)), axis=0)
        t = self.data * self.errors
        new_errors = np.stack(pool.map(expand_args, iter_over_e(t)), axis=0)
        new_errors /= new_data
        if self.totals is None:
            new_totals = None
            new_totals_err = None
        else:
            new_totals = rebin.rebin_nd(
                self.totals, data_rebin_spec, assume_sorted=True
            )
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
            new_x:
                A new binning over X axis.
            new_y: ndarray
                A new binning over Y axis.
            new_z: ndarray
                A new binning over Z axis.
            new_name: int, optional
                name for the rebinned mesh to be created.

        Returns:
            mesh:
                New FMesh object with the rebinned data.
        """
        assert not self.is_cylinder, "Not implemented for cylinder meshes"

        data_rebin_spec = [
            i
            for i in rebin.rebin_spec_composer(
                [self.ibins, self.jbins, self.kbins],
                [new_x, new_y, new_z],
                axes=[1, 2, 3],
            )
        ]
        new_data = rebin.rebin_nd(self.data, iter(data_rebin_spec), assume_sorted=True)
        t = self.data * self.errors
        new_errors = rebin.rebin_nd(t, iter(data_rebin_spec), assume_sorted=True)
        new_errors /= new_data
        if self.totals is None:
            new_totals = None
            new_totals_err = None
        else:
            totals_rebin_spec = [
                i
                for i in rebin.rebin_spec_composer(
                    [self.ibins, self.jbins, self.kbins],
                    [new_x, new_y, new_z],
                    axes=[0, 1, 2],
                )
            ]
            new_totals = rebin.rebin_nd(
                self.totals, iter(totals_rebin_spec), assume_sorted=True
            )
            t = self.totals * self.totals_err
            new_totals_err = rebin.rebin_nd(
                t, iter(totals_rebin_spec), assume_sorted=True
            )
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

    def format_cylinder_origin_and_axis_label(self):
        if self.is_cylinder:
            return "\n  Cylinder origin at {0} {1} {2}, axis in {3} {4} {5} direction\n".format(
                self._geometry_spec.origin[0],
                self._geometry_spec.origin[1],
                self._geometry_spec.origin[2],
                self._geometry_spec.axs[0],
                self._geometry_spec.axs[1],
                self._geometry_spec.axs[2],
            )
        return ""


# noinspection PyTypeChecker,PyProtectedMember
def merge_tallies(
    name: int, kind: int, *tally_weight: Tuple[FMesh, float], comment: str = None
) -> FMesh:
    """Makes superposition of tallies with specific weights.

    Args:
        name:
            Name of new fmesh tally.
        kind:
            Type of new fmesh tally. It can be -1 (or any arbitrary integer).
        tally_weight:
            List of tally-weight pairs (tuples). tally is FMesh instance. weight
            is float.
        comment:
            A comment to assign to the new mesh tally

    Returns:
        result :
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


# def read_meshtally(file_):
#     """Reads fmesh tally from binary file_.
#
#     Parameters
#     ----------
#     file_ : file or str
#         File or filename from which tally should be loaded.
#
#     Returns
#     -------
#     mesh : FMesh
#         Fmesh tally instance.
#     """
#     data = np.load(file_)
#     ne = int(data[0])
#     nx = int(data[1])
#     ny = int(data[2])
#     nz = int(data[3])
#     name = int(data[4])
#     kind = int(data[5])
#     ebins = data[6 : ne + 6]
#     xbins = data[ne + 6 : ne + nx + 6]
#     ybins = data[ne + nx + 6 : ne + nx + ny + 6]
#     zbins = data[ne + nx + ny + 6 : ne + nx + ny + nz + 6]
#     n = (ne - 1) * (nx - 1) * (ny - 1) * (nz - 1)
#     sti = ne + nx + ny + nz + 6
#     data_f = data[sti : sti + n].reshape((ne - 1, nx - 1, ny - 1, nz - 1))
#     data_err = data[sti + n :].reshape((ne - 1, nx - 1, ny - 1, nz - 1))
#     return FMesh(name, kind, xbins, ybins, zbins, ebins, data_f, data_err)


def read_meshtal(stream: TextIO, select=None, mesh_file_info=None) -> List[FMesh]:
    """Reads fmesh tallies from a stream.

    Args:
        stream : The text stream to read.
        select  : predicate
            Selects the meshes actually to process
        mesh_file_info:
            object to collect information from m-file header

    Returns:
        tallies :
            The list of individual fmesh tally.
    """
    next(stream)  # TODO dvp check if we need to store problem time stamp
    next(stream)  # TODO dvp check if we need to store problem title
    line = next(stream)
    nps = int(float((line.strip().split("=")[1])))
    if mesh_file_info is not None:
        mesh_file_info.nps = nps
    return list(iter_meshtal(stream, select))


# noinspection PyTypeChecker
def iter_meshtal(
    fid: TextIO,
    name_select: Callable[[int], bool] = None,
    tally_select: Callable[[FMesh], bool] = None,
) -> Generator[FMesh, None, None]:
    """Iterates fmesh tallies from fid.

    Args:
        fid : A stream to read meshes from.
        name_select:  A function returning True, if tally name is acceptable, otherwise skips tally reading and parsing
        tally_select: A function returning True, if total tally content is acceptable

    Returns:
        iterator: An iterator over meshtal file with proper filtering over names or tallies content.
    """
    try:
        while True:
            # Skip first two comment lines ms version and model title
            # noinspection PyUnresolvedReferences
            name = int(_find_words_after(fid, "Mesh", "Tally", "Number")[0])
            if not name_select or name_select(name):
                # __LOG.debug("Reading mesh tally %s", name)
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
                            for w in _find_words_after(
                                concatv([line], fid), "R", "direction:"
                            )
                        ]
                    )

                    jbins = np.array(
                        [float(w) for w in _find_words_after(fid, "Z", "direction:")]
                    )

                    kbins = np.array(
                        [
                            float(w)
                            for w in _find_words_after(
                                fid, "Theta", "direction", "(revolutions):"
                            )
                        ]
                    )

                    geometry_spec = gc.CylinderGeometrySpec(
                        ibins, jbins, kbins, origin=origin, axs=axis
                    )

                    ebins = np.array(
                        [
                            float(w)
                            for w in _find_words_after(
                                fid, "Energy", "bin", "boundaries:"
                            )
                        ]
                    )
                    with_ebins = check_ebins(
                        fid, ["Energy", "R", "Z", "Th", "Result", "Rel", "Error"]
                    )
                else:
                    xbins = np.array(
                        [
                            float(w)
                            for w in _find_words_after(
                                concatv([line], fid), "X", "direction:"
                            )
                        ]
                    )

                    ybins = np.array(
                        [float(w) for w in _find_words_after(fid, "Y", "direction:")]
                    )

                    zbins = np.array(
                        [float(w) for w in _find_words_after(fid, "Z", "direction:")]
                    )

                    geometry_spec = gc.CartesianGeometrySpec(xbins, ybins, zbins)

                    ebins = np.array(
                        [
                            float(w)
                            for w in _find_words_after(
                                fid, "Energy", "bin", "boundaries:"
                            )
                        ]
                    )
                    with_ebins = check_ebins(
                        fid, ["Energy", "X", "Y", "Z", "Result", "Rel", "Error"]
                    )

                spatial_bins_size = geometry_spec.bins_size
                bins_size = spatial_bins_size * (ebins.size - 1)

                def _iterate_bins(stream, n_, _with_ebins):
                    value_start, value_end = (41, 53) if _with_ebins else (32, 44)
                    for i in range(n_):
                        _line = next(stream)
                        _value = float(_line[value_start:value_end])
                        _error = float(_line[value_end:])
                        if _value < 0.0:
                            _value = _error = 0.0
                        yield _value
                        yield _error

                data_items = np.fromiter(
                    _iterate_bins(fid, bins_size, with_ebins), dtype=float
                )
                data_items = data_items.reshape(bins_size, 2)
                shape = (ebins.size - 1,) + geometry_spec.bins_shape
                data, error = data_items[:, 0].reshape(shape), data_items[:, 1].reshape(
                    shape
                )

                # reading totals for energy
                def _iterate_totals(stream, totals_number):
                    for i in range(totals_number):
                        _line = next(stream).split()
                        # TODO dvp: check for negative values in an MCNP meshtal file
                        assert "Total" == _line[0]
                        for w in _line[4:]:
                            yield float(w)

                if (
                    ebins.size > 2
                ):  # Totals are not output if there's only one bin in energy domain
                    totals_items = np.fromiter(
                        _iterate_totals(fid, spatial_bins_size), dtype=float
                    )
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

    Returns:
        bool: True if energy bins are present, False otherwise.

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
    for line in f:
        words = line.split()
        if 0 < len(words):
            return words
    return None


def _find_words_after(f, *keywords: str) -> List[str]:
    """Searches for words that follow keywords.

    The line from file f is read. Then it is split into words (by spaces).
    If its first words are the same as keywords, then remaining words (up to
    newline character) are returned. Otherwise, new line is read.

    Parameters
    ----------
    f : iterable of text lines
        File in which words are searched.
    *keywords : list of str
        List of keywords after which right words are. The order is important.

    Returns
    -------
    words : list of str
        The list of words that follow keywords.
    """
    for line in f:
        words = line.split()
        i = 0
        for w, kw in zip(words, keywords):
            if w != kw:
                break
            i += 1
        if i >= len(keywords):
            return words[i:]
    raise EOFError


def m_2_npz(
    stream: TextIO,
    prefix: Path,
    *,
    name_select=lambda _: True,
    tally_select=lambda _: True,
    suffix: str = "",
    mesh_file_info=None,
    check_existing_file_strategy=raise_error_when_file_exists_strategy,
):
    """Splits the tallies from the mesh file into separate npz files.

    Args:
        stream: File with MCNP mesh tallies
        prefix: Prefix for separate mesh files names
        name_select: function(int)->bool
            Filter fmesh by names
        tally_select: function(FMesh)->bool
            Filter fmesh by content.
        suffix: srt
            Prefix for separate mesh files names
        mesh_file_info: structure to store meshtal file header info: nps.
        check_existing_file_strategy: what to do if an output file already exists

    Returns:
        total:  Total number of files created
    """
    total = 0
    next(stream)  # TODO dvp check if we need to store problem time stamp
    next(stream)  # TODO dvp check if we need to store problem title
    line = next(stream)
    nps = int(float((line.strip().split("=")[1])))
    if mesh_file_info is not None:
        mesh_file_info.nps = nps

    for t in iter_meshtal(stream, name_select=name_select, tally_select=tally_select):
        t.save_2_npz(prefix / (str(t.name) + suffix), check_existing_file_strategy)
        total += 1

    return total


def fix_mesh_comment(mesh_no: int, comment: str) -> str:
    str_mesh_no = f"{mesh_no}"
    chars_to_remove = len(str_mesh_no) - 3
    if chars_to_remove > 0:
        comment = comment[chars_to_remove:]
    return comment.strip()


def meshes_to_vtk(
    *meshes: FMesh,
    out_dir: Path = None,
    get_mesh_description_strategy: Callable[[FMesh], str] = None,
) -> None:
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
