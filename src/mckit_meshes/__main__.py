from __future__ import annotations

from typing import Annotated, Final, cast

import os
import sys

from dataclasses import dataclass
from pathlib import Path

import cyclopts

from cyclopts import App, Parameter, types  # noqa: TC002
from eliot import start_task

# noinspection package-requirements
from rich.console import Console

from mckit_meshes import __version__
from mckit_meshes.cli import NAME, PREFIX, init_logging
from mckit_meshes.cli.addnpz import add as do_add
from mckit_meshes.cli.invwgt import invwgt as do_invwgt
from mckit_meshes.cli.merge_weights import merge_weights as do_merge_weights
from mckit_meshes.cli.mesh2npz import mesh2npz as do_mesh2npz
from mckit_meshes.cli.mesh2wgt import mesh2wgt as do_mesh2wgt
from mckit_meshes.cli.normalize_weights import normalize_weights as do_normalize_weights
from mckit_meshes.cli.npz2vtk import npz2vtk as do_npz2vtk
from mckit_meshes.cli.split_mesh_file import split as do_split
from mckit_meshes.cli.wgt_drop_ebins import wgt_drop_ebins as do_wgt_drop_ebins

DEFAULT_CONFIG_PATH: Final[Path] = Path(os.getenv("MCKIT_CONFIG", "mckit.toml"))
DEFAULT_ELIOT_LOG_PATH: Final[Path] = PREFIX.with_suffix(".log")
DEFAULT_NPZ = Path("npz")

console = Console()
app = App(
    name=NAME,
    version=__version__,
    console=console,
    config=[
        cyclopts.config.Toml(
            DEFAULT_CONFIG_PATH,
            root_keys=["tool", NAME],
            search_parents=True,
        ),
        cyclopts.config.Env(prefix=NAME.upper() + "_"),
    ],
    help_format="restructuredtext",
)


@Parameter(name="*")  # https://cyclopts.readthedocs.io/en/latest/cookbook/sharing_parameters.html
@dataclass
class Common:
    prefix: Annotated[Path | None, Parameter(name=["--prefix", "-p"])] = None
    "A prefix to prepend output files."

    override: bool = False
    "Override existing output files"

    def __post_init__(self) -> None:
        """Initialize prefix, if not specified."""
        # Should be initialized here, not at field definition,
        # to be set to current directory when the Common instance
        # is created.
        if self.prefix is None:
            self.prefix = Path.cwd()


# noinspection incorrect-docstring
@app.command
def mesh2npz(*mesh_tallies: types.ResolvedExistingFile, common: Common | None = None) -> None:
    """Convert mesh files to npz files.

    By default, output folder (prefix) is "npz".
    If there are many input meshtally files, then meshtally stem is added to prefix.

    Parameters
    ----------
    mesh_tallies
        mesh tally files to process (default: *.m)
    """
    if common is None:
        common = Common(prefix=Path("npz"))
    elif common.prefix is None:
        common.prefix = Path("npz")
    do_mesh2npz(*mesh_tallies, prefix=cast("Path", common.prefix), override=common.override)


# noinspection incorrect-docstring
@app.command
def npz2vtk(*npz_files: types.ResolvedExistingFile, common: Common | None = None) -> None:
    """Convert npz files to VTK files.

    Parameters
    ----------
    npz_files
        .npz files with compressed meshes
    """
    if common is None:
        common = Common(prefix=Path("vtk"))
    elif common.prefix is None:
        common.prefix = Path("vtk")
    do_npz2vtk(*npz_files, prefix=cast("Path", common.prefix), override=common.override)


# noinspection incorrect-docstring
@app.command
def add(
    *npz_files: types.ResolvedExistingFile,
    out: Annotated[types.ResolvedFile | None, Parameter(name=["--out", "-o"])] = None,
    comment: Annotated[str | None, Parameter(name=["--comment", "-c"])] = None,
    number: Annotated[int, Parameter(name=["--number", "-n"])] = 1,
    scale: types.PositiveFloat | None = None,
    common: Common | None = None,
) -> None:
    """Add meshes from npz files.

    Parameters
    ----------
    out
        ... file for created meshtally,   default: computed from the input files stems
    comment
        ... for meshtally, default the comment from the first mesh
    number
        ... of created meshtally
    scale
        multiplication factor for resulting mesh, use on averaging several meshes,
        default: no scaling
    """
    if common is None:
        common = Common()
    do_add(
        *npz_files, out=out, comment=comment, number=number, scale=scale, override=common.override
    )


# noinspection incorrect-docstring
@app.command
def split(meshtally_file: types.ResolvedExistingFile, *, common: Common | None = None) -> None:
    """Split MCNP meshtally file to a number of meshtally files, one for each meshtally.

    Parameters
    ----------
    meshtally_file
        input file to split
    """
    if common is None:
        common = Common()
    do_split(meshtally_file, prefix=common.prefix, override=common.override)


# noinspection incorrect-docstring
@app.command
def invwgt(
    wgtfile: types.ResolvedExistingPath,
    *,
    out: Annotated[types.ResolvedFile | None, Parameter(name=["--out", "-o"])] = None,
    normalization_point: Annotated[
        str, Parameter(name=["--normalization-point", "-n"])
    ] = "610, 0, 57",
    normalization_value: float = 1.0,
    common: Common | None = None,
) -> None:
    """Invert MCNP weight window file.

    All values became reciprocals (w[...] = 1/w[...]).
    Use this for anti-forward weight estimations.

    Features:
        - Zero values remain zeros.
        - After all normalizes the resulting weights, so at given point the weight is 1.0.

    Multiple energy bins are not implemented yet.

    Parameters
    ----------
    out
        output file, default - computed from input file name and saved in current directory
    normalization_point
        Point where to set weight to `normalization_value`
        (default ITER magnetic axis intersection with PY=0: "610, 0, 57")
    normalization_value
        value to be at `normalization point`
    wgtfile
        Weights file to invert
    """
    if common is None:
        common = Common()
    do_invwgt(
        wgtfile,
        normalization_point=normalization_point,
        normalization_value=normalization_value,
        out=out,
        override=common.override,
    )


# noinspection incorrect-docstring
@app.command
def merge_weights(
    *wwinp_files: types.ResolvedExistingFile,
    out: Path,
    merge_spec: Path,
    common: Common | None = None,
) -> None:
    """Merge MCNP weight window meshes.

    Parameters
    ----------
    wwin_files
        weight files to merge
    out
        An output file for merge result
    merge_spec
        A merge specification file
    """
    if common is None:
        common = Common()
    do_merge_weights(*wwinp_files, out=out, merge_spec=merge_spec, override=common.override)


# noinspection incorrect-docstring
@app.command
def mesh2wgt(
    mesh_file: types.ResolvedExistingFile,
    *,
    out: Annotated[
        types.ResolvedFile | None,
        Parameter(
            name=["--out", "-o"],
            help="Output file"
            "[default - compute from mesh_file name and store in current directory]",
        ),
    ] = None,
    beta: int = 5,
    soft: Annotated[
        int | None,
        Parameter(
            name=["--soft", "-s"],
            help="Softening factor:"
            "(power to apply to weight values, typically 0.5, if used) [default: no softening]",
        ),
    ] = None,
    mesh: Annotated[
        int | None,
        Parameter(
            name=["--mesh", "-m"],
            help="Mesh Tally number to use."
            "[default: use the single mesh, which is present in meshtal file]",
        ),
    ] = None,
    common: Common | None = None,
) -> None:
    """Convert mesh tally file to weight mesh file.

    This can be used for GVR weights computing.
    """
    if common is None:
        common = Common()

    do_mesh2wgt(
        mesh_file,
        out=out,
        beta=beta,
        soft=soft,
        override=common.override,
        mesh=mesh,
    )


# noinspection incorrect-docstring
@app.command
def normalize_weights(
    weight_file: types.ResolvedExistingPath,
    out: Annotated[types.ResolvedFile | None, Parameter(name=["--out", "-o"])] = None,
    normalization_point: str = "610, 0, 57",
    normalization_value: float = 1 / 3,
    energy_bin: int = 1,
    common: Common | None = None,
) -> None:
    """Normalize weights file.

    Parameters
    ----------
    weight_file
        weights file
    out
        output file
    normalization_point
        coordinates to normalize the weights at, by default "610, 0, 57"
    normalization_value
        value to set, by default 1/3
    energy_bin
        at which energy bin, by default 1
    """
    if common is None:
        common = Common()
    do_normalize_weights(
        weight_file=weight_file,
        out=out,
        normalization_point=normalization_point,
        normalization_value=normalization_value,
        energy_bin=energy_bin,
        override=common.override,
    )


# noinspection incorrect-docstring
@app.command
def wgt_drop_ebins(
    wgtfile: types.ResolvedExistingPath,
    output: types.ResolvedPath,
    min_energy: float,
    part: int,
    common: Common | None = None,
) -> None:
    """Drop bins with upper boundary below the specified min_energy.

    Use this to drop the too ambitious bins generated with ADVANTG at lower energies.

    Parameters
    ----------
    wgtfile
        input weights file
    output
        ... weights file
    min_energy
        Min energy upper boundary.
    part
        0 - neutron, 1 - photon [default=0]
    """
    if common is None:
        common = Common()
    do_wgt_drop_ebins(
        wgtfile=wgtfile,
        output=output,
        min_energy=min_energy,
        part=part,
        override=common.override,
    )


@app.meta.default
def meta(
    *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
    eliot_log: Path = DEFAULT_ELIOT_LOG_PATH,
    console_log_level: str = "INFO",
) -> None:
    """Transfer meta information from STP to MCNP.

    Parameters
    ----------
    eliot_log
        file for structured eliot logging, by default mapstp.log, optional
    console_log_level
        logging level for console logging
    """
    _console = app.console
    init_logging(_console, eliot_log=eliot_log, console_log_level=console_log_level)
    with start_task(action_type=NAME, version=__version__, working_dir=Path.cwd().absolute()):
        _console.rule("🏁 Start", style="bold yellow1", align="left")
        _console.print(NAME, __version__, style="bold dark_olive_green3")
        _console.print("eliot log: ", eliot_log.absolute(), style="dim")
        if "pytest" in sys.modules:
            app(tokens, result_action="return_value")
        else:
            app(tokens)  # pragma: no cover
        _console.rule("✨ Done :smiley:", style="bold yellow1", align="left")


def main() -> None:  # pragma: no cover
    app.meta()


if __name__ == "__main__":
    main()
