from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Final, cast
import logging
import sys

from pathlib import Path

import cyclopts
from cyclopts import App, Parameter, types  # noqa: TC002 - types are used in run time
from eliot import to_file, start_task, write_traceback
from eliot.stdlib import EliotHandler
from rich.console import Console
from rich.logging import RichHandler


from mckit_meshes import __version__, __name__ as pkg_name
from mckit_meshes.cli.split_mesh_file import split as do_split
from mckit_meshes.cli.addnpz import add as do_add
from mckit_meshes.cli.mesh2npz import mesh2npz as do_mesh2npz
from mckit_meshes.cli.npz2vtk import npz2vtk as do_npz2vtk
from mckit_meshes.cli.invwgt import invwgt as do_invwgt
from mckit_meshes.cli.merge_weights import merge_weights as do_merge_weights
from mckit_meshes.cli.mesh2wgt import mesh2wgt as do_mesh2wgt
from mckit_meshes.cli.normalize_weights import normalize_weights as do_normalize_weights
from mckit_meshes.cli.wgt_drop_ebins import wgt_drop_ebins as do_wgt_drop_ebins


NAME: Final[str] = pkg_name.replace("_", "-")
PREFIX: Final[Path] = Path(NAME)
DEFAULT_CONFIG_PATH: Final[Path] = PREFIX.with_suffix(".toml")
DEFAULT_ELIOT_LOG_PATH: Final[Path] = PREFIX.with_suffix(".log")
DEFAULT_NPZ = Path("npz")

console = Console()
app = App(name=NAME, version=__version__, console=console)


@Parameter(name="*")  # https://cyclopts.readthedocs.io/en/latest/cookbook/sharing_parameters.html
@dataclass
class Common:
    prefix: Annotated[Path | None, Parameter(name=["--prefix", "-p"])] = None
    "A prefix to prepend output files."

    override: bool = False
    "Override existing output files"

    def __post_init__(self):
        """Initialize prefix, if not specified."""
        # Should be initialized here, not at field definition,
        # to be set to current directory when the Common instance
        # is created.
        if self.prefix is None:
            self.prefix = Path.cwd()


@app.command
def mesh2npz(*mesh_tallies: types.ResolvedExistingFile, common: Common | None = None) -> None:
    """Convert mesh files to npz files.

    By default output folder (prefix) is "npz".

    Parameters
    ----------
    mesh_tallies
        mesh tally files to process (default: *.m)
    """
    if common is None:
        common = Common(prefix=Path("npz"))
    if common.prefix is None:
        common.prefix = Path("npz")
    do_mesh2npz(*mesh_tallies, prefix=common.prefix, override=common.override)


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
    if common.prefix is None:
        common.prefix = Path("vtk")
    do_npz2vtk(*npz_files, prefix=common.prefix, override=common.override)


@app.command
def add(
    *npz_files: types.ResolvedExistingFile,
    out: Annotated[types.ResolvedFile | None, Parameter(name=["--out", "-o"])] = None,
    comment: Annotated[str | None, Parameter(name=["--comment", "-c"])] = None,
    number: Annotated[int, Parameter(name=["--number", "-n"])] = 1,
    common: Common | None = None,
) -> None:
    """Add meshes from npz files.

    Parameters
    ----------
    out
        output file for created meshtally,
        if not specified, then it's constructed
        from the input files stems
    comment, optional
        comment for meshtally, default the comment from the first mesh
    number, optional
        number of created meshtally
    """
    if common is None:
        common = Common()
    do_add(*npz_files, out=out, comment=comment, number=number, override=common.override)


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
        - After all normalises the resulting weights, so at given point the weight is 1.0.

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
):
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


@app.command
def normalize_weights(
    weight_file: types.ResolvedExistingPath,
    out: Annotated[types.ResolvedFile | None, Parameter(name=["--out", "-o"])] = None,
    normalization_point: str = "610, 0, 57",
    normalization_value: float = 1 / 3,
    energy_bin: int = 1,
    common: Common | None = None,
):
    """Normalize weights file.

    Parameters
    ----------
    weight_file
        weights file
    out
        output file, optional
    normalization_point, optional
        coordinates to normalize the weights at, by default "610, 0, 57"
    normalization_value, optional
        value to set, by default 1/3
    energy_bin, optional
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
        output weights file
    min_energy
        Min energy upper boundary.
    part
        "Part: 0 - neutron, 1 - photon [default=0]
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


def init_logging(eliot_log: Path | None = None) -> None:
    logging.basicConfig(
        level="NOTSET",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(console=app.console, rich_tracebacks=True, tracebacks_suppress=[cyclopts])
        ],
    )
    if not eliot_log and "pytest" not in sys.modules:
        eliot_log = PREFIX.with_suffix(".log")
    if eliot_log:
        to_file(eliot_log.open(mode="a"))
        # Add Eliot Handler to root Logger. You may wish to only route specific
        # Loggers to Eliot.
        logging.getLogger().addHandler(EliotHandler())


@app.meta.default
def meta(
    *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
    config: types.TomlPath = DEFAULT_CONFIG_PATH,
    eliot_log: Path = DEFAULT_ELIOT_LOG_PATH,
):
    toml_cfg = cyclopts.config.Toml(
        config,
        root_keys=["tool", "character-counter"],
        search_parents=True,
    )
    env_cfg = cyclopts.config.Env(prefix=pkg_name)
    app.config = cast("tuple[str, ...]", (toml_cfg, env_cfg))
    init_logging(eliot_log)
    with start_task(action_type=NAME, version=__version__, working_dir=Path.cwd().absolute()):
        app(tokens)


def main():  # pragma: no cover
    try:
        app.meta()
    except Exception:  # noqa: BLE001
        write_traceback(exc_info=sys.exc_info())
        sys.exit(1)


if __name__ == "__main__":
    main()
