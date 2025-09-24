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
from mckit_meshes.cli.commands.addnpz import add as do_add
from mckit_meshes.cli.commands.mesh2npz import mesh2npz as do_mesh2npz
from mckit_meshes.cli.commands.npz2vtk import npz2vtk as do_npz2vtk


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
        """Initialize prefix, if not specified.

        Should be initialized here, not at field definition,
        to be set to current directory when the Common instance
        is created.
        """
        if self.prefix is None:
            self.prefix = Path.cwd()

@app.command
def mesh2npz(
    *mesh_tallies: types.ResolvedExistingFile,
    common: Common | None = None
) -> None:
    """Converts mesh files to npz files.

    By default output folder (prefix) is "npz".
    
    Parameters
    ----------
    mesh_tallies
        mesh tally files to process (default: *.m)
    """
    if common is None:
       common = Common(prefix=Path("npz"))
    do_mesh2npz(*mesh_tallies, prefix=common.prefix, override=common.override)


@app.command
def npz2vtk(
    *npz_files: types.ResolvedExistingFile,
    common: Common | None = None
) -> None:
    """Converts npz files to VTK files.

    Parameters
    ----------
    npz_files
        .npz files with compressed meshes
    """   
    if common is None:
       common = Common()
    do_npz2vtk(*npz_files, prefix=common.prefix, override=common.override)


@app.command
def add(
    *npz_files: types.ResolvedExistingFile,
    out: Annotated[types.ResolvedFile | None, Parameter(name=["--out", "-o"])] = None,
    comment: Annotated[str | None, Parameter(name=["--comment", "-c"])] = None,
    number: Annotated[int, Parameter(name=["--name", "-n"])] = 1,
    common: Common | None = None
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
       common = Common(prefix=Path("npz"))
    do_add(out, *npz_files, comment=comment, number=number,override=common.override)


@app.command
def split(
    meshtally_file: types.ResolvedExistingFile,
    *,
    common: Common | None = None
) -> None:
    """Split MCNP meshtally file to a number of meshtally files, one for each meshtally.

    Parameters
    ----------
    meshtally_file
        input file to split
    """
    if common is None:
       common = Common()
    do_split(meshtally_file, prefix=common.prefix, override=common.override)


def init_logging(eliot_log: Path | None = None) -> None:
    logging.basicConfig(
        level="NOTSET",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=app.console, rich_tracebacks=True, tracebacks_suppress=[cyclopts])]
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
    with start_task(action_type=NAME, version=__version__, working_dir=Path.cwd()):
        app(tokens)


def main():
    try:
        app.meta()
    except Exception:  # noqa: BLE001
        write_traceback(exc_info=sys.exc_info())
        sys.exit(1)


if __name__ == "__main__":
    main()
