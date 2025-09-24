from __future__ import annotations

from typing import Annotated, Final, cast
import logging
import sys

from pathlib import Path

import cyclopts
from cyclopts import App, Parameter, types  # noqa: TC002 - types are used in run time
from eliot import to_file, start_task, write_traceback
from eliot.stdlib import EliotHandler
from rich.console import Console


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


@app.command
def mesh2npz(
    *mesh_tallies: types.ResolvedExistingFile,
    prefix: Path = DEFAULT_NPZ,
    override: bool = False,
) -> None:
    """Converts mesh files to npz files.

    Parameters
    ----------
    mesh_tallies
        mesh tally files to process (default: *.m)
    prefix
        A prefix to prepend output files,
        output files are also prepended with MESH_TALLY file base name,
        if there are more than 1 input file.
    override
        override existing output files, otherwise raise FileExistsError
    """
    do_mesh2npz(*mesh_tallies, prefix=prefix, override=override)


@app.command
def split(
    meshtally_file: types.ResolvedExistingFile,
    *,
    prefix: Path | None = None,
    override: bool = False,
) -> None:
    """Split MCNP meshtally file to a number of meshtally files, one for each meshtally.

    Parameters
    ----------
        meshtally_file
            input file to split
        prefix
            An output directory for the output files (default: current directory),
            output files are also prepended with a meshtally number.
        override
            override existing output files, otherwise raise FileExistsError

    Returns:
    --------
        0 - on success
    """
    do_split(meshtally_file, prefix=prefix, override=override)


def init_eliot_logging(eliot_log: Path | None = None) -> None:
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
    init_eliot_logging(eliot_log)
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
