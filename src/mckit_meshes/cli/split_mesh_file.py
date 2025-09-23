"""Convert MCNP meshtally file to a number of meshtally files, one for each meshtally."""

from __future__ import annotations

from typing import Literal

import logging
import sys


from pathlib import Path

import textwrap

from cyclopts import App, types  # noqa: TC002 - types are used in run time
from eliot import to_file, start_action, start_task
from eliot.stdlib import EliotHandler
from rich.console import Console

from mckit_meshes import __version__
from mckit_meshes import m_file_iterator


console = Console()
app = App(version=__version__, console=console)


def revise_when_exists_flag(when_exists):
    if when_exists:
        return when_exists
    return "fail"


def init_logger(eliot_log: Path | None = None) -> None:
    if not eliot_log and "pytest" not in sys.modules:
        eliot_log = Path("mckit-meshes.log")
    if eliot_log:
        to_file(eliot_log.open(mode="a"))
        # Add Eliot Handler to root Logger. You may wish to only route specific
        # Loggers to Eliot.
        logging.getLogger().addHandler(EliotHandler())


@app.command
def split(
    meshtally_file: types.ResolvedExistingFile,
    prefix: Path | None = None,
    when_exists: Literal["override", "update", "fail"] = "fail",
    eliot_log: Path | None = None,
) -> int:
    """Split MCNP meshtally file to a number of meshtally files, one for each meshtally.

    Parameters
    ----------
        meshtally_file
            input file to split
        prefix
            An output directory for the output files (default: current directory),
            output files are also prepended with a meshtally number.
        when_exists
            What to do, if output file already exists:

                * "override" - ignore and override
                * "update"   - override if older than input file
                * "fail"     - raise error (default)

        eliot_log
            Path to structured eliot log, default: "mckit-meshes.log

    Returns:
    --------
        0 - on success
    """
    init_logger(eliot_log)
    with start_task(
        action_type="split meshtally file",
        prefix=prefix,
        when_exists=when_exists,
    ):
        meshtally_file_path = Path(meshtally_file)
        source_time = meshtally_file_path.stat().st_mtime_ns
        prefix_path = prefix if prefix else Path.cwd()
        prefix_path.mkdir(parents=True, exist_ok=True)
        with (
            start_action(action_type="loading mesh file", mesh_file=meshtally_file),
            meshtally_file_path.open() as fid,
        ):
            it = m_file_iterator(fid)
            header = next(it)
            for m in it:
                first_line = m[0]
                mesh_tally_number = first_line.split()[3]
                output_path = prefix_path / (mesh_tally_number + ".m")
                if skip_existing_file(source_time, output_path, when_exists):
                    continue
                with (
                    start_action(
                        action_type="saving mesh",
                        tally_number=mesh_tally_number,
                        mesh_file=output_path,
                    ),
                    output_path.open("w") as out,
                ):
                    for line in header:
                        print(line, file=out)
                    print(file=out)
                    for line in m:
                        print(line, file=out)

    return 0


def skip_existing_file(
    source_time: int, output_path: Path, when_exists: Literal["override", "update", "fail"]
) -> bool:
    (
        """Define if the output file should be overriden.

    Parameters
    ----------
    source_time
        Time of source file(s).
    output_path
        Output file
    when_exists
        - `override` - override anyway
        - `update` - need override if the output is older
        - `fail` - raise error if the output already exists

    Returns
    -------
        True - if the output is newer, False - otherwise or if it doesn't exist

    Raises
    ------
    FileExistsError
        if output file exists and option `when_exists` != "override"
    """
        """"""
    )

    with start_action(
        action_type="check if output exists",
        output_path=output_path,
    ) as logger:
        if output_path.exists():
            if when_exists == "fail":
                errmsg = textwrap.dedent(
                    f"""
                    Cannot override existing file \"{output_path}\".
                    Please remove the file or specify
                        --when-exists override
                        or
                        --when-exists update
                    options.
                    """
                )
                raise FileExistsError(errmsg)
            if when_exists == "update":
                t = output_path.stat().st_mtime_ns
                if source_time < t:
                    logger.log(message_type="output file is newer and is not updated")
                    return True
    return False


if __name__ == "__main__":
    try:
        sys.exit(app())
    except Exception:  # noqa: BLE001
        console.print_exception()
        sys.exit(1)
