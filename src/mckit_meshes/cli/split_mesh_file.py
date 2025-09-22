"""Convert MCNP meshtally file to a number of meshtally files, one for each meshtally."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import sys


from pathlib import Path

import textwrap

from cyclopts import App
from eliot import to_file, start_action, start_task
from rich.console import Console

from mckit_meshes import __version__
from mckit_meshes import m_file_iterator


if TYPE_CHECKING:
    from cyclopts import types


console = Console()
app = App(version=__version__, console=console)


def revise_when_exists_flag(when_exists):
    if when_exists:
        return when_exists
    return "fail"


@app.command
def split(
    meshtally_file: types.ResolvedExistingFile,
    prefix: Path | None = None,
    when_exists: Literal["override", "update", "fail"] = "fail",
) -> int:
    """Split MCNP meshtally file to a number of meshtally files, one for each meshtally.

    Parameters
    ----------

        meshtally_file:
            input file to split
        prefix:
            An output directory for the output files (default: current directory),
            output files are also prepended with a meshtally number.
        when_exists:
            What to do, if output file already exists:

                * "override" - ignore and override
                * "update"   - override if older than input file
                * "fail"     - raise error (default)
    """
    to_file("mckit-meshes.split.log", "ab")
    with start_task(
        "split meshtally file",
        prefix=prefix,
        when_exists=when_exists,
    ):
        meshtally_file_path = Path(meshtally_file)
        source_time = meshtally_file_path.stat().st_mtime_ns
        prefix_path = prefix if prefix else Path.cwd()
        prefix_path.mkdir(parents=True, exist_ok=True)
        with (
            start_action("loading mesh file", mesh_file=meshtally_file),
            meshtally_file_path.open() as fid,
        ):
            it = m_file_iterator(fid)
            header = next(it)
            for m in it:
                first_line = m[0]
                mesh_tally_number = first_line.split()[3]
                output_path = prefix_path / (mesh_tally_number + ".m")
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
                        raise ValueError(errmsg)
                    if when_exists == "update":
                        t = output_path.stat().st_mtime_ns
                        if source_time < t:
                            continue
                with (
                    start_action("saving mesh", mesh_file=output_path),
                    output_path.open("w") as out,
                ):
                    for line in header:
                        print(line, file=out)
                    print(file=out)
                    for line in m:
                        print(line, file=out)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(app())
    except Exception:  # noqa: BLE001
        console.print_exception()
        sys.exit(1)
