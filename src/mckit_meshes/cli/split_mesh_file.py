"""Convert MCNP meshtally file to a number of meshtally files, one for each meshtally."""

from __future__ import annotations


from pathlib import Path

import textwrap

from eliot import start_action, start_task

from mckit_meshes import m_file_iterator


def split(
    meshtally_file: Path,
    *,
    prefix: Path | None = None,
    override: bool = False,
) -> None:
    """Split MCNP meshtally file to a number of meshtally files, one for each meshtally.

    Parameters
    ----------
        meshtally_file
            input file to split
        override
            An output directory for the output files (default: current directory),
            output files are also prepended with a meshtally number.
        override
            override existing output files, otherwise raise FileExistsError

        eliot_log
            Path to structured eliot log, default: "mckit-meshes.log
    """
    with start_task(
        action_type="split meshtally file",
        prefix=prefix,
        overide=override,
    ):
        meshtally_file_path = Path(meshtally_file)
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
                check_existing_file(output_path, override=override)
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


def check_existing_file(output_path: Path, *, override: bool = False) -> None:
    (
        """Define if the output file should be overriden.

    Parameters
    ----------
    output_path
        Output file
    override
        override existing file(s)

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
            if override:
                logger.log(message_type="overriding existing output file", output_path=output_path)
            else:
                errmsg = textwrap.dedent(
                    f"""
                    Cannot override existing file \"{output_path}\".
                    Please remove the file or specify
                        "--override"
                    option.
                    """
                )
                raise FileExistsError(errmsg)
