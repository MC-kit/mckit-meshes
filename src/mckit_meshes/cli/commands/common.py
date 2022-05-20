import typing as tp

from pathlib import Path

import click

from mckit_meshes.cli.logging import logger


def check_if_path_exists(path: tp.Union[str, Path], override: bool):
    if isinstance(path, str):
        path = Path(path)
    if not override and path.exists():
        errmsg = f"""\
        Cannot override existing file \"{path}\".
        Please remove the file or specify --override option"""
        logger.error(errmsg)
        raise click.UsageError(errmsg)


# def get_default_output_directory(source, suffix):
#     return Path(Path(source).with_suffix(suffix).name)
