from __future__ import annotations

import logging

from collections.abc import Iterable, Mapping
from pathlib import Path

import click
import click_log

from triniti_ne import __version__
from mckit_meshes.wgtmesh import WgtMesh

__LOG = logging.getLogger(__name__)
click_log.basic_config(__LOG)


def load_merge_descriptor(merge_descriptor_file: Path) -> Mapping[str, int]:
    def convert(line: str) -> tuple[str, int]:
        _id, _nps = line.split("#")[0].split()[:2]
        return _id, int(_nps)

    with merge_descriptor_file.open() as io:

        def do_filter(line: str) -> bool:
            line = line.strip()
            return len(line) > 0 and not line.startswith("#")

        result = dict(map(convert, filter(do_filter, io.readlines())))
        __LOG.info(f"Loaded merge descriptor from {merge_descriptor_file}")
        return result


def create_working_merge_spec(
    merge_descriptor: Mapping[str, int], wgt_mesh_files: Iterable[Path]
) -> Iterable[WgtMesh.MergeSpec]:
    def make_merge_spec(wgt_mesh_file: Path) -> WgtMesh.MergeSpec:
        with wgt_mesh_file.open("rt", encoding="cp1251") as io:
            wgt_mesh: WgtMesh = WgtMesh.read(io)
            __LOG.info(f"Loaded weight mesh from {wgt_mesh_file}")
        _id = wgt_mesh_file.stem
        _nps = merge_descriptor[_id]
        return WgtMesh.MergeSpec(wgt_mesh, _nps)

    return map(make_merge_spec, wgt_mesh_files)


def check_merge_descriptor_is_complete(
    merge_descriptor: Mapping[str, int], files: list[Path]
) -> None:
    keys = set(merge_descriptor.keys())
    for f in files:
        if f.stem not in keys:
            raise ValueError(f"Cannot find  {f.stem} in merge descriptor")


def process(
    out: Path, override: bool, merge_descriptor_file: Path, wwinp_files: Iterable[Path]
) -> None:
    if out.exists() and not override:
        raise ValueError(f"File {out} already exists. Remove it or use --override option.")
    files = list(wwinp_files)
    merge_descriptor = load_merge_descriptor(merge_descriptor_file)
    check_merge_descriptor_is_complete(merge_descriptor, files)
    merge_spec = list(create_working_merge_spec(merge_descriptor, files))
    result = WgtMesh.merge(*merge_spec)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wt") as io:
        result.wm.write(io)
        __LOG.info(f"Merged weight file is saved to {out}")
    spec_file = out.with_suffix(".merge_spec.csv")
    with spec_file.open("wt") as io:
        print(f"{out.stem} {result.nps}", file=io)
        __LOG.info(f"Merge spec for output weight file is saved to {spec_file}")


@click.command(options_metavar="options...")
@click.option(
    "--override/--no-override",
    is_flag=True,
    default=False,
    help="override  already existing output file (default:False)",
)
@click.option(
    "--out",
    "-o",
    required=True,
    help="An output file for merged weight mesh file",
)
@click.option(
    "--merge-spec",
    "-m",
    type=click.Path(exists=True),
    required=True,
    help="A merge specification file",
)
@click.argument(
    "wwinp_files",
    metavar="<wwinp_file...>",
    type=click.Path(exists=True),
    nargs=-1,
    required=True,
)
@click_log.simple_verbosity_option(__LOG, default="INFO")
@click.version_option(__version__)
def merge_weights(
    override: bool,
    out: click.Path,
    merge_spec: click.Path,
    wwinp_files: list[click.Path],
) -> None:
    """The script merges MCNP weight window meshes.

    Algorithm:

    The major assumption is that reciprocal of a given weight is proportional to portion
    of weight incoming or produced in a given mesh voxel passed to a point of interest
    (tally). The portion value is close in sense to probability, and these are
    probabilities we have to merge, not weights.

    The resulting merged weight values are reciprocals or weighted average
    of reciprocals of given weights.

    The zero weight values are passed "as is" and ignored on merging
    with nonzero values.

    Merge specification file:

    The weighting factors are NPS (Number of Particles Samples) achieved on the weight
    meshes generation or whatever a user decide. The factors are provided with
    a specification file, which contains pairs of basename of files and factors.
    The specification file may contain comment lines and trailing comments
    starting with '#'. The empty lines are ignored.
    """

    def make_path(path: click.Path) -> Path:
        return Path(str(path))

    process(make_path(out), override, make_path(merge_spec), map(make_path, wwinp_files))


if __name__ == "__main__":
    merge_weights()
