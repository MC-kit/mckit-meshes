from __future__ import annotations

import logging


from mckit_meshes.wgtmesh import WgtMesh
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Iterable, Mapping

__LOG = logging.getLogger("mckit_meshes.merge_weights")


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


def merge_weights(
    *wwinp_files: Path,
    out: Path,
    merge_spec: Path,
    override: bool = False,
) -> None:
    """Merge MCNP weight window meshes.

    Parameters
    ----------
    wwinp_files
        files to merge
    out
        file to save the merge result
    merge_spec
        file with merge specification
    override
        ignore if output files already exist

    Algorithm
    ---------

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
    if out.exists() and not override:
        raise ValueError(f"File {out} already exists. Remove it or use --override option.")
    files = list(wwinp_files)
    merge_descriptor = load_merge_descriptor(merge_spec)
    check_merge_descriptor_is_complete(merge_descriptor, files)
    merge_spec_list = list(create_working_merge_spec(merge_descriptor, files))
    result = WgtMesh.merge(*merge_spec_list)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wt") as io:
        result.wm.write(io)
        __LOG.info(f"Merged weight file is saved to {out}")
    spec_file = out.with_suffix(".merge_spec.csv")
    with spec_file.open("wt") as io:
        print(f"{out.stem} {result.nps}", file=io)
        __LOG.info(f"Merge spec for output weight file is saved to {spec_file}")
