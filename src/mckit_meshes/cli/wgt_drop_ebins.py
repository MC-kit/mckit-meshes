from __future__ import annotations

from pathlib import Path

import click

from mckit_meshes.wgtmesh import WgtMesh


def load(path: Path) -> WgtMesh:
    with path.open("r") as fid:
        mesh = WgtMesh.read(fid)
        return mesh


def check_output_exists(path: Path, override: bool) -> Path:
    if not override and path.exists():
        raise FileExistsError(path, ' - consider to use "--override" option.')
    return path


def save(mesh: WgtMesh, path: Path) -> None:
    with path.open("wt") as stream:
        mesh.write(stream)


@click.command(options_metavar="options...")
@click.option(
    "--override/--no-override",
    is_flag=True,
    default=False,
    help="override existing file (default:False)",
)
@click.option(
    "--output",
    "-o",
    type=click.STRING,
    help="Output file",
)
@click.option(
    "--min-energy",
    "-e",
    type=click.FLOAT,
    help="Min energy upper boundary",
)
@click.option(
    "--part",
    "-p",
    default=0,
    type=click.INT,
    help="Part: 0 - neutron, 1 - photon [default=0]",
)
@click.argument(
    "wgtfile",
    metavar="<wgtfile>",
    type=click.Path(exists=True),
    nargs=1,
    required=True,
)
def wgt_drop_ebins(
    override: bool, output, min_energy: float, part: int, wgtfile: click.Path
) -> None:
    """Drops bins with upper boundary below the specified min_energy.

    Use this to drop the too ambitious bins generated with ADVANTG at lower energies.
    """
    path = Path(str(wgtfile))
    save(
        load(path).drop_lower_energies(min_energy, part),
        check_output_exists(Path(output), override),
    )


if __name__ == "__main__":
    wgt_drop_ebins()
