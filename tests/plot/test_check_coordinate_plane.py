from __future__ import annotations

from mckit_meshes.plot.check_coordinate_plane import XY, YZ, is_x_plane


def test_is_x_plane() -> None:
    assert is_x_plane(YZ)
    assert not is_x_plane(XY)
