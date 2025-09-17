from __future__ import annotations

import numpy as np
import pytest

from mckit_meshes.cli.runner import mckit_meshes
from mckit_meshes.fmesh import FMesh


@pytest.fixture
def source(data):
    return data / "1.m"


def test_help(runner):
    result = runner.invoke(mckit_meshes, args=["add", "--help"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "Usage: " in result.output


def test_add_with_out_specified(runner, cd_tmpdir, data):
    out = cd_tmpdir / "1+2.npz"
    m1 = data / "1004.npz"  # the two meshes differ only by name
    m2 = data / "2004.npz"
    result = runner.invoke(
        mckit_meshes,
        args=["add", "-o", str(out), str(m1), str(m2)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert out.exists()
    mesh_out = FMesh.load_npz(out)
    mesh1 = FMesh.load_npz(m1)
    mesh2 = FMesh.load_npz(m2)
    assert np.all(mesh1.data + mesh2.data == mesh_out.data)
    idx = np.logical_and(mesh1.errors < 1.0, mesh_out.errors < 1.0)
    assert mesh1.errors[idx] / np.sqrt(2) == pytest.approx(mesh_out.errors[idx])


def test_add_with_out_not_specified(runner, cd_tmpdir, data):
    out = cd_tmpdir / "1004+2004.npz"
    m1 = data / "1004.npz"  # the two meshes differ only by name
    m2 = data / "2004.npz"
    result = runner.invoke(
        mckit_meshes,
        args=["add", str(m1), str(m2)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert out.exists()
