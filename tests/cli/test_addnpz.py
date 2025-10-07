from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest

from mckit_meshes.__main__ import app as mckit_meshes
from mckit_meshes.fmesh import FMesh


@pytest.fixture
def source(data):
    return data / "1.m"


def test_help(cyclopts_runner):
    out = cyclopts_runner(mckit_meshes, ["add", "--help"])
    assert "Usage: " in out


def test_add_with_out_specified(cyclopts_runner, data):
    out = Path.cwd() / "1+2.npz"
    m1 = data / "1004.npz"  # the two meshes differ only by name
    m2 = data / "2004.npz"
    cyclopts_runner(
        mckit_meshes,
        ["add", "-o", str(out), str(m1), str(m2)],
    )
    assert out.exists(), f"Should create output file {out}"
    mesh_out = FMesh.load_npz(out)
    mesh1 = FMesh.load_npz(m1)
    mesh2 = FMesh.load_npz(m2)
    assert np.all(mesh1.data + mesh2.data == mesh_out.data)
    idx = np.logical_and(mesh1.errors < 1.0, mesh_out.errors < 1.0)
    assert mesh1.errors[idx] / np.sqrt(2) == pytest.approx(mesh_out.errors[idx])


def test_add_with_out_not_specified(cyclopts_runner, data):
    out = Path.cwd() / "1004+2004.npz"
    m1 = data / "1004.npz"  # the two meshes differ only by name
    m2 = data / "2004.npz"
    cyclopts_runner(
        mckit_meshes,
        ["add", str(m1), str(m2)],
    )
    assert out.exists()


def test_add_with_override_specified(cyclopts_runner, data):
    out = Path.cwd() / "1004+2004.npz"
    m1 = data / "1004.npz"  # the two meshes differ only by name
    m2 = data / "2004.npz"
    out.touch()
    assert out.stat().st_size == 0
    cyclopts_runner(
        mckit_meshes,
        ["add", "--override", str(m1), str(m2)],
    )
    assert out.stat().st_size > 0
