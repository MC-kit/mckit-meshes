"""CLI implmentation."""
from __future__ import annotations

from .mesh2npz import mesh2npz as do_mesh2npz
from .npz2vtk import npz2vtk as do_npz2vtk

__all__ = ["do_npz2vtk", "do_mesh2npz"]
