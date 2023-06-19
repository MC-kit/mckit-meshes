from __future__ import annotations

from io import StringIO

import numpy as np

from mckit_meshes.mesh2com import mesh2com


def test_mesh2com_simple_cube():
    xbins = ybins = zbins = np.linspace(-1.0, 1.0, 2)
    out = StringIO()
    mesh2com(xbins, ybins, zbins, out=out)
    expected = """
label 0 0
file all
basis 0 1 0 0 0 1 extent 2.0 2.0 origin 0.0 0.0 0.0
basis 1 0 0 0 0 1 extent 2.0 2.0
basis 1 0 0 0 1 0 extent 2.0 2.0
end
""".strip()
    assert expected == out.getvalue().strip()
    out.close()


def test_mesh2com_with_2_bins_along_x():
    xbins = np.linspace(-1.0, 1.0, 3)
    ybins = zbins = np.linspace(-1.0, 1.0, 2)
    out = StringIO()
    mesh2com(xbins, ybins, zbins, out=out)
    expected = """
label 0 0
file all
basis 0 1 0 0 0 1 extent 2.0 2.0 origin -0.5 0.0 0.0
 origin 0.5 0.0 0.0
basis 1 0 0 0 0 1 extent 2.0 2.0 origin 0.0 0.0 0.0
basis 1 0 0 0 1 0 extent 2.0 2.0
end
""".strip()
    assert expected == out.getvalue().strip()
    out.close()


def test_mesh2com_with_bins_along_x_and_y():
    xbins = np.linspace(-5.0, 5.0, 3)
    ybins = np.linspace(-2.0, 2.0, 5)
    zbins = np.linspace(-1.0, 1.0, 2)
    out = StringIO()
    mesh2com(xbins, ybins, zbins, out=out)
    expected = """
label 0 0
file all
basis 0 1 0 0 0 1 extent 4.0 4.0 origin -2.5 0.0 0.0
 origin 2.5 0.0 0.0
basis 1 0 0 0 0 1 extent 10.0 10.0 origin 0.0 -1.5 0.0
 origin 0.0 -0.5 0.0
 origin 0.0 0.5 0.0
 origin 0.0 1.5 0.0
basis 1 0 0 0 1 0 extent 10.0 10.0 origin 0.0 0.0 0.0
end
""".strip()
    assert expected == out.getvalue().strip()
    out.close()


def test_mesh2com_with_bins_along_x_and_z():
    xbins = np.linspace(-5.0, 5.0, 3)
    ybins = np.linspace(-1.0, 1.0, 2)
    zbins = np.linspace(-2.0, 2.0, 5)
    out = StringIO()
    mesh2com(xbins, ybins, zbins, out=out)
    expected = """
label 0 0
file all
basis 0 1 0 0 0 1 extent 4.0 4.0 origin -2.5 0.0 0.0
 origin 2.5 0.0 0.0
basis 1 0 0 0 0 1 extent 10.0 10.0 origin 0.0 0.0 0.0
basis 1 0 0 0 1 0 extent 10.0 10.0 origin 0.0 0.0 -1.5
 origin 0.0 0.0 -0.5
 origin 0.0 0.0 0.5
 origin 0.0 0.0 1.5
end
""".strip()
    assert expected == out.getvalue().strip()
    out.close()
