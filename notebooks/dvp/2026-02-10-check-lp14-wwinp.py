# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Check wwinp from LP#14 results
#
# dvp 2026-02-10

# %%
import sys

from pathlib import Path

print("Python", sys.version, "at", sys.prefix)

# %%
import numpy as np

# %%
import mckit_meshes as mm
from mckit_meshes import ParticleKind
from mckit_meshes import FMesh
from mckit_meshes import WgtMesh

# %%
print(mm.__version__)

# %%
ROOT = Path("/mnt/amarano-2tb/yandex/1-projects/LP#14/D3_MCNP_models&results/11_N2_1E5_section7/11_N2_1E5_compA")

# %%
assert ROOT.is_dir()

# %%
# !ls "{ROOT}"

# %%
inp = ROOT / "wwinp"

# %%
assert inp.is_file()

# %%
with inp.open() as fid:
    weights = WgtMesh.read(fid)

# %%
weights.ibins

# %%
weights.jbins

# %%
weights.kbins

# %%
weights.count_parts

# %%
energies, data = weights.part(ParticleKind.neutron)

# %%
energies

# %%
data.min(), data.max()

# %%
data.shape, weights.ibins.size, weights.jbins.size, weights.kbins.size

# %%
errors = np.ones_like(data)*0.1

# %%
errors.max()

# %%
fmesh = FMesh(4, ParticleKind.neutron, weights.geometry_spec, energies, data, errors) 

# %%
fmesh.save2vtk(str(inp), data_name="wwinp, n")

# %%

# %%
