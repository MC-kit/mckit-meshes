# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Визуализация TRT plotm файла

# %%
import sys

sys.version, sys.prefix

# %%
from pathlib import Path

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt

# %%
# %matplotlib inline

# %%
from mckit_meshes.plot import read_plotm_file

# %%
TRT_ROOT = Path("~/dev/mcnp/trt").expanduser()
assert TRT_ROOT.is_dir()

# %%
PROTOTYPE_DIR = TRT_ROOT / "mcnp-5/prototype"
assert PROTOTYPE_DIR.is_dir()

# %%
