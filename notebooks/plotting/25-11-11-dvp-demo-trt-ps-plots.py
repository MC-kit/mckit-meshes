# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # Визуализация TRT postscript сечений

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
from mckit_meshes.plot import load_plotm_file, Page, plot_ps_page

# %%
TRT_ROOT = Path("~/dev/mcnp/trt").expanduser()
assert TRT_ROOT.is_dir()

# %%
PROTOTYPE_DIR = TRT_ROOT / "mcnp-5/prototype"
assert PROTOTYPE_DIR.is_dir()

# %%
PLOTM_DIR = PROTOTYPE_DIR / "plotm"
assert PLOTM_DIR.is_dir()

# %%
ps_files = list(PLOTM_DIR.glob("*.ps"))
assert ps_files
ps_files

# %%
pages = { p.stem: load_plotm_file(p)[0] for p in ps_files }
len(pages)

# %%
for i, (stem, p) in enumerate(pages.items()):
    fig, axes = plt.subplots()
    axes.set_aspect("equal")
    axes.set_xlim(-p.extent[0], p.extent[0])
    axes.set_ylim(-p.extent[1], p.extent[1])
    axes.set_title(stem)
    plot_ps_page(axes, p)
    plt.savefig((PLOTM_DIR / stem).with_suffix(".png"), dpi=1200)
    plt.show()

# %%
from jupytext.config import global_jupytext_configuration_directories,find_jupytext_configuration_file
list(global_jupytext_configuration_directories())

# %%
find_jupytext_configuration_file(".")

# %%
