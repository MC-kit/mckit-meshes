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
# # Simple ps-files visualization
#
# Demonstrate plotting of ps-files with simple shapes. The files with suffix "-wv.ps" represent files without generated void cells (without voids).
# This files are plotted some lines as red dash-lines. We need to represent them as solid lines.
#

# %%
import os
import sys

print(sys.version, "at", sys.prefix)

# %%
from enum import IntEnum
# %%
from pathlib import Path

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import colormaps as cm
from matplotlib import colors, ticker

# %%
from mckit_meshes.plot import Page, load_plotm_file, plot_ps_page

# %% jupyter={"source_hidden": true}
# %matplotlib inline

# %% [markdown]
# See matplotlib [styles](https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html)

# %%
# plt.style.available

# %%
DEBUG_PLOT = True

class PlotTarget(IntEnum):
    paper = 0,
    presentation = 1,
    jupyter = 2
    

PLOT_TARGET: PlotTarget = PlotTarget.paper



# %%
# font = {
#     "weight": "normal",
#     "size": 12,
# }
# plt.rc("font", **font)
# plt.rcParams["mathtext.default"] = "regular"
PUBLICATION_DPI = 1200 #300  # default resolution for publications images
# A4_WIDTH = 21  # cm
# A4_HEIGHT = 29.7
# A4_WIDTH_WITHOUT_MARGIN = A4_WIDTH - 4
# INCH = 2.54  # cm
# FIG_WIDTH = 16 / INCH
# FIG_HEIGHT = FIG_WIDTH / 1.33
# FIG_SIZE = (FIG_WIDTH, FIG_HEIGHT)
FIG_SIZE = (6.4, 4)
# FIG_WIDTH = int(
#     10.6 / INCH
# )  # Optimal for full page witdh graphs, for 1 column graph in 2 column publications use 8cm.
# FIG_HEIGHT = FIG_WIDTH
# plt.rcParams["figure.figsize"] = (FIG_WIDTH, FIG_HEIGHT)
# plt.style.use("petroff10")
# plt.rc("grid", color="gray", linestyle="solid")
# plt.rc("xtick", direction="out", color="gray")
# plt.rc("ytick", direction="out", color="gray")
# plt.style.use(
#     [
#         "fivethirtyeight"
#     ]
# )
# background = '#f0f0f0' # - default for fivethirtyeight style
# this will plot gray background in Jupyter
# on saving img to png file the background is transparent

my_params = {
    'figure.dpi': int(os.getenv("JUPYTER_DPI", 88)),  # 88 - optimal for ViewSonic 32" screen, WYSWYG for 16 cm figure width
    "mathtext.default": "regular",
    "figure.figsize": FIG_SIZE,
    # 'axes.edgecolor': background,
    # 'axes.facecolor': background,
    # "figure.facecolor": background,
    "savefig.dpi": PUBLICATION_DPI,
    "savefig.transparent": True,
    "savefig.bbox": "tight",
}
plt.rcParams.update(my_params)

markers = "sov^*d"
linestyles = ["-", "--", ":", "-."]


# %%
DATA_ROOT = Path("../../tests/data/plot")
assert DATA_ROOT.is_dir()

# %%
ps_files = list(DATA_ROOT.glob("*.ps"))
assert ps_files
ps_files

# %%
pages = { p.stem: load_plotm_file(p)[0] for p in ps_files }
len(pages)

# %%
p = pages["cube"]
dir(p)


# %%
def get_extent(p: Page, /, zoom: float = 1.0) -> tuple[float, float, float, float]:
    return zoom * (p.origin[0] - p.extent[0]), zoom * (p.origin[0] + p.extent[0]), zoom * (p.origin[1] - p.extent[1]), zoom*(p.origin[1] + p.extent[1])


# %%
def plot_page(ax, stem:str, /, zoom: float = 1.0) -> None:
    p = pages[stem]
    xmin, xmax, ymin, ymax = get_extent(p, zoom)
    ax.set_aspect("equal")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    plot_ps_page(ax, p)
    sns.despine(ax=ax, left=True, bottom=True)
    # axes.get_xaxis().set_visible(False) 
    # axes.get_yaxis().set_visible(False) 
    ax.set_xticks([])
    ax.set_yticks([])    
    # plt.savefig((NPZ_DIR / "total-neutron-flux-pz=50").with_suffix(".png"), dpi=PUBLICATION_DPI, bbox_inches="tight", transparent=True, metadata={"Title": "Total neutron flux at PZ=50"})


# %%
sns.set_theme(context='notebook', style="white", palette='colorblind')


# %%
def scan_stems():
    sorted_stems = sorted(s for s in pages if not s.endswith("-wv"))
    for s in sorted_stems:
        yield s

sorted_stems = [s for s in scan_stems()]
sorted_stems

# %%
fig = plt.figure(figsize=(4,5), dpi=200)
rows = len(sorted_stems)
cols = 2
axes = fig.subplots(rows, cols, squeeze=True)
for i, stem in enumerate(sorted_stems):
    ax = axes[i][0]
    if i == 0:
        ax.set_title("normal")
    plot_page(ax, stem, zoom = 0.15)
    ax = axes[i][1]
    if i == 0:
        ax.set_title("from dashed red lines")
    plot_page(ax, stem + "-wv", zoom = 0.15)
plt.show()
