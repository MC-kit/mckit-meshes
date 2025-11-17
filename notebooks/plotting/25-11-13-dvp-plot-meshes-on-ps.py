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
# # Визуализация мешей на фоне TRT postscript сечений

# %%
import sys

sys.version, sys.prefix

# %%
from pathlib import Path

import numpy as np

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors, colormaps as cm
from matplotlib import ticker

# %%
# %matplotlib inline

# %% [markdown]
# See matplotlib [styles](https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html)

# %%
# plt.style.available

# %%
# mpl.use("Qt5agg")
font = {
    "weight": "normal",
    "size": 12,
}
plt.rc("font", **font)
plt.rcParams["mathtext.default"] = "regular"
INCH = 2.54
FIG_WIDTH = int(
    10.6 / INCH
)  # Optimal for full page witdh graphs, for 1 column graph in 2 column publications use 8cm.
FIG_HEIGHT = FIG_WIDTH
plt.rcParams["figure.figsize"] = (FIG_WIDTH, FIG_HEIGHT)
plt.style.use("petroff10")
# plt.rc("grid", color="gray", linestyle="solid")
# plt.rc("xtick", direction="out", color="gray")
# plt.rc("ytick", direction="out", color="gray")

# %%
from mckit_meshes.plot import load_plotm_file, Page, plot_ps_page, BriefTicksAroundOneTicker, plot_2d_distribution

# %%
from mckit_meshes.fmesh import FMesh

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
# !ls {PROTOTYPE_DIR}/results/heat-3/npz/heat-3-5.7e09

# %% [markdown]
# ## Survey tallies
#
# - 1214 - n flux 
# - 1224 - n actual heating, W/cm3
# - 1234 - n steel heating, W/cm3
# - 1284 - n dose, uSv/h
# - 1294 - p dose, uSv/h
# - 1304 - p flux
# - 1314 - p actual heating, W/cm3
# - 1324 - p steel heating, W/cm3
#

# %% [markdown]
# ## neutron total flux

# %%
NPZ_DIR = PROTOTYPE_DIR / "results/heat-3/npz/heat-3-5.7e09"
assert NPZ_DIR.is_dir()

# %%
neutron_flux_mesh = FMesh.load_npz(NPZ_DIR / "1214.npz")

# %%
neutron_flux_mesh.totals.shape

# %%
eq_mid_height_idx = neutron_flux_mesh.kbins.searchsorted(50)
0.5*(neutron_flux_mesh.kbins[eq_mid_height_idx-1] + neutron_flux_mesh.kbins[eq_mid_height_idx])

# %%
data = neutron_flux_mesh.totals[:,:, eq_mid_height_idx - 1]
data.shape

# %%
x, y = neutron_flux_mesh.ibins, neutron_flux_mesh.jbins
X, Y = np.meshgrid(x, y)


# %%
def mids(x):
    return 0.5*(x[1:] + x[:-1])


# %%
set(cm.keys())


# %%
def plot_2d_distribution(x, y, data, fig, ax,
    *,
    color_bar_title=r"$\frac{1} {cm^{2} \cdot s}$",
    max_log_power=None,
    min_max_log_ratio=1e-4,
    transform=None,
    levels=None,
):
    if max_log_power is None:
        max_log_power = int(np.log10(data.max()))
    vmax = 10.0**max_log_power
    vmin = data.min()
    min_log_power = int(np.log10(vmin)) + 1
    vmin = max(min_max_log_ratio * vmax, 10.0**min_log_power)
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap("hot")
    pcm = ax.pcolormesh(
        x,
        y,
        data,
        norm=norm,
        cmap=cmap,
        # antialiased=True,
        # shading="gouraud",
        shading="flat",
        # transform=transform,
    )
    color_bar = fig.colorbar(pcm, ax=ax, shrink=0.8)
    color_bar.ax.set_title(color_bar_title, fontsize=12)
    tick_formatter = BriefTicksAroundOneTicker()
    color_bar.ax.yaxis.set_major_formatter(tick_formatter)
    color_bar.outline.set_edgecolor("white")
    if levels is not None:
        _colors= "k" # cm.get("Wistia_r")(norm(np.array(levels)))
        contours = ax.contour(
            mids(x),
            mids(y),
            data,
            norm=norm,
            levels=levels,
            colors=_colors,
            linewidths=1.0,
            # alpha=0.5,
        )
        levels = contours.levels
        contour_labeled_levels = levels
        fmt = ticker.LogFormatterMathtext()
        fmt.set_axis(ax)
        ax.clabel(
            contours,
            contour_labeled_levels,
            inline=1.0,
            # fmt="%.1g",
            fmt=fmt,
            # colors="k",
            fontsize=9,
        )
    


# %%
p = pages["pz=50"]
fig = plt.figure(dpi=150)
axes = fig.add_subplot(111)
axes.set_aspect("equal")
axes.set_xlim(x[0], x[-1])
axes.set_ylim(y[0], y[-1])
plot_ps_page(axes, p)
plot_2d_distribution(x, y, data, fig, axes, levels = np.pow(10, np.array([8,9,10,11])))
plt.savefig((NPZ_DIR / "total-neutron-flux-pz=50").with_suffix(".png"), dpi=1200)
plt.show()

# %%
photon_flux_mesh = FMesh.load_npz(NPZ_DIR / "1304.npz")

# %%
photon_data = photon_flux_mesh.totals[:,:, eq_mid_height_idx - 1]

# %%
p = pages["pz=50"]
fig = plt.figure(dpi=150)
axes = fig.add_subplot(111)
axes.set_aspect("equal")
axes.set_xlim(x[0], x[-1])
axes.set_ylim(y[0], y[-1])
plot_ps_page(axes, p)
plot_2d_distribution(x, y, photon_data, fig, axes, levels = np.pow(10, np.array([7, 8, 9, 10])))
plt.savefig((NPZ_DIR / "total-photon-flux-pz=50").with_suffix(".png"), dpi=1200, bbox_inches="tight", transparent=True, metadata={"Title": "Total photon flux at PZ=50"})
plt.show()

# %%
