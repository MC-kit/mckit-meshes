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
#
# Используем меши с максимальным nps на 2025-11-26, nps ~ 1.4e10

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
from matplotlib import colormaps as cm
from matplotlib import colors, ticker

# %%
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
# mpl.use("Qt5agg")
font = {
    "weight": "normal",
    "size": 12,
}
plt.rc("font", **font)
plt.rcParams["mathtext.default"] = "regular"
PUBLICATION_DPI = 1200 #300  # default resolution for publications images
A4_WIDTH = 21  # cm
A4_HEIGHT = 29.7
A4_WIDTH_WITHOUT_MARGIN = A4_WIDTH - 4
INCH = 2.54  # cm
FIG_WIDTH = 16 / INCH
FIG_HEIGHT = FIG_WIDTH / 1.33
# FIG_WIDTH = int(
#     10.6 / INCH
# )  # Optimal for full page witdh graphs, for 1 column graph in 2 column publications use 8cm.
# FIG_HEIGHT = FIG_WIDTH
plt.rcParams["figure.figsize"] = (FIG_WIDTH, FIG_HEIGHT)
# plt.style.use("petroff10")
# plt.rc("grid", color="gray", linestyle="solid")
# plt.rc("xtick", direction="out", color="gray")
# plt.rc("ytick", direction="out", color="gray")
plt.style.use(
    [
        "fivethirtyeight"
    ]
)
background = '#f0f0f0' # - default for fivethirtyeight style
# this will plot gray background in Jupyter
# on saving img to png file the background is transparent

my_params = {
    'figure.dpi': int(os.getenv("JUPYTER_DPI", 88)),  # 88 - optimal for ViewSonic 32" screen, WYSWYG for 16 cm figure width
    "mathtext.default": "regular",
    "figure.figsize": (FIG_WIDTH, FIG_HEIGHT),
    'axes.edgecolor': background,
    'axes.facecolor': background,
    "figure.facecolor": background,
    "savefig.dpi": PUBLICATION_DPI,
    "savefig.transparent": True,
    "savefig.bbox": "tight",
}
plt.rcParams.update(my_params)

markers = "sov^*d"
linestyles = ["-", "--", ":", "-."]


# %%
from mckit_meshes.fmesh import FMesh
# %%
from mckit_meshes.plot import (BriefTicksAroundOneTicker, Page,
                               load_plotm_file, plot_2d_distribution,
                               plot_ps_page)

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
# !ls {PROTOTYPE_DIR}/results/heat-3/npz/heat-3-1.4e10

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
NPZ_DIR = PROTOTYPE_DIR / "results/heat-3/npz/heat-3-1.4e10"
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
print(f"{data.max():.3g}")

# %%
x, y = neutron_flux_mesh.ibins, neutron_flux_mesh.jbins
X, Y = np.meshgrid(x, y)


# %%
def mids(x):
    return 0.5*(x[1:] + x[:-1])


# %%
# set(cm.keys())

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
    color_bar.ax.set_title(color_bar_title, pad=20, fontsize=12)
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
        fmt.create_dummy_axis()
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
# p = pages["pz=50"]
fig = plt.figure()
axes = fig.add_subplot(111)
axes.set_aspect("equal")
axes.set_xlim(x[0], x[-1])
axes.set_ylim(y[0], y[-1])
# plot_ps_page(axes, p)
plot_2d_distribution(x, y, data, fig, axes, levels = np.pow(10, np.array([8,9,10,11])))
plt.savefig((NPZ_DIR / "total-neutron-flux-pz=50").with_suffix(".png"), dpi=PUBLICATION_DPI, bbox_inches="tight", transparent=True, metadata={"Title": "Total neutron flux at PZ=50"})
plt.show()

# %%
photon_flux_mesh = FMesh.load_npz(NPZ_DIR / "1304.npz")

# %%
photon_data = photon_flux_mesh.totals[:,:, eq_mid_height_idx - 1]

# %%
# p = pages["pz=50"]
fig = plt.figure()
axes = fig.add_subplot(111)
axes.set_aspect("equal")
axes.set_xlim(x[0], x[-1])
axes.set_ylim(y[0], y[-1])
# plot_ps_page(axes, p)
plot_2d_distribution(x, y, photon_data, fig, axes, levels = np.pow(10, np.array([7, 8, 9, 10])))
plt.savefig((NPZ_DIR / "total-photon-flux-pz=50").with_suffix(".png"), dpi=PUBLICATION_DPI, bbox_inches="tight", transparent=True, metadata={"Title": "Total photon flux at PZ=50"})
plt.show()

# %%
neutron_dose_mesh = FMesh.load_npz(NPZ_DIR / "1284.npz")

# %%
neutron_dose_mesh.data.shape

# %%
neutron_dose_data = neutron_dose_mesh.data[0,:, :, eq_mid_height_idx - 1]

# %%
# p = pages["pz=50"]
fig = plt.figure()
axes = fig.add_subplot(111)
axes.set_aspect("equal")
axes.set_xlim(x[0], x[-1])
axes.set_ylim(y[0], y[-1])
# plot_ps_page(axes, p)
plot_2d_distribution(x, y, neutron_dose_data/1e6, fig, axes,
    color_bar_title=r"$\frac{Зв} {ч}$",
    levels = np.pow(10, np.array([2, 3, 4]))
)
plt.savefig((NPZ_DIR / "neutron-dose-pz=50").with_suffix(".png"), dpi=PUBLICATION_DPI, bbox_inches="tight", transparent=True, metadata={"Title": "Neutron dose rate at PZ=50"})
plt.show()

# %% [markdown]
# Несусветные дозы, на ITER (Mode-0 Radiation Maps) дают макс 0.1 Зв/ч вокруг установки, а тут порядка 100-1000. нужно разобраться с нормировками.
# Нормировка в fmesh1284 (heat-3.i), похоже, правильная: $pSv/n/cm^{2}\ -> \ \mu Sv/h \ => \ 10^{17} n/cm^{2}s \cdot 3600 s/h \cdot 10^{-6} uSv/pSv = 3.6e14 \mu Sv/h$

# %% [markdown]
# Проверим для сравнения дозу полученную в neutron-4.0 
#
# На домашнем компе ps=50 недосупно, так что без него.

# %%
old_dose_path = TRT_ROOT / "results/4.0/heat+neutron-4.0/dose.npz"
assert old_dose_path.is_file()

# %%
old_dose_mesh = FMesh.load_npz(old_dose_path)

# %%
old_dose_mesh.data.shape

# %%
old_dose_data = old_dose_mesh.data[0,:, :, eq_mid_height_idx - 1]

# %%
# p = pages["pz=50"]
fig = plt.figure()
axes = fig.add_subplot(111)
axes.set_aspect("equal")
axes.set_xlim(x[0], x[-1])
axes.set_ylim(y[0], y[-1])
# plot_ps_page(axes, p)
plot_2d_distribution(x, y, old_dose_data/1e6, fig, axes,
    color_bar_title=r"$\frac{Зв} {ч}$",
    levels = np.pow(10, np.array([2, 3, 4]))
)
plt.savefig((NPZ_DIR / "old-dose-pz=50").with_suffix(".png"), dpi=PUBLICATION_DPI, bbox_inches="tight", transparent=True, metadata={"Title": "Old dose rate at PZ=50"})
plt.show()

# %% [markdown]
# Те же значения. И если прикинуть по функции конверсии (из Recomendations IDM#29PJCT),
# to для потока $10^8н/cm^{2}c$ где-то близко значение.
# Потоки слишком большие, похоже. А с этим непонятно, что делать. Похоже, что это из-за того, что установка компактная и прозрачная.

# %%
