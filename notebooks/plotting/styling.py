from contextlib import contextmanager

import matplotlib.pyplot as plt

# You typically want your plot to be ~1.33x wider than tall.
# Common sizes: (10, 7.5) and (12, 9)
BENQ_WIDTH = 23.5  # inch
BENQ_HEIGHT = 13.2
BENQ_DPI = 163  # BENQ monitor resolution 3840 x 2160 (16:9), dpi for horizontal is 2160/9 = 163 dpi
PUBLICATION_DPI = 600  # default resolution for publications images
DELL_WIDTH = 20.35
DELL_HEIGHT = 12.72
DELL_DPI = 94
A4_WIDTH = 21  # cm
A4_HEIGHT = 29.7
A4_WIDTH_WITHOUT_MARGIN = A4_WIDTH - 4
INCH = 2.54  # cm
FIG_WIDTH = A4_WIDTH_WITHOUT_MARGIN / INCH
FIG_HEIGHT = FIG_WIDTH / 1.33

plt.style.use(
    [
        "seaborn",
        "seaborn-white",
        "seaborn-ticks",
        "seaborn-colorblind",
        "seaborn-paper",
    ]
)


plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["figure.figsize"] = (FIG_WIDTH, FIG_HEIGHT)
# plt.rcParams['ytick.right'] = True
# plt.rcParams['xtick.top'] = True
# plt.rc('grid', color='gray', linestyle='solid')
# plt.rc('xtick', direction='out', color='gray')
# plt.rc('ytick', direction='out', color='gray')


# Tableu colors, see: http://tableaufriction.blogspot.com/2012/11/finally-you-can-use-tableau-data-colors.html
# These are the "Tableau 20" colors as RGB.
tableau20 = [
    (31, 119, 180),
    (174, 199, 232),
    (255, 127, 14),
    (255, 187, 120),
    (44, 160, 44),
    (152, 223, 138),
    (214, 39, 40),
    (255, 152, 150),
    (148, 103, 189),
    (197, 176, 213),
    (140, 86, 75),
    (196, 156, 148),
    (227, 119, 194),
    (247, 182, 210),
    (127, 127, 127),
    (199, 199, 199),
    (188, 189, 34),
    (219, 219, 141),
    (23, 190, 207),
    (158, 218, 229),
]

# See xkcd colors on https://xkcd.com/color/rgb/


# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255.0, g / 255.0, b / 255.0)


@contextmanager
def ne_plotting_style(*additional_styles, after_reset=False):
    styles = [
        "seaborn",
        "seaborn-white",
        "seaborn-ticks",
        "seaborn-colorblind",
        "seaborn-paper",
        *additional_styles,
    ]
    with plt.style.context(styles, after_reset=after_reset):
        yield
