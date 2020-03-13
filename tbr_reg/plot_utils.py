import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interpn


def set_plotting_style(dark=False):
    params = {'backend': 'ps',
              'text.latex.preamble': ['\\usepackage{gensymb}'],
              'axes.labelsize': 16,  # fontsize for x and y labels (was 10)
              'axes.titlesize': 18,
              'font.size': 14,  # was 10
              'legend.fontsize': 16,  # was 10
              'xtick.labelsize': 16,
              'ytick.labelsize': 16,
              'text.usetex': True,
              'figure.figsize': [9, 8],
              'font.family': 'serif'
              }

    plt.rcParams.update(params)

    if dark:
        plt.style.use('dark_background')


# The following utility function is courtesy of:
# https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
def density_scatter(x, y, ax=None, sort=True, bins=20, **kwargs):
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None:
        fig, ax = plt.subplots()
    data, x_e, y_e = np.histogram2d(x, y, bins=bins)
    z = interpn((0.5*(x_e[1:] + x_e[:-1]), 0.5*(y_e[1:]+y_e[:-1])),
                data, np.vstack([x, y]).T, method="splinef2d", bounds_error=False)

    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter(x, y, c=z, **kwargs)
    return ax
