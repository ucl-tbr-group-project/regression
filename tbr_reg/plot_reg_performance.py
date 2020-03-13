import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from scipy.interpolate import interpn


def plot_reg_performance(df, density_bins=80):
    '''
    Display true vs. predicted TBR in a density scatter plot.
    Requires a data frame with tbr and tbr_pred columns.
    If density_bins are None, then a regular (non-density-colored) plot is used.
    '''
    xs = df['tbr'].to_numpy()
    ys = df['tbr_pred'].to_numpy()

    linreg = Ridge().fit(xs.reshape(-1, 1), ys.reshape(-1, 1))
    ys_fit = linreg.predict(xs.reshape(-1, 1))

    fig, ax = plt.subplots()

    if density_bins is None:
        ax.scatter(xs, ys, s=5, label='Data')
    else:
        density_scatter(xs, ys, ax=ax, bins=density_bins, s=5, label='Data')

    ax.plot(xs, xs, '--', c='k', linewidth=1,
            dashes=[5, 10], label='Perfect model')
    ax.plot(xs, ys_fit, '--', c='r', linewidth=1,
            dashes=[5, 10], label='Trained model')

    ax.set_xlabel('True TBR')
    ax.set_ylabel('Predicted TBR')
    ax.legend(loc='lower right')

    return fig, ax


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
