import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

from .plot_utils import density_scatter


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
