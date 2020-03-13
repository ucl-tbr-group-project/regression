import matplotlib.pyplot as plt

from .plot_utils import density_scatter


def plot_params_vs_tbr(df, params, n_rows=3, n_columns=3, density_bins=80):
    '''Plot multiple params vs. TBR. Supplied parameters are expected to be tuples of column names and human-readable names (for labels).'''

    fig = plt.figure()

    for param_idx, (name, human_readable_name) in enumerate(params):
        xs = df[name].to_numpy()
        ys = df['tbr'].to_numpy()

        ax = plt.subplot(n_rows, n_columns, 1 + param_idx)

        if density_bins is None:
            ax.scatter(xs, ys, s=5)
        else:
            density_scatter(xs, ys, ax=ax, bins=density_bins, s=5)

        ax.set_xlabel(human_readable_name)
        ax.set_ylabel('TBR')

    return fig, ax
