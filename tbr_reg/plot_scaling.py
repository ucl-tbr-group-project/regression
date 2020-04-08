import matplotlib.pyplot as plt
import numpy as np
import itertools

from .plot_utils import density_scatter
from .metric_loader import get_metric_factory


def plot_scaling(dfs,
                 size_axis='train_size', size_axis_label='Training set size',
                 performance_axis='mean_score', performance_axis_label='Regression performance',
                 performance_axis_factor=1):
    fig, ax = plt.subplots()

    markers = itertools.cycle(('o', 'v', 's', 'D', 'p', 'x', '+'))

    for (model_label, df), marker in zip(dfs.items(), markers):
        # select N best models
        df = df[[size_axis, performance_axis]] \
            .groupby([size_axis])\
            .agg([np.mean, np.std])

        xs = df.index.to_numpy()
        ys = performance_axis_factor * df[performance_axis]['mean'].to_numpy()
        es = performance_axis_factor * df[performance_axis]['std'].to_numpy()

        ax.errorbar(xs, ys, yerr=es, label=model_label, marker=marker,
                    linestyle='dotted', capsize=4)

    ax.set_xlabel(size_axis_label)
    ax.set_ylabel(performance_axis_label)
    ax.legend()

    return fig, ax
