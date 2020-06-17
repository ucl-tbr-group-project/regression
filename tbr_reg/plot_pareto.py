import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import itertools

from .plot_utils import density_scatter
from .metric_loader import get_metric_factory


def plot_pareto(dfs,
                time_axis='mean_time_pred', time_axis_label='Prediction time per sample [ms]', time_axis_factor=1000, time_axis_limits=[0, 2.15],
                performance_axis='mean_score', performance_axis_label='Regression performance', performance_axis_limits=[0.4, 1],
                performance_ascending=False, training_set_size=None):
    fig, ax = plt.subplots()

    best_xs, best_ys = None, None

    markers = itertools.cycle(('o', 'v', 's', 'D', 'p', 'x', '+'))

    for (model_label, df), marker in zip(dfs.items(), markers):
        # select N best models
        df.sort_values(by=[performance_axis], inplace=True,
                       ascending=performance_ascending)
        df = df.iloc[0:1].copy()

        xs = df[time_axis].to_numpy() * time_axis_factor
        ys = df[performance_axis].to_numpy()

        if best_xs is None:
            best_xs, best_ys = xs[0], ys[0]
        else:
            best_xs = np.c_[best_xs, xs[0]]
            best_ys = np.c_[best_ys, ys[0]]

        p = ax.plot(xs, ys, label=model_label, marker=marker,
                    linestyle='none')
        series_color = p[0].get_color()

    points = np.c_[best_xs.T, best_ys.T]
    hull = ConvexHull(points)

    hull_points = points[hull.vertices, :]
    ax.fill(hull_points[:, 0], hull_points[:, 1], 'k', alpha=0.1)

    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1],
                '-', c='k', linewidth=1, linestyle='dashed')

    ax.set_xlabel(time_axis_label)
    ax.set_ylabel(performance_axis_label)
    ax.legend(loc='lower right')

    ax.set_xlim(time_axis_limits)
    ax.set_ylim(performance_axis_limits)

    if training_set_size is not None:
        ax.set_title(f'Training set size: {training_set_size}')

    return fig, ax
