import matplotlib.pyplot as plt
import itertools

from .plot_utils import density_scatter
from .metric_loader import get_metric_factory


def plot_reg_vs_time(dfs, select_top_n=5,
    time_axis='mean_time_pred', time_axis_label='Prediction time per fold [s]', 
    performance_axis='mean_score', performance_axis_label='Regression performance',
    performance_ascending=False):
    fig, ax = plt.subplots()

    markers = itertools.cycle(('o', 'v', 's', 'D', 'p', 'x', '+')) 

    for (model_label, df), marker in zip(dfs.items(), markers):
        # select N best models
        df.sort_values(by=[performance_axis], inplace=True, ascending=performance_ascending)
        df = df.iloc[0:select_top_n].copy()

        xs = df[time_axis].to_numpy()
        ys = df[performance_axis].to_numpy()

        ax.plot(xs, ys, label=model_label, marker=marker, fillstyle='none', linestyle='none')

    ax.set_xlabel(time_axis_label)
    ax.set_ylabel(performance_axis_label)
    ax.legend()

    return fig, ax
