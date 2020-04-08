import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

from ..plot_scaling import plot_scaling
from ..plot_utils import set_plotting_style
from ..metric_loader import get_metric_factory


def main():
    # parse args
    parser = argparse.ArgumentParser(
        description='Evaluate TBR model benchmark results')
    parser.add_argument('dirs', nargs='+')
    parser.add_argument('--performance-metric', type=str, default='r2',
                        help='metric for evaluating regression performance, supported values: "r2" (default), "mae", "adjusted_r2", "std_error"')
    args = parser.parse_args()

    model_dict = {}
    for i in range(0, len(args.dirs), 2):
        dir_label, dir_name = args.dirs[i], args.dirs[i + 1]
        model_dict[dir_label] = pd.read_csv(
            os.path.join(dir_name, 'benchmark.csv'))

    set_plotting_style()

    if args.performance_metric == 'time_pred':
        fig, ax = plot_scaling(model_dict, performance_axis='mean_time_pred', performance_axis_factor=1e3,
                               performance_axis_label='Prediction time per sample [ms]')
    elif args.performance_metric == 'time_train':
        fig, ax = plot_scaling(model_dict, performance_axis='mean_time_train', performance_axis_factor=1e3,
                               performance_axis_label='Training time per sample [ms]')
    else:
        metric = get_metric_factory()[args.performance_metric]()
        fig, ax = plot_scaling(model_dict,
                               performance_axis='mean_metric_%s' % metric.id, performance_axis_label='Regression performance ($%s$)' % metric.latex_name)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
