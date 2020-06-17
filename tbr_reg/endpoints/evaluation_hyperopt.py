import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

from ..plot_reg_vs_time import plot_reg_vs_time
from ..plot_utils import set_plotting_style
from ..metric_loader import get_metric_factory


def run_search(model_dict, metric, args):
    fig, ax = plot_reg_vs_time(model_dict,
                               performance_axis='mean_metric_%s' % metric.id, performance_axis_label='Regression performance ($%s$)' % metric.latex_name,
                               time_axis_limits=(0, 0.5), performance_axis_limits=(0.5, 1),
                               select_top_n=args.n_top_models)

    plt.tight_layout()
    plt.show()


def run_benchmark(model_dict, metric, args, size_axis='train_size'):
    model_dict_iter = iter(model_dict.items())
    sizes = next(model_dict_iter)[1][size_axis].unique()

    for size_idx, size in enumerate(sizes):
        filtered_model_dict = {model: df[df[size_axis] == size].copy()
                               for model, df in model_dict.items()}
        fig, ax = plot_reg_vs_time(filtered_model_dict,
                                   performance_axis='mean_metric_%s' % metric.id, performance_axis_label='Regression performance ($%s$)' % metric.latex_name,
                                   select_top_n=args.n_top_models,
                                   training_set_size=size)

        plt.tight_layout()
        plt.savefig(f'reg_vs_time_{size_idx}.png')
        plt.show()
        plt.close('all')


def main():
    # parse args
    parser = argparse.ArgumentParser(
        description='Evaluate TBR model search parameters')
    parser.add_argument('dirs', nargs='+')
    parser.add_argument('--performance-metric', type=str, default='r2',
                        help='metric for evaluating regression performance, supported values: "r2" (default), "mae", "adjusted_r2", "std_error"')
    parser.add_argument('--n-top-models', type=int, default=5,
                        help='how many best models to display')
    parser.add_argument('--type', type=str, default='search',
                        help='CSV file type: search or benchmark')
    args = parser.parse_args()

    set_plotting_style()

    file_name = None
    if args.type == 'search':
        file_name = 'search.csv'
    elif args.type == 'benchmark':
        file_name = 'benchmark.csv'
    else:
        raise ValueError(f'Unknown type: {args.type}')

    model_dict = {}
    for i in range(0, len(args.dirs), 2):
        dir_label, dir_name = args.dirs[i], args.dirs[i + 1]
        model_dict[dir_label] = pd.read_csv(os.path.join(dir_name, file_name))

    metric = get_metric_factory()[args.performance_metric]()

    if args.type == 'search':
        run_search(model_dict, metric, args)
    elif args.type == 'benchmark':
        run_benchmark(model_dict, metric, args)


if __name__ == '__main__':
    main()
