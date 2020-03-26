import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

from ..plot_reg_vs_time import plot_reg_vs_time
from ..plot_utils import set_plotting_style
from ..metric_loader import get_metric_factory


def main():
    # parse args
    parser = argparse.ArgumentParser(description='Evaluate TBR model search parameters')
    parser.add_argument('dirs', nargs='+')
    parser.add_argument('--performance-metric', type=str, default='r2',
                        help='metric for evaluating regression performance, supported values: "r2" (default), "mae", "adjusted_r2", "std_error"')
    parser.add_argument('--n-top-models', type=int, default=5,
                        help='how many best models to display')
    args = parser.parse_args()

    model_dict = {}
    for i in range(0, len(args.dirs), 2):
        dir_label, dir_name = args.dirs[i], args.dirs[i + 1]
        model_dict[dir_label] = pd.read_csv(os.path.join(dir_name, 'search.csv'))

    metric = get_metric_factory()[args.performance_metric]()

    set_plotting_style()

    fig, ax = plot_reg_vs_time(model_dict,
        performance_axis='mean_metric_%s' % metric.id, performance_axis_label='Regression performance ($%s$)' % metric.latex_name,
        select_top_n=args.n_top_models)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
