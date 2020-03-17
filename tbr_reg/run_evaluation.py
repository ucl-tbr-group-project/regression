import matplotlib.pyplot as plt
import argparse

import ATE
from .data_utils import load_batches, encode_data_frame, x_y_split
from .model_loader import load_model_from_file
from .plot_reg_performance import plot_reg_performance
from .plot_utils import set_plotting_style


def main():
    # parse args
    parser = argparse.ArgumentParser(description='Evaluate TBR model')
    parser.add_argument('in_dir', type=str,
                        help='directory containing input batches')
    parser.add_argument('batch_low', type=int,
                        help='start batch index (inclusive)')
    parser.add_argument('batch_high', type=int,
                        help='end batch index (exclusive)')
    parser.add_argument('model_file', type=str,
                        help='model file path')
    parser.add_argument('--no-density', default=False, action='store_true',
                        help='disable density coloring')
    args = parser.parse_args()

    model_name, model = load_model_from_file(args.model_file)

    df = load_batches(args.in_dir, (args.batch_low, args.batch_high))
    df_enc = encode_data_frame(df, ATE.Domain())
    X, y_multiple = x_y_split(df_enc)
    y = y_multiple['tbr']
    X = X.sort_index(axis=1)

    df.insert(0, 'tbr_pred', -1.)
    df['tbr_pred'] = model.predict(X)

    set_plotting_style()

    fig, ax = plot_reg_performance(
        df, density_bins=(None if args.no_density else 80))

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
