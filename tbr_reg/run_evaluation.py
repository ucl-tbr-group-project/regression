import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.linear_model import Ridge
from scipy.interpolate import interpn

import ATE
from data_utils import load_batches, encode_data_frame, x_y_split
from model_loader import load_model_from_file


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
    args = parser.parse_args()

    model_name, model = load_model_from_file(args.model_file)

    df = load_batches(args.in_dir, (args.batch_low, args.batch_high))
    df_enc = encode_data_frame(df, ATE.Domain())
    X, y_multiple = x_y_split(df_enc)
    y = y_multiple['tbr']
    X = X.sort_index(axis=1)

    df.insert(0, 'tbr_pred', -1.)
    df['tbr_pred'] = model.predict(X)

    xs = df['tbr'].to_numpy()
    ys = df['tbr_pred'].to_numpy()

    linreg = Ridge().fit(xs.reshape(-1, 1), ys.reshape(-1, 1))
    ys_fit = linreg.predict(xs.reshape(-1, 1))

    fig, ax = plt.subplots()

    density_scatter(xs, ys, ax=ax, bins=80, s=5, label='Data')

    ax.plot(xs, xs, '--', c='k', linewidth=1,
            dashes=[5, 10], label='Perfect model')
    ax.plot(xs, ys_fit, '--', c='r', linewidth=1,
            dashes=[5, 10], label='Trained model')

    ax.set_xlabel('True TBR')
    ax.set_ylabel('Predicted TBR')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.show()


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


if __name__ == '__main__':
    main()
