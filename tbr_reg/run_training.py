import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import ATE
from .data_utils import load_batches, encode_data_frame, x_y_split
from .plot_utils import set_plotting_style
from .plot_reg_performance import plot_reg_performance
from .model_loader import get_model_factory


def main():
    '''Main command line entry point. Trains model with given parameters.'''

    random_state = 1

    # parse args
    parser = argparse.ArgumentParser(description='Train TBR model')
    parser.add_argument('type', type=str,
                        help='which model to train')
    parser.add_argument('in_dir', type=str,
                        help='directory containing input batches')
    parser.add_argument('batch_low', type=int,
                        help='start batch index (inclusive)')
    parser.add_argument('batch_high', type=int,
                        help='end batch index (exclusive)')
    parser.add_argument('test_set_size', type=float,
                        help='fractional size of the test set, set 0 to disable testing and to negative value k for k-fold cross-validation')
    parser.add_argument('plot_perf', type=str,
                        help='set 0 to disable performance plots, set int to enable interactive plots, set anything else to save plots in a file')
    parser.add_argument('feature_def', type=str,
                        help='set 0 to allow all features, otherwise set path to file with line-separated (encoded) whitelisted feature names')
    args = parser.parse_args(sys.argv[1:8])
    model = get_model_factory()[args.type](sys.argv[8:])

    set_plotting_style()

    # load data
    df = load_batches(args.in_dir, (args.batch_low, args.batch_high))
    df_enc = encode_data_frame(df, ATE.Domain())
    X, y_multiple = x_y_split(df_enc)
    y = y_multiple['tbr']

    if args.feature_def != '0':
        with open(args.feature_def, 'r') as f:
            included_features = [line.strip() for line in f.readlines()
                                 if len(line) > 0]
            X = X[included_features].copy()

    X = X.sort_index(axis=1)
    print(f'Features in order are: {list(X.columns)}')

    if args.test_set_size > 0:  # fractional test set size
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_set_size, random_state=random_state)
        train(model, X_train, y_train)
        test(model, X_test, y_test)
        if args.plot_perf != '0':
            plot(args.plot_perf, model, X_test, y_test)
    elif args.test_set_size == 0:  # testing disabled
        X_train, y_train = X, y
        train(model, X_train, y_train)
    else:  # k-fold cross-validation
        k = int(-args.test_set_size)
        kfold = KFold(n_splits=k, shuffle=True, random_state=random_state)
        kfold_scores = []

        fold_idx = 0
        for train_index, test_index in kfold.split(X, y):
            print(f'Starting fold {fold_idx+1} of {k}.')
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            train(model, X_train, y_train)
            evaluation = test(model, X_test, y_test)
            kfold_scores.append(evaluation)

            if args.plot_perf != '0':
                plot(args.plot_perf % fold_idx, model, X_test, y_test)

            print(f'Fold {fold_idx+1} of {k} done.')
            fold_idx += 1

        kfold_scores_arr = np.array(kfold_scores)
        print('K-fold mean scores are:')
        print(np.mean(kfold_scores_arr, axis=0))

    print('Done.')


def train(model, X_train, y_train):
    print(f'Training regressor on set of size {X_train.shape[0]}')
    model.train(X_train.to_numpy(), y_train.to_numpy())


def test(model, X_test, y_test):
    print(f'Testing regressor on set of size {X_test.shape[0]}')
    evaluation = model.evaluate(X_test.to_numpy(), y_test.to_numpy())
    print(
        f'Evaluation on test set of size {X_test.shape[0]} gives result: {evaluation}')
    return evaluation


def plot(save_plot_path, model, X_test, y_test):
    df = X_test.copy()
    df.insert(0, 'tbr', -1.)
    df.insert(0, 'tbr_pred', -1.)
    df['tbr'] = y_test
    df['tbr_pred'] = model.predict(X_test)

    fig, ax = plot_reg_performance(df, density_bins=80)
    plt.tight_layout()

    if save_plot_path == 'int':
        plt.show()
    else:
        plt.savefig('%s.png' % save_plot_path)
        plt.savefig('%s.pdf' % save_plot_path)


if __name__ == '__main__':
    main()
