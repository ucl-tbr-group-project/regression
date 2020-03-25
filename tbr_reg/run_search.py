import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import ATE
from .data_utils import load_batches, encode_data_frame, x_y_split
from .plot_utils import set_plotting_style
from .plot_reg_performance import plot_reg_performance
from .model_loader import get_model_factory, load_model_from_file


def main():
    '''Main command line entry point. Trains model with given parameters.'''

    random_state = 1

    # parse args
    parser = argparse.ArgumentParser(
        description='Search TBR model hyperparameter space')
    parser.add_argument('in_dir', type=str,
                        help='directory containing input batches')
    parser.add_argument('batch_low', type=int,
                        help='start batch index (inclusive)')
    parser.add_argument('batch_high', type=int,
                        help='end batch index (exclusive)')
    parser.add_argument('k_folds', type=int,
                        help='k for k-fold cross-validation')
    parser.add_argument('out_dir', type=str,
                        help='output directory')
    parser.add_argument('feature_def', type=str,
                        help='set 0 to allow all features, otherwise set path to file with line-separated (encoded) whitelisted feature names')
    args = parser.parse_args()

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

    model_creator, model_space = prepare_model_domain()
    scores = grid_search(X, y, args.k_folds, random_state,
                         model_space, model_creator, args.out_dir)

    scores.to_csv(os.path.join(args.out_dir, 'grid_search.csv'))

    print('Done.')


def prepare_model_domain():
    model_type = 'krg'
    creator = get_model_factory()[model_type]
    search_space = {
        'poly': ['constant', 'linear', 'quadratic'],
        'theta0': [0.01, 0.02, 0.05],
    }

    return creator, search_space


def model_space_product(model_space):
    product_args = []
    product_input = []

    for name, values in model_space.items():
        product_args.append(name)

        if isinstance(values, tuple):  # linear subspace
            low, high, n_steps = values
            product_input.append(np.linspace(low, high, n_steps).tolist())
        else:  # exhaustive list of options
            product_input.append(values)

    for arg_values in itertools.product(*product_input):
        yield {name: value for name, value in zip(product_args, arg_values)}


def grid_search(X, y, k_folds, random_state, model_space, model_creator, out_dir):
    data = {column: [] for column in model_space.keys()}
    data['mean_score'] = []
    for i in range(k_folds):
        data['score%d' % i] = []

    model_idx = 0
    for model_args in model_space_product(model_space):
        print('Trying model configuration: %s' % model_args)

        # prepare output directory
        model_dir = os.path.join(out_dir, '%04d_running' % model_idx)

        if os.path.exists(model_dir):
            print('WARNING: path "%s" exists, deleting previous contents' % model_dir)
            os.removedirs(model_dir)
            os.rmdir(model_dir)

        os.makedirs(model_dir)

        # run evaluation
        model = model_creator(arg_dict=model_args)
        model_scores, model_mean_scores = evaluate_model(
            model, model_dir, X, y, k_folds, random_state + model_idx)

        # rename output directory to include model score
        new_model_dir = os.path.join(
            out_dir, '%04d_%0.06f' % (model_idx, model_mean_scores[0, 0]))
        os.rename(model_dir, new_model_dir)

        # save scores
        for name, value in model_args.items():
            data[name].append(value)

        data['mean_score'].append(model_mean_scores)
        for i in range(k_folds):
            data['score%d' % i].append(model_scores[i])

        model_idx += 1

    return pd.DataFrame(data=data)


def evaluate_model(model, model_dir, X, y, k_folds, random_state):
    kfold = KFold(n_splits=k_folds, shuffle=True,
                  random_state=random_state)
    kfold_scores = []

    fold_idx = 0
    for train_index, test_index in kfold.split(X, y):
        print(f'Starting fold {fold_idx+1} of {k_folds}.')
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        try:
            train(model, X_train, y_train)
            evaluation = test(model, X_test, y_test)
            kfold_scores.append(evaluation)

            plot_perf_path = os.path.join(model_dir, 'fold%d' % fold_idx)
            plot(plot_perf_path, model, X_test, y_test)
        except Exception as e:
            print(
                f'WARNING: Fold {fold_idx+1} of {k_folds} failed with error: {e}')

        print(f'Fold {fold_idx+1} of {k_folds} done.')
        fold_idx += 1

    if len(kfold_scores) == 0:
        # all folds failed
        return np.nan * np.ones((k_folds, 1)), np.nan * np.ones((1, 1))

    kfold_scores_arr = np.array(kfold_scores)
    kfold_scores_mean = np.mean(kfold_scores_arr, axis=0)
    print('K-fold mean scores are: %f' % kfold_scores_mean)

    if isinstance(kfold_scores_mean, float):
        kfold_scores_mean = np.array([[kfold_scores_mean]])

    return kfold_scores_arr, kfold_scores_mean


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
