import os
import sys
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold

import ATE
from ..data_utils import load_batches, encode_data_frame, x_y_split
from ..plot_utils import set_plotting_style
from ..plot_reg_performance import plot_reg_performance
from ..model_loader import get_model_factory, load_model_from_file
from ..hyperopt.grid_search import grid_search
from ..hyperopt.bayesian_optimization import bayesian_optimization
from ..metric_loader import get_metric_factory


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
    parser.add_argument('out_dir', type=str,
                        help='output directory')
    parser.add_argument('search_space_config', type=str,
                        help='path to YAML search space configuration file')

    parser.add_argument('--feature-def', type=str,
                        help='path to file with line-separated (encoded) whitelisted feature names')
    parser.add_argument('--k-folds', type=int, default=5,
                        help='k for k-fold cross-validation')
    parser.add_argument('--score', type=str, default='r2',
                        help='metric for model quality evaluation, supported values: "r2" (default), "mae"')
    parser.add_argument('--strategy', type=str, default='grid',
                        help='algorithm used for search, supported values: "grid" (default), "bayesian"')
    args = parser.parse_args()

    set_plotting_style()

    # load data
    df = load_batches(args.in_dir, (args.batch_low, args.batch_high))
    df_enc = encode_data_frame(df, ATE.Domain())
    X, y_multiple = x_y_split(df_enc)
    y = y_multiple['tbr']

    if args.feature_def is not None:
        with open(args.feature_def, 'r') as f:
            included_features = [line.strip() for line in f.readlines()
                                 if len(line) > 0]
            X = X[included_features].copy()

    X = X.sort_index(axis=1)
    print(f'Features in order are: {list(X.columns)}')

    with open(args.search_space_config) as f:
        search_space_dict = yaml.load(f.read())

    metric = get_metric_factory()[args.score]()

    model_type = search_space_dict['model_type']
    model_creator = get_model_factory()[model_type]
    model_space = search_space_dict['search_space']

    def get_model_dir(model_idx):
        return os.path.join(args.out_dir, '%04d_running' % model_idx)

    def get_new_model_dir(model_idx, model_mean_score):
        return os.path.join(
            args.out_dir, '%04d_%0.06f' % (model_idx, model_mean_score))

    def get_output_file():
        return os.path.join(args.out_dir, 'search.csv')

    def args_handler(model_idx, model_args):
        model_dir = get_model_dir(model_idx)

        # prepare output directory
        if os.path.exists(model_dir):
            print('WARNING: path "%s" exists, deleting previous contents' % model_dir)
            os.removedirs(model_dir)

        os.makedirs(model_dir)
        return model_args

    def evaluation_handler(model_idx, model, X, y, k_folds, random_state):
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
                evaluation = test(model, X_test, y_test, metric=metric)
                kfold_scores.append(evaluation)

                plot_perf_path = os.path.join(
                    get_model_dir(model_idx), 'fold%d' % fold_idx)
                plot(plot_perf_path, model, X_test, y_test)
            except Exception as e:
                print(
                    f'WARNING: Fold {fold_idx+1} of {k_folds} failed with error: {e}')

            print(f'Fold {fold_idx+1} of {k_folds} done.')
            fold_idx += 1

        if len(kfold_scores) == 0:
            # all folds failed
            return np.nan * np.ones((k_folds, 1)), np.nan

        kfold_scores_arr = np.array(kfold_scores)
        kfold_scores_mean = np.mean(kfold_scores_arr, axis=0)
        print('K-fold mean scores are: %f' % kfold_scores_mean)

        if not isinstance(kfold_scores_mean, float):
            kfold_scores_mean = kfold_scores_mean[0]

        return kfold_scores_arr, kfold_scores_mean

    def post_evaluation_handler(model_idx, data, model_mean_score):
        if model_idx >= 0:
            # rename output directory to include model score
            model_dir = get_model_dir(model_idx)
            new_model_dir = get_new_model_dir(model_idx, model_mean_score)

            if os.path.exists(new_model_dir):
                print('WARNING: path "%s" exists, deleting previous contents' %
                      new_model_dir)
                os.removedirs(new_model_dir)

            os.rename(model_dir, new_model_dir)

        # save checkpoint
        if model_idx < 0 or model_idx % 5 == 0:
            out_file = get_output_file()
            print(f'Writing {model_idx+1} results to {out_file}')
            scores = pd.DataFrame(data=data)
            scores.to_csv(out_file)

    if args.strategy == 'grid':
        search_algorithm = grid_search
    elif args.strategy == 'bayesian':
        search_algorithm = bayesian_optimization
    else:
        raise ValueError(f'Unknown search strategy "{args.search_strategy}"')

    scores = search_algorithm(X, y, args.k_folds, random_state, model_space, model_creator, metric,
                              evaluation_handler, args_handler=args_handler, post_evaluation_handler=post_evaluation_handler)

    print('Search completed.')
    print('=====================================================')
    print('')

    best_idx = metric.rank(scores["mean_score"]).idxmin()
    print(
        f'Best found model index: {best_idx} with mean score: {scores.iloc[best_idx]["mean_score"]}')
    print('Best found model parameters:')
    print(scores.iloc[best_idx])

    print('Done.')


def train(model, X_train, y_train):
    print(f'Training regressor on set of size {X_train.shape[0]}')
    model.train(X_train.to_numpy(), y_train.to_numpy())


def test(model, X_test, y_test, metric):
    print(f'Testing regressor on set of size {X_test.shape[0]}')
    y_pred = model.predict(X_test.to_numpy())
    y_test = y_test.to_numpy()

    evaluation = metric.evaluate(y_test, y_pred)
    print(
        f'Evaluation on test set of size {X_test.shape[0]} gives {metric.name} result: {evaluation}')
    return evaluation


def plot(save_plot_path, model, X_test, y_test):
    df = X_test.copy()
    df.insert(0, 'tbr', -1.)
    df.insert(0, 'tbr_pred', -1.)
    df['tbr'] = y_test
    df['tbr_pred'] = model.predict(X_test)

    fig, ax = plot_reg_performance(df, density_bins=80)
    plt.tight_layout()

    plt.savefig('%s.png' % save_plot_path)
    plt.savefig('%s.pdf' % save_plot_path)
    plt.close()


if __name__ == '__main__':
    main()
