import pandas as pd
from joblib import Parallel, delayed
from itertools import zip_longest

from .model_space import model_space_product


def grouper(n, iterable, fillvalue=None):
    # courtesy of https://stackoverflow.com/a/8290490
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


def evaluate_single_point(X, y, k_folds, random_state, model_args, model_creator, evaluation_handler, args_handler, model_idx):
    print('Trying model configuration: %s' % model_args)

    if args_handler is not None:
        model_args = args_handler(model_idx, model_args)

    # run evaluation
    model_scores, model_mean_score, extra_values = evaluation_handler(
        model_idx, model_creator, model_args, X, y, k_folds, random_state + model_idx)

    return model_args, model_mean_score, model_scores, model_idx, extra_values


def grid_search(X, y, k_folds, random_state, model_space, model_creator, metric,
                evaluation_handler, args_handler=None, post_evaluation_handler=None,
                extra_columns=[], n_parallel=1):
    data = {column: [] for column in model_space.keys()}
    data['mean_score'] = []
    for i in range(k_folds):
        data['score%d' % i] = []
    for column in extra_columns:
        data[column] = []

    model_idx = 0
    for model_args_batch in grouper(n_parallel, model_space_product(model_space)):
        results = Parallel(n_jobs=n_parallel)(
            delayed(evaluate_single_point)(
                X, y, k_folds, random_state, model_args, model_creator,
                evaluation_handler, args_handler, model_idx + parallel_offset)
            for model_args, parallel_offset in zip(model_args_batch, range(n_parallel))
        )

        # save scores
        for model_args, model_mean_score, model_scores, results_model_idx, extra_values in results:
            for arg_name in data.keys():
                if arg_name in model_args:
                    data[arg_name].append(model_args[arg_name])

            for column, value in extra_values.items():
                data[column].append(value)

            data['mean_score'].append(model_mean_score)
            for i in range(k_folds):
                data['score%d' % i].append(model_scores[i])

            if post_evaluation_handler is not None:
                post_evaluation_handler(
                    results_model_idx, data, model_mean_score)

        model_idx += n_parallel

    if post_evaluation_handler is not None:
        post_evaluation_handler(-model_idx, data, None)

    return pd.DataFrame(data=data)
