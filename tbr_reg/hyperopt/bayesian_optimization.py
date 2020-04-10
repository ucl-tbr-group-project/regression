import pandas as pd
from skopt import Optimizer
from skopt.space.space import Real, Integer, Categorical
from joblib import Parallel, delayed


def model_space_to_dims(model_space):
    dim_names = sorted(model_space.keys())

    def map_dim(values):
        if isinstance(values, tuple):  # linear subspace
            low, high, n_steps, value_type = values

            if value_type == 'i':
                return Integer(low, high)
            elif value_type == 'f':
                return Real(low, high)
            else:
                raise ValueError(f'Unknown value type "{value_type}"')
        else:  # exhaustive list of options
            return Categorical(values)

    dims = [map_dim(model_space[name]) for name in dim_names]
    return dim_names, dims


def evaluate_single_point(X, y, k_folds, random_state, suggested, model_creator, evaluation_handler, args_handler, dim_names, model_idx):
    model_args = {name: value for name, value in zip(dim_names, suggested)}
    print('Trying model configuration: %s' % model_args)

    if args_handler is not None:
        model_args = args_handler(model_idx, model_args)

    # run evaluation
    model_scores, model_mean_score, extra_values = evaluation_handler(
        model_idx, model_creator, model_args, X, y, k_folds, random_state + model_idx)

    return model_args, model_mean_score, model_scores, model_idx, extra_values


def bayesian_optimization(X, y, k_folds, random_state, model_space, model_creator, metric,
                          evaluation_handler, args_handler=None, post_evaluation_handler=None,
                          n_iterations=1000, extra_columns=[], n_parallel=1):
    dim_names, dims = model_space_to_dims(model_space)
    opt = Optimizer(dims, random_state=random_state)

    data = {column: [] for column in model_space.keys()}
    data['mean_score'] = []
    for i in range(k_folds):
        data['score%d' % i] = []
    for column in extra_columns:
        data[column] = []

    model_idx = 0
    for i in range(n_iterations):
        suggested = opt.ask(n_points=n_parallel)

        results = Parallel(n_jobs=n_parallel)(
            delayed(evaluate_single_point)(
                X, y, k_folds, random_state, point, model_creator,
                evaluation_handler, args_handler, dim_names, model_idx + parallel_offset)
            for point, parallel_offset in zip(suggested, range(n_parallel))
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

        opt.tell(suggested, [
            metric.rank(model_mean_score)
            for model_args, model_mean_score, model_scores, results_model_idx, extra_values in results])

        model_idx += n_parallel

    if post_evaluation_handler is not None:
        post_evaluation_handler(-model_idx, data, None)

    return pd.DataFrame(data=data)
