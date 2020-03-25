import pandas as pd
from skopt import Optimizer
from skopt.space.space import Real, Integer, Categorical


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


def bayesian_optimization(X, y, k_folds, random_state, model_space, model_creator, metric,
                          evaluation_handler, args_handler=None, post_evaluation_handler=None):
    dim_names, dims = model_space_to_dims(model_space)
    opt = Optimizer(dims)

    data = {column: [] for column in model_space.keys()}
    data['mean_score'] = []
    for i in range(k_folds):
        data['score%d' % i] = []

    model_idx = 0
    for i in range(20):
        suggested = opt.ask()
        model_args = {name: value for name, value in zip(dim_names, suggested)}
        print('Trying model configuration: %s' % model_args)

        if args_handler is not None:
            model_args = args_handler(model_idx, model_args)

        # run evaluation
        model = model_creator(arg_dict=model_args)
        model_scores, model_mean_score = evaluation_handler(
            model_idx, model, X, y, k_folds, random_state + model_idx)

        # save scores
        for name, value in model_args.items():
            data[name].append(value)

        data['mean_score'].append(model_mean_score)
        for i in range(k_folds):
            data['score%d' % i].append(model_scores[i])

        opt.tell(suggested, metric.rank(model_mean_score))

        if post_evaluation_handler is not None:
            post_evaluation_handler(model_idx, data, model_mean_score)

        model_idx += 1

    if post_evaluation_handler is not None:
        post_evaluation_handler(-model_idx, data, None)

    return pd.DataFrame(data=data)
