import pandas as pd

from .model_space import model_space_product


def grid_search(X, y, k_folds, random_state, model_space, model_creator, metric,
                evaluation_handler, args_handler=None, post_evaluation_handler=None):
    data = {column: [] for column in model_space.keys()}
    data['mean_score'] = []
    for i in range(k_folds):
        data['score%d' % i] = []

    model_idx = 0
    for model_args in model_space_product(model_space):
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

        if post_evaluation_handler is not None:
            post_evaluation_handler(model_idx, data, model_mean_score)

        model_idx += 1

    if post_evaluation_handler is not None:
        post_evaluation_handler(-model_idx, data, None)

    return pd.DataFrame(data=data)
