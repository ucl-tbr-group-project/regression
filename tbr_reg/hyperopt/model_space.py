import itertools
import numpy as np


def model_space_product(model_space):
    '''
    Iterate over Cartesian product of hyperparameters.
    Expects dictionary of parameters, for instance:
    {
        'param1': ['value1', 'value2'],
        'param2': ['value3', 'value4']
    }

    For the above input, the following configurations are yielded:
      1. {'param1': 'value1', 'param2': 'value3'}
      2. {'param1': 'value1', 'param2': 'value4'}
      3. {'param1': 'value2', 'param2': 'value3'}
      4. {'param1': 'value2', 'param2': 'value4'}

    The values in the input dictionary can be either:
      - lists--these are treated as categorical values, or
      - 4-tuples--these are passed to np.linspace to discretize a linear subspace (excluding the last component).
    '''

    product_args = []
    product_input = []
    is_integer = []

    for name, values in model_space.items():
        product_args.append(name)

        if isinstance(values, tuple):  # linear subspace
            low, high, n_steps, value_type = values
            product_input.append(np.linspace(low, high, n_steps).tolist())
            is_integer.append(value_type == 'i')
        else:  # exhaustive list of options
            product_input.append(values)
            is_integer.append(False)

    for arg_values in itertools.product(*product_input):
        yield {name: (int(value) if should_quantize else value)
               for name, value, should_quantize in zip(product_args, arg_values, is_integer)}
