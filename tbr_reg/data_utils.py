import os
import sys

import pandas as pd
import numpy as np


def load_batches(in_dir, batch_range, file_template='batch%d_out.csv'):
    '''
    Load multiple batches as data frames and concatenate them into a single data frame.
    Errors are handled gracefully: messages are printed for each data frame that fails.
    '''
    file_paths = [os.path.join(in_dir, file_template % batch_idx)
                  for batch_idx in range(batch_range[0], batch_range[1])]

    def graceful_load(path):
        try:
            return pd.read_csv(path)
        except:
            print('Skipping %s due to error: %s' %
                  (path, str(sys.exc_info()[0])))
            return None

    loaded_frames = [graceful_load(file_path) for file_path in file_paths]
    return pd.concat([frame for frame in loaded_frames if frame is not None])


def encode_data_frame(df, domain):
    '''
    Encode data frame into format suitable for regression.
    This maps continuous columns with identity, and one-hot-encodes discrete columns.
    '''
    column_map = [param.transform_columns() for param in domain.params]
    one_hot = pd.get_dummies(df)
    zero_columns = [column for columns in column_map for column in columns]
    for column in zero_columns:
        if column not in one_hot.columns:
            one_hot[column] = 0.
    return one_hot


def x_y_split(df, drop_invalid=True):
    '''
    Split encoded data frame into regression inputs (X) and outputs (y).
    '''
    y_params = ['tbr', 'tbr_error']
    drop_params = ['sim_time', '']
    X_params = list(set(df.columns.tolist()) - set(y_params + drop_params))

    df_copy = df.copy()

    if drop_invalid:
        df_copy[y_params] = df_copy[y_params].replace(-1., np.nan)
        df_copy = df_copy.dropna()

    X, y = df_copy[X_params].copy(), df_copy[y_params].copy()
    return X, y


def c_d_split(df, drop_invalid=True):
    '''
    Split encoded input data frame into continuous (c) and discrete (d) inputs.
    '''
    drop_params = ['sim_time', 'tbr', 'tbr_error', '']
    d_params = ['firstwall_armour_material',
                'firstwall_structural_material',
                'firstwall_coolant_material',
                'blanket_structural_material',
                'blanket_breeder_material',
                'blanket_multiplier_material',
                'blanket_coolant_material']
    c_params = list(set(df.columns.tolist()) -
                    set(drop_params + d_params))

    df_copy = df.copy()

    if drop_invalid:
        df_copy = df_copy.dropna()

    c, d = df_copy[c_params].copy(), df_copy[d_params].copy()
    return c, d

def c_d_y_split(df, drop_invalid=True):
    '''
    Split encoded data frame into continuous (c) and discrete (d) inputs, and outputs (y).
    '''
    y_params = ['tbr', 'tbr_error']
    drop_params = ['sim_time', '']
    d_params = ['firstwall_armour_material',
                'firstwall_structural_material',
                'firstwall_coolant_material',
                'blanket_structural_material',
                'blanket_breeder_material',
                'blanket_multiplier_material',
                'blanket_coolant_material']
    c_params = list(set(df.columns.tolist()) -
                    set(y_params + drop_params + d_params))

    df_copy = df.copy()

    if drop_invalid:
        df_copy[y_params] = df_copy[y_params].replace(-1., np.nan)
        df_copy = df_copy.dropna()

    c, d, y = df_copy[c_params].copy(
    ), df_copy[d_params].copy(), df_copy[y_params].copy()
    return c, d, y
