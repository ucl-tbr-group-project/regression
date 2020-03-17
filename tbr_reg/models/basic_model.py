import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class RegressionModel:
    '''Base class for all regression models. Meant to be inherited and overridden.'''

    def __init__(self, name,
                 scaling='standard'):  # standard|minmax|none
        '''Initialize model with name.'''
        self.name = name
        self.scaling = scaling

        scalers = RegressionModel.get_scalers()
        if self.scaling in scalers:
            self.scaler = scalers[self.scaling]()
        else:
            raise ValueError(f'Unknown scaler "{self.scaling}".')

    @staticmethod
    def get_scalers():
        return {
            'standard': lambda: StandardScaler(),
            'minmax': lambda: MinMaxScaler(),
            'none': lambda: None
        }

    def join_sets(self, X, y):
        Xy = np.zeros([X.shape[0], X.shape[1] + 1])
        last_column_idx = Xy.shape[1] - 1
        Xy[:, 0:last_column_idx] = X
        Xy[:, last_column_idx] = y.ravel()
        return Xy

    def split_sets(self, Xy):
        last_column_idx = Xy.shape[1] - 1
        X = Xy[:, 0:last_column_idx]
        y = Xy[:, last_column_idx]
        return X, y

    def scale_training_set(self, X_train, y_train, out_scaler_file=None):
        if self.scaler is not None:
            Xy_in = self.join_sets(X_train, y_train)
            Xy_out = self.scaler.fit_transform(Xy_in)
            X_train, y_train = self.split_sets(Xy_out)
            if out_scaler_file is not None:
                joblib.dump(self.scaler, out_scaler_file)

        return X_train, y_train

    def scale_testing_set(self, X_test, y_test):
        if self.scaler is not None:
            y_test = y_test if y_test is not None \
                else np.zeros((X_test.shape[0], 1))
            Xy_in = self.join_sets(X_test, y_test)
            Xy_out = self.scaler.transform(Xy_in)
            X_test, y_test = self.split_sets(Xy_out)
        return X_test, y_test

    def inverse_scale_predictions(self, y_pred):
        if self.scaler is not None:
            X_dummy = np.zeros(
                (y_pred.shape[0], self.scaler.scale_.shape[0]-1))
            Xy_in = self.join_sets(X_dummy, y_pred)
            Xy_out = self.scaler.inverse_transform(Xy_in)
            X_dummy, y_pred = self.split_sets(Xy_out)
        return y_pred

    def inverse_scale_errors(self, errs):
        if self.scaler is not None:
            # TODO: this will only work for scalers which multiply the result with scale_
            errs = self.scaler.scale_[self.scaler.scale_.shape[0]-1] * errs
        return errs

    def train(self, X_train, y_train):
        '''Train model on the given set of inputs and expected outputs.'''
        pass

    def evaluate(self, X_test, y_test):
        '''Compare model's predictions on the given inputs with the given outputs.'''
        pass

    def predict(self, X):
        '''Let model predict on the given set of inputs.'''
        pass
