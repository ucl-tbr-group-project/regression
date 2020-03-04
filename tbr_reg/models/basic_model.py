import joblib
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

    def scale_training_set(self, X_train, out_scaler_file=None):
        if self.scaler is not None:
            X_train = self.scaler.fit_transform(X_train)
            if out_scaler_file is not None:
                joblib.dump(self.scaler, out_scaler_file)

        return X_train

    def scale_testing_set(self, X_test):
        if self.scaler is not None:
            X_test = self.scaler.transform(X_test)
        return X_test

    def train(self, X_train, y_train):
        '''Train model on the given set of inputs and expected outputs.'''
        pass

    def evaluate(self, X_test, y_test):
        '''Compare model's predictions on the given inputs with the given outputs.'''
        pass

    def predict(self, X):
        '''Let model predict on the given set of inputs.'''
        pass
