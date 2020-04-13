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
        self.scaler_fitted = False
        self.scaler = self.create_scaler()

        self.renormalize = False
        self.saved_renormalization_sets = None, None
        self.renormalization_threshold = None
        self.should_retrain_on_saved_set = False

    @staticmethod
    def get_scalers():
        return {
            'standard': lambda: StandardScaler(),
            'minmax': lambda: MinMaxScaler(),
            'none': lambda: None
        }

    def create_scaler(self):
        scalers = RegressionModel.get_scalers()
        if self.scaling in scalers:
            return scalers[self.scaling]()
        else:
            raise ValueError(f'Unknown scaler "{self.scaling}".')

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
        self.save_training_set_for_renormalization(X_train, y_train)
        self.should_retrain_on_saved_set = self.update_scaler_if_needed()

        if self.scaler is not None:
            Xy_in = self.join_sets(X_train, y_train)
            print(type(Xy_in))

            if not self.scaler_fitted:
                Xy_out = self.scaler.fit_transform(Xy_in)
                self.scaler_fitted = True
            else:
                Xy_out = self.scaler.transform(Xy_in)

            X_train, y_train = self.split_sets(Xy_out)
            if out_scaler_file is not None:
                joblib.dump(self.scaler, out_scaler_file)

        return X_train, y_train

    def get_saved_renormalization_set_for_retraining(self):
        if not self.renormalize:
            return None, None

        X_train, y_train = self.saved_renormalization_sets

        if self.scaler is not None:
            Xy_in = self.join_sets(X_train, y_train)
            Xy_out = self.scaler.transform(Xy_in)
            X_train, y_train = self.split_sets(Xy_out)

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

    def enable_renormalization(self, threshold):
        self.renormalize = True
        self.saved_renormalization_sets = None, None
        self.renormalization_threshold = threshold

    def save_training_set_for_renormalization(self, X_train, y_train):
        if not self.renormalize:
            return

        X_prev, y_prev = self.saved_renormalization_sets
        if X_prev is None:
            self.saved_renormalization_sets = X_train, y_train
            return

        # concatenate sets, this could get ugly fast!
        X_prev = np.r_[X_prev, X_train]
        y_prev = np.r_[y_prev, y_train]
        self.saved_renormalization_sets = X_prev, y_prev

    def fit_new_scaler(self):
        if not self.renormalize:
            return None

        X_saved, y_saved = self.saved_renormalization_sets
        if X_saved is None:
            return None

        alt_scaler = self.create_scaler()
        Xy_in = self.join_sets(X_saved, y_saved)
        alt_scaler.fit(Xy_in)
        return alt_scaler

    @staticmethod
    def scaler_similarity(a, b):
        if isinstance(a, StandardScaler):
            return np.linalg.norm(a.mean_ - b.mean_) + np.linalg.norm(a.var_ - b.var_)
        elif isinstance(a, MinMaxScaler):
            return np.linalg.norm(a.data_min_ - b.data_min_) + np.linalg.norm(a.data_max_ - b.data_max_)
        else:
            raise ValueError('Unsupported scaler type.')

    def update_scaler_if_needed(self):
        alt_scaler = self.fit_new_scaler()

        if self.renormalize and self.scaler_fitted and \
                self.scaler is not None and alt_scaler is not None:
            similarity = RegressionModel.scaler_similarity(
                self.scaler, alt_scaler)

            if similarity < self.renormalization_threshold:
                print(
                    f'Scaler similarity {similarity} is below set threshold {self.renormalization_threshold}, reusing previous scaler')
            else:
                print(
                    f'Scaler similarity {similarity} exceeds set threshold {self.renormalization_threshold}, replacing scaler and training from scratch')
                self.scaler = alt_scaler
                self.scaler_fitted = True
                return True

        return False

    def train(self, X_train, y_train):
        '''Train model on the given set of inputs and expected outputs.'''
        pass

    def evaluate(self, X_test, y_test):
        '''Compare model's predictions on the given inputs with the given outputs.'''
        pass

    def predict(self, X):
        '''Let model predict on the given set of inputs.'''
        pass
