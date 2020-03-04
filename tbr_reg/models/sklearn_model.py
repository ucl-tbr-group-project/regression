import argparse
import joblib
from sklearn.metrics import mean_absolute_error

from models.basic_model import RegressionModel


class SKLearnModel(RegressionModel):
    '''A base class for all SciKit models.'''

    def __init__(self,
                 name,
                 scaling='standard',  # standard|minmax|none
                 out=None,  # overrides all below
                 out_model_file=None,
                 out_scaler_file=None
                 ):
        RegressionModel.__init__(self, name, scaling=scaling)

        if out is not None:
            out_model_file = '%s.svm.pkl' % out
            out_scaler_file = '%s.scaler.pkl' % out

        self.out_scaler_file = out_scaler_file
        self.out_model_file = out_model_file

        self.sklearn_model = None

    @staticmethod
    def load(derived, model, scaler=None):
        derived.sklearn_model = joblib.load(model)
        derived.scaler = joblib.load(scaler) if scaler is not None else None
        return derived

    @staticmethod
    def create_cli_parser(description):
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument('--scaling', type=str,
                            help='how to scale data before training')
        parser.add_argument('--out', type=str,
                            help='where to save all outputs')
        parser.add_argument('--out-model-file', type=str,
                            help='where to save trained model')
        parser.add_argument('--out-scaler-file', type=str,
                            help='where to save scaler')
        return parser

    def train(self, X_train, y_train):
        X_train = self.scale_training_set(
            X_train, out_scaler_file=self.out_scaler_file)

        self.sklearn_model.fit(X_train, y_train)

        # save trained model
        if self.out_model_file is not None:
            joblib.dump(self.sklearn_model, self.out_model_file)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return mean_absolute_error(y_test, y_pred)

    def predict(self, X):
        X = self.scale_testing_set(X)
        return self.sklearn_model.predict(X)
