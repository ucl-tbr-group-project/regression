import argparse
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR

from models.basic_model import RegressionModel


class SupportVectorModel(RegressionModel):
    '''A support vector machine with kernel trick, implemented by SciKit.'''

    def __init__(self,
                 scaling='standard',  # standard|minmax|none
                 out=None,  # overrides all below
                 out_model_file=None,
                 out_scaler_file=None
                 ):
        RegressionModel.__init__(self, 'SVM')

        if out is not None:
            out_model_file = '%s.svm.h5' % out
            out_scaler_file = '%s.scaler.pkl' % out

        self.scaling = scaling
        self.out_scaler_file = out_scaler_file
        self.out_model_file = out_model_file

        self.svr = None

        if self.scaling == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaling == 'none':
            self.scaler = None
        else:
            raise ValueError('Unknown scaler.')

    @staticmethod
    def load(model, scaler=None):
        loaded = SupportVectorModel()
        loaded.svr = joblib.load(model)

        if scaler is not None:
            loaded.scaler = joblib.load(scaler)

        return loaded

    @staticmethod
    def parse_cli_args(args):
        parser = argparse.ArgumentParser(
            description='Train support vector machine')
        parser.add_argument('--scaling', type=str,
                            help='how to scale data before training')
        parser.add_argument('--out', type=str,
                            help='where to save all outputs')
        parser.add_argument('--out-model-file', type=str,
                            help='where to save trained model')
        parser.add_argument('--out-scaler-file', type=str,
                            help='where to save scaler')

        return {key: value
                for key, value in vars(parser.parse_args(args)).items()
                if value is not None}

    def train(self, X_train, y_train):
        if self.scaler is not None:
            X_train = self.scaler.fit_transform(X_train)
            if self.out_scaler_file is not None:
                joblib.dump(self.scaler, self.out_scaler_file)

        self.svr = SVR(verbose=True).fit(X_train, y_train)

        # save trained model
        if self.out_model_file is not None:
            joblib.dump(self.svr, self.out_model_file)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return mean_absolute_error(y_test, y_pred)

    def predict(self, X):
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.svr.predict(X)
