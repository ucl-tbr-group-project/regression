import joblib
from sklearn.gaussian_process import GaussianProcessRegressor

from models.sklearn_model import SKLearnModel


class GaussianProcessModel(SKLearnModel):
    '''A Gaussian process regressor, implemented by SciKit.'''

    def __init__(self,
                 scaling='standard',  # standard|minmax|none
                 out=None,  # overrides all below
                 out_model_file=None,
                 out_scaler_file=None
                 ):
        SKLearnModel.__init__(self, 'GPR', 'gpr',
                              scaling=scaling,
                              out=out,
                              out_model_file=out_model_file,
                              out_scaler_file=out_scaler_file)

    @staticmethod
    def load(model, scaler=None):
        return SKLearnModel.load(GaussianProcessModel(), model, scaler=scaler)

    @staticmethod
    def parse_cli_args(args):
        parser = SKLearnModel.create_cli_parser(
            'Train Gaussian process regressor')

        return {key: value
                for key, value in vars(parser.parse_args(args)).items()
                if value is not None}

    def train(self, X_train, y_train):
        self.sklearn_model = GaussianProcessRegressor()
        super(GaussianProcessModel, self).train(X_train, y_train)
