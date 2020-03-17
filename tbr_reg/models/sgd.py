from sklearn.linear_model import SGDRegressor

from .sklearn_poly_model import SKLearnPolyModel


class SGDModel(SKLearnPolyModel):
    '''Stochastic gradient descent with polynomial features, implemented by SciKit.'''

    def __init__(self,
                 degree=3,
                 scaling='standard',  # standard|minmax|none
                 out=None,  # overrides all below
                 out_model_file=None,
                 out_scaler_file=None
                 ):
        SKLearnPolyModel.__init__(self, 'Stochastic gradient descent', 'sgd',
                                  scaling=scaling,
                                  out=out,
                                  out_model_file=out_model_file,
                                  out_scaler_file=out_scaler_file)

    @staticmethod
    def load(model, scaler=None):
        return SKLearnPolyModel.load(SGDModel(), model, scaler=scaler)

    @staticmethod
    def parse_cli_args(args):
        parser = SKLearnPolyModel.create_cli_parser(
            'Train stochastic gradient descent with polynomial features')

        return {key: value
                for key, value in vars(parser.parse_args(args)).items()
                if value is not None}

    def train(self, X_train, y_train):
        self.sklearn_linear_model = SGDRegressor()
        super(SGDModel, self).train(X_train, y_train)
