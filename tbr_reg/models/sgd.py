from sklearn.linear_model import SGDRegressor

from .sklearn_poly_model import SKLearnPolyModel


class SGDFactory:
    def __init__(self):
        self.extension = SGDModel().extension

    def get_suffix(self):
        return '.%s.pkl' % self.extension

    def init_model(self, cli_args=None, arg_dict=None):
        if cli_args is None and arg_dict is None:
            raise ValueError('CLI or dictionary arguments must be provided.')
        return SGDModel(**SGDModel.parse_cli_args(cli_args)) if arg_dict is None \
            else SGDModel(**arg_dict)

    def load_model(self, fname):
        return SGDModel.load(
            '%s.%s.pkl' % (fname, self.extension),
            scaler='%s.scaler.pkl' % fname)


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
                                  degree=degree,
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
