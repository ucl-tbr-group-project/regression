from sklearn.ensemble import GradientBoostingRegressor

from .sklearn_model import SKLearnModel


class GradientBoostingFactory:
    def __init__(self):
        self.extension = GradientBoostingModel().extension

    def get_suffix(self):
        return '.%s.pkl' % self.extension

    def init_model(self, args):
        return GradientBoostingModel(**GradientBoostingModel.parse_cli_args(args))

    def load_model(self, fname):
        return GradientBoostingModel.load(
            '%s.%s.pkl' % (fname, self.extension),
            scaler='%s.scaler.pkl' % fname)


class GradientBoostingModel(SKLearnModel):
    '''A gradient boosted tree regressor, implemented by SciKit.'''

    def __init__(self,
                 random_state=0,
                 scaling='standard',  # standard|minmax|none
                 out=None,  # overrides all below
                 out_model_file=None,
                 out_scaler_file=None
                 ):
        SKLearnModel.__init__(self, 'GBT', 'gbt',
                              scaling=scaling,
                              out=out,
                              out_model_file=out_model_file,
                              out_scaler_file=out_scaler_file)

        self.random_state = random_state

    @staticmethod
    def load(model, scaler=None):
        return SKLearnModel.load(GradientBoostingModel(), model, scaler=scaler)

    @staticmethod
    def parse_cli_args(args):
        parser = SKLearnModel.create_cli_parser('Train gradient boosted trees')

        parser.add_argument('--random-state', type=int,
                            help='seed for PRNG used in training')

        return {key: value
                for key, value in vars(parser.parse_args(args)).items()
                if value is not None}

    def train(self, X_train, y_train):
        self.sklearn_model = GradientBoostingRegressor(
            random_state=self.random_state
        )
        super(GradientBoostingModel, self).train(X_train, y_train)
