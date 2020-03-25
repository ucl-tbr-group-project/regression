from sklearn.ensemble import ExtraTreesRegressor

from .sklearn_model import SKLearnModel


class ExtremelyRandomizedTreesFactory:
    def __init__(self):
        self.extension = ExtremelyRandomizedTreesModel().extension

    def get_suffix(self):
        return '.%s.pkl' % self.extension

    def init_model(self, args):
        return ExtremelyRandomizedTreesModel(**ExtremelyRandomizedTreesModel.parse_cli_args(args))

    def load_model(self, fname):
        return ExtremelyRandomizedTreesModel.load(
            '%s.%s.pkl' % (fname, self.extension),
            scaler='%s.scaler.pkl' % fname)


class ExtremelyRandomizedTreesModel(SKLearnModel):
    '''An extremely randomized trees regressor, implemented by SciKit.'''

    def __init__(self,
                 random_state=0,
                 scaling='standard',  # standard|minmax|none
                 out=None,  # overrides all below
                 out_model_file=None,
                 out_scaler_file=None
                 ):
        SKLearnModel.__init__(self, 'ERT', 'ert',
                              scaling=scaling,
                              out=out,
                              out_model_file=out_model_file,
                              out_scaler_file=out_scaler_file)

        self.random_state = random_state

    @staticmethod
    def load(model, scaler=None):
        return SKLearnModel.load(ExtremelyRandomizedTreesModel(), model, scaler=scaler)

    @staticmethod
    def parse_cli_args(args):
        parser = SKLearnModel.create_cli_parser(
            'Train extremely randomized trees')

        parser.add_argument('--random-state', type=int,
                            help='seed for PRNG used in training')

        return {key: value
                for key, value in vars(parser.parse_args(args)).items()
                if value is not None}

    def train(self, X_train, y_train):
        self.sklearn_model = ExtraTreesRegressor(
            verbose=True,
            random_state=self.random_state
        )
        super(ExtremelyRandomizedTreesModel, self).train(X_train, y_train)
