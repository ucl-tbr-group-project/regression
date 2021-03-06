from sklearn.ensemble import AdaBoostRegressor

from .sklearn_model import SKLearnModel


class AdaBoostFactory:
    def __init__(self):
        self.extension = AdaBoostModel().extension

    def get_suffix(self):
        return '.%s.pkl' % self.extension

    def init_model(self, cli_args=None, arg_dict=None):
        if cli_args is None and arg_dict is None:
            raise ValueError('CLI or dictionary arguments must be provided.')
        return AdaBoostModel(**AdaBoostModel.parse_cli_args(cli_args)) if arg_dict is None \
            else AdaBoostModel(**arg_dict)

    def load_model(self, fname):
        return AdaBoostModel.load(
            '%s.%s.pkl' % (fname, self.extension),
            scaler='%s.scaler.pkl' % fname)


class AdaBoostModel(SKLearnModel):
    '''A AdaBoost regressor, implemented by SciKit.'''

    def __init__(self,
                 n_estimators=50,
                 learning_rate=1.0,
                 loss='linear',  # linear|square|exponential
                 random_state=0,
                 scaling='standard',  # standard|minmax|none
                 out=None,  # overrides all below
                 out_model_file=None,
                 out_scaler_file=None
                 ):
        SKLearnModel.__init__(self, 'AB', 'ab',
                              scaling=scaling,
                              out=out,
                              out_model_file=out_model_file,
                              out_scaler_file=out_scaler_file)

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state

    @staticmethod
    def load(model, scaler=None):
        return SKLearnModel.load(AdaBoostModel(), model, scaler=scaler)

    @staticmethod
    def parse_cli_args(args):
        parser = SKLearnModel.create_cli_parser('Train AdaBoost')

        parser.add_argument('--n-estimators', type=int)
        parser.add_argument('--learning-rate', type=float)
        parser.add_argument('--loss', type=str)
        parser.add_argument('--random-state', type=int,
                            help='seed for PRNG used in training')

        return {key: value
                for key, value in vars(parser.parse_args(args)).items()
                if value is not None}

    def train(self, X_train, y_train):
        self.sklearn_model = AdaBoostRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            loss=self.loss,
            random_state=self.random_state
        )
        super(AdaBoostModel, self).train(X_train, y_train)
