from sklearn.gaussian_process import GaussianProcessRegressor

from .sklearn_model import SKLearnModel


class GaussianProcessFactory:
    def __init__(self):
        self.extension = GaussianProcessModel().extension

    def get_suffix(self):
        return '.%s.pkl' % self.extension

    def init_model(self, cli_args=None, arg_dict=None):
        if cli_args is None and arg_dict is None:
            raise ValueError('CLI or dictionary arguments must be provided.')
        return GaussianProcessModel(**GaussianProcessModel.parse_cli_args(cli_args)) if arg_dict is None \
            else GaussianProcessModel(**arg_dict)

    def load_model(self, fname):
        return GaussianProcessModel.load(
            '%s.%s.pkl' % (fname, self.extension),
            scaler='%s.scaler.pkl' % fname)


class GaussianProcessModel(SKLearnModel):
    '''A Gaussian process regressor, implemented by SciKit.'''

    def __init__(self,
                 alpha=1e-10,
                 optimizer="fmin_l_bfgs_b",
                 n_restarts_optimizer=0,
                 random_state=0,
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

        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state

    @staticmethod
    def load(model, scaler=None):
        return SKLearnModel.load(GaussianProcessModel(), model, scaler=scaler)

    @staticmethod
    def parse_cli_args(args):
        parser = SKLearnModel.create_cli_parser(
            'Train Gaussian process regressor')

        parser.add_argument('--alpha', type=float)
        parser.add_argument('--optimizer', type=str)
        parser.add_argument('--n-restarts-optimizer', type=int)
        parser.add_argument('--random-state', type=int,
                            help='seed for PRNG used in training')

        return {key: value
                for key, value in vars(parser.parse_args(args)).items()
                if value is not None}

    def train(self, X_train, y_train):
        self.sklearn_model = GaussianProcessRegressor(
            alpha=self.alpha,
            optimizer=self.optimizer,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state
        )
        super(GaussianProcessModel, self).train(X_train, y_train)
