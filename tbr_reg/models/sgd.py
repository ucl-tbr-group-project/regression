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
                 loss='squared_loss',  # squared_loss|huber|epsilon_insensitive|squared_epsilon_insensitive
                 penalty='l2',  # l2|l1|elasticnet
                 alpha=0.0001,
                 l1_ratio=0.15,
                 fit_intercept=True,
                 max_iter=1000,
                 tol=0.001,
                 shuffle=True,
                 epsilon=0.1,
                 random_state=0,
                 learning_rate='invscaling',  # constant|optimal|invscaling|adaptive
                 eta0=0.01,
                 power_t=0.25,
                 early_stopping=False,
                 validation_fraction=0.1,
                 n_iter_no_change=5,
                 warm_start=False,
                 average=False,
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

        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.epsilon = epsilon
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.warm_start = warm_start
        self.average = average

    @staticmethod
    def load(model, scaler=None):
        return SKLearnPolyModel.load(SGDModel(), model, scaler=scaler)

    @staticmethod
    def parse_cli_args(args):
        parser = SKLearnPolyModel.create_cli_parser(
            'Train stochastic gradient descent with polynomial features')

        parser.add_argument('--loss', type=str)
        parser.add_argument('--penalty', type=str)
        parser.add_argument('--alpha', type=float)
        parser.add_argument('--l1-ratio', type=float)
        parser.add_argument('--fit-intercept', type=bool)
        parser.add_argument('--max-iter', type=int)
        parser.add_argument('--tol', type=float)
        parser.add_argument('--shuffle', type=bool)
        parser.add_argument('--epsilon', type=float)
        parser.add_argument('--random-state', type=int)
        parser.add_argument('--learning-rate', type=str)
        parser.add_argument('--eta0', type=float)
        parser.add_argument('--power-t', type=float)
        parser.add_argument('--early-stopping', type=bool)
        parser.add_argument('--validation-fraction', type=float)
        parser.add_argument('--n-iter-no-change', type=int)
        parser.add_argument('--warm-start', type=bool)
        parser.add_argument('--average', type=bool)

        return {key: value
                for key, value in vars(parser.parse_args(args)).items()
                if value is not None}

    def train(self, X_train, y_train):
        self.sklearn_linear_model = SGDRegressor(
            loss=self.loss,
            penalty=self.penalty,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            shuffle=self.shuffle,
            epsilon=self.epsilon,
            random_state=self.random_state,
            learning_rate=self.learning_rate,
            eta0=self.eta0,
            power_t=self.power_t,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            warm_start=self.warm_start,
            average=self.average,
            verbose=True
        )
        super(SGDModel, self).train(X_train, y_train)
