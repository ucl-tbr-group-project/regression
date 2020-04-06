from sklearn.linear_model import Ridge

from .sklearn_poly_model import SKLearnPolyModel


class RidgeFactory:
    def __init__(self):
        self.extension = RidgeModel().extension

    def get_suffix(self):
        return '.%s.pkl' % self.extension

    def init_model(self, cli_args=None, arg_dict=None):
        if cli_args is None and arg_dict is None:
            raise ValueError('CLI or dictionary arguments must be provided.')
        return RidgeModel(**RidgeModel.parse_cli_args(cli_args)) if arg_dict is None \
            else RidgeModel(**arg_dict)

    def load_model(self, fname):
        return RidgeModel.load(
            '%s.%s.pkl' % (fname, self.extension),
            scaler='%s.scaler.pkl' % fname)


class RidgeModel(SKLearnPolyModel):
    '''Ridge regression with polynomial features, implemented by SciKit.'''

    def __init__(self,
                 alpha=1.0,
                 fit_intercept=True,
                 normalize=False,
                 copy_X=True,
                 max_iter=None,
                 tol=0.001,
                 solver='auto',  # auto|svd|cholesky|lsqr|sparse_cg|sag|saga
                 random_state=0,
                 degree=3,
                 scaling='standard',  # standard|minmax|none
                 out=None,  # overrides all below
                 out_model_file=None,
                 out_scaler_file=None
                 ):
        SKLearnPolyModel.__init__(self, 'Ridge regression', 'ridge',
                                  degree=degree,
                                  scaling=scaling,
                                  out=out,
                                  out_model_file=out_model_file,
                                  out_scaler_file=out_scaler_file)

        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.random_state = random_state

    @staticmethod
    def load(model, scaler=None):
        return SKLearnPolyModel.load(RidgeModel(), model, scaler=scaler)

    @staticmethod
    def parse_cli_args(args):
        parser = SKLearnPolyModel.create_cli_parser(
            'Train ridge regression model with polynomial features')

        parser.add_argument('--alpha', type=float)
        parser.add_argument('--fit-intercept', type=bool)
        parser.add_argument('--normalize', type=bool)
        parser.add_argument('--max-iter', type=int)
        parser.add_argument('--tol', type=float)
        parser.add_argument('--solver', type=str)
        parser.add_argument('--random-state', type=int)

        return {key: value
                for key, value in vars(parser.parse_args(args)).items()
                if value is not None}

    def train(self, X_train, y_train):
        self.sklearn_linear_model = Ridge(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            normalize=self.normalize,
            copy_X=self.copy_X,
            max_iter=self.max_iter,
            tol=self.tol,
            solver=self.solver,
            random_state=self.random_state
        )
        super(RidgeModel, self).train(X_train, y_train)
