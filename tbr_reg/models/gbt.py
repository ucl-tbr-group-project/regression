from sklearn.ensemble import GradientBoostingRegressor

from .sklearn_model import SKLearnModel


class GradientBoostingFactory:
    def __init__(self):
        self.extension = GradientBoostingModel().extension

    def get_suffix(self):
        return '.%s.pkl' % self.extension

    def init_model(self, cli_args=None, arg_dict=None):
        if cli_args is None and arg_dict is None:
            raise ValueError('CLI or dictionary arguments must be provided.')
        return GradientBoostingModel(**GradientBoostingModel.parse_cli_args(cli_args)) if arg_dict is None \
            else GradientBoostingModel(**arg_dict)

    def load_model(self, fname):
        return GradientBoostingModel.load(
            '%s.%s.pkl' % (fname, self.extension),
            scaler='%s.scaler.pkl' % fname)


class GradientBoostingModel(SKLearnModel):
    '''A gradient boosted tree regressor, implemented by SciKit.'''

    def __init__(self,
                 loss='ls',  # ls|lad|huber|quantile
                 learning_rate=0.1,
                 n_estimators=100,
                 subsample=1.0,
                 criterion='friedman_mse',  # friedman_mse|mse|mae
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_depth=3,
                 min_impurity_decrease=0.,
                 max_features=None,  # auto|sqrt|log2 or int or float
                 alpha=0.9,
                 max_leaf_nodes=None,
                 warm_start=False,
                 validation_fraction=0.1,
                 ccp_alpha=0.0,
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

        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.alpha = alpha
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.validation_fraction = validation_fraction
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state

    @staticmethod
    def load(model, scaler=None):
        return SKLearnModel.load(GradientBoostingModel(), model, scaler=scaler)

    @staticmethod
    def parse_cli_args(args):
        parser = SKLearnModel.create_cli_parser('Train gradient boosted trees')

        parser.add_argument('--loss', type=str)
        parser.add_argument('--learning-rate', type=float)
        parser.add_argument('--n-estimators', type=int)
        parser.add_argument('--subsample', type=float)
        parser.add_argument('--criterion', type=str)
        parser.add_argument('--min-samples-split')
        parser.add_argument('--min-samples-leaf')
        parser.add_argument('--min-weight-fraction-leaf', type=float)
        parser.add_argument('--max-depth', type=int)
        parser.add_argument('--min-impurity-decrease', type=float)
        parser.add_argument('--max-features')
        parser.add_argument('--alpha', type=float)
        parser.add_argument('--max-leaf-nodes', type=int)
        parser.add_argument('--warm-start', type=bool)
        parser.add_argument('--validation-fraction', type=float)
        parser.add_argument('--ccp-alpha', type=float)
        parser.add_argument('--random-state', type=int,
                            help='seed for PRNG used in training')

        return {key: value
                for key, value in vars(parser.parse_args(args)).items()
                if value is not None}

    def train(self, X_train, y_train):
        self.sklearn_model = GradientBoostingRegressor(
            loss=self.loss,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            criterion=self.criterion,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_depth=self.max_depth,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features,
            alpha=self.alpha,
            max_leaf_nodes=self.max_leaf_nodes,
            warm_start=self.warm_start,
            validation_fraction=self.validation_fraction,
            ccp_alpha=self.ccp_alpha,
            random_state=self.random_state,
            verbose=True
        )
        super(GradientBoostingModel, self).train(X_train, y_train)
