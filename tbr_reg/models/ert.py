from sklearn.ensemble import ExtraTreesRegressor

from .sklearn_model import SKLearnModel


class ExtremelyRandomizedTreesFactory:
    def __init__(self):
        self.extension = ExtremelyRandomizedTreesModel().extension

    def get_suffix(self):
        return '.%s.pkl' % self.extension

    def init_model(self, cli_args=None, arg_dict=None):
        if cli_args is None and arg_dict is None:
            raise ValueError('CLI or dictionary arguments must be provided.')
        return ExtremelyRandomizedTreesModel(**ExtremelyRandomizedTreesModel.parse_cli_args(cli_args)) if arg_dict is None \
            else ExtremelyRandomizedTreesModel(**arg_dict)

    def load_model(self, fname):
        return ExtremelyRandomizedTreesModel.load(
            '%s.%s.pkl' % (fname, self.extension),
            scaler='%s.scaler.pkl' % fname)


class ExtremelyRandomizedTreesModel(SKLearnModel):
    '''An extremely randomized trees regressor, implemented by SciKit.'''

    def __init__(self,
                 n_estimators=10,
                 criterion='mse',  # mse|mae,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features='auto',  # auto|sqrt|log2 or None or int or float
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 bootstrap=False,
                 oob_score=False,
                 warm_start=False,
                 ccp_alpha=0.0,
                 max_samples=None,
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

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        self.random_state = random_state

    @staticmethod
    def load(model, scaler=None):
        return SKLearnModel.load(ExtremelyRandomizedTreesModel(), model, scaler=scaler)

    @staticmethod
    def parse_cli_args(args):
        parser = SKLearnModel.create_cli_parser(
            'Train extremely randomized trees')

        parser.add_argument('--n-estimators', type=int)
        parser.add_argument('--criterion', type=str)
        parser.add_argument('--max-depth', type=int)
        parser.add_argument('--min-samples-split')
        parser.add_argument('--min-samples-leaf', type=float)
        parser.add_argument('--min-weight-fraction-leaf')
        parser.add_argument('--max-features')
        parser.add_argument('--max-leaf-nodes', type=int)
        parser.add_argument('--min-impurity-decrease', type=float)
        parser.add_argument('--bootstrap', type=bool)
        parser.add_argument('--oob-score', type=bool)
        parser.add_argument('--warm-start', type=bool)
        parser.add_argument('--ccp-alpha', type=float)
        parser.add_argument('--max-samples')
        parser.add_argument('--random-state', type=int,
                            help='seed for PRNG used in training')

        return {key: value
                for key, value in vars(parser.parse_args(args)).items()
                if value is not None}

    def train(self, X_train, y_train):
        self.sklearn_model = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            warm_start=self.warm_start,
            ccp_alpha=self.ccp_alpha,
            max_samples=self.max_samples,
            random_state=self.random_state,
            verbose=True
        )
        super(ExtremelyRandomizedTreesModel, self).train(X_train, y_train)
