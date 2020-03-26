from sklearn.neighbors import KNeighborsRegressor

from .sklearn_model import SKLearnModel


class NearestNeighboursFactory:
    def __init__(self):
        self.extension = NearestNeighboursModel().extension

    def get_suffix(self):
        return '.%s.pkl' % self.extension

    def init_model(self, cli_args=None, arg_dict=None):
        if cli_args is None and arg_dict is None:
            raise ValueError('CLI or dictionary arguments must be provided.')
        return NearestNeighboursModel(**NearestNeighboursModel.parse_cli_args(cli_args)) if arg_dict is None \
            else NearestNeighboursModel(**arg_dict)

    def load_model(self, fname):
        return NearestNeighboursModel.load(
            '%s.%s.pkl' % (fname, self.extension),
            scaler='%s.scaler.pkl' % fname)


class NearestNeighboursModel(SKLearnModel):
    '''A k-nearest neighbours regressor, implemented by SciKit.'''

    def __init__(self,
                 n_neighbors=5,
                 weights='uniform',  # uniform|distance
                 algorithm='auto',  # auto|ball_tree|kd_tree|brute
                 leaf_size=30,
                 p=2,
                 metric='minkowski',
                 metric_params=None,
                 scaling='standard',  # standard|minmax|none
                 out=None,  # overrides all below
                 out_model_file=None,
                 out_scaler_file=None):
        SKLearnModel.__init__(self, 'k-NN', 'knn',
                              scaling=scaling,
                              out=out,
                              out_model_file=out_model_file,
                              out_scaler_file=out_scaler_file)

        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params

    @staticmethod
    def load(model, scaler=None):
        return SKLearnModel.load(NearestNeighboursModel(), model, scaler=scaler)

    @staticmethod
    def parse_cli_args(args):
        parser = SKLearnModel.create_cli_parser('Train k-nearest neighbours')

        parser.add_argument('--n_neighbors', type=int)
        parser.add_argument('--weights', type=str)
        parser.add_argument('--algorithm', type=str)
        parser.add_argument('--leaf_size', type=int)
        parser.add_argument('--p', type=int)
        parser.add_argument('--metric', type=str)

        return {key: value
                for key, value in vars(parser.parse_args(args)).items()
                if value is not None}

    def train(self, X_train, y_train):
        self.sklearn_model = KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            p=self.p,
            metric=self.metric,
            metric_params=self.metric_params
        )
        super(NearestNeighboursModel, self).train(X_train, y_train)
