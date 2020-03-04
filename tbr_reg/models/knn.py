from sklearn.neighbors import KNeighborsRegressor

from models.sklearn_model import SKLearnModel


class NearestNeighboursModel(SKLearnModel):
    '''A k-nearest neighbours regressor, implemented by SciKit.'''

    def __init__(self,
                 scaling='standard',  # standard|minmax|none
                 out=None,  # overrides all below
                 out_model_file=None,
                 out_scaler_file=None
                 ):
        SKLearnModel.__init__(self, 'kNN', 'knn',
                              scaling=scaling,
                              out=out,
                              out_model_file=out_model_file,
                              out_scaler_file=out_scaler_file)

    @staticmethod
    def load(model, scaler=None):
        return SKLearnModel.load(NearestNeighboursModel(), model, scaler=scaler)

    @staticmethod
    def parse_cli_args(args):
        parser = SKLearnModel.create_cli_parser('Train k-nearest neighbours')

        return {key: value
                for key, value in vars(parser.parse_args(args)).items()
                if value is not None}

    def train(self, X_train, y_train):
        self.sklearn_model = KNeighborsRegressor()
        super(NearestNeighboursModel, self).train(X_train, y_train)
