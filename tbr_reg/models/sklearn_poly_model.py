from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from models.sklearn_model import SKLearnModel


class SKLearnPolyModel(SKLearnModel):
    '''Base class for all dimension-lifted SciKit models.'''

    def __init__(self,
                 name, extension,
                 degree=3,
                 scaling='standard',  # standard|minmax|none
                 out=None,  # overrides all below
                 out_model_file=None,
                 out_scaler_file=None):
        SKLearnModel.__init__(self, name, extension,
                              scaling=scaling,
                              out=out,
                              out_model_file=out_model_file,
                              out_scaler_file=out_scaler_file)

        self.degree = degree
        self.sklearn_linear_model = None

    @staticmethod
    def create_cli_parser(description):
        parser = SKLearnModel.create_cli_parser(description)
        parser.add_argument('--degree', type=int,
                            help='degree of polynomial features')
        return parser

    def train(self, X_train, y_train):
        self.sklearn_model = Pipeline([('poly', PolynomialFeatures(degree=self.degree)),
                                       ('linear', self.sklearn_linear_model)])
        super(SKLearnPolyModel, self).train(X_train, y_train)
