import joblib
from smt.surrogate_models import RBF

from models.smt_model import SMTModel


class RBFModel(SMTModel):
    '''Radial basis functions model, implemented by SMT.'''

    def __init__(self,
                 d0=1.0,
                 poly_degree=-1,  # -1|0|1
                 reg=1e-10,
                 scaling='standard',  # standard|minmax|none
                 out=None,  # overrides all below
                 out_model_file=None,
                 out_scaler_file=None):
        SMTModel.__init__(self, 'Radial basis functions', 'rbf',
                          scaling=scaling,
                          out=out,
                          out_model_file=out_model_file,
                          out_scaler_file=out_scaler_file)

        self.d0 = d0
        self.poly_degree = poly_degree
        self.reg = reg

    @staticmethod
    def load(model, scaler=None):
        return SMTModel.load(RBFModel(), model, scaler=scaler)

    @staticmethod
    def parse_cli_args(args):
        parser = SMTModel.create_cli_parser('Train RBF model')

        parser.add_argument('--d0', type=float,
                            help='basis function scaling parameter in exp(-d^2 / d0^2)')
        parser.add_argument('--poly-degree', type=int,
                            help='-1 means no global polynomial, 0 means constant, 1 means linear trend')
        parser.add_argument('--reg', type=float,
                            help='Regularization coeff.')

        return {key: value
                for key, value in vars(parser.parse_args(args)).items()
                if value is not None}

    def train(self, X_train, y_train):
        self.smt_model = RBF(d0=d0,
                             poly_degree=poly_degree,
                             reg=reg)

        super(RBFModel, self).train(X_train, y_train)
