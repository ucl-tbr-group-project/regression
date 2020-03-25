import joblib
from smt.surrogate_models import RBF

from .smt_model import SMTModel


class RBFFactory:
    def __init__(self):
        self.extension = RBFModel().extension

    def get_suffix(self):
        return '.%s.pkl' % self.extension

    def init_model(self, cli_args=None, arg_dict=None):
        if cli_args is None and arg_dict is None:
            raise ValueError('CLI or dictionary arguments must be provided.')
        return RBFModel(**RBFModel.parse_cli_args(cli_args)) if arg_dict is None \
            else RBFModel(**arg_dict)

    def load_model(self, fname):
        return RBFModel.load(
            '%s.%s.pkl' % (fname, self.extension),
            scaler='%s.scaler.pkl' % fname)


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
        self.smt_model = RBF(d0=self.d0,
                             poly_degree=self.poly_degree,
                             reg=self.reg)

        super(RBFModel, self).train(X_train, y_train)
