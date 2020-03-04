import joblib
from smt.surrogate_models import KRG, KPLS, KPLSK, GEKPLS

from models.smt_model import SMTModel


class KrigingModel(SMTModel):
    '''Kriging model, implemented by SMT.'''

    def __init__(self,
                 flavour='plain',  # plain|pls|plsk|gepls
                 poly='constant',  # constant|linear|quadratic
                 corr='squar_exp',  # abs_exp|squar_exp
                 theta0=0.01,
                 n_comp=1,
                 xlimits=None,
                 delta_x=0.0001,
                 extra_points=0,
                 scaling='standard',  # standard|minmax|none
                 out=None,  # overrides all below
                 out_model_file=None,
                 out_scaler_file=None):
        SMTModel.__init__(self, 'Kriging', 'krg',
                          scaling=scaling,
                          out=out,
                          out_model_file=out_model_file,
                          out_scaler_file=out_scaler_file)

        self.flavour = flavour
        self.poly = poly
        self.corr = corr
        self.theta0 = [theta0]
        self.n_comp = n_comp
        self.xlimits = xlimits
        self.delta_x = delta_x
        self.extra_points = extra_points

    @staticmethod
    def load(model, scaler=None):
        return SMTModel.load(KrigingModel(), model, scaler=scaler)

    @staticmethod
    def parse_cli_args(args):
        parser = SMTModel.create_cli_parser('Train Kriging model')

        parser.add_argument('--flavour', type=str,
                            help='type of Kriging model used')
        parser.add_argument('--poly', type=str,
                            help='regression function type')
        parser.add_argument('--corr', type=str,
                            help='correlation function type')
        parser.add_argument('--theta0', type=float,
                            help='initial hyperparameter')
        parser.add_argument('--n-comp', type=int,
                            help='number of principal components')
        parser.add_argument('--delta-x', type=float,
                            help='step used in the FOTA')
        parser.add_argument('--extra-points', type=int,
                            help='number of extra points per training point')

        return {key: value
                for key, value in vars(parser.parse_args(args)).items()
                if value is not None}

    def train(self, X_train, y_train):
        if self.flavour == 'plain':
            self.smt_model = KRG(
                poly=self.poly,
                corr=self.corr,
                theta0=self.theta0)
        elif self.flavour == 'pls':
            self.smt_model = KPLS(
                poly=self.poly,
                corr=self.corr,
                theta0=self.theta0,
                n_comp=self.n_comp)
        elif self.flavour == 'plsk':
            self.smt_model = KPLSK(
                poly=self.poly,
                corr=self.corr,
                theta0=self.theta0,
                n_comp=self.n_comp)
        elif self.flavour == 'gepls':
            self.smt_model = GEKPLS(
                poly=self.poly,
                corr=self.corr,
                theta0=self.theta0,
                n_comp=self.n_comp,
                xlimits=self.xlimits,
                delta_x=self.delta_x,
                extra_points=self.extra_points)

        super(KrigingModel, self).train(X_train, y_train)
