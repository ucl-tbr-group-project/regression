from smt.surrogate_models import RMTB, RMTC

from .smt_model import SMTModel


class RMTSFactory:
    def __init__(self):
        self.extension = RMTSModel().extension

    def get_suffix(self):
        return '.%s.pkl' % self.extension

    def init_model(self, cli_args=None, arg_dict=None):
        if cli_args is None and arg_dict is None:
            raise ValueError('CLI or dictionary arguments must be provided.')
        return RMTSModel(**RMTSModel.parse_cli_args(cli_args)) if arg_dict is None \
            else RMTSModel(**arg_dict)

    def load_model(self, fname):
        return RMTSModel.load(
            '%s.%s.pkl' % (fname, self.extension),
            scaler='%s.scaler.pkl' % fname)


class RMTSModel(SMTModel):
    '''Regularized minimal-energy tensor-product splines model, implemented by SMT.'''

    def __init__(self,
                 flavour='bspline',  # bspline|cubic
                 xlimits=None,
                 smoothness=1.0,
                 approx_order=4,
                 line_search='backtracking',  # backtracking|bracketed|quadratic|cubic|null
                 order=3,
                 num_ctrl_pts=15,
                 num_elements=4,
                 scaling='standard',  # standard|minmax|none
                 out=None,  # overrides all below
                 out_model_file=None,
                 out_scaler_file=None):
        SMTModel.__init__(self, 'Regularized minimal-energy tensor-product splines', 'rmts',
                          scaling=scaling,
                          out=out,
                          out_model_file=out_model_file,
                          out_scaler_file=out_scaler_file)

        self.flavour = flavour

        self.xlimits = xlimits
        self.smoothness = smoothness
        self.approx_order = approx_order
        self.line_search = line_search
        self.order = order
        self.num_ctrl_pts = num_ctrl_pts
        self.num_elements = num_elements

    @staticmethod
    def load(model, scaler=None):
        return SMTModel.load(RMTSModel(), model, scaler=scaler)

    @staticmethod
    def parse_cli_args(args):
        parser = SMTModel.create_cli_parser('Train RMTS model')

        parser.add_argument('--smoothness', type=float,
                            help='Smoothness parameter in each dimension - length nx. None implies uniform')
        parser.add_argument('--approx-order', type=int,
                            help='Exponent in the approximation term')
        parser.add_argument('--line-search', type=str,
                            help=' 	Line search algorithm')
        parser.add_argument('--order', type=int,
                            help='B-spline order in each dimension - length [nx]')
        parser.add_argument('--num-ctrl-pts', type=int,
                            help='# B-spline control points in each dimension - length [nx]')
        parser.add_argument('--num-elements', type=int,
                            help='# elements in each dimension - ndarray [nx]')

        return {key: value
                for key, value in vars(parser.parse_args(args)).items()
                if value is not None}

    def train(self, X_train, y_train):
        if self.flavour == 'bspline':
            self.smt_model = RMTB(
                xlimits=self.xlimits,
                smoothness=self.smoothness,
                approx_order=self.approx_order,
                line_search=self.line_search,
                order=self.order,
                num_ctrl_pts=self.num_ctrl_pts)
        if self.flavour == 'cubic':
            self.smt_model = RMTC(
                xlimits=self.xlimits,
                smoothness=self.smoothness,
                approx_order=self.approx_order,
                line_search=self.line_search,
                order=self.order,
                num_elements=self.num_elements)

        super(RMTSModel, self).train(X_train, y_train)
