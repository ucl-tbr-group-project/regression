from smt.surrogate_models import IDW

from .smt_model import SMTModel


class IDWFactory:
    def __init__(self):
        self.extension = IDWModel().extension

    def get_suffix(self):
        return '.%s.pkl' % self.extension

    def init_model(self, cli_args=None, arg_dict=None):
        if cli_args is None and arg_dict is None:
            raise ValueError('CLI or dictionary arguments must be provided.')
        return IDWModel(**IDWModel.parse_cli_args(cli_args)) if arg_dict is None \
            else IDWModel(**arg_dict)

    def load_model(self, fname):
        return IDWModel.load(
            '%s.%s.pkl' % (fname, self.extension),
            scaler='%s.scaler.pkl' % fname)


class IDWModel(SMTModel):
    '''Inverse distance weighting model, implemented by SMT.'''

    def __init__(self,
                 p=2.5,
                 scaling='standard',  # standard|minmax|none
                 out=None,  # overrides all below
                 out_model_file=None,
                 out_scaler_file=None):
        SMTModel.__init__(self, 'Inverse distance weighting', 'idw',
                          scaling=scaling,
                          out=out,
                          out_model_file=out_model_file,
                          out_scaler_file=out_scaler_file)

        self.p = p

    @staticmethod
    def load(model, scaler=None):
        return SMTModel.load(IDWModel(), model, scaler=scaler)

    @staticmethod
    def parse_cli_args(args):
        parser = SMTModel.create_cli_parser('Train IDW model')

        parser.add_argument('--p', type=float,
                            help='order of distance norm')

        return {key: value
                for key, value in vars(parser.parse_args(args)).items()
                if value is not None}

    def train(self, X_train, y_train):
        self.smt_model = IDW(p=self.p)

        super(IDWModel, self).train(X_train, y_train)
