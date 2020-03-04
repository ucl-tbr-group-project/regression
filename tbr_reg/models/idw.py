from smt.surrogate_models import IDW

from models.smt_model import SMTModel


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
