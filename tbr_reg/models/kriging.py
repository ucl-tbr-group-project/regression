import joblib
from smt.surrogate_models import KRG

from models.smt_model import SMTModel


class KrigingModel(SMTModel):
    '''Kriging model, implemented by SMT.'''

    def __init__(self,
                 scaling='standard',  # standard|minmax|none
                 out=None,  # overrides all below
                 out_model_file=None,
                 out_scaler_file=None
                 ):
        SMTModel.__init__(self, 'Kriging', 'krg',
                          scaling=scaling,
                          out=out,
                          out_model_file=out_model_file,
                          out_scaler_file=out_scaler_file)

    @staticmethod
    def load(model, scaler=None):
        return SMTModel.load(KrigingModel(), model, scaler=scaler)

    @staticmethod
    def parse_cli_args(args):
        parser = SMTModel.create_cli_parser('Train Kriging model')

        return {key: value
                for key, value in vars(parser.parse_args(args)).items()
                if value is not None}

    def train(self, X_train, y_train):
        self.smt_model = KRG()
        super(KrigingModel, self).train(X_train, y_train)
