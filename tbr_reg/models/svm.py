from sklearn.svm import SVR

from .sklearn_model import SKLearnModel


class SupportVectorFactory:
    def __init__(self):
        self.extension = SupportVectorModel().extension

    def get_suffix(self):
        return '.%s.pkl' % self.extension

    def init_model(self, cli_args=None, arg_dict=None):
        if cli_args is None and arg_dict is None:
            raise ValueError('CLI or dictionary arguments must be provided.')
        return SupportVectorModel(**SupportVectorModel.parse_cli_args(cli_args)) if arg_dict is None \
            else SupportVectorModel(**arg_dict)

    def load_model(self, fname):
        return SupportVectorModel.load(
            '%s.%s.pkl' % (fname, self.extension),
            scaler='%s.scaler.pkl' % fname)


class SupportVectorModel(SKLearnModel):
    '''A support vector machine with kernel trick, implemented by SciKit.'''

    def __init__(self,
                 scaling='standard',  # standard|minmax|none
                 out=None,  # overrides all below
                 out_model_file=None,
                 out_scaler_file=None
                 ):
        SKLearnModel.__init__(self, 'SVM', 'svm',
                              scaling=scaling,
                              out=out,
                              out_model_file=out_model_file,
                              out_scaler_file=out_scaler_file)

    @staticmethod
    def load(model, scaler=None):
        return SKLearnModel.load(SupportVectorModel(), model, scaler=scaler)

    @staticmethod
    def parse_cli_args(args):
        parser = SKLearnModel.create_cli_parser('Train support vector machine')

        return {key: value
                for key, value in vars(parser.parse_args(args)).items()
                if value is not None}

    def train(self, X_train, y_train):
        self.sklearn_model = SVR(verbose=True)
        super(SupportVectorModel, self).train(X_train, y_train)
