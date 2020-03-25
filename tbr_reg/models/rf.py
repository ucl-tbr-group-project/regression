from sklearn.ensemble import RandomForestRegressor

from .sklearn_model import SKLearnModel


class RandomForestFactory:
    def __init__(self):
        self.extension = RandomForestModel().extension

    def get_suffix(self):
        return '.%s.pkl' % self.extension

    def init_model(self, args):
        return RandomForestModel(**RandomForestModel.parse_cli_args(args))

    def load_model(self, fname):
        return RandomForestModel.load(
            '%s.%s.pkl' % (fname, self.extension),
            scaler='%s.scaler.pkl' % fname)


class RandomForestModel(SKLearnModel):
    '''A random forest ensemble regressor, implemented by SciKit.'''

    def __init__(self,
                 scaling='standard',  # standard|minmax|none
                 out=None,  # overrides all below
                 out_model_file=None,
                 out_scaler_file=None
                 ):
        SKLearnModel.__init__(self, 'RF', 'rf',
                              scaling=scaling,
                              out=out,
                              out_model_file=out_model_file,
                              out_scaler_file=out_scaler_file)

    @staticmethod
    def load(model, scaler=None):
        return SKLearnModel.load(RandomForestModel(), model, scaler=scaler)

    @staticmethod
    def parse_cli_args(args):
        parser = SKLearnModel.create_cli_parser('Train random forest')

        return {key: value
                for key, value in vars(parser.parse_args(args)).items()
                if value is not None}

    def train(self, X_train, y_train):
        self.sklearn_model = RandomForestRegressor(verbose=True)
        super(RandomForestModel, self).train(X_train, y_train)
