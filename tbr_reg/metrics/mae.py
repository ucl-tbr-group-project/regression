from sklearn.metrics import mean_absolute_error

from .basic_metric import RegressionMetric


class MAEFactory:
    def __init__(self):
        self.id = MAE().id

    def init_metric(self):
        return MAE()


class MAE(RegressionMetric):
    '''Mean absolute error, implemented by SciKit.'''

    def __init__(self):
        RegressionMetric.__init__(self, 'MAE', 'mae')

    def evaluate(self, X, y_test, y_pred):
        return mean_absolute_error(y_test, y_pred)

    def rank(self, values):
        return values
