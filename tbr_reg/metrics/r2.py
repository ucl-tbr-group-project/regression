from sklearn.metrics import r2_score

from .basic_metric import RegressionMetric


class R2Factory:
    def __init__(self):
        self.id = R2Score().id

    def init_metric(self):
        return R2Score()


class R2Score(RegressionMetric):
    '''R2 score, implemented by SciKit.'''

    def __init__(self):
        RegressionMetric.__init__(self, 'R2', 'r2')

    def evaluate(self, y_test, y_pred):
        return r2_score(y_test, y_pred)

    def rank(self, values):
        return (values - 1.0) ** 2
