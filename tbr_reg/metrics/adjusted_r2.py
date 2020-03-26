from sklearn.metrics import r2_score

from .basic_metric import RegressionMetric


class AdjustedR2Factory:
    def __init__(self):
        self.id = AdjustedR2Score().id

    def init_metric(self):
        return AdjustedR2Score()


class AdjustedR2Score(RegressionMetric):
    '''Adjusted R2 score, implemented as an extension of the standard SciKit implementation.'''

    def __init__(self):
        RegressionMetric.__init__(self, 'R2(adj)', 'adjusted_r2')

    def evaluate(self, X, y_test, y_pred):
        R2 = r2_score(y_test, y_pred)
        n = X.shape[0]
        p = X.shape[1]
        return 1 - (1-R2) * (n-1) / (n-p-1)

    def rank(self, values):
        return (values - 1.0) ** 2
