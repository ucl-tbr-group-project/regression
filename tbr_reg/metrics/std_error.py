import numpy as np

from .basic_metric import RegressionMetric


class StandardErrorFactory:
    def __init__(self):
        self.id = StandardErrorScore().id

    def init_metric(self):
        return StandardErrorScore()


class StandardErrorScore(RegressionMetric):
    '''Standard regression error score, in part implemented by SciKit.'''

    def __init__(self):
        RegressionMetric.__init__(self, 'S', '{\\rm S}', 'std_error')

    def evaluate(self, X, y_test, y_pred):
        # could not find exact definition, did the next best thing
        # according to: https://support.minitab.com/en-us/minitab/18/help-and-how-to/modeling-statistics/regression/how-to/fit-regression-model/interpret-the-results/all-statistics-and-graphs/model-summary-table/
        # "S represents the standard deviation of the distance between
        #  the data values and the fitted values. S is measured in the
        #  units of the response."
        return np.std(np.abs(y_test - y_pred))

    def rank(self, values):
        return values
