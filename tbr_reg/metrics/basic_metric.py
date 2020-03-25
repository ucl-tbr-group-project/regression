class RegressionMetric:
    '''Base class for all regression metrics. Meant to be inherited and overridden.'''

    def __init__(self, name, id):
        '''Initialize metric with name and ID.'''
        self.name = name
        self.id = id

    def evaluate(self, y_test, y_pred):
        '''Evaluate the metric for multiple predictions, given their corresponding true values.'''
        pass

    def rank(self, values):
        '''Transform metric values to sortable keys wherein the lowest key value corresponds to the best-performing model.'''
        pass
