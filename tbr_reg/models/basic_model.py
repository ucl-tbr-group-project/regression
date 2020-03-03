class RegressionModel:
    '''Base class for all regression models. Meant to be inherited and overridden.'''

    def __init__(self, name):
        '''Initialize model with name.'''
        self.name = name

    def train(self, X_train, y_train):
        '''Train model on the given set of inputs and expected outputs.'''
        pass

    def evaluate(self, X_test, y_test):
        '''Compare model's predictions on the given inputs with the given outputs.'''
        pass

    def predict(self, X):
        '''Let model predict on the given set of inputs.'''
        pass
