import pandas as pd


def apply_on_y_columns(X, y, fn):
    '''Call fn on individual columns of y and return its results indexed by column names.'''

    return {column: fn(X, y[[column]], column)
            for column in y.columns.tolist()}


def fit_multiple(X, y, models):
    '''Fit multiple models on X and individual columns of y. Return trained models.'''

    def fitter(X_in, y_in, column):
        model = models[column]
        return model.fit(X_in, y_in)

    return apply_on_y_columns(X, y, fitter)


def predict_multiple(X, models):
    '''Predict values from the input using multiple given models. Return a data frame with predictions.'''

    data = {column: model.predict(X).ravel().tolist()
            for column, model in models.items()}
    return pd.DataFrame(data=data)
