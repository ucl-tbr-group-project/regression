import pandas as pd


def apply_on_y_columns(X, y, fn):
    return {column: fn(X, y[[column]], column)
            for column in y.columns.tolist()}


def fit_multiple(X, y, models):
    def fitter(X_in, y_in, column):
        model = models[column]
        return model.fit(X_in, y_in)

    return apply_on_y_columns(X, y, fitter)


def predict_multiple(X, models):
    data = {column: model.predict(X).ravel().tolist()
            for column, model in models.items()}
    return pd.DataFrame(data=data)
