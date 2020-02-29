from .. import UniformSamplingStrategy, Domain, Samplerun, encode_data_frame, x_y_split, fit_multiple, predict_multiple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor


def test_split():
    csv_path = 'output/10uniform.csv'
    df = encode_data_frame(pd.read_csv(csv_path), Domain())
    X, y = x_y_split(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    initial_models = {
        'tbr': GaussianProcessRegressor(),
        'tbr_error': GaussianProcessRegressor()
    }

    fitted_models = fit_multiple(X_train, y_train, initial_models)
    y_pred = predict_multiple(X_test, fitted_models)


test_split()
