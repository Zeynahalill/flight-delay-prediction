import numpy as np
import pandas as pd

from flight_delay import config, training
from flight_delay.preprocessing import Preprocessor


def _sample_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "id": range(n),
        "Airline": rng.choice(["AA", "DL", "UA"], n),
        "Flight": rng.integers(100, 999, n),
        "AirportFrom": rng.choice(["JFK", "LAX", "ORD", "ATL", "SEA"], n),
        "AirportTo": rng.choice(["JFK", "LAX", "ORD", "ATL", "SEA"], n),
        "DayOfWeek": rng.integers(1, 8, n),
        "Time": rng.integers(0, 1440, n),
        "Length": rng.integers(30, 400, n),
        "Delay": rng.integers(0, 2, n),
    })


def test_tune_random_forest_returns_row_per_max_depth():
    df = _sample_df()
    train_df, val_df, test_df = training.split_data(df)
    preprocessor = Preprocessor().fit(train_df)

    best_depth, tuning_rows = training.tune_random_forest(preprocessor, train_df, val_df)

    assert len(tuning_rows) == len(config.RF_MAX_DEPTH_GRID)
    for row in tuning_rows:
        assert set(row.keys()) == {"model", "max_depth", "validation_f1"}
        assert row["model"] == "random_forest"
    assert best_depth in config.RF_MAX_DEPTH_GRID


def test_train_all_returns_tuning_rows_alongside_models():
    df = _sample_df()
    train_df, val_df, test_df = training.split_data(df)

    models, preprocessor, X_test, y_test, tuning_rows = training.train_all(train_df, val_df, test_df)

    assert set(models.keys()) == {"baseline", "logistic_regression", "random_forest"}
    assert len(tuning_rows) == len(config.RF_MAX_DEPTH_GRID)
