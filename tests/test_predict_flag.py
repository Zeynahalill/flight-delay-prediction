import numpy as np
import pandas as pd
import pytest

from flight_delay import modeling, predict
from flight_delay.feature_engineering import add_features
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


def _dummy_artifact():
    df = _sample_df()
    preprocessor = Preprocessor().fit(df)
    df_t = preprocessor.transform(df)
    df_t = add_features(df_t)

    feature_columns = [
        "Airline_encoded", "AirportFrom_encoded", "AirportTo_encoded",
        "DayOfWeek", "Time", "Length", "IsEveningFlight", "IsBusyDay",
    ]
    model = modeling.get_random_forest(max_depth=5, n_estimators=20)
    model.fit(df_t[feature_columns], df_t["Delay"])

    return {"model": model, "preprocessor": preprocessor, "feature_columns": feature_columns}


def test_predict_new_with_flag_matches_predict_new():
    artifact = _dummy_artifact()
    sample = {
        "Airline": "AA", "AirportFrom": "JFK", "AirportTo": "LAX",
        "DayOfWeek": 5, "Time": 1140, "Length": 240,
    }

    proba, label = predict.predict_new(sample, artifact)
    proba2, label2, prediction = predict.predict_new_with_flag(sample, artifact)

    assert proba == pytest.approx(proba2)
    assert label == label2
    assert prediction in (0, 1)
    assert prediction == int(proba >= 0.5)
