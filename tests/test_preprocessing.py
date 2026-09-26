import pandas as pd

from flight_delay.preprocessing import clean_data, Preprocessor


def _sample_df(n=50):
    import numpy as np
    rng = np.random.default_rng(0)
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


def test_clean_data_drops_id_and_flight():
    df = _sample_df()
    cleaned = clean_data(df)
    assert "id" not in cleaned.columns
    assert "Flight" not in cleaned.columns


def test_preprocessor_fit_transform_produces_encoded_columns():
    df = _sample_df()
    pre = Preprocessor().fit(df)
    transformed = pre.transform(df)

    for col in ["Airline_encoded", "AirportFrom_encoded", "AirportTo_encoded"]:
        assert col in transformed.columns
        assert transformed[col].dtype.kind in "iu"  # integer encoding


def test_preprocessor_handles_unseen_category_gracefully():
    df = _sample_df()
    pre = Preprocessor().fit(df)

    new_row = pd.DataFrame([{
        "Airline": "ZZ_NEVER_SEEN",  # eğitimde görülmemiş kod
        "AirportFrom": "JFK",
        "AirportTo": "LAX",
        "DayOfWeek": 3,
        "Time": 500,
        "Length": 120,
    }])

    transformed = pre.transform(new_row)  # exception fırlatmamalı
    assert "Airline_encoded" in transformed.columns
