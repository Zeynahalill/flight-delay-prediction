import pandas as pd
import pytest

from flight_delay.data_validation import validate_schema, DataValidationError


def _valid_df():
    return pd.DataFrame({
        "id": [1, 2],
        "Airline": ["AA", "DL"],
        "Flight": [101, 202],
        "AirportFrom": ["JFK", "LAX"],
        "AirportTo": ["ORD", "ATL"],
        "DayOfWeek": [1, 5],
        "Time": [600, 1200],
        "Length": [120, 90],
        "Delay": [0, 1],
    })


def test_valid_schema_passes():
    validate_schema(_valid_df())  # hata fırlatmamalı


def test_missing_column_raises():
    df = _valid_df().drop(columns=["Length"])
    with pytest.raises(DataValidationError):
        validate_schema(df)


def test_invalid_target_values_raises():
    df = _valid_df()
    df["Delay"] = [0, 5]  # 5 geçersiz bir değer
    with pytest.raises(DataValidationError):
        validate_schema(df)
