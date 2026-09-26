import pandas as pd

from flight_delay.feature_engineering import add_features


def test_is_evening_flight_flag():
    df = pd.DataFrame({
        "Time": [10 * 60, 18 * 60],   # 10:00 ve 18:00
        "DayOfWeek": [1, 1],
    })
    out = add_features(df)
    assert out["IsEveningFlight"].tolist() == [0, 1]


def test_is_busy_day_flag():
    df = pd.DataFrame({
        "Time": [600, 600, 600],
        "DayOfWeek": [3, 5, 7],  # Çar, Cum, Paz
    })
    out = add_features(df)
    assert out["IsBusyDay"].tolist() == [0, 1, 1]
