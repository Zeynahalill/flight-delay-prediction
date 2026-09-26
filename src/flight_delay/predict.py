"""Kayıtlı model artifact'ini yükleyip yeni veri üzerinde tahmin yapar."""

import joblib
import pandas as pd

from . import config
from .feature_engineering import add_features


def load_artifact(path=None) -> dict:
    path = path or config.MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Eğitilmiş model bulunamadı: {path}\n"
            "Önce 'python scripts/train.py' ile modeli eğit."
        )
    return joblib.load(path)


def predict_new(sample: dict, artifact: dict = None):
    """sample örneği:
        {"Airline": "AA", "AirportFrom": "JFK", "AirportTo": "LAX",
         "DayOfWeek": 5, "Time": 1140, "Length": 240}
    """
    artifact = artifact or load_artifact()

    df = pd.DataFrame([sample])
    df_t = artifact["preprocessor"].transform(df)
    df_t = add_features(df_t)

    X = df_t[artifact["feature_columns"]]
    proba = artifact["model"].predict_proba(X)[0, 1]
    label = "GECİKME BEKLENİYOR" if proba >= 0.5 else "ZAMANINDA BEKLENİYOR"
    return proba, label
