"""Training / evaluation / prediction sonuçlarını CSV dosyalarına kaydeder.

Bu modülün TEK sorumluluğu raporlama (CSV yazma)'dır. Model eğitimi,
preprocessing veya split mantığına dokunmaz; sadece başka modüllerin
ürettiği sonuçları (dict/list/DataFrame) diske yazar. Böylece mevcut
pipeline'ın davranışı değişmeden yeni bir "yan çıktı" eklenmiş olur.
"""

import csv
from datetime import datetime, timezone

import pandas as pd

from . import config


def save_model_results_csv(results: dict, path=None) -> None:
    """results: {"baseline": {"accuracy":..., "precision":..., "recall":...,
    "f1":..., "auc":...}, "logistic_regression": {...}, "random_forest": {...}}
    -> outputs/evaluation/model_results.csv
    """
    path = path or config.MODEL_RESULTS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "auc": metrics["auc"],
        }
        for model_name, metrics in results.items()
    ]
    df = pd.DataFrame(rows, columns=["model", "accuracy", "precision", "recall", "f1", "auc"])
    df.to_csv(path, index=False)
    print(f"[Kaydedildi] {path}")


def save_hyperparameter_results_csv(tuning_rows: list, path=None) -> None:
    """tuning_rows: [{"model": "random_forest", "max_depth": 5, "validation_f1": 0.4142}, ...]
    -> outputs/evaluation/hyperparameter_results.csv
    """
    path = path or config.HYPERPARAMETER_RESULTS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(tuning_rows, columns=["model", "max_depth", "validation_f1"])
    df.to_csv(path, index=False)
    print(f"[Kaydedildi] {path}")


def save_dataset_summary_csv(
    total_rows: int, train_rows: int, validation_rows: int, test_rows: int,
    delay_rate: float, path=None,
) -> None:
    """-> outputs/evaluation/dataset_summary.csv"""
    path = path or config.DATASET_SUMMARY_CSV
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([{
        "total_rows": total_rows,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "test_rows": test_rows,
        "delay_rate": delay_rate,
    }])
    df.to_csv(path, index=False)
    print(f"[Kaydedildi] {path}")


def save_feature_importance_csv(feature_names, importances, path=None) -> None:
    """Büyükten küçüğe sıralı -> outputs/evaluation/feature_importance.csv"""
    path = path or config.FEATURE_IMPORTANCE_CSV
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"feature": list(feature_names), "importance": list(importances)})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"[Kaydedildi] {path}")


# Test set örneklerinde satır satır kaydedilecek orijinal (encode edilmemiş) feature'lar.
TEST_PREDICTIONS_FEATURE_COLUMNS = [
    "Airline", "AirportFrom", "AirportTo", "DayOfWeek", "Time", "Length",
]


def save_test_predictions_csv(raw_test_df, predicted, probability, path=None) -> None:
    """Test setindeki HER örnek için satır satır sonuç kaydeder
    (-> outputs/evaluation/test_predictions.csv). Hiçbir satır atlanmaz;
    kolonlar: Airline, AirportFrom, AirportTo, DayOfWeek, Time, Length,
    actual, predicted, probability. Train/validation verisi dahil edilmez;
    sadece `raw_test_df` (split_data'nın döndürdüğü test seti, encode
    edilmemiş hali) ve ona karşılık gelen model tahminleri kullanılır.
    Satır sırası `raw_test_df` ile birebir aynı korunur.
    """
    path = path or config.TEST_PREDICTIONS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)

    predicted = list(predicted)
    probability = list(probability)

    if len(raw_test_df) != len(predicted) or len(raw_test_df) != len(probability):
        raise ValueError(
            "raw_test_df, predicted ve probability aynı uzunlukta olmalı: "
            f"{len(raw_test_df)} vs {len(predicted)} vs {len(probability)}"
        )

    df_out = raw_test_df[TEST_PREDICTIONS_FEATURE_COLUMNS].reset_index(drop=True).copy()
    df_out["actual"] = pd.Series(raw_test_df[config.TARGET_COLUMN].values).astype(int)
    df_out["predicted"] = pd.Series(predicted).astype(int)
    df_out["probability"] = pd.Series(probability).astype(float)

    df_out.to_csv(path, index=False)
    print(f"[Kaydedildi] {path} ({len(df_out)} satır)")


def append_prediction_csv(sample: dict, proba: float, prediction: int, path=None) -> None:
    """Her `predict.py` çalıştırmasında outputs/predictions.csv dosyasına
    yeni bir satır EKLER (mevcut satırları korur). Terminal çıktısının
    yerine geçmez; ona ek olarak çalışır."""
    path = path or config.PREDICTIONS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "airline": sample["Airline"],
        "airport_from": sample["AirportFrom"],
        "airport_to": sample["AirportTo"],
        "day": sample["DayOfWeek"],
        "time": sample["Time"],
        "flight_length": sample["Length"],
        "delay_probability": round(float(proba), 4),
        "prediction": int(prediction),
    }
    fieldnames = list(row.keys())

    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"[Kaydedildi] {path}")
