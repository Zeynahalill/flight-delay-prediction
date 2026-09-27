import pandas as pd

from flight_delay import reporting


def test_save_model_results_csv_creates_file_with_expected_columns(tmp_path):
    results = {
        "baseline": {"accuracy": 0.55, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.5},
        "logistic_regression": {"accuracy": 0.61, "precision": 0.58, "recall": 0.42, "f1": 0.49, "auc": 0.63},
        "random_forest": {"accuracy": 0.68, "precision": 0.66, "recall": 0.55, "f1": 0.60, "auc": 0.72},
    }
    out_path = tmp_path / "model_results.csv"

    reporting.save_model_results_csv(results, path=out_path)

    assert out_path.exists()
    df = pd.read_csv(out_path)
    assert list(df.columns) == ["model", "accuracy", "precision", "recall", "f1", "auc"]
    assert set(df["model"]) == {"baseline", "logistic_regression", "random_forest"}
    assert len(df) == 3


def test_save_hyperparameter_results_csv_creates_file_with_expected_columns(tmp_path):
    tuning_rows = [
        {"model": "random_forest", "max_depth": 5, "validation_f1": 0.4142},
        {"model": "random_forest", "max_depth": 8, "validation_f1": 0.4663},
        {"model": "random_forest", "max_depth": 12, "validation_f1": 0.5206},
        {"model": "random_forest", "max_depth": None, "validation_f1": 0.5569},
    ]
    out_path = tmp_path / "hyperparameter_results.csv"

    reporting.save_hyperparameter_results_csv(tuning_rows, path=out_path)

    assert out_path.exists()
    df = pd.read_csv(out_path)
    assert list(df.columns) == ["model", "max_depth", "validation_f1"]
    assert len(df) == 4


def test_save_dataset_summary_csv_creates_file_with_expected_columns(tmp_path):
    out_path = tmp_path / "dataset_summary.csv"

    reporting.save_dataset_summary_csv(
        total_rows=100, train_rows=60, validation_rows=20, test_rows=20,
        delay_rate=0.4321, path=out_path,
    )

    assert out_path.exists()
    df = pd.read_csv(out_path)
    assert list(df.columns) == [
        "total_rows", "train_rows", "validation_rows", "test_rows", "delay_rate",
    ]
    assert len(df) == 1
    row = df.iloc[0]
    assert row["total_rows"] == 100
    assert row["train_rows"] + row["validation_rows"] + row["test_rows"] == 100


def test_save_feature_importance_csv_sorted_descending(tmp_path):
    out_path = tmp_path / "feature_importance.csv"

    reporting.save_feature_importance_csv(
        feature_names=["a", "b", "c"], importances=[0.1, 0.5, 0.4], path=out_path,
    )

    assert out_path.exists()
    df = pd.read_csv(out_path)
    assert list(df.columns) == ["feature", "importance"]
    assert list(df["feature"]) == ["b", "c", "a"]
    assert df["importance"].is_monotonic_decreasing


def test_append_prediction_csv_creates_file_with_expected_columns(tmp_path):
    out_path = tmp_path / "predictions.csv"
    sample = {
        "Airline": "CO", "AirportFrom": "ATL", "AirportTo": "JFK",
        "DayOfWeek": 3, "Time": 1200, "Length": 120,
    }

    reporting.append_prediction_csv(sample, proba=0.595, prediction=1, path=out_path)

    assert out_path.exists()
    df = pd.read_csv(out_path)
    assert list(df.columns) == [
        "timestamp", "airline", "airport_from", "airport_to",
        "day", "time", "flight_length", "delay_probability", "prediction",
    ]
    assert len(df) == 1
    assert df.iloc[0]["airline"] == "CO"
    assert df.iloc[0]["prediction"] == 1


def test_append_prediction_csv_appends_without_overwriting(tmp_path):
    out_path = tmp_path / "predictions.csv"
    sample = {
        "Airline": "AA", "AirportFrom": "JFK", "AirportTo": "LAX",
        "DayOfWeek": 5, "Time": 1140, "Length": 240,
    }

    reporting.append_prediction_csv(sample, proba=0.30, prediction=0, path=out_path)
    reporting.append_prediction_csv(sample, proba=0.80, prediction=1, path=out_path)

    df = pd.read_csv(out_path)
    assert len(df) == 2
    assert list(df["prediction"]) == [0, 1]
