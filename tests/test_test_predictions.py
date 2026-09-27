import numpy as np
import pandas as pd

from flight_delay import config, evaluation, reporting, training


def _sample_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "id": range(n),
        "Airline": rng.choice(["AA", "DL", "UA", "CO"], n),
        "Flight": rng.integers(100, 999, n),
        "AirportFrom": rng.choice(["JFK", "LAX", "ORD", "ATL", "SEA"], n),
        "AirportTo": rng.choice(["JFK", "LAX", "ORD", "ATL", "SEA"], n),
        "DayOfWeek": rng.integers(1, 8, n),
        "Time": rng.integers(0, 1440, n),
        "Length": rng.integers(30, 400, n),
        "Delay": rng.integers(0, 2, n),
    })


def _raw_test_df_sample(n=25, seed=1):
    """clean_data sonrası, encode edilmemiş bir test seti örneği (id/Flight
    zaten düşürülmüş, tıpkı split_data()'nın döndürdüğü gibi)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Airline": rng.choice(["AA", "DL", "UA", "CO"], n),
        "AirportFrom": rng.choice(["JFK", "LAX", "ORD", "ATL", "SEA"], n),
        "AirportTo": rng.choice(["JFK", "LAX", "ORD", "ATL", "SEA"], n),
        "DayOfWeek": rng.integers(1, 8, n),
        "Time": rng.integers(0, 1440, n),
        "Length": rng.integers(30, 400, n),
        "Delay": rng.integers(0, 2, n),
    })


def test_save_test_predictions_csv_creates_file_with_expected_columns_and_full_row_count(tmp_path):
    n = 37
    raw_test_df = _raw_test_df_sample(n=n)
    rng = np.random.default_rng(2)
    predicted = rng.integers(0, 2, n)
    probability = rng.random(n)
    out_path = tmp_path / "test_predictions.csv"

    reporting.save_test_predictions_csv(raw_test_df, predicted, probability, path=out_path)

    assert out_path.exists()
    df = pd.read_csv(out_path)

    assert list(df.columns) == [
        "Airline", "AirportFrom", "AirportTo", "DayOfWeek", "Time", "Length",
        "actual", "predicted", "probability",
    ]
    # Hiçbir test satırı atlanmamalı: satır sayısı test setiyle birebir aynı olmalı.
    assert len(df) == n == len(raw_test_df)

    assert set(df["actual"].unique()).issubset({0, 1})
    assert set(df["predicted"].unique()).issubset({0, 1})
    assert df["probability"].between(0.0, 1.0).all()

    # Sıra korunmalı: 'actual' kolonu raw_test_df'deki Delay ile birebir eşleşmeli.
    assert list(df["actual"]) == list(raw_test_df["Delay"])


def test_save_test_predictions_csv_rejects_length_mismatch(tmp_path):
    raw_test_df = _raw_test_df_sample(n=10)
    out_path = tmp_path / "test_predictions.csv"

    try:
        reporting.save_test_predictions_csv(raw_test_df, predicted=[0, 1, 1], probability=[0.1, 0.2, 0.3], path=out_path)
        assert False, "Uzunluk uyuşmazlığında hata bekleniyordu"
    except ValueError:
        pass


def test_evaluate_all_writes_test_predictions_for_the_full_test_set(monkeypatch, tmp_path):
    """evaluate_all() gerçekten test setindeki TÜM örnekleri (train/val hariç)
    reporting.save_test_predictions_csv'ye iletiyor mu? Dosya sistemine
    dokunmadan (fonksiyonlar mock'lanarak) doğrulanır."""
    df = _sample_df(n=300)
    train_df, val_df, test_df = training.split_data(df)
    models, preprocessor, X_test, y_test, tuning_rows = training.train_all(train_df, val_df, test_df)

    # Grafikler gerçek repo'ya yazılmasın diye geçici klasöre yönlendiriliyor.
    monkeypatch.setattr(config, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(evaluation.reporting, "save_model_results_csv", lambda *a, **k: None)
    monkeypatch.setattr(evaluation.reporting, "save_feature_importance_csv", lambda *a, **k: None)

    captured = {}

    def fake_save_test_predictions_csv(raw_test_df, predicted, probability, path=None):
        captured["raw_test_df"] = raw_test_df
        captured["predicted"] = list(predicted)
        captured["probability"] = list(probability)

    monkeypatch.setattr(evaluation.reporting, "save_test_predictions_csv", fake_save_test_predictions_csv)

    evaluation.evaluate_all(
        models, X_test, y_test, feature_names=config.FEATURE_COLUMNS, raw_test_df=test_df,
    )

    assert captured["raw_test_df"] is test_df
    # Test setindeki HİÇBİR satır atlanmamalı; sadece test seti (train/val değil).
    assert len(captured["predicted"]) == len(test_df)
    assert len(captured["probability"]) == len(test_df)
    assert set(captured["predicted"]).issubset({0, 1})
    assert all(0.0 <= p <= 1.0 for p in captured["probability"])
