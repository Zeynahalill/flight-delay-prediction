"""CLI: python scripts/train.py

Tüm pipeline'ı sırayla çalıştırır:
DATA LOADING -> VALIDATION -> EDA -> SPLIT -> TRAINING -> TUNING ->
EVALUATION -> MODEL SAVING
"""

from flight_delay import config, data_loading, data_validation, eda, training, evaluation


def main():
    print("1) Veri yükleniyor...")
    df = data_loading.load_raw_data()

    print("\n2) Şema doğrulanıyor...")
    data_validation.validate_schema(df)

    print("\n3) EDA çalıştırılıyor...")
    eda.run_eda(df)

    print("\n4) Train/Validation/Test split...")
    train_df, val_df, test_df = training.split_data(df)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    print("\n5) Modeller eğitiliyor + hyperparameter tuning...")
    models, preprocessor, X_test, y_test = training.train_all(train_df, val_df, test_df)

    print("\n6) Test seti üzerinde değerlendirme...")
    results = evaluation.evaluate_all(models, X_test, y_test, feature_names=config.FEATURE_COLUMNS)

    best_name = max(results, key=lambda k: results[k]["f1"])
    print(f"\nEn iyi model (F1'e göre): {best_name}")

    print("\n7) Model kaydediliyor...")
    training.save_artifact(models[best_name], preprocessor)
    print(f"[OK] Model kaydedildi: {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
