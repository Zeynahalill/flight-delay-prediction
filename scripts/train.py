"""CLI: python scripts/train.py

Tüm pipeline'ı sırayla çalıştırır:
DATA LOADING -> VALIDATION -> EDA -> SPLIT -> TRAINING -> TUNING ->
EVALUATION -> MODEL SAVING
"""

from flight_delay import config, data_loading, data_validation, eda, training, evaluation, reporting
from flight_delay.preprocessing import clean_data


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

    cleaned_df = clean_data(df)
    reporting.save_dataset_summary_csv(
        total_rows=len(cleaned_df),
        train_rows=len(train_df),
        validation_rows=len(val_df),
        test_rows=len(test_df),
        delay_rate=cleaned_df[config.TARGET_COLUMN].mean(),
    )

    print("\n5) Modeller eğitiliyor + hyperparameter tuning...")
    models, preprocessor, X_test, y_test, tuning_rows = training.train_all(train_df, val_df, test_df)
    reporting.save_hyperparameter_results_csv(tuning_rows)

    print("\n6) Test seti üzerinde değerlendirme...")
    results = evaluation.evaluate_all(
        models, X_test, y_test, feature_names=config.FEATURE_COLUMNS, raw_test_df=test_df
    )

    best_name = max(results, key=lambda k: results[k]["f1"])
    print(f"\nEn iyi model (F1'e göre): {best_name}")

    print("\n7) Model kaydediliyor...")
    training.save_artifact(models[best_name], preprocessor)
    print(f"[OK] Model kaydedildi: {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
