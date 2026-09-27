"""Train/val/test split, hyperparameter tuning, model eğitimi, model kaydetme."""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from . import config, modeling
from .preprocessing import Preprocessor, clean_data
from .feature_engineering import add_features


def split_data(df: pd.DataFrame):
    """%60 train / %20 validation / %20 test, stratify edilmiş."""
    df = clean_data(df)

    train_val, test = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
        stratify=df[config.TARGET_COLUMN],
    )
    val_ratio = config.VAL_SIZE / (config.TRAIN_SIZE + config.VAL_SIZE)
    train, val = train_test_split(
        train_val, test_size=val_ratio, random_state=config.RANDOM_STATE,
        stratify=train_val[config.TARGET_COLUMN],
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def build_xy(preprocessor: Preprocessor, df: pd.DataFrame):
    df_t = preprocessor.transform(df)
    df_t = add_features(df_t)
    X = df_t[config.FEATURE_COLUMNS]
    y = df_t[config.TARGET_COLUMN]
    return X, y


def tune_random_forest(preprocessor, train_df, val_df):
    """Random Forest'ın max_depth hyperparametresini validation set
    üzerinde arar. Diğer hyperparametreler (n_estimators vb.) sabit
    tutulur -- workshop'ta 'parametre vs hyperparametre' farkını
    tek bir eksende net göstermek için.

    Returns:
        best_depth: en iyi max_depth değeri (davranış değişmedi).
        tuning_rows: denenen her max_depth için
            {"model": "random_forest", "max_depth": depth, "validation_f1": f1}
            şeklinde satırlar (outputs/evaluation/hyperparameter_results.csv
            için kullanılır).
    """
    X_train, y_train = build_xy(preprocessor, train_df)
    X_val, y_val = build_xy(preprocessor, val_df)

    best_depth, best_f1 = None, -1.0
    tuning_rows = []
    for depth in config.RF_MAX_DEPTH_GRID:
        model = modeling.get_random_forest(max_depth=depth)
        model.fit(X_train, y_train)
        f1 = f1_score(y_val, model.predict(X_val))
        print(f"  max_depth={depth}: validation F1 = {f1:.4f}")
        tuning_rows.append({"model": "random_forest", "max_depth": depth, "validation_f1": f1})
        if f1 > best_f1:
            best_f1, best_depth = f1, depth

    print(f"En iyi max_depth: {best_depth} (F1={best_f1:.4f})")
    return best_depth, tuning_rows


def train_all(train_df, val_df, test_df):
    """Preprocessor'ı SADECE train_df üzerinde fit eder, sonra train+val ile
    modelleri final olarak eğitir. Test seti hiçbir eğitim adımına girmez."""
    preprocessor = Preprocessor().fit(train_df)

    best_depth, tuning_rows = tune_random_forest(preprocessor, train_df, val_df)

    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    X_trainval, y_trainval = build_xy(preprocessor, train_val_df)
    X_test, y_test = build_xy(preprocessor, test_df)

    models = modeling.get_models(rf_max_depth=best_depth)
    for name, model in models.items():
        model.fit(X_trainval, y_trainval)
        print(f"'{name}' eğitildi.")

    return models, preprocessor, X_test, y_test, tuning_rows


def save_artifact(model, preprocessor: Preprocessor, path=None) -> None:
    path = path or config.MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "preprocessor": preprocessor,
        "feature_columns": config.FEATURE_COLUMNS,
    }, path)
