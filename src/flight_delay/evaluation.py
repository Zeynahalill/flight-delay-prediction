"""Model değerlendirme: metrikler + grafikler (config.EVAL_DIR altına)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
)

from . import config, reporting

sns.set_style("whitegrid")


def evaluate_all(models: dict, X_test, y_test, feature_names=None, raw_test_df=None) -> dict:
    """
    raw_test_df: split_data()'nın döndürdüğü, encode edilmemiş test seti
    (Airline, AirportFrom, AirportTo, DayOfWeek, Time, Length, Delay
    kolonlarını içerir). Verilirse ve random_forest modeli varsa, test
    setindeki HER örnek için satır satır sonuçlar
    outputs/evaluation/test_predictions.csv olarak kaydedilir.
    """
    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    predictions = {}
    probas = {}

    for name, model in models.items():
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        predictions[name] = pred
        probas[name] = proba

        results[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "auc": roc_auc_score(y_test, proba),
        }
        print(f"\n--- {name} ---")
        for k, v in results[name].items():
            print(f"{k}: {v:.4f}")

    _plot_confusion_matrices(predictions, y_test)
    _plot_model_comparison(results)
    _plot_roc_curves(probas, y_test)

    if "random_forest" in models and feature_names is not None:
        _plot_feature_importance(models["random_forest"], feature_names)
        reporting.save_feature_importance_csv(
            feature_names, models["random_forest"].feature_importances_
        )

    if "random_forest" in models and raw_test_df is not None:
        reporting.save_test_predictions_csv(
            raw_test_df, predictions["random_forest"], probas["random_forest"]
        )

    reporting.save_model_results_csv(results)

    return results


def _plot_confusion_matrices(predictions: dict, y_test):
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (name, pred) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Zamanında", "Gecikti"],
                    yticklabels=["Zamanında", "Gecikti"])
        ax.set_title(name)
        ax.set_xlabel("Tahmin")
        ax.set_ylabel("Gerçek")
    plt.tight_layout()
    out = config.EVAL_DIR / "confusion_matrices.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"[Kaydedildi] {out}")


def _plot_model_comparison(results: dict):
    df = pd.DataFrame(results).T
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    x = np.arange(len(metrics))
    width = 0.8 / len(df)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, row) in enumerate(df.iterrows()):
        ax.bar(x + i * width, row[metrics], width, label=name)
    ax.set_xticks(x + width * (len(df) - 1) / 2)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
    ax.set_ylim(0, 1)
    ax.set_title("Model Karşılaştırması (Test Seti)")
    ax.legend()
    plt.tight_layout()
    out = config.EVAL_DIR / "model_comparison.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"[Kaydedildi] {out}")


def _plot_roc_curves(probas: dict, y_test):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Rastgele")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Eğrisi")
    ax.legend()
    plt.tight_layout()
    out = config.EVAL_DIR / "roc_curve.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"[Kaydedildi] {out}")


def _plot_feature_importance(rf_model, feature_names):
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": rf_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=importance_df, x="importance", y="feature", ax=ax, color="#55A868")
    ax.set_title("Feature Importance - Random Forest")
    plt.tight_layout()
    out = config.EVAL_DIR / "feature_importance.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"[Kaydedildi] {out}")
