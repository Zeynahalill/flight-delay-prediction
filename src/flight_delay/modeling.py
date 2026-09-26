"""Model tanımları.

Yeni bir classification modeli eklemek için buraya yeni bir get_xxx()
fonksiyonu eklemek yeterlidir; training.py'yi değiştirmeye gerek yoktur.
"""

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from . import config


def get_baseline():
    """En basit referans model: her zaman en sık sınıfı tahmin eder.
    Diğer modellerin gerçekten bir şey öğrenip öğrenmediğini ölçmek için."""
    return DummyClassifier(strategy="most_frequent", random_state=config.RANDOM_STATE)


def get_logistic_regression():
    return LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)


def get_random_forest(max_depth=None, n_estimators=200):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )


def get_models(rf_max_depth=None):
    return {
        "baseline": get_baseline(),
        "logistic_regression": get_logistic_regression(),
        "random_forest": get_random_forest(max_depth=rf_max_depth),
    }
