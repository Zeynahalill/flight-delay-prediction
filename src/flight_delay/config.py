"""Merkezi konfigürasyon: path'ler, sabitler, feature listeleri.

Diğer hiçbir modülde hardcoded path/sabit olmamalı; hepsi buradan import edilir.
"""

from pathlib import Path

# --- Path'ler ---
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw"
RAW_DATA_PATH = DATA_DIR / "Airlines.csv"

MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"

OUTPUT_DIR = ROOT_DIR / "outputs"
EDA_DIR = OUTPUT_DIR / "eda"
EVAL_DIR = OUTPUT_DIR / "evaluation"

# --- Genel ---
RANDOM_STATE = 42

# --- Dataset şeması (Kaggle: jimschacko/airlines-dataset-to-predict-a-delay) ---
EXPECTED_COLUMNS = [
    "id", "Airline", "Flight", "AirportFrom", "AirportTo",
    "DayOfWeek", "Time", "Length", "Delay",
]
TARGET_COLUMN = "Delay"

# Prediction için anlamsız / yüksek kardinalite riski taşıyan kolonlar
DROP_COLUMNS = ["id", "Flight"]

# Yüksek kardiniteli havalimanı kolonlarında kaç tanesini ayrı tutacağımız
# (geri kalanı "OTHER" bucket'ına düşer)
TOP_N_AIRPORTS = 20

# --- Modele giden nihai feature seti ---
FEATURE_COLUMNS = [
    "Airline_encoded",
    "AirportFrom_encoded",
    "AirportTo_encoded",
    "DayOfWeek",
    "Time",
    "Length",
    "IsEveningFlight",
    "IsBusyDay",
]

# --- Split oranları (toplam = 1.0) ---
TRAIN_SIZE = 0.6
VAL_SIZE = 0.2
TEST_SIZE = 0.2

# --- Hyperparameter arama uzayı ---
RF_MAX_DEPTH_GRID = [5, 8, 12, None]
