"""Ham veriyi diskten okuma. Tek sorumluluğu bu."""

import pandas as pd

from . import config


def load_raw_data(path=None) -> pd.DataFrame:
    """Kaggle'dan indirilen Airlines.csv dosyasını okur.

    path verilmezse config.RAW_DATA_PATH kullanılır
    (varsayılan: data/raw/Airlines.csv).
    """
    path = path or config.RAW_DATA_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset bulunamadı: {path}\n"
            "Lütfen README.md'deki 'Dataset Kurulumu' bölümüne bakarak "
            "CSV'yi Kaggle'dan indirip data/raw/ klasörüne koy."
        )

    return pd.read_csv(path)
