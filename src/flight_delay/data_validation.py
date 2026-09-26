"""Ham verinin beklenen şemaya uyup uymadığını kontrol eder.

Pipeline'ın en başında çalışır: kolon eksikse / target değeri bozuksa
burada, anlamlı bir hata mesajıyla durur -- sessizce yanlış sonuç
üretmek yerine.
"""

import pandas as pd

from . import config


class DataValidationError(Exception):
    """Dataset beklenen şemaya uymadığında fırlatılır."""


def validate_schema(df: pd.DataFrame) -> None:
    missing_cols = set(config.EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise DataValidationError(
            f"Beklenen kolonlar bulunamadı: {sorted(missing_cols)}\n"
            f"Mevcut kolonlar: {list(df.columns)}\n"
            "Kaggle CSV'sinin kolon isimleri değişmiş olabilir; "
            "config.EXPECTED_COLUMNS'u gerçek dosyaya göre güncelle."
        )

    target_values = set(df[config.TARGET_COLUMN].dropna().unique())
    if not target_values.issubset({0, 1}):
        raise DataValidationError(
            f"'{config.TARGET_COLUMN}' kolonu 0/1 dışında değer içeriyor: "
            f"{target_values}"
        )

    null_counts = df[config.EXPECTED_COLUMNS].isnull().sum()
    if null_counts.any():
        print("[Uyarı] Eksik değer bulundu:")
        print(null_counts[null_counts > 0])
    else:
        print("[OK] Eksik değer yok.")

    if df.duplicated().sum() > 0:
        print(f"[Uyarı] {df.duplicated().sum()} adet tam tekrar eden satır var.")

    print(f"[OK] Şema doğrulandı. Satır sayısı: {len(df)}, Kolon sayısı: {df.shape[1]}")
