"""Ham kolonlardan türetilmiş, yorumlanabilir feature'lar.

Preprocessor.transform() sonrası çağrılır. Training ve prediction'da
AYNI fonksiyon kullanılır.
"""

import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["IsEveningFlight"] = (df["Time"] >= 17 * 60).astype(int)
    df["IsBusyDay"] = df["DayOfWeek"].isin([5, 7]).astype(int)
    return df
