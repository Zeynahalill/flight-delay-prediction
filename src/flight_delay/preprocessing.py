"""Veri temizleme ve encoding.

Preprocessor sınıfı, sklearn'ün fit/transform mantığını taklit eder:
  - fit(): SADECE training verisinden öğrenir (hangi havalimanları "top N",
    encoder'ların bildiği kategoriler vb.)
  - transform(): öğrenilen durumu, training'de de prediction'da da AYNI
    şekilde uygular.

Bu sınıf, model ile birlikte joblib'e kaydedilir; böylece prediction anında
training'dekiyle birebir aynı dönüşüm garanti edilir (train/serve skew yok).
Ayrıca prediction anında daha önce görülmemiş bir havalimanı/havayolu
gelirse çökmeden, bilinen bir kategoriye güvenle düşer (bkz. _safe_transform).
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from . import config


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """id/Flight gibi kullanılmayacak kolonları düşürür, target'ı olmayan
    satırları atar. Hem training hem prediction verisi için güvenlidir
    (Delay kolonu prediction verisinde zaten bulunmaz)."""
    df = df.drop(columns=[c for c in config.DROP_COLUMNS if c in df.columns])
    if config.TARGET_COLUMN in df.columns:
        df = df.dropna(subset=[config.TARGET_COLUMN])
    return df.reset_index(drop=True)


class Preprocessor:
    def __init__(self):
        self.top_airports_from = None
        self.top_airports_to = None
        self.airline_encoder = LabelEncoder()
        self.airport_from_encoder = LabelEncoder()
        self.airport_to_encoder = LabelEncoder()
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        df = clean_data(df)

        self.top_airports_from = set(
            df["AirportFrom"].value_counts().nlargest(config.TOP_N_AIRPORTS).index
        )
        self.top_airports_to = set(
            df["AirportTo"].value_counts().nlargest(config.TOP_N_AIRPORTS).index
        )

        df = self._bucket_airports(df)

        self.airline_encoder.fit(df["Airline"])
        self.airport_from_encoder.fit(df["AirportFrom"])
        self.airport_to_encoder.fit(df["AirportTo"])
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Preprocessor henüz fit edilmedi.")

        df = clean_data(df)
        df = self._bucket_airports(df)

        df["Airline_encoded"] = self._safe_transform(self.airline_encoder, df["Airline"])
        df["AirportFrom_encoded"] = self._safe_transform(self.airport_from_encoder, df["AirportFrom"])
        df["AirportTo_encoded"] = self._safe_transform(self.airport_to_encoder, df["AirportTo"])
        return df

    def _bucket_airports(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["AirportFrom"] = df["AirportFrom"].where(
            df["AirportFrom"].isin(self.top_airports_from), other="OTHER")
        df["AirportTo"] = df["AirportTo"].where(
            df["AirportTo"].isin(self.top_airports_to), other="OTHER")
        return df

    @staticmethod
    def _safe_transform(encoder: LabelEncoder, series: pd.Series):
        """Prediction anında encoder'ın hiç görmediği bir kategori gelirse
        (örn. yeni bir havayolu kodu), çökmek yerine bilinen ilk kategoriye
        düşürür. Gerçek üretimde bu, sessiz bir hata yerine tahmin
        üretebilmeyi sağlar."""
        known = set(encoder.classes_)
        safe_series = series.where(series.isin(known), encoder.classes_[0])
        return encoder.transform(safe_series)
