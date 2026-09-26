"""Keşifsel veri analizi. Sadece OKUR, veriyi değiştirmez.

Grafikleri config.EDA_DIR altına kaydeder.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import config

sns.set_style("whitegrid")


def run_eda(df: pd.DataFrame) -> None:
    config.EDA_DIR.mkdir(parents=True, exist_ok=True)

    print("Genel gecikme oranı:")
    print(df[config.TARGET_COLUMN].value_counts(normalize=True))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    df[config.TARGET_COLUMN].value_counts().rename({0: "Zamanında", 1: "Gecikti"}).plot(
        kind="bar", ax=axes[0, 0], color=["#4C72B0", "#C44E52"])
    axes[0, 0].set_title("Genel Gecikme Dağılımı")
    axes[0, 0].tick_params(axis="x", rotation=0)

    airline_delay = df.groupby("Airline")[config.TARGET_COLUMN].mean().sort_values(ascending=False)
    airline_delay.plot(kind="bar", ax=axes[0, 1], color="#DD8452")
    axes[0, 1].set_title("Havayoluna Göre Gecikme Oranı")
    axes[0, 1].tick_params(axis="x", rotation=45)

    hour_bucket = (df["Time"] // 60).clip(0, 23)
    hour_delay = df.groupby(hour_bucket)[config.TARGET_COLUMN].mean()
    hour_delay.plot(kind="line", marker="o", ax=axes[1, 0], color="#55A868")
    axes[1, 0].set_title("Saate Göre Gecikme Oranı")
    axes[1, 0].set_xlabel("Kalkış saati (0-23)")

    day_delay = df.groupby("DayOfWeek")[config.TARGET_COLUMN].mean()
    day_delay.plot(kind="bar", ax=axes[1, 1], color="#8172B2")
    axes[1, 1].set_title("Haftanın Gününe Göre Gecikme Oranı")
    axes[1, 1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    out_path = config.EDA_DIR / "eda_overview.png"
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"[Kaydedildi] {out_path}")
