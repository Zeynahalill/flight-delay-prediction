# Flight Delay Prediction

Bir uçuşun (havayolu, kalkış/varış havalimanı, haftanın günü, planlanan
kalkış saati, planlanan uçuş süresi) bilgilerine bakarak **15+ dakika
gecikip gecikmeyeceğini** tahmin eden klasik/tabular bir Machine Learning
projesi.

Bu proje bir RobotyLab Community ML workshop'u için hazırlanmıştır.

## Problem Tanımı

- **Problem tipi:** Binary Classification
- **Target:** `Delay` (0 = zamanında, 1 = gecikti)
- **Feature'lar:** Havayolu, kalkış/varış havalimanı, haftanın günü,
  planlanan kalkış saati, planlanan uçuş süresi -- hepsi uçuş kalkmadan
  **önce** bilinen bilgiler (data leakage yok).

## Dataset

**Kaynak:** [Kaggle - Airlines Dataset to Predict a Delay](https://www.kaggle.com/datasets/jimschacko/airlines-dataset-to-predict-a-delay)

### Kurulum
1. Yukarıdaki linkten `Airlines.csv` dosyasını indir.
2. Dosyayı `data/raw/Airlines.csv` olarak kaydet.
3. Dataset repoya dahil değildir (`.gitignore`'da) -- bu adımı her klonlamada
   tekrarlaman gerekir.

## Ortam Kurulumu

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                 # src/flight_delay paketini importable yapar
```

## Projeyi Çalıştırma

### Model eğitimi (tüm pipeline)
```bash
python scripts/train.py
```
Bu komut sırasıyla: veriyi yükler → şemayı doğrular → EDA grafiklerini
üretir (`outputs/eda/`) → train/val/test split yapar → Logistic Regression
ve Random Forest'ı eğitip Random Forest için `max_depth` hyperparameter
tuning yapar → test setinde değerlendirir (`outputs/evaluation/`) → en iyi
modeli `models/model.joblib` olarak kaydeder.

### Tahmin
```bash
python scripts/predict.py --airline AA --from JFK --to LAX --day 5 --time 1140 --length 240
```

### Testler
```bash
pytest tests/
```

### Workshop Notebook'u
```bash
jupyter notebook notebooks/workshop_walkthrough.ipynb
```
Notebook, `src/flight_delay` içindeki gerçek modülleri import edip
kullanır -- ayrı bir "deneme kodu" değildir. Workshop'ta EDA'yı, feature
engineering'i ve canlı tahmini adım adım göstermek için kullanılır.

## Proje Yapısı

```
src/flight_delay/   -> tekrar kullanılabilir kaynak kod (modüler pipeline)
scripts/            -> CLI giriş noktaları (train.py, predict.py)
notebooks/          -> workshop anlatımı için (sadece keşif/demo)
tests/              -> en riskli 3 nokta için birim testler
data/raw/           -> CSV buraya (gitignored)
models/             -> eğitilmiş model (gitignored)
outputs/            -> EDA ve evaluation grafikleri (gitignored)
```

## Tasarım Notları

- `id` ve `Flight` kolonları düşürülür: `id` tamamen anlamsız bir satır
  numarası; `Flight` (~6000 farklı değer) çok yüksek kardiniteli ve modelin
  genellemek yerine belirli uçuş numaralarını ezberlemesine yol açabilir.
- `AirportFrom`/`AirportTo` (~293 kategori) en sık görülen N havalimanına
  indirgenip geri kalanı "OTHER" olarak bucket'lanır.
- Preprocessing, `Preprocessor` sınıfı olarak stateful yazılmıştır ve
  modelle birlikte tek bir `.joblib` dosyasında saklanır -- böylece
  training'de öğrenilen encoding, prediction anında birebir aynı şekilde
  uygulanır (train/serve skew riski yok) ve daha önce görülmemiş
  kategoriler çökme yaratmadan ele alınır.
