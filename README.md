# Flight Delay Prediction

Gerçek Kaggle **Airlines** dataset'i kullanılarak bir uçuşun (havayolu,
kalkış/varış havalimanı, haftanın günü, planlanan kalkış saati, planlanan
uçuş süresi) bilgilerine bakarak **gecikip gecikmeyeceğini** tahmin eden
klasik/tabular bir Machine Learning projesi.

Amaç, dataset'i kendi bilgisayarına indirip **training ve test
işlemlerini uçtan uca kendi makinende çalıştırmandır** -- proje
hazır/paylaşılmış bir sonuç içermez.

## 1. Projenin Amacı

- **Problem tipi:** Binary Classification
- **Hedef (target):** `Delay` (0 = zamanında, 1 = gecikti)
- **Amaç:** Bir uçuşun kalkmadan **önce** bilinen bilgilerine (havayolu,
  kalkış/varış havalimanı, haftanın günü, planlanan kalkış saati, planlanan
  uçuş süresi) bakarak gecikme olasılığını tahmin etmek (data leakage yok).
- **Kullanılan veri:** Gerçek Kaggle `Airlines.csv` dataset'i -- sentetik
  veya örnek/demo veri değildir.

## 2. Dataset

- **Dosya adı:** `Airlines.csv`
- **Kaynak:** [Kaggle - Airlines Dataset to Predict a Delay](https://www.kaggle.com/datasets/jimschacko/airlines-dataset-to-predict-a-delay)
- **Boyut:** 539.383 satır, 9 kolon (`id`, `Airline`, `Flight`, `AirportFrom`,
  `AirportTo`, `DayOfWeek`, `Time`, `Length`, `Delay`)
- **Önemli:** Dataset repoya dahil DEĞİLDİR (`.gitignore`'da) -- boyutu
  nedeniyle GitHub'a eklenmez. Her klonlamada aşağıdaki adımı tekrarlaman
  gerekir.

### Dataset kurulumu

1. Yukarıdaki Kaggle linkinden `Airlines.csv` dosyasını indir.
2. İndirdiğin dosyayı, adını değiştirmeden, **tam olarak** şu konuma koy:

   ```
   data/raw/Airlines.csv
   ```

Bunun dışında başka bir dosyaya (örneğin hazır `train.csv` / `test.csv`)
ihtiyaç yoktur -- tüm train/validation/test split'i `Airlines.csv`'den
`python scripts/train.py` çalıştırıldığında kod tarafından otomatik olarak
üretilir.

## 3. Kurulum

### macOS / Linux

```bash
git clone <bu-repo-url>
cd flight-delay-prediction

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .          # src/flight_delay paketini importable yapar
```

### Windows (PowerShell)

```powershell
git clone <bu-repo-url>
cd flight-delay-prediction

python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -e .          # src/flight_delay paketini importable yapar
```

> PowerShell "execution policy" hatası verirse (script çalıştırma engeli),
> yönetici olmayan bir PowerShell'de tek seferlik şunu çalıştırabilirsin:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

Kurulumdan sonra `data/raw/Airlines.csv` dosyasının yerinde olduğundan
emin ol (bkz. Bölüm 2).

## 4. Training ve Test Akışı

**Önemli:** Bu projede hazır bir train/test dosyası indirilmez. Sadece
`Airlines.csv` indirilir; train/validation/test split'i, model eğitimi ve
test değerlendirmesi tamamen `python scripts/train.py` çalıştırıldığında,
senin kendi bilgisayarında, kod tarafından gerçekleştirilir:

> **Training ve testing işlemleri hazır sonuçlar üzerinden değil,
> kullanıcının kendi bilgisayarında çalıştırdığı kod üzerinden
> gerçekleştirilir.**

### Çalıştırma

```bash
python scripts/train.py
```

Bu tek komut, sırasıyla aşağıdaki adımları otomatik olarak yapar:

1. **Veri yükleme** -- `data/raw/Airlines.csv` okunur.
2. **Data validation** -- beklenen kolonlar ve `Delay` değerleri doğrulanır.
3. **EDA (keşifsel veri analizi)** -- genel istatistikler ve grafikler
   üretilir.
4. **Train / Validation / Test split** -- %60 / %20 / %20 oranında,
   stratify edilmiş şekilde bölünür.
5. **Hyperparameter tuning** -- Random Forest için farklı `max_depth`
   değerleri validation set üzerinde denenir.
6. **Model training** -- Baseline, Logistic Regression ve Random Forest
   (en iyi `max_depth` ile) eğitilir.
7. **Test evaluation** -- eğitilen modeller, hiç görmedikleri test seti
   üzerinde değerlendirilir.
8. **Grafiklerin oluşturulması** -- `outputs/eda/` ve `outputs/evaluation/`
   altına kaydedilir.
9. **CSV raporlarının oluşturulması** -- tüm önemli sayısal sonuçlar
   `outputs/evaluation/` altına CSV olarak kaydedilir (bkz. Bölüm 7).
10. **Modelin kaydedilmesi** -- en iyi model (F1'e göre)
    `models/model.joblib` olarak kaydedilir.

### Split oranları ve gerçek satır sayıları

Gerçek `Airlines.csv` (539.383 satır) ile çalıştırıldığında split şu
şekilde gerçekleşir:

| Küme | Oran | Satır sayısı |
|---|---|---|
| Train | %60 | 323.629 |
| Validation | %20 | 107.877 |
| Test | %20 | 107.877 |

Bu sayılar `dataset_summary.csv` içine de otomatik olarak yazılır (bkz.
Bölüm 7) -- kodda hiçbir yerde hard-code edilmemiştir; her çalıştırmada
gerçek veriden yeniden hesaplanır.

## 5. Modeller

Proje üç modeli karşılaştırır:

1. **Baseline** -- her zaman en sık görülen sınıfı tahmin eden referans
   model (diğer modellerin gerçekten bir şey öğrenip öğrenmediğini ölçmek
   için).
2. **Logistic Regression**
3. **Random Forest**

Her model için test seti üzerinde accuracy, precision, recall, F1 ve
ROC-AUC hesaplanır.

## 6. Hyperparameter Tuning

Random Forest için `max_depth` hyperparametresi şu değerler denenerek
aranır:

```
max_depth ∈ {5, 8, 12, None}
```

Her `max_depth` değeri için model **validation set** üzerinde eğitilip
değerlendirilir; **Validation F1 skoruna göre** en iyi `max_depth` seçilir
ve final modeller bu değerle eğitilir. Diğer hyperparametreler
(`n_estimators` vb.) sabit tutulur. Denenen tüm değerler ve sonuçları
`hyperparameter_results.csv` dosyasına kaydedilir.

## 7. Outputs (Çıktılar)

`python scripts/train.py` çalıştırıldıktan sonra aşağıdaki dosyalar
**otomatik olarak** oluşturulur (klasörler yoksa otomatik oluşturulur,
elle oluşturmaya gerek yoktur):

```
outputs/eda/eda_overview.png
outputs/evaluation/
├── model_results.csv
├── hyperparameter_results.csv
├── dataset_summary.csv
├── feature_importance.csv
├── test_predictions.csv
├── confusion_matrices.png
├── model_comparison.png
├── roc_curve.png
└── feature_importance.png
```

| Dosya | İçerik |
|---|---|
| `eda_overview.png` | Genel gecikme dağılımı, havayoluna/saate/güne göre gecikme oranları |
| `model_results.csv` | Her model için accuracy, precision, recall, f1, auc (test seti) |
| `hyperparameter_results.csv` | Random Forest tuning sırasında denenen her `max_depth` için validation F1 |
| `dataset_summary.csv` | Toplam satır sayısı, train/validation/test satır sayıları, genel gecikme oranı |
| `feature_importance.csv` | Random Forest feature importance değerleri (büyükten küçüğe sıralı) |
| `test_predictions.csv` | **Test setindeki TÜM örneklerin** satır satır tahmin sonuçları -- her satır bir test uçuşuna karşılık gelir (`Airline`, `AirportFrom`, `AirportTo`, `DayOfWeek`, `Time`, `Length`, `actual`, `predicted`, `probability`); hiçbir test örneği atlanmaz, satır sayısı test seti satır sayısıyla birebir aynıdır (gerçek dataset ile 107.877 satır) |
| `confusion_matrices.png` | Her model için confusion matrix |
| `model_comparison.png` | Modellerin metrik bazında karşılaştırması |
| `roc_curve.png` | Modellerin ROC eğrileri |
| `feature_importance.png` | Random Forest feature importance grafiği |

Bu değerlerin hepsi her çalıştırmada gerçek dataset ve gerçek split
sonucundan otomatik hesaplanır; hiçbiri hard-code edilmemiştir.

## 8. Prediction

Eğitilmiş modeli (`models/model.joblib`) kullanarak **tek bir yeni uçuş**
için gecikme olasılığı tahmini yapabilirsin:

```bash
python scripts/predict.py --airline CO --from ATL --to JFK --day 3 --time 1200 --length 120
```

Parametreler:

| Argüman | Açıklama |
|---|---|
| `--airline` | Havayolu kodu (örn. `CO`, `AA`, `DL`) |
| `--from` | Kalkış havalimanı kodu (örn. `ATL`) |
| `--to` | Varış havalimanı kodu (örn. `JFK`) |
| `--day` | Haftanın günü (1-7) |
| `--time` | Planlanan kalkış saati (gece yarısından itibaren dakika) |
| `--length` | Planlanan uçuş süresi (dakika) |

`predict.py`, `models/model.joblib` içindeki kaydedilmiş model ve
preprocessing durumunu yükler, verdiğin uçuş bilgilerini aynı
transformasyonlardan geçirir ve **tek bir tahmin** üretir. Sonuç (gecikme
olasılığı ve etiket) terminale yazdırılır; ayrıca her çalıştırmada
`outputs/predictions.csv` dosyasına yeni bir satır eklenir (dosya yoksa
otomatik oluşturulur, varsa üzerine yazılmaz -- sadece satır eklenir).
Bunun için önce `python scripts/train.py` ile bir model eğitmiş olman
gerekir.

## 9. Testler

```bash
pytest -q
```

Testler; preprocessing, feature engineering, veri doğrulama,
hyperparameter tuning ve `outputs/evaluation/` altındaki tüm CSV
raporlama fonksiyonlarını (dosyaların oluşması, kolonların doğru olması,
satır sayılarının test setiyle eşleşmesi, prediction CSV'sinin satır
eklemesi) kapsar. Testler gerçek `Airlines.csv` dosyasına ihtiyaç
duymaz; küçük sentetik veriler üzerinde çalışır, bu yüzden dataset'i
indirmeden de çalıştırılabilir. Tüm testlerin geçmesi beklenir.

## 10. Notebook

```bash
jupyter notebook notebooks/workshop_walkthrough.ipynb
```

Notebook, `src/flight_delay` içindeki gerçek modülleri import edip
kullanır -- ayrı bir "deneme kodu" değildir. EDA'yı, feature
engineering'i ve canlı tahmini adım adım görmek için kullanılabilir.

## 11. Proje Yapısı

```
flight-delay-prediction/
├── data/
│   └── raw/                 -> Airlines.csv buraya konur (gitignored)
├── models/
│   └── model.joblib          -> eğitilmiş model (gitignored, train.py ile üretilir)
├── notebooks/
│   └── workshop_walkthrough.ipynb
├── outputs/
│   ├── eda/                  -> EDA grafikleri (gitignored)
│   ├── evaluation/           -> evaluation grafikleri + CSV raporları (gitignored)
│   └── predictions.csv       -> predict.py çıktıları (gitignored)
├── scripts/
│   ├── train.py               -> CLI: tüm training/evaluation pipeline'ı
│   └── predict.py              -> CLI: tek bir uçuş için tahmin
├── src/flight_delay/
│   ├── config.py               -> merkezi konfigürasyon (path'ler, sabitler)
│   ├── data_loading.py         -> ham veriyi diskten okuma
│   ├── data_validation.py      -> şema/veri doğrulama
│   ├── eda.py                  -> keşifsel veri analizi ve grafikler
│   ├── preprocessing.py        -> temizleme + encoding (Preprocessor sınıfı)
│   ├── feature_engineering.py  -> türetilmiş feature'lar
│   ├── modeling.py             -> model tanımları (baseline/LogReg/RF)
│   ├── training.py             -> split, hyperparameter tuning, eğitim
│   ├── evaluation.py           -> metrikler + grafikler
│   ├── reporting.py            -> tüm CSV raporlarının kaydedilmesi
│   └── predict.py              -> kaydedilmiş modelle tahmin
├── tests/                    -> birim testler (pytest)
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 12. Reproducibility

Projeyi sıfırdan, kendi bilgisayarında çalıştırmak için izlemen gereken
adımlar tam olarak şunlardır:

1. **Kodu clone et:**
   ```bash
   git clone <bu-repo-url>
   cd flight-delay-prediction
   ```
2. **Ortamı kur** (bkz. Bölüm 3): virtual environment oluştur,
   `pip install -r requirements.txt` ve `pip install -e .` çalıştır.
3. **Kaggle'dan `Airlines.csv` dosyasını indir** (bkz. Bölüm 2).
4. **İndirilen dosyayı `data/raw/Airlines.csv` olarak yerleştir.**
5. **`python scripts/train.py` çalıştır.**
6. Training ve testing tamamen **senin kendi bilgisayarında**
   gerçekleşir -- data validation, EDA, train/validation/test split,
   hyperparameter tuning, model eğitimi ve test değerlendirmesi hiçbir
   hazır/paylaşılmış sonuç kullanmadan, o an çalıştırdığın kod üzerinden
   üretilir.
7. **Kendi sonuçlarını `outputs/evaluation/` altında gör:** grafikler ve
   CSV raporları (`model_results.csv`, `hyperparameter_results.csv`,
   `dataset_summary.csv`, `feature_importance.csv`, `test_predictions.csv`)
   bu adımda senin verinden, senin makinende üretilmiş olur.
8. (Opsiyonel) `python scripts/predict.py ...` ile kendi eğittiğin
   modeli kullanarak yeni bir uçuş için tahmin yap.
9. (Opsiyonel) `pytest -q` ile proje bileşenlerinin doğru çalıştığını
   doğrula.

Bu proje hiçbir aşamada hazır/paylaşılmış bir sonucun indirilmesini
gerektirmez; tek indirilecek dosya `Airlines.csv`'dir, geri kalan her
şey (split, eğitim, değerlendirme, raporlar) `scripts/train.py` ile
senin bilgisayarında üretilir.

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
