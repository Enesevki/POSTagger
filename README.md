# HMM + Viterbi Tabanlı Türkçe POS-Tagger

**Okul Projesi** — 1200 cümlelik dataset, Stanza ile etiketleme, temiz/modüler Python kodu

## 📋 Proje Hakkında

Bu proje, Hidden Markov Models (HMM) ve Viterbi algoritması kullanarak Türkçe metinlerde Part-of-Speech (POS) etiketleme yapmaktadır. Stanza kütüphanesi ile önceden etiketlenmiş veriler kullanılarak HMM modeli eğitilir ve Viterbi algoritması ile decode işlemi gerçekleştirilir.

## 🚀 Hızlı Başlangıç

### 1. Ortam Kurulumu
```bash
# Virtual environment oluştur
python -m venv pos_tagger_env
source pos_tagger_env/bin/activate  # Linux/Mac
# pos_tagger_env\Scripts\activate  # Windows

# Paketleri yükle
pip install -r requirements.txt
```

### 2. Adım Adım Çalıştırma (IDE'den)
1. **`scripts/01_preprocess_data.py`** → Excel'i işle, Stanza ile etiketle
2. **`scripts/02_train_model.py`** → HMM modelini eğit  
3. **`scripts/03_test_model.py`** → Test setinde tahmin yap
4. **`scripts/04_evaluate.py`** → Performansı değerlendir
5. **`scripts/05_interactive_demo.py`** → İnteraktif test

### 3. Tek Seferde Çalıştırma
```bash
python run_all_pipeline.py
```

### 4. Web Demo
```bash
python web/app.py
# Tarayıcıda http://localhost:5000
```

## 📁 Proje Yapısı

```
pos_tagger_tr/
├── data/
│   ├── raw/                 # Ham Excel verisi
│   ├── processed/           # CoNLL-U format veriler
│   └── results/             # Sonuçlar ve raporlar
├── core/                    # Ana modüller
│   ├── corpus.py            # CoNLL-U reader
│   ├── counts.py            # HMM sayımları
│   ├── model.py             # HMM model
│   └── viterbi.py           # Viterbi decoder
├── scripts/                 # Çalıştırılabilir scriptler
├── models/                  # Eğitilmiş modeller
├── web/                     # Web demo
│   ├── templates/           # HTML şablonları
│   ├── static/              # CSS, JS dosyaları
│   └── app.py               # Flask uygulaması
└── run_all_pipeline.py      # Tam pipeline script
```

## 🎯 Özellikler

- **HMM Tabanlı Modelleme**: Geçiş ve emisyon olasılıklarıyla
- **Viterbi Algoritması**: Optimal etiket dizisi bulma
- **Smoothing**: Sparse data problemi için
- **OOV Handling**: Bilinmeyen kelimeler için Türkçe suffix analizi
- **Web Demo**: Basit Flask arayüzü
- **Modüler Kod**: Temiz ve genişletilebilir yapı

## 📊 Beklenen Performans

- **Accuracy**: ~88-92%
- **Macro F1**: ~0.85+
- **OOV Accuracy**: ~70-80%

## 🔧 Kullanılan Teknolojiler

- **Python 3.8+**
- **Stanza**: POS etiketleme için
- **scikit-learn**: Değerlendirme metrikleri
- **Flask**: Web demo
- **pandas**: Veri işleme

## 📝 Lisans

Bu proje eğitim amaçlı olarak geliştirilmiştir. 