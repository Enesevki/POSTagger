# CRF Tabanlı Türkçe Cümle Öğesi Etiketleme (POS Tagging)

[![CI](https://github.com/Enesevki/POSTagger/actions/workflows/ci.yml/badge.svg)](https://github.com/Enesevki/POSTagger/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Bu proje, Türkçe metinler için **Koşullu Rastgele Alanlar (Conditional Random Field - CRF)** tabanlı bir Cümle Öğesi Etiketleme (Part-of-Speech Tagging) modeli geliştirmek, eğitmek ve analiz etmek için kapsamlı bir araç seti sunar. Projenin temel amacı, bir kelime dizisi verildiğinde her kelimeye ait dilbilgisel rolü (`NOUN`, `VERB`, `ADJ` vb.) doğru bir şekilde atamaktır.

CRF, özellikle POS tagging gibi sıralı veri etiketleme görevleri için güçlü bir modeldir çünkü bir kelimenin etiketini tahmin ederken sadece o kelimenin özelliklerine değil, aynı zamanda çevre kelimelerin etiketlerine ve özelliklerine de bakar. Bu sayede cümlenin bütünündeki bağlamı daha etkili bir şekilde yakalar.

<br>

## Canlı Demo ve Tanıtım Videosu
- Kendi cümlelerinize POS Tagging uygulamak isterseniz geliştirdiğimiz site üzerinden projeye ulaşabilirsinşz: https://turkishpostagger.streamlit.app
- YouTube videomuzda projenin arka planını detaylı bir şekilde açıkladık: https://www.youtube.com/watch?v=AQlYbdYh_bM

## ✨ Temel Özellikler

Bu proje, bir makine öğrenmesi modelinin yaşam döngüsündeki kritik adımları yönetmek için tasarlanmış modüler araçlar içerir:

-   **Veri Yönetimi**: Ham `.conll` formatındaki veriyi okuma, yazma ve standart Eğitim/Geliştirme/Test setlerine ayırma.
-   **Özelleştirilebilir Model Eğitimi**: L1 (`c1`) ve L2 (`c2`) düzenlileştirme gibi kritik hiperparametreleri ayarlayarak modeller eğitme.
-   **Kapsamlı Hiperparametre Değerlendirmesi**: K-katmanlı Çapraz Doğrulama (K-Fold Cross-Validation) ile bir model mimarisinin genelleme performansını güvenilir bir şekilde ölçme.
-   **Öğrenme Potansiyeli Analizi**: Öğrenme ve Hata Eğrileri (`learning_curve`, `error_curve`) çizerek modelin veri miktarına olan duyarlılığını, aşırı/eksik öğrenme (overfitting/underfitting) eğilimlerini analiz etme.
-   **Derinlemesine Nihai Model Analizi**: Eğitilmiş bir modelin "karnesini" çıkarma:
    -   Detaylı Sınıflandırma Raporları (Precision, Recall, F1-score).
    -   Görselleştirilmiş Karmaşıklık Matrisleri (Confusion Matrix).
    -   Sınıflandırma güvenini ölçen ROC Eğrileri ve AUC skorları.
    -   Modelin "beynini" anlamak için en önemli özellik ve geçişlerin listesi.
    -   Hatalı tahmin edilen örnekler için detaylı bir rapor.

<br>

## 📂 Proje Mimarisi ve Dosya Açıklamaları

Proje, her birinin net bir sorumluluğu olan betik ve modüllerden oluşur. Bu yapı, kodun anlaşılabilirliğini ve bakımını kolaylaştırır.

```
pos_tagger_crf/
├── data/
│   ├── raw/
│   │   └── stanza_tagged_output_3200     # İşlenmemiş tüm verinin birleşimi
│   │   └── stanza_tagged_output_1600
│   └── processed/
│       ├── train.conll         # Eğitim seti
│       ├── dev.conll           # Geliştirme/doğrulama seti
│       └── test.conll          # Test seti
│       └── test_300Ssentences.txt  # 300 cümlelik Test seti
│
├── outputs/
│   ├── training_analysis/      # Öğrenme eğrisi gibi eğitim analizi çıktıları
│   ├── model_analysis/         # Nihai model analizinin çıktıları
│   ├── diagnose_v2/            # Modeli script içinde üretip kaydetmeden değerlendirme yapmak için
│   └── models/
│       └── crf_final.pkl       # Eğitilmiş ve kaydedilmiş model
│
├── src/
│   ├── data_loader.py          # Veri okuma, yazma ve bölme işlemleri
│   ├── features.py             # Cümlelerden özellik çıkarma mantığı
│   ├── model.py                # CRF modelini oluşturma ve eğitme fonksiyonları
│   ├── utils.py                # Yardımcı fonksiyonlar (JSON/pickle kaydetme vb.)
│   ├── train.py                # Model eğitimi ve hiperparametre değerlendirme betiği
│   └── analyze_model.py        # Eğitilmiş bir modelin detaylı analizi için betik
│   └── diagnose_v2.py          # Modeli kaydetmeden kendi içinde eğiterek anlık detaylı analiz için betik
│
├── requirements.txt            # Proje bağımlılıkları
└── README.md                   # Bu dosya
```

| Dosya Adı          | Açıklama                                                                                                                                              |
| :----------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data_loader.py`   | `.conll` formatındaki veriyi okur, yazar ve train/dev/test setlerine böler. Bu betik, veri hazırlığı için bir komut satırı aracı olarak kullanılabilir. |
| `features.py`      | Bir cümledeki kelimelerden CRF modelinin kullanacağı özellikleri (`word shape`, `suffix`, `prefix`, bağlam vb.) çıkaran fonksiyonları içerir.              |
| `model.py`         | `sklearn-crfsuite` kütüphanesini kullanarak CRF modelini inşa etme, eğitme ve yükleme gibi temel modelleme fonksiyonlarını barındırır.                  |
| `utils.py`         | JSON/Pickle dosyalarını kaydetme/yükleme, klasör oluşturma gibi genel yardımcı fonksiyonları içerir.                                                   |
| **`train.py`** | **Model Geliştirme Betiği.** Hiperparametreleri test etmek (CV), öğrenme eğrilerini çizmek ve nihai modeli eğitmek için kullanılır.                      |
| **`analyze_model.py`** | **Model Analiz Betiği.** Önceden eğitilmiş bir `.pkl` modelinin performansını derinlemesine analiz etmek ve raporlamak için kullanılır.                   |
| **`diagnose_v2.py`** | **Model Tanılama Betiği.** Model kaydetmeden betik içinde eğitilip performansını derinlemesine analiz etmek ve raporlamak için kullanılır.                   |
| `requirements.txt` | Projenin çalışması için gereken Python kütüphanelerini listeler.                                                                                      |

<br>

## 💾 Veri Seti

Bu projede kullanılan veri seti, yaklaşık 3200 cümleden oluşmaktadır. Bu veri setinin yarısı (1600 cümle) proje kapsamında tarafımızca manuel olarak hazırlanmıştır. Diğer yarısı ise iki farklı akademik veri setinin birleşiminden elde edilmiştir:

1.  **English/Turkish Wikipedia Named-Entity Recognition and Text Categorization Dataset**: Türkçe Wikipedia'dan otomatik olarak etiketlenmiş cümlelerden oluşan bir koleksiyon.
    > *Şahin, H. Bahadır; Eren, Mustafa Tolga; Tırkaz, Çağlar; Sönmez, Ozan; Yıldız, Eray (2017), “English/Turkish Wikipedia Named-Entity Recognition and Text Categorization Dataset”, Mendeley Data, V1, doi: 10.17632/cdcztymf4k.1*

2.  **Compilation of Bilkent Turkish Writings Dataset**: Bilkent Üniversitesi'nin Türkçe 101 ve 102 derslerinde 2014'ten bu yana öğrenciler tarafından oluşturulan yaratıcı yazı metinlerinden oluşan bir derleme.
    > *Yılmaz, Selim F. (2025), “Compilation of Bilkent Turkish Writings Dataset”, Zenodo, V2, doi: 10.5281/zenodo.15498155*

<br>

## 🚀 Kurulum

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin.

1.  **Sanal Ortam Oluşturun (Önerilir)**: Proje bağımlılıklarını sisteminizdeki diğer paketlerden izole etmek için bir sanal ortam oluşturun.
    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux için
    # venv\Scripts\activate    # Windows için
    ```

2.  **Bağımlılıkları Yükleyin**: Projenin ana dizinindeyken `requirements.txt` dosyasını kullanarak gerekli tüm kütüphaneleri yükleyin.
    ```bash
    pip install -r requirements.txt
    ```

<br>

## 🛠️ Kullanım ve Adım Adım İş Akışı

Bu proje, en iyi sonuçları elde etmek için yapılandırılmış bir iş akışı sunar.

### Adım 1: Veri Hazırlama

Bu adım, projenin en başında yalnızca bir kez yapılır. Tüm verinizi içeren büyük bir `.conll` dosyasını, modelin eğitimi ve değerlendirmesi için standart setlere ayırın.

```bash
python -m src.data_loader --input data/raw/all_data.conll --outdir data/processed
```
> Bu komut, `data/processed/` klasörüne `train.conll`, `dev.conll` ve `test.conll` dosyalarını oluşturur. **Test seti (`test.conll`) modelin performansını en son ölçeceğimiz, dokunulmaz veridir.**

### Adım 2: Model Geliştirme ve Hiperparametre Optimizasyonu

Bu aşamada `train.py` betiğini kullanarak model mimarinizi test eder ve en iyi hiperparametreleri ararsınız.

**a) Hızlı Performans Değerlendirmesi (Cross-Validation)**

Farklı `c1` ve `c2` değerlerinin genelleme performansı üzerindeki etkisini hızlıca görmek için `--mode evaluate_hparams` kullanın.

```bash
# c1=2.0 ve c2=2.0 değerlerini 5-katmanlı CV ile test et
python -m src.train --mode evaluate_hparams --data data/processed/train.conll --c1 2.0 --c2 2.0
```
> Bu komut, verilen hiperparametreler için K-katmanlı Çapraz Doğrulama çalıştırır ve konsola ortalama F1 skorunu basar. Bu işlemi farklı `c1`, `c2` değerleri için tekrarlayarak en iyi kombinasyonu bulmaya çalışın.

**b) Aşırı/Eksik Öğrenme Analizi (Learning Curve)**

Seçtiğiniz bir hiperparametre setinin davranışını daha detaylı anlamak için öğrenme eğrilerini çizin.

```bash
# c1=2.0 ve c2=2.0 için öğrenme ve hata eğrilerini çiz
python -m src.train --mode learning_curve --data data/processed/train.conll --c1 2.0 --c2 2.0
```
> Bu komut, `outputs/training_analysis` klasörüne grafikler oluşturur. **Grafikteki eğitim ve doğrulama çizgileri arasındaki büyük bir fark, aşırı öğrenmeye (overfitting) işaret eder.**

### Adım 3: Nihai Modelin Eğitimi

En iyi performansı verdiğine karar kıldığınız hiperparametrelerle nihai modelinizi eğitmek için `--mode train` kullanın. Bu komut, eğitim verisinin tamamını kullanarak tek bir model eğitir ve kaydeder.

```bash
python -m src.train --mode train --data data/processed/train.conll --out-model outputs/models/crf_final.pkl --c1 2.0 --c2 2.0 --verbose
```
> Bu komutun çıktısı, `outputs/models/crf_final.pkl` dosyasında saklanan, projenizin nihai ürünüdür.

### Adım 4: Nihai Modelin Kapsamlı Analizi

Artık eğitilmiş bir modeliniz var. `analyze_model.py` ile bu modelin "karnesini" çıkarın. Bu betik, **sadece ve sadece** `--model` ile verdiğiniz `.pkl` dosyasını analiz eder.

```bash
python -m src.analyze_model --model outputs/models/crf_final.pkl --train data/processed/train.conll --test data/processed/test.conll --output-dir outputs/final_model_analysis
```

Bu komutun sonucunda:
* **Konsolda** modelin genel parametrelerini, öğrendiği en önemli kuralları (geçişler ve özellikler) ve hem eğitim hem de test setleri için detaylı performans raporlarını görürsünüz.
* **`outputs/final_model_analysis` klasöründe** ise bu analizlerin kaydedilmiş hallerini (JSON raporları, PNG grafikleri, CSV hata listesi) bulabilirsiniz.

<br>

## 📈 Gelecek Çalışmalar ve İyileştirme Fırsatları

Bu proje sağlam bir temel sunsa da, performansı daha da artırmak için potansiyel iyileştirme alanları mevcuttur:

* **Gelişmiş Özellik Mühendisliği**: Türkçe'nin morfolojik zenginliğinden faydalanmak için `features.py`'a kelimenin kökü (lemma), hal ekleri, iyelik ekleri gibi daha detaylı dilbilimsel özellikler eklenebilir.
* **Hiperparametre Optimizasyon Araçları**: En iyi `c1` ve `c2` değerlerini manuel olarak aramak yerine `Optuna` veya `Scikit-learn`'in `GridSearchCV` gibi araçlarını entegre ederek bu süreci otomatikleştirmek.
* **Farklı Algoritmalar**: `sklearn-crfsuite`'in sunduğu farklı optimizasyon algoritmalarını (`lbfgs` yerine `l2sgd`, `ap`, `pa` gibi) denemek.
* **Kelime Vektörleri (Word Embeddings)**: Özellik setine `Word2Vec` veya `FastText` gibi önceden eğitilmiş kelime vektörlerini dahil ederek modelin kelime anlamlarını daha iyi yakalamasını sağlamak.
