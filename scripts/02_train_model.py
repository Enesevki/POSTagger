#!/usr/bin/env python3
"""
HMM Model Training Script

Bu script CoNLL-U formatındaki training verisini kullanarak
HMM (Hidden Markov Model) eğitir ve modeli kaydeder.

Usage:
    python scripts/02_train_model.py
    
Input:
    - data/processed/train.conllu
    - data/processed/dev.conllu (opsiyonel validation)
    
Output:
    - models/hmm_model.pkl
    - models/training_report.txt
"""

import os
import sys
import time
from datetime import datetime

# Proje core modüllerini import et
sys.path.append(os.path.abspath('.'))
from core import CoNLLUReader, HMMCounts, HMMModel

# Konfigürasyon
TRAIN_FILE = 'data/processed/train.conllu'
DEV_FILE = 'data/processed/dev.conllu'
MODEL_DIR = 'models/'
MODEL_FILE = f'{MODEL_DIR}/hmm_model.pkl'
REPORT_FILE = f'{MODEL_DIR}/training_report.txt'

# HMM Hiperparameterleri
SMOOTHING_ALPHA = 0.1        # Add-k smoothing parametresi
N_GRAM_ORDER = 2              # Bigram kullan (2-gram)
MIN_WORD_FREQ = 2             # OOV threshold
USE_MORPHOLOGY = True          # Türkçe morphology kullan


def check_input_files():
    """
    Gerekli input dosyalarının varlığını kontrol eder.
    
    Returns:
        tuple: (train_exists, dev_exists)
    """
    train_exists = os.path.exists(TRAIN_FILE)
    dev_exists = os.path.exists(DEV_FILE)
    
    print(f"📂 Input dosyaları kontrol ediliyor:")
    print(f"   {'✅' if train_exists else '❌'} Training: {TRAIN_FILE}")
    print(f"   {'✅' if dev_exists else '❌'} Dev (opsiyonel): {DEV_FILE}")
    
    if not train_exists:
        raise FileNotFoundError(f"Training dosyası bulunamadı: {TRAIN_FILE}")
    
    return train_exists, dev_exists


def load_training_data():
    """
    Training ve dev verilerini yükler ve istatistikleri gösterir.
    
    Returns:
        tuple: (train_reader, dev_reader)
    """
    print(f"\n📖 Training verisi yükleniyor: {TRAIN_FILE}")
    train_reader = CoNLLUReader(TRAIN_FILE)
    
    print(f"✅ Training verisi yüklendi:")
    print(f"   Cümle sayısı: {train_reader.get_sentence_count():,}")
    print(f"   Token sayısı: {train_reader.get_token_count():,}")
    print(f"   Kelime çeşitliliği: {len(train_reader.get_vocabulary()):,}")
    print(f"   POS tag çeşitliliği: {len(train_reader.get_tag_set()):,}")
    
    # Dev verisi varsa yükle
    dev_reader = None
    if os.path.exists(DEV_FILE):
        print(f"\n📖 Dev verisi yükleniyor: {DEV_FILE}")
        dev_reader = CoNLLUReader(DEV_FILE)
        print(f"✅ Dev verisi yüklendi:")
        print(f"   Cümle sayısı: {dev_reader.get_sentence_count():,}")
        print(f"   Token sayısı: {dev_reader.get_token_count():,}")
    
    return train_reader, dev_reader


def show_data_statistics(train_reader, dev_reader=None):
    """
    Veri istatistiklerini detaylı şekilde gösterir.
    
    Args:
        train_reader: Training data reader
        dev_reader: Dev data reader (opsiyonel)
    """
    print(f"\n📊 Veri İstatistikleri:")
    print(f"{'='*60}")
    
    # Training stats
    vocab = train_reader.get_vocabulary()
    tags = train_reader.get_tag_set()
    
    print(f"Training Set:")
    print(f"  Cümle sayısı    : {train_reader.get_sentence_count():,}")
    print(f"  Token sayısı    : {train_reader.get_token_count():,}")
    print(f"  Kelime çeşitliliği: {len(vocab):,}")
    print(f"  POS tag sayısı  : {len(tags):,}")
    
    # POS tag dağılımı
    print(f"\n🏷️  POS Tag Dağılımı (ilk 10):")
    tag_stats = train_reader.get_tag_statistics()
    for tag, count in sorted(tag_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
        percentage = (count / train_reader.get_token_count()) * 100
        print(f"   {tag:8s}: {count:6,d} (%{percentage:4.1f})")
    
    # En sık kelimeler
    print(f"\n🔤 En Sık Kelimeler (ilk 10):")
    word_stats = train_reader.get_word_statistics()
    for word, count in sorted(word_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
        percentage = (count / train_reader.get_token_count()) * 100
        print(f"   {word:15s}: {count:4,d} (%{percentage:4.1f})")
    
    # Dev set varsa
    if dev_reader:
        print(f"\nDev Set:")
        print(f"  Cümle sayısı    : {dev_reader.get_sentence_count():,}")
        print(f"  Token sayısı    : {dev_reader.get_token_count():,}")
        
        # OOV analizi
        dev_vocab = dev_reader.get_vocabulary()
        oov_words = dev_vocab - vocab
        oov_rate = len(oov_words) / len(dev_vocab) * 100
        print(f"  OOV kelime sayısı: {len(oov_words):,} (%{oov_rate:.1f})")


def train_hmm_model(train_reader):
    """
    HMM modelini eğitir.
    
    Args:
        train_reader: Training data reader
    
    Returns:
        HMMModel: Eğitilmiş model
    """
    print(f"\n🤖 HMM Model Eğitimi Başlıyor")
    print(f"{'='*50}")
    print(f"Hiperparametreler:")
    print(f"  Smoothing Alpha : {SMOOTHING_ALPHA}")
    print(f"  N-gram Order    : {N_GRAM_ORDER}")
    print(f"  Min Word Freq   : {MIN_WORD_FREQ}")
    print(f"  Morphology      : {'Evet' if USE_MORPHOLOGY else 'Hayır'}")
    
    start_time = time.time()
    
    # Model oluştur ve eğit
    model = HMMModel()
    
    print(f"\n🔢 N-gram sayımları hesaplanıyor...")
    model.train(train_reader, smoothing=SMOOTHING_ALPHA)
    
    training_time = time.time() - start_time
    print(f"✅ Model eğitimi tamamlandı ({training_time:.2f} saniye)")
    
    return model


def evaluate_on_dev(model, dev_reader):
    """
    Modeli dev set üzerinde değerlendirir.
    
    Args:
        model: Eğitilmiş HMM model
        dev_reader: Dev data reader
    
    Returns:
        dict: Değerlendirme sonuçları
    """
    if not dev_reader:
        return None
    
    print(f"\n🔍 Dev Set Üzerinde Değerlendirme")
    print(f"{'='*40}")
    
    total_tokens = 0
    correct_tags = 0
    
    start_time = time.time()
    
    first = True  # Sadece ilk cümle için detaylı çıktı
    for sentence_data in dev_reader:
        words = [token['form'] for token in sentence_data]
        true_tags = [token['upos'] for token in sentence_data]
        
        # Viterbi ile tahmin yap
        predicted_tags = model.viterbi_decode(words)
        
        # Uzunluk kontrolü ve örnek çıktı
        if len(words) != len(predicted_tags):
            print(f"UYARI: {len(words)} kelime, {len(predicted_tags)} tag")
        if first:
            print("\nİlk cümle için karşılaştırma:")
            for w, t, p in zip(words, true_tags, predicted_tags):
                print(f"{w:15s} | {t:8s} | {p:8s}")
            first = False
        
        # Doğruluğu hesapla
        total_tokens += len(words)
        correct_tags += sum(1 for true, pred in zip(true_tags, predicted_tags) if true == pred)
    
    eval_time = time.time() - start_time
    accuracy = (correct_tags / total_tokens) * 100
    
    results = {
        'total_tokens': total_tokens,
        'correct_tags': correct_tags,
        'accuracy': accuracy,
        'evaluation_time': eval_time
    }
    
    print(f"Total tokens: {total_tokens:,}")
    print(f"Correct tags: {correct_tags:,}")
    print(f"Accuracy    : {accuracy:.2f}%")
    print(f"Eval time   : {eval_time:.2f} seconds")
    
    return results


def save_model_and_report(model, train_reader, dev_reader=None, dev_results=None):
    """
    Modeli ve eğitim raporunu kaydeder.
    
    Args:
        model: Eğitilmiş HMM model
        train_reader: Training data reader
        dev_reader: Dev data reader
        dev_results: Dev değerlendirme sonuçları
    """
    # Model dizinini oluştur
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print(f"\n💾 Model kaydediliyor: {MODEL_FILE}")
    model.save(MODEL_FILE)
    print(f"✅ Model kaydedildi")
    
    # Training report oluştur
    print(f"📄 Eğitim raporu oluşturuluyor: {REPORT_FILE}")
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("HMM POS Tagger - Training Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Konfigürasyon
        f.write("Configuration:\n")
        f.write(f"  Smoothing Alpha    : {SMOOTHING_ALPHA}\n")
        f.write(f"  N-gram Order       : {N_GRAM_ORDER}\n")
        f.write(f"  Min Word Frequency : {MIN_WORD_FREQ}\n")
        f.write(f"  Turkish Morphology : {USE_MORPHOLOGY}\n\n")
        
        # Training data stats
        f.write("Training Data:\n")
        f.write(f"  File              : {TRAIN_FILE}\n")
        f.write(f"  Sentences         : {train_reader.get_sentence_count():,}\n")
        f.write(f"  Tokens            : {train_reader.get_token_count():,}\n")
        f.write(f"  Vocabulary Size   : {len(train_reader.get_vocabulary()):,}\n")
        f.write(f"  POS Tags          : {len(train_reader.get_tag_set()):,}\n")
        f.write(f"  Tags: {', '.join(sorted(train_reader.get_tag_set()))}\n\n")
        
        # Model stats
        model_stats = model.get_model_statistics()
        f.write("Model Statistics:\n")
        f.write(f"  Emission Entries  : {model_stats.get('emission_pairs', 0):,}\n")
        f.write(f"  Transition Entries: {model_stats.get('transition_pairs', 0):,}\n")
        f.write(f"  Vocabulary Size   : {model_stats.get('vocab_size', 0):,}\n")
        f.write(f"  Tag Set Size      : {model_stats.get('tag_count', 0):,}\n\n")
        
        # Dev results varsa
        if dev_results:
            f.write("Development Set Evaluation:\n")
            f.write(f"  File          : {DEV_FILE}\n")
            f.write(f"  Total Tokens  : {dev_results['total_tokens']:,}\n")
            f.write(f"  Correct Tags  : {dev_results['correct_tags']:,}\n")
            f.write(f"  Accuracy      : {dev_results['accuracy']:.2f}%\n")
            f.write(f"  Eval Time     : {dev_results['evaluation_time']:.2f}s\n\n")
        
        # Top POS tags
        tag_stats = train_reader.get_tag_statistics()
        f.write("POS Tag Distribution:\n")
        for tag, count in sorted(tag_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / train_reader.get_token_count()) * 100
            f.write(f"  {tag:8s}: {count:6,d} ({percentage:4.1f}%)\n")
    
    print(f"✅ Eğitim raporu kaydedildi")


def show_model_info(model):
    """
    Eğitilmiş model hakkında bilgi gösterir.
    
    Args:
        model: Eğitilmiş HMM model
    """
    print(f"\n📋 Model Bilgileri:")
    print(f"{'='*40}")
    
    stats = model.get_model_statistics()
    print(f"Emission parametreleri : {stats.get('emission_pairs', 0):,}")
    print(f"Transition parametreleri: {stats.get('transition_pairs', 0):,}")
    print(f"Vocabulary boyutu       : {stats.get('vocab_size', 0):,}")
    print(f"POS tag sayısı          : {stats.get('tag_count', 0):,}")
    
    # Dosya boyutu
    if os.path.exists(MODEL_FILE):
        model_size = os.path.getsize(MODEL_FILE) / 1024  # KB
        print(f"Model dosya boyutu      : {model_size:.1f} KB")


def main():
    """
    Ana eğitim fonksiyonu.
    """
    print("🚀 HMM Model Eğitimi Başlıyor")
    print("=" * 50)
    
    try:
        # 1. Input dosyalarını kontrol et
        train_exists, dev_exists = check_input_files()
        
        # 2. Veriyi yükle
        train_reader, dev_reader = load_training_data()
        
        # 3. Veri istatistiklerini göster
        show_data_statistics(train_reader, dev_reader)
        
        # 4. Modeli eğit
        model = train_hmm_model(train_reader)
        
        # 5. Dev set'te değerlendir (varsa)
        dev_results = evaluate_on_dev(model, dev_reader) if dev_reader else None
        
        # 6. Model ve raporu kaydet
        save_model_and_report(model, train_reader, dev_reader, dev_results)
        
        # 7. Model bilgilerini göster
        show_model_info(model)
        
        print(f"\n🎉 Model eğitimi başarıyla tamamlandı!")
        print(f"Model dosyası: {MODEL_FILE}")
        print(f"Rapor dosyası: {REPORT_FILE}")
        print(f"\nSonraki adım: python scripts/03_test_model.py")
        
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main() 