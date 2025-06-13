#!/usr/bin/env python3
"""
HMM Model Testing Script

Bu script eğitilmiş HMM modelini yükleyerek Türkçe cümleleri 
POS tagging yapar. Hem tekli hem de batch test modu sunar.

Usage:
    python scripts/03_test_model.py
    
Input:
    - models/hmm_model.pkl (eğitilmiş model)
    - Test cümleleri (script içinde veya kullanıcıdan)
    
Output:
    - POS tag tahminleri
    - Güven skorları
    - Alternatif tahminler (k-best)
"""

import os
import sys
import time
import stanza

# Proje core modüllerini import et
sys.path.append(os.path.abspath('.'))
from core import HMMModel, ViterbiDecoder

# Konfigürasyon
MODEL_FILE = 'models/hmm_model.pkl'
TEST_SENTENCES_FILE = 'data/processed/test.conllu'  # Opsiyonel
K_BEST = 3                    # En iyi K tahmin
SHOW_PROBABILITIES = True     # Olasılık skorlarını göster
ENABLE_DEBUG = False          # Debug modunu aktif et

# Test cümleleri (model yoksa bunlar kullanılır)
SAMPLE_SENTENCES = [
    "Bu kitap çok güzel.",
    "Ankara Türkiye'nin başkentidir.",
    "Öğrenciler okula gitti.",
    "Hava bugün çok soğuk.",
    "Ben eve gidiyorum.",
    "Kedi bahçede oynuyor.",
    "Televizyonda haber izliyoruz.",
    "Merhaba nasılsın?",
    "Türkçe öğrenmek istiyorum.",
    "Su içmek sağlıklıdır."
]

# Stanza pipeline'ı başlat (tokenizer için)
stanza_tokenizer = stanza.Pipeline('tr', processors='tokenize', use_gpu=False, verbose=False)

def stanza_tokenize(sentence):
    """
    Stanza ile Türkçe tokenizasyonu yapar.
    Args:
        sentence (str): Cümle
    Returns:
        list: Token listesi
    """
    # Cümle sonu noktalama yoksa ekle
    if not sentence.strip().endswith(('.', '!', '?')):
        sentence = sentence.strip() + '.'
    doc = stanza_tokenizer(sentence)
    tokens = []
    for sent in doc.sentences:
        tokens.extend([word.text for word in sent.words])
    return tokens

def load_model():
    """
    Eğitilmiş HMM modelini yükler.
    
    Returns:
        HMMModel: Yüklenmiş model
    """
    print(f"📂 Model yükleniyor: {MODEL_FILE}")
    
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {MODEL_FILE}")
    
    try:
        model = HMMModel.load(MODEL_FILE)
        print(f"✅ Model başarıyla yüklendi")
        
        # Model bilgileri
        stats = model.get_model_statistics()
        print(f"   Vocabulary: {stats.get('vocab_size', 0):,} kelime")
        print(f"   POS Tags  : {stats.get('tag_count', 0):,} tag")
        print(f"   Model Date: {model.metadata.get('training_date', 'N/A')}")
        
        return model
        
    except Exception as e:
        raise RuntimeError(f"Model yüklenemedi: {e}")


def preprocess_sentence(sentence):
    """
    Cümleyi temizler ve tokenize eder (Stanza ile).
    """
    return stanza_tokenize(sentence)


def tag_sentence(model, sentence, k_best=K_BEST, show_probs=SHOW_PROBABILITIES):
    """
    Tek bir cümleyi POS tagging yapar.
    
    Args:
        model: HMM model
        sentence (str): Tag'lenecek cümle
        k_best (int): En iyi K tahmin
        show_probs (bool): Olasılıkları göster
    
    Returns:
        dict: Tagging sonuçları
    """
    # Cümleyi hazırla
    words = preprocess_sentence(sentence)
    
    if not words:
        return {'error': 'Boş cümle'}
    
    start_time = time.time()
    
    try:
        # Viterbi ile en iyi tahmin
        best_tags = model.viterbi_decode(words)
        
        # K-best tahminler (eğer destekleniyorsa)
        k_best_results = []
        if hasattr(model, 'viterbi_k_best') and k_best > 1:
            try:
                k_best_results = model.viterbi_k_best(words, k_best)
            except:
                k_best_results = [(best_tags, 0.0)]  # Fallback
        else:
            k_best_results = [(best_tags, 0.0)]
        
        processing_time = time.time() - start_time
        
        results = {
            'sentence': sentence,
            'words': words,
            'best_tags': best_tags,
            'k_best': k_best_results,
            'processing_time': processing_time,
            'word_count': len(words)
        }
        
        return results
        
    except Exception as e:
        return {'error': f'Tagging hatası: {e}'}


def display_tagging_results(results):
    """
    Tagging sonuçlarını güzel bir formatta gösterir.
    
    Args:
        results (dict): tag_sentence() sonucu
    """
    if 'error' in results:
        print(f"❌ {results['error']}")
        return
    
    print(f"\n🔤 Cümle: {results['sentence']}")
    print(f"⏱️  İşlem süresi: {results['processing_time']:.3f} saniye")
    print(f"📊 Kelime sayısı: {results['word_count']}")
    print(f"{'='*60}")
    
    # Ana tagging sonucu
    words = results['words']
    tags = results['best_tags']
    
    print(f"🏷️  POS Tagging Sonucu:")
    for i, (word, tag) in enumerate(zip(words, tags), 1):
        print(f"   {i:2d}. {word:15s} → {tag}")
    
    # K-best sonuçları (eğer varsa ve birden fazlaysa)
    if len(results['k_best']) > 1:
        print(f"\n🔍 Alternatif Tahminler (Top {len(results['k_best'])}):")
        for rank, (tag_sequence, prob) in enumerate(results['k_best'], 1):
            tag_str = ' '.join(tag_sequence)
            if SHOW_PROBABILITIES and prob != 0.0:
                print(f"   {rank}. {tag_str} (score: {prob:.4f})")
            else:
                print(f"   {rank}. {tag_str}")


def test_sample_sentences(model):
    """
    Önceden tanımlı örnek cümleleri test eder.
    
    Args:
        model: HMM model
    """
    print(f"\n🧪 Örnek Cümleler Test Ediliyor ({len(SAMPLE_SENTENCES)} cümle)")
    print(f"{'='*70}")
    
    total_time = 0
    total_words = 0
    
    for i, sentence in enumerate(SAMPLE_SENTENCES, 1):
        print(f"\n--- Test {i}/{len(SAMPLE_SENTENCES)} ---")
        
        results = tag_sentence(model, sentence)
        display_tagging_results(results)
        
        if 'processing_time' in results:
            total_time += results['processing_time']
            total_words += results['word_count']
    
    # Özet istatistikler
    if total_words > 0:
        avg_time_per_word = total_time / total_words
        avg_time_per_sentence = total_time / len(SAMPLE_SENTENCES)
        
        print(f"\n📊 Test Özeti:")
        print(f"   Toplam cümle: {len(SAMPLE_SENTENCES)}")
        print(f"   Toplam kelime: {total_words}")
        print(f"   Toplam süre: {total_time:.3f} saniye")
        print(f"   Cümle başına: {avg_time_per_sentence:.3f} saniye")
        print(f"   Kelime başına: {avg_time_per_word:.3f} saniye")


def test_from_file(model, file_path):
    """
    Test dosyasından cümleleri okuyup test eder.
    
    Args:
        model: HMM model
        file_path (str): CoNLL-U test dosyası
    """
    if not os.path.exists(file_path):
        print(f"⚠️  Test dosyası bulunamadı: {file_path}")
        return
    
    print(f"\n📁 Test dosyasından cümleler test ediliyor: {file_path}")
    
    try:
        # Core modulünü kullanarak CoNLL-U dosyasını oku
        from core.corpus import CoNLLUReader
        reader = CoNLLUReader(file_path)
        
        sentence_count = 0
        total_accuracy = 0
        total_tokens = 0
        
        print(f"Dosyada {reader.get_sentence_count()} cümle bulundu")
        print(f"İlk 5 cümle test ediliyor...")
        
        for sentence_data in reader:
            if sentence_count >= 5:  # İlk 5 cümleyi test et
                break
                
            # Cümle verisini hazırla
            words = [token['form'] for token in sentence_data]
            true_tags = [token['upos'] for token in sentence_data]
            sentence_text = ' '.join(words)
            
            print(f"\n--- Test Dosyası Cümle {sentence_count + 1} ---")
            print(f"🔤 Orijinal: {sentence_text}")
            print(f"🎯 Gerçek taglar: {' '.join(true_tags)}")
            
            # Model ile tahmin yap
            results = tag_sentence(model, sentence_text, k_best=1, show_probs=False)
            
            if 'error' not in results:
                predicted_tags = results['best_tags']
                print(f"🤖 Tahmin edilen: {' '.join(predicted_tags)}")
                
                # Accuracy hesapla
                correct = sum(1 for true, pred in zip(true_tags, predicted_tags) if true == pred)
                accuracy = (correct / len(true_tags)) * 100
                print(f"✅ Doğruluk: {correct}/{len(true_tags)} (%{accuracy:.1f})")
                
                total_accuracy += correct
                total_tokens += len(true_tags)
            else:
                print(f"❌ {results['error']}")
            
            sentence_count += 1
        
        # Genel accuracy
        if total_tokens > 0:
            overall_accuracy = (total_accuracy / total_tokens) * 100
            print(f"\n📊 Genel Test Sonucu:")
            print(f"   Test edilen cümle: {sentence_count}")
            print(f"   Toplam token: {total_tokens}")
            print(f"   Doğru tahmin: {total_accuracy}")
            print(f"   Genel doğruluk: %{overall_accuracy:.2f}")
            
    except Exception as e:
        print(f"❌ Test dosyası okunamadı: {e}")


def interactive_test(model):
    """
    Kullanıcıdan cümle alarak interaktif test yapar.
    
    Args:
        model: HMM model
    """
    print(f"\n🎮 İnteraktif Test Modu")
    print(f"{'='*50}")
    print(f"Türkçe cümleler girin (çıkmak için 'quit' veya 'exit'):")
    print(f"Örnek: Bu kitap çok güzel.")
    
    while True:
        try:
            # Kullanıcıdan input al
            user_input = input(f"\n📝 Cümle: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'çık', 'q']:
                print(f"👋 İnteraktif test sonlandırıldı")
                break
            
            if not user_input:
                print(f"⚠️  Lütfen bir cümle girin")
                continue
            
            # Test et
            results = tag_sentence(model, user_input)
            display_tagging_results(results)
            
        except KeyboardInterrupt:
            print(f"\n\n👋 İnteraktif test sonlandırıldı (Ctrl+C)")
            break
        except Exception as e:
            print(f"❌ Hata: {e}")


def show_model_capabilities(model):
    """
    Modelin yetenekleri hakkında bilgi gösterir.
    
    Args:
        model: HMM model
    """
    print(f"\n🔧 Model Yetenekleri:")
    print(f"{'='*40}")
    
    # Tag set'i göster
    if hasattr(model, 'counts') and hasattr(model.counts, 'tag_set'):
        tags = sorted(model.counts.tag_set)
        print(f"🏷️  Desteklenen POS Tagları ({len(tags)} adet):")
        # Tagları satır satır göster (8'li gruplar)
        for i in range(0, len(tags), 8):
            tag_group = tags[i:i+8]
            print(f"   {' '.join(f'{tag:8s}' for tag in tag_group)}")
    
    # Özellikler
    print(f"\n🔍 Model Özellikleri:")
    print(f"   ✅ Viterbi algoritması")
    print(f"   ✅ OOV kelime desteği")
    print(f"   ✅ Türkçe morphology")
    print(f"   ✅ Smoothing (Add-k)")
    
    if hasattr(model, 'viterbi_k_best'):
        print(f"   ✅ K-best tahminler")
    else:
        print(f"   ❌ K-best tahminler")


def main():
    """
    Ana test fonksiyonu.
    """
    print("🧪 HMM Model Test Başlıyor")
    print("=" * 50)
    
    try:
        # 1. Modeli yükle
        model = load_model()
        
        # 2. Model yeteneklerini göster
        show_model_capabilities(model)
        
        # 3. Örnek cümleleri test et
        test_sample_sentences(model)
        
        # 4. Test dosyası varsa test et
        if os.path.exists(TEST_SENTENCES_FILE):
            test_from_file(model, TEST_SENTENCES_FILE)
        
        # 5. İnteraktif test
        print(f"\n" + "="*70)
        interactive_test(model)
        
        print(f"\n🎉 Test tamamlandı!")
        print(f"Sonraki adım: python scripts/04_evaluate.py")
        
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main() 