#!/usr/bin/env python3
"""
HMM POS Tagger - Interactive Demo

Bu script eğitilmiş HMM modelini kullanarak interaktif bir
POS tagging demo'su sunar. Çeşitli test modları ve analiz 
seçenekleri içerir.

Usage:
    python scripts/05_interactive_demo.py
    
Input:
    - models/hmm_model.pkl (eğitilmiş model)
    
Features:
    - Tek cümle tagging
    - Batch tagging
    - Model analizi
    - Performance benchmarking
    - Örnek cümleler
"""

import os
import sys
import time
import json
from datetime import datetime
import stanza

# Proje core modüllerini import et
sys.path.append(os.path.abspath('.'))
from core import HMMModel

# Konfigürasyon
MODEL_FILE = 'models/hmm_model.pkl'
DEMO_LOG_FILE = 'models/demo_session.log'

# Örnek Türkçe cümleler (çeşitli zorluk seviyeleri)
SAMPLE_SENTENCES = {
    'Basit': [
        "Ben eve gidiyorum.",
        "Kedi uyuyor.",
        "Su içtim.",
        "Okul büyük.",
        "Kitap masa üzerinde."
    ],
    'Orta': [
        "Bugün hava çok güzel.",
        "Öğrenciler derse katılmıyor.",
        "Türkiye'nin başkenti Ankara'dır.",
        "Pazartesi günü işe gitmek zorundayım.",
        "Annem bana hediye almış."
    ],
    'Karmaşık': [
        "Yapay zeka teknolojilerinin gelişmesiyle birlikte doğal dil işleme alanında önemli ilerlemeler kaydedilmektedir.",
        "Üniversitedeki araştırmacılar yeni bir algoritma geliştirerek bu problemi çözmeye çalışıyorlar.",
        "Küreselleşmenin etkisiyle ekonomik dengeler değişirken, toplumsal yapılar da bu durumdan etkilenmektedir.",
        "Bilgisayar mühendisliği öğrencilerinin programlama becerilerini geliştirmeleri için sürekli pratik yapmaları gerekmektedir.",
        "Çevre kirliliğinin artması nedeniyle yenilenebilir enerji kaynaklarına olan ilgi her geçen gün artmaktadır."
    ]
}

# Demo seçenekleri
DEMO_OPTIONS = {
    '1': 'Tek Cümle Tagging',
    '2': 'Batch Tagging',
    '3': 'Örnek Cümleler Testi',
    '4': 'Model Analizi',
    '5': 'Performance Benchmark',
    '6': 'Model İstatistikleri',
    '7': 'Hakkında',
    'q': 'Çıkış'
}

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

class DemoSession:
    """Demo oturumu için log ve istatistik tutma."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.tagged_sentences = 0
        self.total_tokens = 0
        self.total_time = 0.0
        self.session_log = []
    
    def log_tagging(self, sentence, tags, processing_time):
        """Tagging işlemini logla."""
        self.tagged_sentences += 1
        self.total_tokens += len(tags)
        self.total_time += processing_time
        
        log_entry = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'sentence': sentence,
            'tags': tags,
            'processing_time': processing_time,
            'token_count': len(tags)
        }
        self.session_log.append(log_entry)
    
    def get_stats(self):
        """Oturum istatistiklerini döndür."""
        duration = (datetime.now() - self.start_time).total_seconds()
        avg_time_per_sentence = self.total_time / self.tagged_sentences if self.tagged_sentences > 0 else 0
        avg_time_per_token = self.total_time / self.total_tokens if self.total_tokens > 0 else 0
        
        return {
            'session_duration': duration,
            'tagged_sentences': self.tagged_sentences,
            'total_tokens': self.total_tokens,
            'total_processing_time': self.total_time,
            'avg_time_per_sentence': avg_time_per_sentence,
            'avg_time_per_token': avg_time_per_token
        }
    
    def save_log(self):
        """Oturum logunu dosyaya kaydet."""
        if not self.session_log:
            return
        
        log_data = {
            'session_info': {
                'start_time': self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'stats': self.get_stats()
            },
            'tagging_log': self.session_log
        }
        
        try:
            with open(DEMO_LOG_FILE, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            print(f"📝 Oturum logu kaydedildi: {DEMO_LOG_FILE}")
        except Exception as e:
            print(f"⚠️  Log kaydedilemedi: {e}")


def load_model():
    """HMM modelini yükler."""
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Model dosyası bulunamadı: {MODEL_FILE}")
        print(f"Önce modeli eğitin: python scripts/02_train_model.py")
        return None
    
    try:
        print(f"📂 Model yükleniyor: {MODEL_FILE}")
        model = HMMModel.load(MODEL_FILE)
        print(f"✅ Model başarıyla yüklendi")
        return model
    except Exception as e:
        print(f"❌ Model yüklenemedi: {e}")
        return None


def print_banner():
    """Demo başlık banner'ını yazdırır."""
    print("╔" + "="*70 + "╗")
    print("║" + " "*25 + "HMM POS TAGGER DEMO" + " "*25 + "║")
    print("║" + " "*22 + "Türkçe Kelime Türü Etiketleyici" + " "*17 + "║")
    print("╚" + "="*70 + "╝")
    print()


def print_menu():
    """Ana menüyü yazdırır."""
    print(f"\n{'📋 ANA MENÜ':^50}")
    print(f"{'='*50}")
    
    for key, description in DEMO_OPTIONS.items():
        emoji = "🚪" if key == 'q' else f"{key}️⃣"
        print(f"  {emoji} {key:2s} - {description}")
    
    print(f"{'='*50}")


def simple_tokenize(sentence):
    return stanza_tokenize(sentence)


def display_tagging_result(sentence, words, tags, processing_time):
    """Tagging sonucunu güzel formatta gösterir."""
    print(f"\n🔤 Cümle: {sentence}")
    print(f"⏱️  İşlem süresi: {processing_time:.3f} saniye")
    print(f"📊 Kelime sayısı: {len(words)}")
    print(f"{'='*60}")
    
    print(f"🏷️  POS Tagging Sonucu:")
    for i, (word, tag) in enumerate(zip(words, tags), 1):
        print(f"   {i:2d}. {word:20s} → {tag}")


def single_sentence_demo(model, session):
    """Tek cümle tagging demosu."""
    print(f"\n🔤 TEK CÜMLE TAGGING")
    print(f"{'='*50}")
    print(f"Türkçe bir cümle girin (boş enter ile ana menüye dönün):")
    
    while True:
        try:
            sentence = input(f"\n📝 Cümle: ").strip()
            
            if not sentence:
                break
            
            # Tokenize
            words = simple_tokenize(sentence)
            
            if not words:
                print(f"⚠️  Geçerli bir cümle girin")
                continue
            
            # Tag
            start_time = time.time()
            tags = model.viterbi_decode(words)
            processing_time = time.time() - start_time
            
            # Sonucu göster
            display_tagging_result(sentence, words, tags, processing_time)
            
            # Session'a kaydet
            session.log_tagging(sentence, tags, processing_time)
            
        except KeyboardInterrupt:
            print(f"\n👋 Ana menüye dönülüyor...")
            break
        except Exception as e:
            print(f"❌ Hata: {e}")


def batch_tagging_demo(model, session):
    """Batch tagging demosu."""
    print(f"\n📦 BATCH TAGGING")
    print(f"{'='*50}")
    print(f"Cümleleri girin (her satıra bir cümle, boş satır ile bitirin):")
    
    sentences = []
    print(f"\n📝 Cümleler:")
    
    while True:
        try:
            sentence = input().strip()
            if not sentence:
                break
            sentences.append(sentence)
        except KeyboardInterrupt:
            print(f"\n👋 Ana menüye dönülüyor...")
            return
    
    if not sentences:
        print(f"⚠️  Hiç cümle girilmedi")
        return
    
    print(f"\n🔄 {len(sentences)} cümle işleniyor...")
    
    total_start = time.time()
    
    for i, sentence in enumerate(sentences, 1):
        print(f"\n--- Cümle {i}/{len(sentences)} ---")
        
        words = simple_tokenize(sentence)
        if not words:
            print(f"⚠️  Geçersiz cümle atlanıyor: {sentence}")
            continue
        
        start_time = time.time()
        tags = model.viterbi_decode(words)
        processing_time = time.time() - start_time
        
        display_tagging_result(sentence, words, tags, processing_time)
        session.log_tagging(sentence, tags, processing_time)
    
    total_time = time.time() - total_start
    print(f"\n📊 Batch İşlem Özeti:")
    print(f"   İşlenen cümle: {len(sentences)}")
    print(f"   Toplam süre: {total_time:.3f} saniye")
    print(f"   Cümle başına: {total_time/len(sentences):.3f} saniye")


def sample_sentences_demo(model, session):
    """Örnek cümleler demosu."""
    print(f"\n🧪 ÖRNEK CÜMLELER TESTİ")
    print(f"{'='*50}")
    
    for difficulty, sentences in SAMPLE_SENTENCES.items():
        print(f"\n📈 Zorluk Seviyesi: {difficulty}")
        print(f"{'-'*40}")
        
        for i, sentence in enumerate(sentences, 1):
            print(f"\n--- {difficulty} - {i}/{len(sentences)} ---")
            
            words = simple_tokenize(sentence)
            start_time = time.time()
            tags = model.viterbi_decode(words)
            processing_time = time.time() - start_time
            
            display_tagging_result(sentence, words, tags, processing_time)
            session.log_tagging(sentence, tags, processing_time)
        
        # Kullanıcı onayı
        if difficulty != list(SAMPLE_SENTENCES.keys())[-1]:  # Son değilse
            try:
                input(f"\n⏸️  Devam etmek için Enter'a basın...")
            except KeyboardInterrupt:
                print(f"\n👋 Ana menüye dönülüyor...")
                return


def model_analysis_demo(model):
    """Model analizi demosu."""
    print(f"\n🔬 MODEL ANALİZİ")
    print(f"{'='*50}")
    
    # Model istatistikleri
    stats = model.get_model_statistics()
    print(f"📊 Model İstatistikleri:")
    print(f"   Vocabulary boyutu    : {stats['vocabulary_size']:,}")
    print(f"   POS tag sayısı       : {stats['tag_set_size']:,}")
    print(f"   Emission parametreler: {stats['emission_params']:,}")
    print(f"   Transition parametr. : {stats['transition_params']:,}")
    
    # Model metadata
    print(f"\n📋 Model Bilgileri:")
    for key, value in model.metadata.items():
        print(f"   {key:20s}: {value}")
    
    # Tag set analizi
    if hasattr(model, 'counts') and hasattr(model.counts, 'tag_set'):
        tags = sorted(model.counts.tag_set)
        print(f"\n🏷️  Desteklenen POS Tagları ({len(tags)} adet):")
        
        # Tagları satır satır göster (8'li gruplar)
        for i in range(0, len(tags), 8):
            tag_group = tags[i:i+8]
            print(f"   {' '.join(f'{tag:8s}' for tag in tag_group)}")
    
    # Model dosya bilgisi
    if os.path.exists(MODEL_FILE):
        file_size = os.path.getsize(MODEL_FILE) / 1024  # KB
        creation_time = datetime.fromtimestamp(os.path.getctime(MODEL_FILE))
        print(f"\n💾 Dosya Bilgileri:")
        print(f"   Dosya boyutu: {file_size:.1f} KB")
        print(f"   Oluşturma tar: {creation_time.strftime('%Y-%m-%d %H:%M:%S')}")


def performance_benchmark(model):
    """Performance benchmark demosu."""
    print(f"\n⚡ PERFORMANCE BENCHMARK")
    print(f"{'='*50}")
    
    # Test cümleleri hazırla
    test_sentences = []
    for sentences in SAMPLE_SENTENCES.values():
        test_sentences.extend(sentences)
    
    # Warm-up
    print(f"🔥 Warm-up işlemi...")
    for sentence in test_sentences[:3]:
        words = simple_tokenize(sentence)
        model.viterbi_decode(words)
    
    # Benchmark
    print(f"🏃‍♂️ Benchmark başlıyor ({len(test_sentences)} cümle)...")
    
    processing_times = []
    total_tokens = 0
    
    benchmark_start = time.time()
    
    for sentence in test_sentences:
        words = simple_tokenize(sentence)
        total_tokens += len(words)
        
        start_time = time.time()
        model.viterbi_decode(words)
        processing_time = time.time() - start_time
        
        processing_times.append(processing_time)
    
    total_benchmark_time = time.time() - benchmark_start
    
    # Sonuçları analiz et
    avg_sentence_time = sum(processing_times) / len(processing_times)
    avg_token_time = sum(processing_times) / total_tokens
    tokens_per_second = total_tokens / total_benchmark_time
    
    min_time = min(processing_times)
    max_time = max(processing_times)
    
    print(f"\n📊 Benchmark Sonuçları:")
    print(f"   Test cümle sayısı    : {len(test_sentences)}")
    print(f"   Toplam token sayısı  : {total_tokens:,}")
    print(f"   Toplam süre          : {total_benchmark_time:.3f} saniye")
    print(f"   Ortalama cümle süresi: {avg_sentence_time:.3f} saniye")
    print(f"   Ortalama token süresi: {avg_token_time:.4f} saniye")
    print(f"   Token/saniye         : {tokens_per_second:.1f}")
    print(f"   En hızlı cümle       : {min_time:.3f} saniye")
    print(f"   En yavaş cümle       : {max_time:.3f} saniye")
    
    # Performance kategorisi
    if tokens_per_second > 1000:
        category = "🚀 Çok Hızlı"
    elif tokens_per_second > 500:
        category = "⚡ Hızlı"
    elif tokens_per_second > 100:
        category = "✅ Normal"
    else:
        category = "🐌 Yavaş"
    
    print(f"   Performance kategorisi: {category}")


def show_session_stats(session):
    """Oturum istatistiklerini gösterir."""
    stats = session.get_stats()
    
    print(f"\n📊 OTURUM İSTATİSTİKLERİ")
    print(f"{'='*50}")
    print(f"⏰ Oturum süresi      : {stats['session_duration']:.1f} saniye")
    print(f"📝 Etiketlenen cümle  : {stats['tagged_sentences']}")
    print(f"🔤 Toplam token       : {stats['total_tokens']}")
    print(f"⚡ İşlem süresi       : {stats['total_processing_time']:.3f} saniye")
    
    if stats['tagged_sentences'] > 0:
        print(f"📊 Cümle başına       : {stats['avg_time_per_sentence']:.3f} saniye")
        print(f"🎯 Token başına       : {stats['avg_time_per_token']:.4f} saniye")
        
        efficiency = (stats['total_processing_time'] / stats['session_duration']) * 100
        print(f"📈 İşlem verimi       : %{efficiency:.1f}")


def show_about():
    """Hakkında bilgilerini gösterir."""
    print(f"\n📖 HAKKINDA")
    print(f"{'='*50}")
    print(f"🔧 HMM POS Tagger - Türkçe Kelime Türü Etiketleyici")
    print(f"📅 Geliştirme Tarihi: 2024")
    print(f"🏷️  Desteklenen Dil: Türkçe")
    print(f"🧠 Algoritma: Hidden Markov Model + Viterbi")
    print(f"⚙️  Framework: Python, Stanza, scikit-learn")
    print(f"📊 Özellikler:")
    print(f"   • Türkçe morphology desteği")
    print(f"   • OOV (Out-of-Vocabulary) kelime işleme")
    print(f"   • Add-k smoothing")
    print(f"   • Viterbi optimal path finding")
    print(f"   • CoNLL-U format desteği")
    print(f"   • Batch processing")
    print(f"   • Performance benchmarking")
    print(f"\n📁 Proje Yapısı:")
    print(f"   • core/        - Ana modüller")
    print(f"   • scripts/     - Çalışabilir scriptler")
    print(f"   • data/        - Veri dosyaları")
    print(f"   • models/      - Eğitilmiş modeller")
    print(f"   • web/         - Web arayüzü")
    print(f"\n🔗 Gelişmiş Kullanım:")
    print(f"   • Web demo: python web/app.py")
    print(f"   • Evaluation: python scripts/04_evaluate.py")
    print(f"   • Training: python scripts/02_train_model.py")


def main():
    """Ana demo fonksiyonu."""
    print_banner()
    
    # Model yükle
    model = load_model()
    if not model:
        return
    
    # Demo session başlat
    session = DemoSession()
    
    print(f"🎮 Demo başlatıldı! Model hazır.")
    
    try:
        while True:
            print_menu()
            
            try:
                choice = input(f"\n🎯 Seçiminiz: ").strip().lower()
                
                if choice == 'q' or choice == 'quit' or choice == 'exit':
                    break
                elif choice == '1':
                    single_sentence_demo(model, session)
                elif choice == '2':
                    batch_tagging_demo(model, session)
                elif choice == '3':
                    sample_sentences_demo(model, session)
                elif choice == '4':
                    model_analysis_demo(model)
                elif choice == '5':
                    performance_benchmark(model)
                elif choice == '6':
                    show_session_stats(session)
                elif choice == '7':
                    show_about()
                else:
                    print(f"❌ Geçersiz seçim: {choice}")
                    
            except KeyboardInterrupt:
                print(f"\n\n👋 Demo sonlandırılıyor...")
                break
    
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Session sonuçlarını göster ve kaydet
        if session.tagged_sentences > 0:
            print(f"\n" + "="*60)
            print(f"🎊 Demo Tamamlandı!")
            show_session_stats(session)
            session.save_log()
        
        print(f"👋 Görüşmek üzere!")


if __name__ == '__main__':
    main() 