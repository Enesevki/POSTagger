#!/usr/bin/env python3
"""
HMM Model Evaluation Script

Bu script eğitilmiş HMM modelini test seti üzerinde kapsamlı şekilde 
değerlendirir ve detaylı performans raporları oluşturur.

Usage:
    python scripts/04_evaluate.py
    
Input:
    - models/hmm_model.pkl (eğitilmiş model)
    - data/processed/test.conllu (test seti)
    
Output:
    - models/evaluation_report.txt (detaylı rapor)
    - models/confusion_matrix.png (confusion matrix)
    - models/performance_plots.png (performans grafikleri)
"""

import os
import sys
import time
import json
from collections import defaultdict, Counter
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score

# Import için gerekli path ayarı
sys.path.append(os.path.abspath('.'))
from core import HMMModel, CoNLLUReader

# Plotting için
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️  Matplotlib/Seaborn bulunamadı. Grafikler oluşturulamayacak.")

# Konfigürasyon
MODEL_FILE = 'models/hmm_model.pkl'
TEST_FILE = 'data/processed/test.conllu'
OUTPUT_DIR = 'models/'
EVALUATION_REPORT = f'{OUTPUT_DIR}/evaluation_report.txt'
CONFUSION_MATRIX_PNG = f'{OUTPUT_DIR}/confusion_matrix.png'
PERFORMANCE_PLOTS_PNG = f'{OUTPUT_DIR}/performance_plots.png'
RESULTS_JSON = f'{OUTPUT_DIR}/evaluation_results.json'

# Evaluation ayarları
MAX_SENTENCES = None          # None = tüm test setini kullan
DETAILED_ERRORS = True        # Hata örneklerini göster
CREATE_PLOTS = True           # Grafik oluştur (eğer kütüphaneler varsa)


def load_model_and_test_data():
    """
    Model ve test verisini yükler.
    
    Returns:
        tuple: (model, test_reader)
    """
    print(f"📂 Model ve test verisi yükleniyor...")
    
    # Model kontrolü
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {MODEL_FILE}")
    
    # Test dosyası kontrolü
    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"Test dosyası bulunamadı: {TEST_FILE}")
    
    # Model yükle
    print(f"   📖 Model: {MODEL_FILE}")
    model = HMMModel.load(MODEL_FILE)
    
    # Test verisi yükle
    print(f"   📖 Test: {TEST_FILE}")
    test_reader = CoNLLUReader(TEST_FILE)
    
    print(f"✅ Model ve test verisi yüklendi")
    print(f"   Test cümle sayısı: {test_reader.get_sentence_count():,}")
    print(f"   Test token sayısı: {test_reader.get_token_count():,}")
    
    return model, test_reader


def evaluate_model(model, test_reader):
    """
    Modeli test seti üzerinde değerlendirir.
    
    Args:
        model: HMM model
        test_reader: Test verisi reader
    
    Returns:
        dict: Değerlendirme sonuçları
    """
    print(f"\n🔍 Model Değerlendirme Başlıyor")
    print(f"{'='*50}")
    
    # Sonuç toplama değişkenleri
    total_tokens = 0
    correct_predictions = 0
    sentences_processed = 0
    
    # POS tag bazlı istatistikler
    tag_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    # Confusion matrix için
    true_tags = []
    predicted_tags = []
    
    # Hata analizi için
    error_examples = []
    
    # Zaman ölçümü
    start_time = time.time()
    processing_times = []
    
    print(f"📊 Test setindeki cümleler işleniyor...")
    
    for sentence_idx, sentence_data in enumerate(test_reader):
        if MAX_SENTENCES and sentence_idx >= MAX_SENTENCES:
            break
        
        # Cümle verilerini hazırla
        words = [token['form'] for token in sentence_data]
        gold_tags = [token['upos'] for token in sentence_data]
        
        # Model tahminini al
        sentence_start = time.time()
        try:
            pred_tags = model.viterbi_decode(words)
            sentence_time = time.time() - sentence_start
            processing_times.append(sentence_time)
            
            # İstatistikleri güncelle
            sentences_processed += 1
            total_tokens += len(words)
            
            # Token bazlı accuracy
            for gold_tag, pred_tag in zip(gold_tags, pred_tags):
                true_tags.append(gold_tag)
                predicted_tags.append(pred_tag)
                
                tag_stats[gold_tag]['total'] += 1
                
                if gold_tag == pred_tag:
                    correct_predictions += 1
                    tag_stats[gold_tag]['correct'] += 1
                else:
                    # Hata örneği kaydet
                    if len(error_examples) < 50:  # İlk 50 hatayı kaydet
                        error_examples.append({
                            'sentence_id': sentence_idx + 1,
                            'word': words[gold_tags.index(gold_tag)] if gold_tag in gold_tags else 'UNK',
                            'true_tag': gold_tag,
                            'predicted_tag': pred_tag,
                            'context': ' '.join(words)
                        })
            
            # Progress göster
            if sentences_processed % 100 == 0:
                current_accuracy = (correct_predictions / total_tokens) * 100
                print(f"   İşlenen: {sentences_processed:4d} cümle, "
                      f"Geçici accuracy: %{current_accuracy:.2f}")
                
        except Exception as e:
            print(f"❌ Cümle {sentence_idx + 1} işlenemedi: {e}")
            continue
    
    total_time = time.time() - start_time
    
    # Sonuçları hesapla
    overall_accuracy = (correct_predictions / total_tokens) * 100 if total_tokens > 0 else 0
    avg_sentence_time = sum(processing_times) / len(processing_times) if processing_times else 0
    tokens_per_second = total_tokens / total_time if total_time > 0 else 0
    
    # Tüm tokenlar için overall precision, recall, f1 (macro average)
    overall_precision = precision_score(true_tags, predicted_tags, average='macro', zero_division=0)
    overall_recall = recall_score(true_tags, predicted_tags, average='macro', zero_division=0)
    overall_f1 = f1_score(true_tags, predicted_tags, average='macro', zero_division=0)
    
    print(f"\n✅ Değerlendirme tamamlandı!")
    print(f"   İşlenen cümle: {sentences_processed:,}")
    print(f"   İşlenen token: {total_tokens:,}")
    print(f"   Doğru tahmin: {correct_predictions:,}")
    print(f"   Genel accuracy: %{overall_accuracy:.2f}")
    print(f"   Toplam süre: {total_time:.2f} saniye")
    print(f"   Token/saniye: {tokens_per_second:.1f}")
    
    # Sonuçları dönder
    results = {
        'sentences_processed': sentences_processed,
        'total_tokens': total_tokens,
        'correct_predictions': correct_predictions,
        'overall_accuracy': overall_accuracy / 100,  # 0-1 arası olacak şekilde
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1': overall_f1,
        'total_time': total_time,
        'avg_sentence_time': avg_sentence_time,
        'tokens_per_second': tokens_per_second,
        'tag_stats': dict(tag_stats),
        'true_tags': true_tags,
        'predicted_tags': predicted_tags,
        'error_examples': error_examples,
        'processing_times': processing_times
    }
    
    return results


def analyze_per_tag_performance(results):
    """
    POS tag bazlı performans analizi yapar.
    
    Args:
        results (dict): Değerlendirme sonuçları
    
    Returns:
        dict: Tag bazlı performans metrikleri
    """
    print(f"\n📈 POS Tag Bazlı Performans Analizi")
    print(f"{'='*60}")
    
    tag_stats = results['tag_stats']
    tag_performance = {}
    
    print(f"{'Tag':10s} {'Total':8s} {'Correct':8s} {'Accuracy':10s} {'Precision':10s} {'Recall':10s} {'F1':8s}")
    print(f"{'-'*70}")
    
    # Her tag için precision, recall, f1 hesapla
    for tag in sorted(tag_stats.keys()):
        total = tag_stats[tag]['total']
        correct = tag_stats[tag]['correct']
        accuracy = (correct / total * 100) if total > 0 else 0
        
        # Precision ve recall için true/false positives hesapla
        true_positives = correct
        false_positives = results['predicted_tags'].count(tag) - true_positives
        false_negatives = total - true_positives
        
        precision = (true_positives / (true_positives + false_positives)) * 100 if (true_positives + false_positives) > 0 else 0
        recall = (true_positives / (true_positives + false_negatives)) * 100 if (true_positives + false_negatives) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        
        tag_performance[tag] = {
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        print(f"{tag:10s} {total:8d} {correct:8d} {accuracy:9.2f}% {precision:9.2f}% {recall:9.2f}% {f1:7.2f}")
    
    return tag_performance


def create_confusion_matrix(results):
    """
    Confusion matrix oluşturur ve kaydeder.
    
    Args:
        results (dict): Değerlendirme sonuçları
    """
    if not PLOTTING_AVAILABLE or not CREATE_PLOTS:
        print(f"⚠️  Grafik kütüphaneleri bulunamadı veya grafik oluşturma kapalı")
        return
    
    print(f"\n📊 Confusion Matrix oluşturuluyor...")
    
    try:
        true_tags = results['true_tags']
        pred_tags = results['predicted_tags']
        
        # Unique tagları al
        all_tags = sorted(list(set(true_tags + pred_tags)))
        
        # Confusion matrix oluştur
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_tags, pred_tags, labels=all_tags)
        
        # Normalize et
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot oluştur
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, 
                   xticklabels=all_tags, 
                   yticklabels=all_tags,
                   annot=True, 
                   fmt='.2f', 
                   cmap='Blues',
                   cbar_kws={'label': 'Normalized Frequency'})
        
        plt.title('POS Tagging Confusion Matrix (Normalized)')
        plt.xlabel('Predicted Tags')
        plt.ylabel('True Tags')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        
        # Kaydet
        plt.savefig(CONFUSION_MATRIX_PNG, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Confusion matrix kaydedildi: {CONFUSION_MATRIX_PNG}")
        
    except Exception as e:
        print(f"❌ Confusion matrix oluşturulamadı: {e}")


def create_performance_plots(results, tag_performance):
    """
    Performans grafikleri oluşturur.
    
    Args:
        results (dict): Değerlendirme sonuçları
        tag_performance (dict): Tag bazlı performans
    """
    if not PLOTTING_AVAILABLE or not CREATE_PLOTS:
        return
    
    print(f"\n📈 Performans grafikleri oluşturuluyor...")
    
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Tag frequency vs accuracy
        tags = list(tag_performance.keys())
        frequencies = [tag_performance[tag]['total'] for tag in tags]
        accuracies = [tag_performance[tag]['accuracy'] for tag in tags]
        
        ax1.scatter(frequencies, accuracies, alpha=0.7)
        ax1.set_xlabel('Tag Frequency')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Tag Frequency vs Accuracy')
        ax1.grid(True, alpha=0.3)
        
        # 2. Precision vs Recall
        precisions = [tag_performance[tag]['precision'] for tag in tags]
        recalls = [tag_performance[tag]['recall'] for tag in tags]
        
        ax2.scatter(recalls, precisions, alpha=0.7)
        ax2.set_xlabel('Recall (%)')
        ax2.set_ylabel('Precision (%)')
        ax2.set_title('Precision vs Recall by Tag')
        ax2.grid(True, alpha=0.3)
        
        # Diagonal line ekle
        ax2.plot([0, 100], [0, 100], 'r--', alpha=0.5)
        
        # 3. Top 10 tags by frequency
        top_tags = sorted(tags, key=lambda x: tag_performance[x]['total'], reverse=True)[:10]
        top_freqs = [tag_performance[tag]['total'] for tag in top_tags]
        
        ax3.bar(range(len(top_tags)), top_freqs)
        ax3.set_xlabel('POS Tags')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Top 10 Most Frequent Tags')
        ax3.set_xticks(range(len(top_tags)))
        ax3.set_xticklabels(top_tags, rotation=45)
        
        # 4. Processing time distribution
        times = results['processing_times']
        ax4.hist(times, bins=30, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Processing Time (seconds)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Sentence Processing Time Distribution')
        ax4.axvline(np.mean(times), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(times):.3f}s')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(PERFORMANCE_PLOTS_PNG, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Performans grafikleri kaydedildi: {PERFORMANCE_PLOTS_PNG}")
        
    except Exception as e:
        print(f"❌ Performans grafikleri oluşturulamadı: {e}")


def save_evaluation_report(results, tag_performance, model):
    """
    Detaylı değerlendirme raporu oluşturur ve kaydeder.
    
    Args:
        results (dict): Değerlendirme sonuçları
        tag_performance (dict): Tag bazlı performans
        model: HMM model
    """
    print(f"\n📄 Değerlendirme raporu oluşturuluyor: {EVALUATION_REPORT}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(EVALUATION_REPORT, 'w', encoding='utf-8') as f:
        f.write("HMM POS Tagger - Evaluation Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Model bilgileri
        f.write("Model Information:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Model File: {MODEL_FILE}\n")
        f.write(f"Test File: {TEST_FILE}\n")
        
        model_stats = model.get_model_statistics()
        f.write(f"Vocabulary Size: {model_stats.get('vocabulary_size', 0):,}\n")
        f.write(f"Tag Set Size: {model_stats.get('tag_set_size', 0):,}\n")
        f.write(f"Training Date: {model.metadata.get('training_date', 'N/A')}\n\n")
        
        # Genel performans
        f.write("Overall Performance:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Sentences Processed: {results['sentences_processed']:,}\n")
        f.write(f"Total Tokens: {results['total_tokens']:,}\n")
        f.write(f"Correct Predictions: {results['correct_predictions']:,}\n")
        f.write(f"Overall Accuracy: {results['overall_accuracy']:.2f}%\n")
        f.write(f"Total Time: {results['total_time']:.2f} seconds\n")
        f.write(f"Average Sentence Time: {results['avg_sentence_time']:.3f} seconds\n")
        f.write(f"Tokens per Second: {results['tokens_per_second']:.1f}\n\n")
        
        # Tag bazlı performans
        f.write("Per-Tag Performance:\n")
        f.write("-" * 30 + "\n")
        f.write(f"{'Tag':10s} {'Total':8s} {'Correct':8s} {'Acc%':8s} {'Prec%':8s} {'Rec%':8s} {'F1':8s}\n")
        f.write("-" * 70 + "\n")
        
        for tag in sorted(tag_performance.keys()):
            perf = tag_performance[tag]
            f.write(f"{tag:10s} {perf['total']:8d} {perf['correct']:8d} "
                   f"{perf['accuracy']:7.2f} {perf['precision']:7.2f} "
                   f"{perf['recall']:7.2f} {perf['f1']:7.2f}\n")
        
        # En iyi ve en kötü taglar
        f.write("\nBest Performing Tags (by F1 score):\n")
        best_tags = sorted(tag_performance.items(), 
                          key=lambda x: x[1]['f1'], reverse=True)[:5]
        for tag, perf in best_tags:
            f.write(f"  {tag:10s}: F1={perf['f1']:5.2f} (Acc={perf['accuracy']:5.2f}%)\n")
        
        f.write("\nWorst Performing Tags (by F1 score):\n")
        worst_tags = sorted(tag_performance.items(), 
                           key=lambda x: x[1]['f1'])[:5]
        for tag, perf in worst_tags:
            f.write(f"  {tag:10s}: F1={perf['f1']:5.2f} (Acc={perf['accuracy']:5.2f}%)\n")
        
        # Hata örnekleri
        if DETAILED_ERRORS and results['error_examples']:
            f.write("\nError Examples (First 20):\n")
            f.write("-" * 40 + "\n")
            for i, error in enumerate(results['error_examples'][:20], 1):
                f.write(f"{i:2d}. Word: '{error['word']}' "
                       f"True: {error['true_tag']} → Pred: {error['predicted_tag']}\n")
                f.write(f"    Context: {error['context'][:100]}...\n\n")
    
    print(f"✅ Değerlendirme raporu kaydedildi")


def save_results_json(results, tag_performance):
    """
    Sonuçları JSON formatında kaydeder.
    
    Args:
        results (dict): Değerlendirme sonuçları
        tag_performance (dict): Tag bazlı performans
    """
    print(f"💾 Sonuçlar JSON olarak kaydediliyor: {RESULTS_JSON}")
    
    # JSON serializable hale getir
    json_results = {
        'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'overall_performance': {
            'sentences_processed': results['sentences_processed'],
            'total_tokens': results['total_tokens'],
            'correct_predictions': results['correct_predictions'],
            'overall_accuracy': results['overall_accuracy'],
            'overall_precision': results['overall_precision'],
            'overall_recall': results['overall_recall'],
            'overall_f1': results['overall_f1'],
            'total_time': results['total_time'],
            'avg_sentence_time': results['avg_sentence_time'],
            'tokens_per_second': results['tokens_per_second']
        },
        'tag_performance': tag_performance,
        'error_count': len(results['error_examples'])
    }
    
    with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ JSON sonuçları kaydedildi")


def print_summary(results, tag_performance):
    """
    Özet sonuçları ekrana yazdırır.
    
    Args:
        results (dict): Değerlendirme sonuçları  
        tag_performance (dict): Tag bazlı performans
    """
    print(f"\n📋 DEĞERLENDIRME ÖZETİ")
    print(f"{'='*60}")
    print(f"🎯 Genel Accuracy     : %{results['overall_accuracy']:.2f}")
    print(f"⚡ İşlem Hızı        : {results['tokens_per_second']:.1f} token/saniye")
    print(f"📊 İşlenen Token     : {results['total_tokens']:,}")
    print(f"✅ Doğru Tahmin      : {results['correct_predictions']:,}")
    print(f"❌ Yanlış Tahmin     : {results['total_tokens'] - results['correct_predictions']:,}")
    
    # En iyi/kötü taglar
    best_tag = max(tag_performance.items(), key=lambda x: x[1]['f1'])
    worst_tag = min(tag_performance.items(), key=lambda x: x[1]['f1'])
    
    print(f"\n🏆 En İyi Tag         : {best_tag[0]} (F1: {best_tag[1]['f1']:.2f})")
    print(f"⚠️  En Zor Tag         : {worst_tag[0]} (F1: {worst_tag[1]['f1']:.2f})")
    
    # Çıktı dosyaları
    print(f"\n📂 Oluşturulan Dosyalar:")
    print(f"   📄 Rapor: {EVALUATION_REPORT}")
    print(f"   💾 JSON: {RESULTS_JSON}")
    if PLOTTING_AVAILABLE and CREATE_PLOTS:
        print(f"   📊 Confusion Matrix: {CONFUSION_MATRIX_PNG}")
        print(f"   📈 Performans Grafikleri: {PERFORMANCE_PLOTS_PNG}")


def plot_overall_metrics(results):
    """
    Overall metrikleri (accuracy, precision, recall, F1) bar grafik olarak görselleştirir.
    """
    metrics = {
        'Accuracy': results['overall_accuracy'],
        'Precision': results['overall_precision'],
        'Recall': results['overall_recall'],
        'F1': results['overall_f1']
    }
    plt.figure(figsize=(10, 6))
    plt.bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'red', 'purple'])
    plt.title('Overall Model Performance Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    for i, v in enumerate(metrics.values()):
        plt.text(i, v + 0.01, f'{v:.2f}', ha='center')
    plt.tight_layout()
    plt.savefig('models/overall_metrics.png')
    plt.close()


def plot_tag_heatmap(tag_performance):
    """
    Tag bazlı performans metriklerini (accuracy, precision, recall, F1) heatmap olarak görselleştirir.
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    data = []
    for tag, perf in tag_performance.items():
        row = [perf[m] for m in metrics]
        data.append(row)
    data = np.array(data)
    plt.figure(figsize=(12, 8))
    sns.heatmap(data, annot=True, fmt='.2f', cmap='YlGnBu', xticklabels=metrics, yticklabels=tag_performance.keys())
    plt.title('Tag-based Performance Metrics Heatmap')
    plt.tight_layout()
    plt.savefig('models/tag_heatmap.png')
    plt.close()


def main():
    """
    Ana değerlendirme fonksiyonu.
    """
    print("🔬 HMM Model Değerlendirmesi Başlıyor")
    print("=" * 60)
    
    try:
        # 1. Model ve test verisini yükle
        model, test_reader = load_model_and_test_data()
        
        # 2. Modeli değerlendir
        results = evaluate_model(model, test_reader)
        
        # 3. Tag bazlı analiz
        tag_performance = analyze_per_tag_performance(results)
        
        # 4. Görselleştirmeler oluştur
        if PLOTTING_AVAILABLE and CREATE_PLOTS:
            create_confusion_matrix(results)
            create_performance_plots(results, tag_performance)
        
        # 5. Raporları kaydet
        save_evaluation_report(results, tag_performance, model)
        save_results_json(results, tag_performance)
        plot_overall_metrics(results)
        plot_tag_heatmap(tag_performance)
        
        # 6. Özet yazdır
        print_summary(results, tag_performance)
        
        print(f"\n🎉 Değerlendirme başarıyla tamamlandı!")
        print(f"Sonraki adım: python scripts/05_interactive_demo.py")
        
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main() 