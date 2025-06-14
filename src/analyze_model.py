# src/analyze_model.py

import argparse
import logging
import os
import sys
import csv
from dataclasses import dataclass
from itertools import cycle
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report

from src.data_loader import read_conll
from src.features import sent2features, sent2labels, sent2tokens
from src.model import load_crf_model, predict_tags
from src.utils import ensure_dir, save_json

# Proje kök dizinini PYTHONPATH'e ekleyerek modül yükleme sorunlarını önler
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Tip tanımlamalarıyla kod okunabilirliğini artırır
Sentence = List[Tuple[str, str]]
Corpus = List[Sentence]
FeatureDict = Dict[str, Any]
SentenceFeatures = List[FeatureDict]
Labels = List[str]


@dataclass
class AnalysisConfig:
    """
    Model analizi sırasında kullanılması gereken yollar ve parametreler.
    """
    model_path: str
    train_path: str
    test_path: str
    output_dir: str
    top_n: int = 15
    cm_normalize: Optional[str] = 'true'
    misclass_max_samples: int = 100


# ----------------------------
# Konsol Çıktısı Yardımcıları
# ----------------------------

def _print_header(title: str):
    print("\n" + "=" * 80)
    print(f"--- {title.upper()} ---")
    print("=" * 80)


def _print_classification_report_console(report_dict: Dict[str, Any]):
    # Başlık satırını ve çizgiyi yazar
    headers = ["", "precision", "recall", "f1-score", "support"]
    print(f"{headers[0]:<15}{headers[1]:>12}{headers[2]:>12}{headers[3]:>12}{headers[4]:>12}")
    print("-" * 65)

    # Her etiket için metrikleri listeler
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            precision = metrics['precision']
            recall = metrics['recall']
            f1 = metrics['f1-score']
            support = int(metrics.get('support', 0))
            print(f"{label:<15}{precision:>12.4f}{recall:>12.4f}{f1:>12.4f}{support:>12}")

    print("-" * 65)
    # Makro ortalama gibi sayısal olmayan girdileri yazar
    for key, val in report_dict.items():
        if not isinstance(val, dict):
            print(f"{key:<51} {val:.4f}")


def _print_top_features(crf: CRF, top_n: int):
    # Modelin en yüksek ve en düşük ağırlıklı özelliklerini gösterir
    _print_header(f"TOP {top_n} FEATURE WEIGHTS")
    weights = crf.state_features_
    sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)

    print(">>> En Çok Artıran Özellikler:")
    for (feat, tag), w in sorted_items[:top_n]:
        print(f"{w:+.4f}  {tag:<10}  {feat}")

    print("\n>>> En Çok Azaltan Özellikler:")
    for (feat, tag), w in sorted_items[-top_n:]:
        print(f"{w:+.4f}  {tag:<10}  {feat}")


def _print_top_transitions(crf: CRF, top_n: int):
    # En olası ve en az olası etiket geçişlerini listeler
    _print_header(f"TOP {top_n} TRANSITIONS")
    trans = crf.transition_features_
    sorted_trans = sorted(trans.items(), key=lambda x: x[1], reverse=True)

    print(">>> En Olası Geçişler:")
    for (from_tag, to_tag), w in sorted_trans[:top_n]:
        print(f"{w:+.4f}  {from_tag:<10} -> {to_tag}")

    print("\n>>> En Olası Olmayan Geçişler:")
    for (from_tag, to_tag), w in sorted_trans[-top_n:]:
        print(f"{w:+.4f}  {from_tag:<10} -> {to_tag}")


# --------------------
# Veri Hazırlama
# --------------------

def _load_and_prepare_data(conll_path: str) -> Tuple[List[SentenceFeatures], List[Labels], Corpus]:
    sentences = read_conll(conll_path)
    X = [sent2features(s) for s in sentences]
    y = [sent2labels(s) for s in sentences]
    return X, y, sentences


# --------------------
# Grafik ve Matris
# --------------------

def _plot_and_save_confusion_matrix(
        cm: np.ndarray,
        labels: List[str],
        title: str,
        output_path: str,
        normalize: Optional[str] = None
    ):
    # Normalize edip matris verisini hazırlar
    with np.errstate(divide='ignore', invalid='ignore'):
        if normalize == 'true':
            cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        elif normalize == 'pred':
            cm = cm.astype(float) / cm.sum(axis=0)[np.newaxis, :]
            fmt = '.2f'
        else:
            fmt = 'd'

    cm = np.nan_to_num(cm)

    # Grafik oluşturma
    fig, ax = plt.subplots(figsize=(max(12, len(labels)//2), max(10, len(labels)//2.5)))
    sns.heatmap(
        cm,
        annot=(len(labels) <= 25),
        fmt=fmt,
        ax=ax,
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    ax.set_xlabel('Tahmin Edilen Etiket')
    ax.set_ylabel('Gerçek Etiket')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# --------------------
# Değerlendirme Adımları
# --------------------

def evaluate_split(
        model: CRF,
        sentences: Corpus,
        X: List[SentenceFeatures],
        y_true: List[Labels],
        split_name: str,
        config: AnalysisConfig
    ):
    logger = logging.getLogger("analyze_model")
    logger.info(f"{split_name.capitalize()} set üzerinde değerlendirme başlatıldı.")

    y_pred = predict_tags(model, X)
    labels = sorted(model.classes_)
    labels = [l for l in labels if any(l in seq for seq in (y_true + y_pred))]

    # Sınıflandırma raporunu kaydet ve konsola yazdır
    report = flat_classification_report(
        y_true, y_pred,
        labels=labels,
        digits=4,
        output_dict=True,
        zero_division=0
    )
    save_json(report, os.path.join(config.output_dir, f"{split_name}_report.json"))
    _print_header(f"CLASSIFICATION REPORT ({split_name.upper()})")
    _print_classification_report_console(report)

    # Karmaşıklık matrisini oluştur ve kaydet
    y_true_flat = [tag for seq in y_true for tag in seq]
    y_pred_flat = [tag for seq in y_pred for tag in seq]
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=labels)

    raw_path = os.path.join(config.output_dir, f"{split_name}_confusion_matrix_raw.png")
    _plot_and_save_confusion_matrix(cm, labels, f"{split_name} Ham Matris", raw_path, normalize=None)
    logger.info(f"{split_name} ham matris kaydedildi: {raw_path}")

    if config.cm_normalize:
        norm_path = os.path.join(config.output_dir, f"{split_name}_confusion_matrix_norm.png")
        _plot_and_save_confusion_matrix(cm, labels, f"{split_name} Normalize Matris", norm_path, normalize=config.cm_normalize)
        logger.info(f"{split_name} normalize matris kaydedildi: {norm_path}")


def plot_detailed_roc_curves(model: CRF, config: AnalysisConfig):
    logger = logging.getLogger("analyze_model")
    logger.info("ROC eğrileri hesaplanıyor.")

    X_test, y_true_seqs, _ = _load_and_prepare_data(config.test_path)
    labels = sorted(model.classes_)
    n_classes = len(labels)

    # Etiketleri ikili formata çevir
    y_true_flat = [tag for seq in y_true_seqs for tag in seq]
    y_true_bin = label_binarize(y_true_flat, classes=labels)

    # Model olasılıklarını al
    marginals = model.predict_marginals(X_test)
    probs = [token_probs[l] for seq in marginals for token_probs in seq for l in labels]
    y_score = np.array(probs).reshape(-1, n_classes)

    # Mikro ve makro ROC
    fpr, tpr, roc_auc = {}, {}, {}
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    for i, label in enumerate(labels):
        if label in y_true_flat:
            fpr[label], tpr[label], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[label] = auc(fpr[label], tpr[label])
        else:
            fpr[label], tpr[label], roc_auc[label] = np.array([0]), np.array([0]), 0.0

    all_fpr = np.unique(np.concatenate([fpr[l] for l in labels]))
    mean_tpr = sum(np.interp(all_fpr, fpr[l], tpr[l]) for l in labels) / n_classes

    # Ortalama ROC grafiği
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr["micro"], tpr["micro"], label=f'Micro (AUC={roc_auc["micro"]:.3f})', linestyle=':', linewidth=4)
    ax.plot(all_fpr, mean_tpr, label=f'Macro (AUC={auc(all_fpr, mean_tpr):.3f})', linestyle=':', linewidth=4)
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('Micro ve Macro ROC Eğrileri')
    ax.legend(loc="lower right"); ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(config.output_dir, "roc_curve_averages.png"), dpi=300)
    plt.close(fig)
    logger.info("Ortalama ROC eğrisi kaydedildi.")

    # Sınıf bazlı ROC grafiği
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = cycle(plt.get_cmap('tab20').colors)
    for label, color in zip(labels, colors):
        ax.plot(fpr[label], tpr[label], lw=2, label=f'{label} (AUC={roc_auc[label]:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('Sınıf Bazlı ROC Eğrileri')
    ax.legend(loc='lower right', prop={'size': 8}); ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(config.output_dir, "roc_curve_per_class.png"), dpi=300)
    plt.close(fig)
    logger.info("Sınıf bazlı ROC eğrisi kaydedildi.")


def generate_misclassification_report(model: CRF, config: AnalysisConfig):
    logger = logging.getLogger("analyze_model")
    logger.info("Hatalı sınıflandırmaları CSV olarak kaydediyor.")

    X, y_true, sentences = _load_and_prepare_data(config.test_path)
    y_pred = predict_tags(model, X)

    header = ["Sentence ID", "Token ID", "Token", "Previous", "Next", "True Label", "Pred Label"]
    rows = []

    for i, (sent, true_seq, pred_seq) in enumerate(zip(sentences, y_true, y_pred)):
        if len(rows) >= config.misclass_max_samples:
            break
        tokens = sent2tokens(sent)
        for j, (tok, true_lbl, pred_lbl) in enumerate(zip(tokens, true_seq, pred_seq)):
            if true_lbl != pred_lbl:
                prev_tok = tokens[j-1] if j > 0 else "<BOS>"
                next_tok = tokens[j+1] if j < len(tokens)-1 else "<EOS>"
                rows.append([i, j, tok, prev_tok, next_tok, true_lbl, pred_lbl])
                if len(rows) >= config.misclass_max_samples:
                    break

    mis_path = os.path.join(config.output_dir, "misclassifications.csv")
    with open(mis_path, 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    logger.info(f"Misclassification raporu kaydedildi: {mis_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Detaylı CRF model analiz aracı",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", required=True, help="Analiz edilecek eğitilmiş model dosyası (.pkl).")
    parser.add_argument("--train", required=True, help="Eğitim verisi konll dosyası.")
    parser.add_argument("--test", required=True, help="Test verisi konll dosyası.")
    parser.add_argument("--output-dir", default="outputs/model_analysis", help="Analiz çıktılarının kaydedileceği dizin.")
    parser.add_argument("--top-n", type=int, default=15, help="Konsola yazdırılacak en iyi N özellik/geçiş sayısı.")
    parser.add_argument("--cm-normalize", choices=['true', 'pred'], default='true', help="Karmaşıklık matrisi normalizasyonu.")
    parser.add_argument("--misclass-max-samples", type=int, default=100, help="CSV raporundaki maksimum hata örneği sayısı.")
    parser.add_argument("--verbose", action="store_true", help="Detaylı bilgi loglamayı etkinleştir.")

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
        level=logging.INFO if args.verbose else logging.WARNING,
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    config = AnalysisConfig(
        model_path=args.model,
        train_path=args.train,
        test_path=args.test,
        output_dir=args.output_dir,
        top_n=args.top_n,
        cm_normalize=args.cm_normalize,
        misclass_max_samples=args.misclass_max_samples
    )

    ensure_dir(config.output_dir)
    logger = logging.getLogger("analyze_model")
    logger.info(f"Model yükleniyor: {config.model_path}")
    crf_model = load_crf_model(config.model_path)

    # Veri yükleme
    logger.info("Veriler yükleniyor...")
    X_train, y_train, sents_train = _load_and_prepare_data(config.train_path)
    X_test, y_test, sents_test = _load_and_prepare_data(config.test_path)
    logger.info("Veri yükleme tamamlandı.")

    # Model özet bilgileri
    _print_header("MODEL OVERVIEW")
    print(f"Model Sınıfı: {type(crf_model).__name__}")
    print("Model Parametreleri:", crf_model.get_params())
    _print_top_transitions(crf_model, config.top_n)
    _print_top_features(crf_model, config.top_n)

    # Rapor ve matris oluşturma
    evaluate_split(crf_model, sents_train, X_train, y_train, "train", config)
    evaluate_split(crf_model, sents_test, X_test, y_test, "test", config)

    # ROC eğrileri ve hata örnekleri
    plot_detailed_roc_curves(crf_model, config)
    generate_misclassification_report(crf_model, config)

    logger.info(f"Analiz tamamlandı. Çıktılar: {config.output_dir}")


if __name__ == "__main__":
    main()
