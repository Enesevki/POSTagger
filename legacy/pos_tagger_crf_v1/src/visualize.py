# src/visualize.py

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Projenin src klasörünü PYTHONPATH'e ekleyelim
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_PATH)

from data_loader import read_conll
from features import sent2features, sent2labels
from model import load_crf_model, predict_tags

def plot_confusion_matrix(
    model_path: str = "outputs/models/crf.pkl",
    test_conll: str = "data/processed/test.conll",
    labels: list = None
):
    # 1) Modeli ve veriyi yükle
    crf = load_crf_model(model_path)
    test_sents = read_conll(test_conll)

    # 2) Özellik ve etiket listelerini hazırla
    X_test = [sent2features(s) for s in test_sents]
    y_true = [sent2labels(s)   for s in test_sents]
    y_pred = predict_tags(crf, X_test)

    # 3) Flatten ve Confusion Matrix
    if labels is None:
        labels = sorted(crf.classes_)
    y_true_flat = [tag for sent in y_true for tag in sent]
    y_pred_flat = [tag for sent in y_pred for tag in sent]
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=labels)

    # 4) Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, aspect='auto')  # varsayılan colormap kullanılır
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('POS Tagging Confusion Matrix')
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize POS Tagging Confusion Matrix")
    parser.add_argument("--model", default="outputs/models/crf.pkl", help="Path to trained CRF model")
    parser.add_argument("--test",  default="data/processed/test.conll", help="Path to test ConLL file")
    parser.add_argument(
        "--labels", nargs="+",
        help="Optional list of labels to include (default: all sorted labels)"
    )
    args = parser.parse_args()

    plot_confusion_matrix(
        model_path=args.model,
        test_conll=args.test,
        labels=args.labels
    )
