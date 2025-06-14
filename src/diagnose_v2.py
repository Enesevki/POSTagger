# src/diagnose_v2.py

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
from sklearn.model_selection import cross_val_score, KFold, learning_curve
from sklearn.metrics import confusion_matrix, make_scorer, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_f1_score, flat_classification_report

from src.data_loader import read_conll
from src.features import sent2features, sent2labels, sent2tokens
from src.model import load_crf_model, predict_tags, build_crf_model
from src.utils import ensure_dir, save_json

# Type aliases for readability
Sentence = List[Tuple[str, str]]
Corpus = List[Sentence]
FeatureDict = Dict[str, Any]
SentenceFeatures = List[FeatureDict]
Labels = List[str]


@dataclass
class DiagnosticsConfig:
    """
    Configuration parameters for the diagnostic workflow.
    """
    model_path: str
    train_path: str
    test_path: str
    output_dir: str
    cv_folds: int = 5
    n_jobs: int = -1
    seed: int = 42
    c1: float = 0.1
    c2: float = 0.1
    max_iter: int = 100
    cm_normalize: Optional[str] = 'true'
    misclass_max_samples: int = 100
    top_n: int = 15  # Number of top features/transitions to display


# -----------------------------
# Console Output Helper Methods
# -----------------------------

def _print_header(title: str):
    """
    Print a standardized section header to the console.
    """
    print("\n" + "=" * 80)
    print(f"--- {title.upper()} ---")
    print("=" * 80)


def _print_classification_report_console(report: Dict[str, Any]):
    """
    Render the classification report as a formatted table in the console.
    """
    cols = ["", "precision", "recall", "f1-score", "support"]
    print(f"{cols[0]:<15}{cols[1]:>12}{cols[2]:>12}{cols[3]:>12}{cols[4]:>12}")
    print("-" * 65)

    for label, metrics in report.items():
        if isinstance(metrics, dict):
            precision = metrics['precision']
            recall = metrics['recall']
            f1 = metrics['f1-score']
            support = int(metrics.get('support', 0) or 0)
            print(f"{label:<15}{precision:>12.4f}{recall:>12.4f}{f1:>12.4f}{support:>12}")

    print("-" * 65)
    # Print aggregated metrics (accuracy, macro avg, weighted avg)
    for key, value in report.items():
        if not isinstance(value, dict):
            print(f"{key:<51}{value:.4f}")


def _print_top_features(crf: CRF, top_n: int = 15):
    """
    Display the highest- and lowest-weighted state features learned by the CRF.
    """
    _print_header(f"TOP {top_n} FEATURE WEIGHTS")
    weights = crf.state_features_
    items = sorted(weights.items(), key=lambda x: x[1], reverse=True)

    print(">>> Top Positive Features:")
    for (feat, tag), w in items[:top_n]:
        print(f"{w:+.4f}  {tag:<10}  {feat}")

    print("\n>>> Top Negative Features:")
    for (feat, tag), w in items[-top_n:]:
        print(f"{w:+.4f}  {tag:<10}  {feat}")


def _print_top_transitions(crf: CRF, top_n: int = 15):
    """
    Display the most and least probable label transitions.
    """
    _print_header(f"TOP {top_n} TRANSITIONS")
    trans = crf.transition_features_
    items = sorted(trans.items(), key=lambda x: x[1], reverse=True)

    print(">>> Most Likely Transitions:")
    for (from_tag, to_tag), w in items[:top_n]:
        print(f"{w:+.4f}  {from_tag:<10} -> {to_tag}")

    print("\n>>> Least Likely Transitions:")
    for (from_tag, to_tag), w in items[-top_n:]:
        print(f"{w:+.4f}  {from_tag:<10} -> {to_tag}")


# ---------------------
# Data Preparation
# ---------------------

def _load_and_prepare_data(conll_path: str) -> Tuple[List[SentenceFeatures], List[Labels], Corpus]:
    """
    Read a CoNLL file, extract features and labels for each sentence.
    """
    sentences = read_conll(conll_path)
    X = [sent2features(s) for s in sentences]
    y = [sent2labels(s) for s in sentences]
    return X, y, sentences


# ---------------------
# Confusion Matrix Plot
# ---------------------

def _plot_and_save_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str,
    output_path: str,
    normalize: Optional[str] = None
):
    """
    Plot and save a confusion matrix with optional normalization.
    """
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
    fig, ax = plt.subplots(figsize=(max(12, len(labels)//2), max(10, len(labels)//2.5)))
    annotate = len(labels) <= 25

    sns.heatmap(
        cm,
        annot=annotate,
        fmt=fmt,
        ax=ax,
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# -------------------------
# Evaluation & Reporting
# -------------------------

def evaluate_split(
    model: CRF,
    sents: Corpus,
    X: List[SentenceFeatures],
    y_true: List[Labels],
    split_name: str,
    config: DiagnosticsConfig
):
    """
    Evaluate the model on a dataset split, save JSON report and confusion matrix.
    """
    logger = logging.getLogger("diagnose")
    logger.info(f"Evaluating {split_name} split...")

    y_pred = predict_tags(model, X)
    all_labels = sorted(model.classes_)
    labels = [
        l for l in all_labels
        if any(l in seq for seq in y_true) or any(l in seq for seq in y_pred)
    ]

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

    y_true_flat = [tag for seq in y_true for tag in seq]
    y_pred_flat = [tag for seq in y_pred for tag in seq]
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=labels)

    raw_path = os.path.join(config.output_dir, f"{split_name}_confusion_matrix_raw.png")
    _plot_and_save_confusion_matrix(cm, labels, f'{split_name} Raw Confusion Matrix', raw_path)
    logger.info(f"Saved raw confusion matrix to: {raw_path}")

    if config.cm_normalize:
        norm_path = os.path.join(config.output_dir, f"{split_name}_confusion_matrix_norm.png")
        _plot_and_save_confusion_matrix(cm, labels, f'{split_name} Normalized Confusion Matrix', norm_path, normalize=config.cm_normalize)
        logger.info(f"Saved normalized confusion matrix to: {norm_path}")


def cross_validation(config: DiagnosticsConfig):
    """
    Perform k-fold cross-validation and save F1 scores report.
    """
    logger = logging.getLogger("diagnose")
    logger.info(f"Running {config.cv_folds}-fold cross-validation...")

    X, y, _ = _load_and_prepare_data(config.train_path)
    crf = build_crf_model(c1=config.c1, c2=config.c2, max_iterations=config.max_iter)
    labels = sorted({tag for seq in y for tag in seq})
    scorer = make_scorer(flat_f1_score, average='weighted', labels=labels, zero_division=0)
    cv = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)

    scores = cross_val_score(crf, X, y, cv=cv, scoring=scorer, n_jobs=config.n_jobs)
    report = {
        "fold_scores": scores.tolist(),
        "mean_f1": float(np.mean(scores)),
        "std_f1": float(np.std(scores))
    }
    save_json(report, os.path.join(config.output_dir, "cross_val_report.json"))
    _print_header("CROSS-VALIDATION RESULTS")
    print(f"Fold scores: {np.round(scores, 4)}")
    print(f"Mean F1: {scores.mean():.4f}")
    print(f"Std F1: {scores.std():.4f}")


def plot_learning_curve_and_error(config: DiagnosticsConfig):
    """
    Generate and save learning curve (F1) and error rate plots.
    """
    logger = logging.getLogger("diagnose")
    logger.info("Generating learning and error curves...")

    X, y, _ = _load_and_prepare_data(config.train_path)
    crf = build_crf_model(c1=config.c1, c2=config.c2, max_iterations=config.max_iter)
    labels = sorted({tag for seq in y for tag in seq})
    scorer = make_scorer(flat_f1_score, average='weighted', labels=labels, zero_division=0)
    cv = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)

    train_sizes, train_scores, valid_scores = learning_curve(
        crf, X, y, cv=cv, scoring=scorer,
        train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=config.n_jobs
    )
    train_mean, train_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
    valid_mean, valid_std = np.mean(valid_scores, axis=1), np.std(valid_scores, axis=1)

    # Learning curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    ax.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, alpha=0.1)
    ax.plot(train_sizes, train_mean, 'o-', label='Training F1')
    ax.plot(train_sizes, valid_mean, 'o-', label='Validation F1')
    ax.set_xlabel('Training Samples')
    ax.set_ylabel('F1 Score')
    ax.set_title('Learning Curve (F1 Score)')
    ax.legend(loc='best'); ax.grid(True)
    fig.tight_layout(); fig.savefig(os.path.join(config.output_dir, "learning_curve_f1.png"), dpi=300)
    plt.close(fig)
    logger.info("Saved learning curve plot.")

    # Error curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, 1 - train_mean, 'o-', label='Training Error')
    ax.plot(train_sizes, 1 - valid_mean, 'o-', label='Validation Error')
    ax.set_xlabel('Training Samples')
    ax.set_ylabel('Error Rate')
    ax.set_title('Error Curve')
    ax.legend(loc='best'); ax.grid(True)
    fig.tight_layout(); fig.savefig(os.path.join(config.output_dir, "error_curve.png"), dpi=300)
    plt.close(fig)
    logger.info("Saved error curve plot.")


def plot_detailed_roc_curves(model: CRF, config: DiagnosticsConfig):
    """
    Plot micro, macro, and per-class ROC curves for the test set.
    """
    logger = logging.getLogger("diagnose")
    logger.info("Computing detailed ROC curves...")

    X_test, y_true_seqs, _ = _load_and_prepare_data(config.test_path)
    labels = sorted(model.classes_)
    n_classes = len(labels)

    y_true_flat = [tag for seq in y_true_seqs for tag in seq]
    y_true_bin = label_binarize(y_true_flat, classes=labels)

    marginals = model.predict_marginals(X_test)
    probs = [token_probs[l] for seq in marginals for token_probs in seq for l in labels]
    y_score = np.array(probs).reshape(-1, n_classes)

    fpr, tpr, roc_auc = {}, {}, {}
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    for i, label in enumerate(labels):
        if label in y_true_flat:
            fpr[label], tpr[label], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[label] = auc(fpr[label], tpr[label])
        else:
            fpr[label], tpr[label], roc_auc[label] = np.array([0]), np.array([0]), 0.0

    all_fpr = np.unique(np.concatenate(list(fpr.values())))
    mean_tpr = sum(np.interp(all_fpr, fpr[l], tpr[l]) for l in labels) / n_classes
    fpr["macro"], tpr["macro"], roc_auc["macro"] = all_fpr, mean_tpr, auc(all_fpr, mean_tpr)

    # Macro and micro curves
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr["micro"], tpr["micro"], linestyle=':', linewidth=4, label=f'Micro (AUC={roc_auc["micro"]:.3f})')
    ax.plot(fpr["macro"], tpr["macro"], linestyle=':', linewidth=4, label=f'Macro (AUC={roc_auc["macro"]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('Micro & Macro ROC Curves')
    ax.legend(loc='lower right'); ax.grid(True)
    fig.tight_layout(); fig.savefig(os.path.join(config.output_dir, "roc_curve_averages.png"), dpi=300)
    plt.close(fig)
    logger.info("Saved average ROC curves.")

    # Per-class curves
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = cycle(plt.get_cmap('tab20').colors)
    for label, color in zip(labels, colors):
        ax.plot(fpr[label], tpr[label], lw=2, label=f'{label} (AUC={roc_auc[label]:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('Per-Class ROC Curves')
    ax.legend(loc='lower right', prop={'size': 8}); ax.grid(True)
    fig.tight_layout(); fig.savefig(os.path.join(config.output_dir, "roc_curve_per_class.png"), dpi=300)
    plt.close(fig)
    logger.info("Saved per-class ROC curves.")


def generate_misclassification_report(model: CRF, config: DiagnosticsConfig):
    """
    Save a CSV report detailing individual token misclassifications.
    """
    logger = logging.getLogger("diagnose")
    logger.info("Generating misclassification report...")

    X, y_true, sentences = _load_and_prepare_data(config.test_path)
    y_pred = predict_tags(model, X)

    header = ["Sentence ID", "Token ID", "Token", "Previous Token", "Next Token", "True Label", "Predicted Label"]
    rows = []

    for i, (sent, true_seq, pred_seq) in enumerate(zip(sentences, y_true, y_pred)):
        if len(rows) >= config.misclass_max_samples:
            break
        tokens = sent2tokens(sent)
        for j, (tok, t_lbl, p_lbl) in enumerate(zip(tokens, true_seq, pred_seq)):
            if t_lbl != p_lbl:
                prev_tok = tokens[j-1] if j > 0 else "<BOS>"
                next_tok = tokens[j+1] if j < len(tokens)-1 else "<EOS>"
                rows.append([i, j, tok, prev_tok, next_tok, t_lbl, p_lbl])
                if len(rows) >= config.misclass_max_samples:
                    break

    mis_path = os.path.join(config.output_dir, "misclassifications.csv")
    with open(mis_path, 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    logger.info(f"Misclassification report saved to: {mis_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive diagnostic script for CRF models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", required=True, help="Path to trained CRF model (.pkl).")
    parser.add_argument("--train", required=True, help="Path to training data (.conll).")
    parser.add_argument("--test", required=True, help="Path to test data (.conll).")
    parser.add_argument("--output-dir", default="outputs/diagnose_v2", help="Directory for all diagnostic outputs.")
    parser.add_argument("--top-n", type=int, default=15, help="Number of top features/transitions to display.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of folds for cross-validation.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs to run (-1 uses all cores).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--c1", type=float, default=0.1, help="L1 regularization coefficient.")
    parser.add_argument("--c2", type=float, default=0.1, help="L2 regularization coefficient.")
    parser.add_argument("--max-iter", type=int, default=100, help="Maximum training iterations.")
    parser.add_argument("--cm-normalize", choices=['true', 'pred'], default='true',
                        help="Normalization method for confusion matrix.")
    parser.add_argument("--misclass-max-samples", type=int, default=100,
                        help="Max number of misclassification samples in report.")
    parser.add_argument("--verbose", action="store_true", help="Enable INFO-level logging.")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
        level=logging.INFO if args.verbose else logging.WARNING,
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    config = DiagnosticsConfig(
        model_path=args.model,
        train_path=args.train,
        test_path=args.test,
        output_dir=args.output_dir,
        cv_folds=args.cv_folds,
        n_jobs=args.n_jobs,
        seed=args.seed,
        c1=args.c1,
        c2=args.c2,
        max_iter=args.max_iter,
        cm_normalize=args.cm_normalize,
        misclass_max_samples=args.misclass_max_samples,
        top_n=args.top_n
    )

    ensure_dir(config.output_dir)
    logger = logging.getLogger("diagnose")
    logger.info("Loading model and data...")
    crf_model = load_crf_model(config.model_path)
    X_train, y_train, sents_train = _load_and_prepare_data(config.train_path)
    X_test, y_test, sents_test = _load_and_prepare_data(config.test_path)
    logger.info("Load complete.")

    _print_header("MODEL OVERVIEW")
    print(f"Model class: {type(crf_model).__name__}")
    print("Model parameters:", crf_model.get_params())
    _print_top_transitions(crf_model, config.top_n)
    _print_top_features(crf_model, config.top_n)

    evaluate_split(crf_model, sents_train, X_train, y_train, "train", config)
    evaluate_split(crf_model, sents_test, X_test, y_test, "test", config)
    cross_validation(config)
    plot_learning_curve_and_error(config)
    plot_detailed_roc_curves(crf_model, config)
    generate_misclassification_report(crf_model, config)

    logger.info("All diagnostics complete.")

if __name__ == "__main__":
    main()
