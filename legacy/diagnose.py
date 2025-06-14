# src/diagnose.py

import argparse
import logging
import os
import sys
import csv

# ensure project src is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold, learning_curve
from sklearn.metrics import (
    confusion_matrix, make_scorer, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn_crfsuite.metrics import flat_f1_score, flat_classification_report

from src.data_loader import read_conll
from src.features import sent2features, sent2labels, sent2tokens
from src.model import (
    load_crf_model, predict_tags, build_crf_model
)
from src.utils import ensure_dir, save_json


def evaluate_split(model_path, conll_path, split_name, output_dir):
    logger = logging.getLogger("diagnose")
    logger.info(f"Evaluating on {split_name} split")

    crf    = load_crf_model(model_path)
    sents  = read_conll(conll_path)
    X      = [sent2features(s) for s in sents]
    y_true = [sent2labels(s)   for s in sents]
    y_pred = predict_tags(crf, X)

    # filter labels that actually appear in true or pred
    all_labels = sorted(crf.classes_)
    labels = [
        l for l in all_labels
        if any(l in seq for seq in y_true) or any(l in seq for seq in y_pred)
    ]

    # classification report
    report = flat_classification_report(
        y_true, y_pred,
        labels=labels,
        digits=4,
        output_dict=True
    )
    report_path = os.path.join(output_dir, f"{split_name}_report.json")
    save_json(report, report_path)
    logger.info(f"{split_name} report saved: {report_path}")

    # confusion matrix
    y_true_flat = [t for seq in y_true for t in seq]
    y_pred_flat = [t for seq in y_pred for t in seq]
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=labels)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(cm, aspect='auto')
    ax.set_xticks(np.arange(len(labels))); ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90); ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix ({split_name})')
    fig.tight_layout()
    cm_path = os.path.join(output_dir, f"{split_name}_confusion_matrix.png")
    fig.savefig(cm_path)
    plt.close(fig)
    logger.info(f"{split_name} confusion matrix saved: {cm_path}")


def cross_validation(conll_path, cv_folds, n_jobs, seed, c1, c2, max_iter, output_dir):
    logger = logging.getLogger("diagnose")
    logger.info(f"Running {cv_folds}-fold cross-validation")
    sents = read_conll(conll_path)
    X = [sent2features(s) for s in sents]
    y = [sent2labels(s)   for s in sents]

    crf = build_crf_model(c1=c1, c2=c2, max_iterations=max_iter)
    unique_labels = sorted({tag for seq in y for tag in seq})
    scorer = make_scorer(flat_f1_score, average='weighted', labels=unique_labels)

    cv     = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(crf, X, y, cv=cv, scoring=scorer, n_jobs=n_jobs)
    report = {
        "fold_scores": scores.tolist(),
        "mean_f1":     float(np.mean(scores)),
        "std_f1":      float(np.std(scores))
    }
    cv_path = os.path.join(output_dir, "cross_val_report.json")
    save_json(report, cv_path)
    logger.info(f"Cross-validation report saved: {cv_path}")


def plot_learning_curve(conll_path, cv_folds, n_jobs, seed, c1, c2, max_iter, output_dir):
    logger = logging.getLogger("diagnose")
    logger.info("Generating learning curve")
    sents = read_conll(conll_path)
    X = [sent2features(s) for s in sents]
    y = [sent2labels(s)   for s in sents]

    crf = build_crf_model(c1=c1, c2=c2, max_iterations=max_iter)
    unique_labels = sorted({tag for seq in y for tag in seq})
    scorer = make_scorer(flat_f1_score, average='weighted', labels=unique_labels)
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    train_sizes, train_scores, valid_scores = learning_curve(
        crf, X, y, cv=cv, scoring=scorer,
        train_sizes=[0.1,0.3,0.5,0.7,1.0], n_jobs=n_jobs
    )

    # F1 learning curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Train F1')
    ax.plot(train_sizes, np.mean(valid_scores, axis=1), 'o-', label='Valid F1')
    ax.set_xlabel('Training Set Size'); ax.set_ylabel('F1 Score')
    ax.set_title('Learning Curve (F1)')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "learning_curve_f1.png"))
    plt.close(fig)
    logger.info("Learning curve (F1) saved")

    # Error curve = 1 - F1
    train_error = 1 - np.mean(train_scores, axis=1)
    valid_error = 1 - np.mean(valid_scores, axis=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_sizes, train_error, 'o-', label='Train Error')
    ax.plot(train_sizes, valid_error, 'o-', label='Valid Error')
    ax.set_xlabel('Training Set Size'); ax.set_ylabel('Error (1 - F1)')
    ax.set_title('Train/Test Error Curve')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "error_curve.png"))
    plt.close(fig)
    logger.info("Error curve saved")


def plot_roc_curve(model_path, test_path, output_dir):
    logger = logging.getLogger("diagnose")
    logger.info("Plotting ROC curve")
    crf = load_crf_model(model_path)
    sents = read_conll(test_path)
    X_test = [sent2features(s) for s in sents]
    y_true_seqs = [sent2labels(s) for s in sents]
    labels = sorted(crf.classes_)

    y_true_flat = [tag for seq in y_true_seqs for tag in seq]
    y_true_bin  = label_binarize(y_true_flat, classes=labels)

    marginals = crf.predict_marginals(X_test)
    probs = []
    for seq in marginals:
        for token_probs in seq:
            probs.append([token_probs[l] for l in labels])
    y_score = np.array(probs)

    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'micro-average ROC (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('Micro-average ROC Curve')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close(fig)
    logger.info("ROC curve saved")


def misclassification_report(model_path, test_path, output_dir, max_samples=100):
    logger = logging.getLogger("diagnose")
    logger.info("Generating misclassification report")
    crf = load_crf_model(model_path)
    sents = read_conll(test_path)
    X = [sent2features(s) for s in sents]
    y_true = [sent2labels(s) for s in sents]
    y_pred = predict_tags(crf, X)

    rows = []
    for i, (sent, true_seq, pred_seq) in enumerate(zip(sents, y_true, y_pred)):
        tokens = sent2tokens(sent)
        for j, (tok, t, p) in enumerate(zip(tokens, true_seq, pred_seq)):
            if t != p:
                prev_tok = tokens[j-1] if j > 0 else ""
                next_tok = tokens[j+1] if j < len(tokens)-1 else ""
                rows.append([i, j, tok, prev_tok, next_tok, t, p])
                if len(rows) >= max_samples:
                    break
        if len(rows) >= max_samples:
            break

    mis_path = os.path.join(output_dir, "misclassifications.csv")
    with open(mis_path, 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "SentenceIndex","TokenIndex","Token",
            "PrevToken","NextToken","TrueLabel","PredLabel"
        ])
        writer.writerows(rows)
    logger.info("Misclassification report saved")


def main():
    parser = argparse.ArgumentParser(description="Model diagnostics for POS tagger")
    parser.add_argument("--model",      required=True, help="Trained CRF model .pkl")
    parser.add_argument("--train",      required=True, help="Train split .conll")
    parser.add_argument("--test",       required=True, help="Test split .conll")
    parser.add_argument("--cv-folds",   type=int, default=5, help="Number of CV folds")
    parser.add_argument("--n-jobs",     type=int, default=1, help="Parallel jobs for CV")
    parser.add_argument("--seed",       type=int, default=42, help="Random seed")
    parser.add_argument("--c1",         type=float, default=0.1, help="CRF L1 regularization")
    parser.add_argument("--c2",         type=float, default=0.1, help="CRF L2 regularization")
    parser.add_argument("--max-iter",   type=int, default=100, help="CRF max iterations")
    parser.add_argument("--output-dir", default="outputs/diagnose", help="Diagnostics output directory")
    parser.add_argument("--verbose",    action="store_true", help="Enable INFO logging")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO if args.verbose else logging.WARNING
    )
    ensure_dir(args.output_dir)

    evaluate_split(args.model, args.train, "train", args.output_dir)
    evaluate_split(args.model, args.test,  "test",  args.output_dir)
    cross_validation(
        conll_path=args.train,
        cv_folds=args.cv_folds,
        n_jobs=args.n_jobs,
        seed=args.seed,
        c1=args.c1, c2=args.c2,
        max_iter=args.max_iter,
        output_dir=args.output_dir
    )
    plot_learning_curve(
        conll_path=args.train,
        cv_folds=args.cv_folds,
        n_jobs=args.n_jobs,
        seed=args.seed,
        c1=args.c1, c2=args.c2,
        max_iter=args.max_iter,
        output_dir=args.output_dir
    )
    plot_roc_curve(
        model_path=args.model,
        test_path=args.test,
        output_dir=args.output_dir
    )
    misclassification_report(
        model_path=args.model,
        test_path=args.test,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
