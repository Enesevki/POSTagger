# src/train.py

import argparse
import logging
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_val_score, learning_curve
from sklearn.metrics import make_scorer
from sklearn_crfsuite.metrics import flat_f1_score

# Proje içinden importlar
from src.data_loader import read_conll
from src.features import sent2features, sent2labels
from src.model import build_crf_model, train_crf
from src.utils import ensure_dir

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a CRF model and/or evaluate its hyperparameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # DEĞİŞTİRİLDİ: Mod sayısı ikiye indirildi.
    parser.add_argument(
        "--mode", required=True, choices=['train', 'evaluate_hparams'],
        help=(
            "'train': Train a final model and generate its learning curves. "
            "'evaluate_hparams': Run K-fold CV to quickly evaluate hyperparameters."
        )
    )
    parser.add_argument("--data", required=True, help="Path to the .conll data file for training or evaluation.")
    parser.add_argument("--out-model", help="Path to save the trained model (required in 'train' mode).")
    parser.add_argument("--out-dir", default="outputs/training_analysis", help="Directory to save analysis plots like learning curves.")
    
    # Hiperparametreler
    parser.add_argument("--c1", type=float, default=0.1, help="CRF L1 regularization.")
    parser.add_argument("--c2", type=float, default=0.1, help="CRF L2 regularization.")
    parser.add_argument("--max-iter", type=int, default=100, help="Max iterations for training.")
    
    # CV'ye özel argüman
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of folds for CV or Learning Curve.")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs to use.")
    parser.add_argument("--verbose", action="store_true", help="Enable INFO level logging.")
    
    args = parser.parse_args()
    if args.mode == 'train' and not args.out_model:
        parser.error("--out-model is required when mode is 'train'")
    
    return args

def _load_and_prepare_data(conll_path: str):
    """Veriyi yükleyip özellik ve etiketleri çıkaran yardımcı fonksiyon."""
    sents = read_conll(conll_path)
    X = [sent2features(s) for s in sents]
    y = [sent2labels(s) for s in sents]
    return X, y

def run_cross_validation(args: argparse.Namespace):
    """K-katmanlı CV çalıştırır ve skorları konsola basar. Model veya grafik üretmez."""
    logger = logging.getLogger("train.evaluate_hparams")
    logger.info(f"Starting {args.cv_folds}-fold cross-validation on {args.data}")
    
    X, y = _load_and_prepare_data(args.data)
    crf = build_crf_model(c1=args.c1, c2=args.c2, max_iterations=args.max_iter)
    
    unique_labels = sorted({tag for seq in y for tag in seq})
    scorer = make_scorer(flat_f1_score, average='weighted', labels=unique_labels, zero_division=0)
    cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    
    logger.info("Calculating scores... (This may take a while)")
    scores = cross_val_score(crf, X, y, cv=cv, scoring=scorer, n_jobs=args.n_jobs)
    
    print("\n--- Cross-Validation Results ---")
    print(f"Hyperparameters: c1={args.c1}, c2={args.c2}, max_iter={args.max_iter}")
    print(f"F1 Scores per fold: {np.round(scores, 4)}")
    print(f"Mean F1 Score: {scores.mean():.4f}")
    print(f"Std Dev of F1 Score: {scores.std():.4f}")
    print("--------------------------------")
    logger.info("Evaluation complete.")

def train_final_model(args: argparse.Namespace):
    """Verilen veriyle tek bir nihai model eğitir ve kaydeder."""
    logger = logging.getLogger("train.train")
    logger.info(f"Training final model on {args.data}")

    X_train, y_train = _load_and_prepare_data(args.data)

    logger.info("Building CRF model with hyperparameters: c1=%.2f, c2=%.2f", args.c1, args.c2)
    crf = build_crf_model(c1=args.c1, c2=args.c2, max_iterations=args.max_iter)
    
    logger.info("Fitting the final model...")
    crf = train_crf(crf, X_train, y_train, save_path=args.out_model)
    logger.info(f"Final model training complete. Model saved to {args.out_model}")

def plot_learning_and_error_curves(args: argparse.Namespace):
    """Verilen hiperparametrelerle öğrenme ve hata eğrilerini oluşturur."""
    logger = logging.getLogger("train.curves")
    logger.info("Generating learning and error curves...")
    ensure_dir(args.out_dir)

    X, y = _load_and_prepare_data(args.data)
    crf = build_crf_model(c1=args.c1, c2=args.c2, max_iterations=args.max_iter)
    unique_labels = sorted({tag for seq in y for tag in seq})
    scorer = make_scorer(flat_f1_score, average='weighted', labels=unique_labels, zero_division=0)
    cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    train_sizes_abs, train_scores, valid_scores = learning_curve(
        crf, X, y, cv=cv, scoring=scorer,
        train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=args.n_jobs
    )
    
    # F1 Öğrenme Eğrisi Çizimi
    fig, ax = plt.subplots(figsize=(10, 6))
    train_scores_mean = np.mean(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    ax.plot(train_sizes_abs, train_scores_mean, 'o-', color="r", label='Eğitim F1 Skoru')
    ax.plot(train_sizes_abs, valid_scores_mean, 'o-', color="g", label='Çapraz Doğrulama F1 Skoru')
    ax.set_title(f'Learning Curve (c1={args.c1}, c2={args.c2})')
    ax.set_xlabel('Eğitim Örneği Sayısı'); ax.set_ylabel('F1 Skoru')
    ax.legend(loc='best'); ax.grid(True)
    save_path = os.path.join(args.out_dir, f"learning_curve_c1_{args.c1}_c2_{args.c2}.png")
    fig.savefig(save_path, dpi=300); plt.close(fig)
    logger.info(f"Learning curve saved to {save_path}")

    # Hata Eğrisi Çizimi
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes_abs, 1 - train_scores_mean, 'o-', label='Eğitim Hatası (1 - F1)')
    ax.plot(train_sizes_abs, 1 - valid_scores_mean, 'o-', label='Doğrulama Hatası (1 - F1)')
    ax.set_title(f'Error Curve (c1={args.c1}, c2={args.c2})')
    ax.set_xlabel('Eğitim Örneği Sayısı'); ax.set_ylabel('Hata Oranı'); ax.legend(loc='best'); ax.grid(True)
    save_path = os.path.join(args.out_dir, f"error_curve_c1_{args.c1}_c2_{args.c2}.png")
    fig.savefig(save_path, dpi=300); plt.close(fig)
    logger.info(f"Error curve saved to {save_path}")


def main():
    args = parse_args()
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO if args.verbose else logging.WARNING)
    logger = logging.getLogger("train")

    if args.mode == 'evaluate_hparams':
        run_cross_validation(args)
    elif args.mode == 'train':
        # Önce grafikleri çiz
        logger.info("Step 1/2: Generating learning curves for the specified hyperparameters...")
        plot_learning_and_error_curves(args)
        
        # Sonra nihai modeli eğit
        logger.info("Step 2/2: Training the final model with the same hyperparameters...")
        train_final_model(args)
        
        logger.info("Process complete.")

if __name__ == "__main__":
    main()