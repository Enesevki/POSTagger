# src/train.py

import argparse
import logging
import os

from data_loader import read_conll, train_dev_test_split, write_conll
from features import sent2features, sent2labels
from model import build_crf_model, train_crf
from utils import ensure_dir

def parse_args():
    parser = argparse.ArgumentParser(description="Train a CRF-based POS tagger")
    parser.add_argument("--train",    required=True, help="Path to train.conll (or raw .conll to be split)")
    parser.add_argument("--dev",      required=False, help="Path to dev.conll (no longer used in training)")
    parser.add_argument("--test",     required=False, help="Path to test.conll (optional; not used here)")
    parser.add_argument("--out-model", required=True, help="Where to save the trained model (outputs/models/crf.pkl)")
    parser.add_argument("--split",    action="store_true", help="If set, split raw --train into train/dev/test")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="When --split: fraction for training")
    parser.add_argument("--dev-ratio",   type=float, default=0.1, help="When --split: fraction for dev")
    parser.add_argument("--test-ratio",  type=float, default=0.1, help="When --split: fraction for test")
    parser.add_argument("--c1", type=float, default=0.1, help="CRF L1 regularization coefficient")
    parser.add_argument("--c2", type=float, default=0.1, help="CRF L2 regularization coefficient")
    parser.add_argument("--max-iter", type=int, default=100, help="Maximum number of training iterations")
    parser.add_argument("--verbose", action="store_true", help="Turn on INFO logging")
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO if args.verbose else logging.WARNING
    )
    logger = logging.getLogger("train")

    # Eğer raw .conll dosyasını split edeceksek
    if args.split:
        logger.info("Splitting raw data into train/dev/test")
        all_sents = read_conll(args.train)
        train_sents, dev_sents, test_sents = train_dev_test_split(
            all_sents,
            train_ratio=args.train_ratio,
            dev_ratio=args.dev_ratio,
            test_ratio=args.test_ratio
        )
        processed_dir = os.path.abspath(os.path.join(os.path.dirname(args.out_model), "..", "data", "processed"))
        ensure_dir(processed_dir)
        write_conll(train_sents, os.path.join(processed_dir, "train.conll"))
        write_conll(dev_sents,   os.path.join(processed_dir, "dev.conll"))
        if args.test:
            write_conll(test_sents, os.path.join(processed_dir, "test.conll"))
        train_path = os.path.join(processed_dir, "train.conll")
    else:
        train_path = args.train

    # Train verisini oku
    logger.info(f"Loading train data from {train_path}")
    train_sents = read_conll(train_path)

    # Özellik çıkarımı
    logger.info("Extracting features")
    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s)   for s in train_sents]

    # Model inşa et ve eğit
    logger.info("Building CRF model")
    crf = build_crf_model(
        c1=args.c1,
        c2=args.c2,
        max_iterations=args.max_iter
    )
    logger.info("Training CRF model")
    crf = train_crf(
        crf,
        X_train, y_train,
        save_path=args.out_model       # **Artık burası tek argüman**
    )
    logger.info(f"Model training complete — saved to {args.out_model}")

if __name__ == "__main__":
    main()
