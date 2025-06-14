# src/evaluate.py

import argparse
import logging
import os

from src.data_loader import read_conll
from src.features import sent2features, sent2labels
from src.model import load_crf_model, evaluate_model
from src.utils import save_json, ensure_dir

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained CRF POS tagger")
    parser.add_argument(
        "--model", required=True,
        help="Yüklenecek modelin yolu (örn. outputs/models/crf.pkl)"
    )
    parser.add_argument(
        "--test", required=True,
        help="Test seti ConLL dosyası (data/processed/test.conll)"
    )
    parser.add_argument(
        "--out-report", required=True,
        help="Değerlendirme raporunu kaydedeceğimiz JSON dosyası (örn. outputs/reports/eval.json)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="INFO log seviyesini açar"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO if args.verbose else logging.WARNING
    )
    logger = logging.getLogger("evaluate")

    # 1) Modeli yükle
    logger.info(f"Loading model from {args.model}")
    crf = load_crf_model(args.model)

    # 2) Test verisini yükle
    logger.info(f"Loading test data from {args.test}")
    test_sents = read_conll(args.test)

    # 3) Özellik ve etiket listelerini hazırla
    logger.info("Extracting features and labels for test set")
    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s)   for s in test_sents]

    # 4) Değerlendirmeyi yap
    logger.info("Evaluating model")
    report = evaluate_model(crf, X_test, y_test)

    # 5) Raporu kaydet
    out_path = os.path.abspath(args.out_report)
    ensure_dir(os.path.dirname(out_path))
    save_json(report, out_path)
    logger.info(f"Evaluation report saved to {out_path}")

    # 6) Kısa özet (terminale bas)
    acc = report.get('accuracy', None)
    if acc is not None:
        print(f"Token-level accuracy: {acc:.4f}")
    else:
        # accuracy metriği flat_classification_report içinde yoksa
        # toplam doğru/tüm token sayısını hesaplamak gerekir
        pass

if __name__ == "__main__":
    main()
