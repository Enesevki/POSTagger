# src/data_loader.py

import os
import random
from typing import List, Tuple
import argparse 
def read_conll(file_path: str) -> List[List[Tuple[str, str]]]:
    """
    CoNLL-formatlı bir dosyayı okuyup her bir cümleyi
    token–etiket çiftlerinden oluşan bir liste olarak döner.
    """
    sentences = []
    current = []
    with open(file_path, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current:
                    sentences.append(current)
                    current = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    token, tag = parts[0], parts[-1]
                    current.append((token, tag))
        # Son cümle kontrolü
        if current:
            sentences.append(current)
    return sentences

def write_conll(sentences: List[List[Tuple[str, str]]], out_path: str) -> None:
    """
    Token–etiket çiftlerinden oluşan cümle listesini
    CoNLL formatında bir dosyaya yazar.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf8') as f:
        for sent in sentences:
            for token, tag in sent:
                f.write(f"{token} {tag}\n")
            f.write("\n")

def train_dev_test_split(
    data: List[List[Tuple[str, str]]],
    train_ratio: float = 0.8,
    dev_ratio:   float = 0.1,
    test_ratio:  float = 0.1,
    shuffle:    bool  = True,
    seed:       int   = 42
) -> Tuple[
    List[List[Tuple[str, str]]],
    List[List[Tuple[str, str]]],
    List[List[Tuple[str, str]]]
]:
    """
    data içindeki cümleleri train/dev/test oranlarına göre böler.
    """
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."
    if shuffle:
        random.seed(seed)
        random.shuffle(data)

    n = len(data)
    n_train = int(n * train_ratio)
    n_dev   = int(n * dev_ratio)

    train_sents = data[:n_train]
    dev_sents   = data[n_train:n_train + n_dev]
    test_sents  = data[n_train + n_dev:]
    return train_sents, dev_sents, test_sents

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a single CoNLL file into train, dev, and test sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input",  required=True, help="Path to the single large .conll file to be split.")
    parser.add_argument("--outdir", required=True, help="Directory to save the output files (e.g., data/processed).")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Fraction for the training set.")
    parser.add_argument("--dev-ratio",   type=float, default=0.1, help="Fraction for the development (validation) set.")
    parser.add_argument("--test-ratio",  type=float, default=0.1, help="Fraction for the test set.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    args = parser.parse_args()

    if not abs(args.train_ratio + args.dev_ratio + args.test_ratio - 1.0) < 1e-6:
        raise ValueError("The sum of train, dev, and test ratios must be 1.0.")

    # 1) Ham veriyi oku
    print(f"Reading all sentences from: {args.input}")
    all_sents = read_conll(args.input)
    print(f"Total sentences found: {len(all_sents)}")

    # 2) Veriyi böl
    train_sents, dev_sents, test_sents = train_dev_test_split(
        all_sents,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    print(f"Splitting data into Train/Dev/Test: {len(train_sents)}/{len(dev_sents)}/{len(test_sents)}")

    # 3) Bölünmüş dosyaları yaz
    os.makedirs(args.outdir, exist_ok=True)
    write_conll(train_sents, os.path.join(args.outdir, "train.conll"))
    write_conll(dev_sents,   os.path.join(args.outdir, "dev.conll"))
    write_conll(test_sents,  os.path.join(args.outdir, "test.conll"))

    print(f"Data successfully split and saved in: {args.outdir}")
