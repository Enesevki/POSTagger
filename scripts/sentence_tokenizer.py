import pandas as pd
import re

def tokenize_sentences(excel_path, output_path):
    """Excel'den cümleleri okuyup tokenleyerek conll formatında txt'ye yazar."""
    df = pd.read_excel(excel_path)
    column_name = df.columns[0]
    with open(output_path, "w", encoding="utf-8") as f:
        for sentence in df[column_name].dropna():
            tokens = re.findall(r'\w+|[^\s\w]', sentence, re.UNICODE)
            for token in tokens:
                f.write(f"{token}\t_\n")
            f.write("\n")
