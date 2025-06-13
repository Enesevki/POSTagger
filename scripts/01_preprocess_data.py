#!/usr/bin/env python3
"""
Data Preprocessing Script - Excel to CoNLL-U Conversion

Bu script Excel dosyasındaki Türkçe cümleleri Stanza ile işleyerek
CoNLL-U formatına dönüştürür ve train/dev/test olarak böler.

Usage:
    python scripts/01_preprocess_data.py
    
Input:
    - data/raw/dataset.xlsx (Excel dosyası)
    
Output:
    - data/processed/train.conllu
    - data/processed/dev.conllu  
    - data/processed/test.conllu
"""

import os
import sys
import pandas as pd
import stanza
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Konfigürasyon
INPUT_EXCEL = 'data/raw/dataset.xlsx'
OUTPUT_DIR = 'data/processed/'
SENTENCE_COLUMN = 'Cümle'  # Excel'deki cümle sütunu adı
BATCH_SIZE = 50               # Stanza işleme batch boyutu
TEST_SIZE = 0.2              # %20 test
DEV_SIZE = 0.5               # Test'in %50'si dev (toplam %10)
RANDOM_STATE = 42            # Reproducible split


def setup_stanza():
    """
    Stanza Türkçe modelini yükler. Gerekirse indirir.
    
    Returns:
        stanza.Pipeline: Türkçe NLP pipeline
    """
    print("🔧 Stanza Türkçe modeli yükleniyor...")
    
    try:
        # Önce model var mı kontrol et
        nlp = stanza.Pipeline('tr', processors='tokenize,pos', verbose=False)
        print("✅ Stanza modeli başarıyla yüklendi")
        return nlp
    except Exception as e:
        print(f"⚠️  Stanza modeli bulunamadı, indiriliyor: {e}")
        try:
            stanza.download('tr')
            nlp = stanza.Pipeline('tr', processors='tokenize,pos', verbose=False)
            print("✅ Stanza modeli indirildi ve yüklendi")
            return nlp
        except Exception as e:
            raise RuntimeError(f"Stanza modeli yüklenemedi: {e}")


def load_excel_data(filepath):
    """
    Excel dosyasından cümleleri yükler ve temizler.
    
    Args:
        filepath (str): Excel dosyası yolu
    
    Returns:
        list: Temizlenmiş cümle listesi
    """
    print(f"📊 Excel dosyası okunuyor: {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Excel dosyası bulunamadı: {filepath}")
    
    try:
        # Excel dosyasını oku
        df = pd.read_excel(filepath)
        print(f"Excel dosyası okundu - {len(df)} satır")
        
        # Cümle sütununu kontrol et
        if SENTENCE_COLUMN not in df.columns:
            available_columns = ', '.join(df.columns.tolist())
            raise ValueError(f"'{SENTENCE_COLUMN}' sütunu bulunamadı. Mevcut sütunlar: {available_columns}")
        
        # Cümleleri çıkar ve temizle
        sentences = df[SENTENCE_COLUMN].dropna().tolist()
        
        # Temel temizlik
        cleaned_sentences = []
        for sent in sentences:
            if isinstance(sent, str):
                sent = sent.strip()
                # Çok kısa veya çok uzun cümleleri filtrele
                if 5 <= len(sent) <= 500 and len(sent.split()) >= 2:
                    cleaned_sentences.append(sent)
        
        print(f"✅ {len(cleaned_sentences)} geçerli cümle bulundu")
        return cleaned_sentences
        
    except Exception as e:
        raise RuntimeError(f"Excel dosyası okunamadı: {e}")


def process_sentences_with_stanza(sentences, nlp, batch_size=BATCH_SIZE):
    """
    Cümleleri Stanza ile işleyerek token bilgilerini çıkarır.
    
    Args:
        sentences (list): İşlenecek cümle listesi
        nlp: Stanza pipeline
        batch_size (int): Batch boyutu
    
    Returns:
        list: İşlenmiş cümle listesi (her cümle token dict'leri içerir)
    """
    print(f"🔤 Stanza ile {len(sentences)} cümle işleniyor...")
    
    processed_data = []
    failed_count = 0
    
    # Progress bar ile batch'ler halinde işle
    for i in tqdm(range(0, len(sentences), batch_size), desc="Processing batches"):
        batch = sentences[i:i+batch_size]
        
        for sent_text in batch:
            try:
                # Stanza ile işle
                doc = nlp(sent_text)
                tokens = []
                
                # Her sentence'daki word'leri al
                for sentence in doc.sentences:
                    for word in sentence.words:
                        tokens.append({
                            'text': word.text,
                            'lemma': word.lemma if word.lemma else word.text,
                            'upos': word.upos if word.upos else 'X'
                        })
                
                # En az 2 token olması gerekiyor
                if len(tokens) >= 2:
                    processed_data.append(tokens)
                else:
                    failed_count += 1
                    
            except Exception as e:
                failed_count += 1
                if failed_count <= 5:  # İlk 5 hatayı göster
                    print(f"⚠️  İşleme hatası: {sent_text[:50]}... -> {e}")
    
    print(f"✅ {len(processed_data)} cümle başarıyla işlendi")
    if failed_count > 0:
        print(f"⚠️  {failed_count} cümle işlenemedi")
    
    return processed_data


def to_conllu_format(processed_sentences):
    """
    İşlenmiş cümleleri CoNLL-U formatına dönüştürür.
    
    Args:
        processed_sentences (list): İşlenmiş cümle listesi
    
    Returns:
        str: CoNLL-U format string
    """
    conllu_lines = []
    
    for sent_id, sentence in enumerate(processed_sentences, 1):
        # Sentence metadata
        sentence_text = ' '.join([token['text'] for token in sentence])
        conllu_lines.append(f"# sent_id = {sent_id}")
        conllu_lines.append(f"# text = {sentence_text}")
        
        # Tokens
        for token_id, token in enumerate(sentence, 1):
            # CoNLL-U format: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
            line = (f"{token_id}\t{token['text']}\t{token['lemma']}\t{token['upos']}\t"
                   f"_\t_\t0\troot\t_\t_")
            conllu_lines.append(line)
        
        # Boş satır (sentence separator)
        conllu_lines.append("")
    
    return "\n".join(conllu_lines)


def split_and_save_data(processed_sentences):
    """
    Veriyi train/dev/test olarak böler ve CoNLL-U dosyalarını kaydeder.
    
    Args:
        processed_sentences (list): İşlenmiş cümle listesi
    """
    print(f"📂 Veri bölünüyor ve kaydediliyor...")
    
    # Çıktı dizinini oluştur
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Veriyi böl
    train_data, temp_data = train_test_split(
        processed_sentences, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    
    dev_data, test_data = train_test_split(
        temp_data, 
        test_size=DEV_SIZE, 
        random_state=RANDOM_STATE
    )
    
    # CoNLL-U dosyalarını kaydet
    datasets = [
        (train_data, f'{OUTPUT_DIR}/train.conllu', 'Training'),
        (dev_data, f'{OUTPUT_DIR}/dev.conllu', 'Development'),
        (test_data, f'{OUTPUT_DIR}/test.conllu', 'Test')
    ]
    
    total_tokens = 0
    for data, filename, description in datasets:
        conllu_content = to_conllu_format(data)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(conllu_content)
        
        tokens_in_split = sum(len(sent) for sent in data)
        total_tokens += tokens_in_split
        
        print(f"✅ {description:12s}: {len(data):4d} cümle, {tokens_in_split:5d} token → {filename}")
    
    print(f"\n📊 Veri Bölme Özeti:")
    print(f"{'='*50}")
    print(f"Toplam cümle: {len(processed_sentences):,}")
    print(f"Toplam token: {total_tokens:,}")
    print(f"Train: {len(train_data):,} cümle (%{len(train_data)/len(processed_sentences)*100:.1f})")
    print(f"Dev  : {len(dev_data):,} cümle (%{len(dev_data)/len(processed_sentences)*100:.1f})")
    print(f"Test : {len(test_data):,} cümle (%{len(test_data)/len(processed_sentences)*100:.1f})")


def validate_output_files():
    """
    Oluşturulan CoNLL-U dosyalarını doğrular.
    """
    print(f"\n🔍 Çıktı dosyaları doğrulanıyor...")
    
    files_to_check = [
        f'{OUTPUT_DIR}/train.conllu',
        f'{OUTPUT_DIR}/dev.conllu', 
        f'{OUTPUT_DIR}/test.conllu'
    ]
    
    for filepath in files_to_check:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            sentences = len([line for line in lines if line.startswith('# sent_id')])
            tokens = len([line for line in lines if line and not line.startswith('#') and line.strip()])
            
            file_size = os.path.getsize(filepath) / 1024  # KB
            print(f"✅ {os.path.basename(filepath):12s}: {sentences:3d} cümle, {tokens:4d} token, {file_size:.1f} KB")
        else:
            print(f"❌ {filepath} dosyası bulunamadı!")


def print_sample_output():
    """
    Örnek çıktı gösterir.
    """
    train_file = f'{OUTPUT_DIR}/train.conllu'
    if os.path.exists(train_file):
        print(f"\n📋 Örnek çıktı (train.conllu'dan ilk birkaç satır):")
        print(f"{'='*60}")
        
        with open(train_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # İlk cümleyi göster
        for i, line in enumerate(lines):
            if i >= 15:  # İlk 15 satır
                break
            print(line.rstrip())


def main():
    """
    Ana işlem fonksiyonu.
    """
    print("🚀 Veri Ön İşleme Başlıyor")
    print("=" * 50)
    
    try:
        # 1. Stanza'yı hazırla
        nlp = setup_stanza()
        
        # 2. Excel verilerini yükle
        sentences = load_excel_data(INPUT_EXCEL)
        
        # 3. Stanza ile işle
        processed_sentences = process_sentences_with_stanza(sentences, nlp)
        
        if not processed_sentences:
            raise ValueError("Hiç cümle işlenemedi!")
        
        # 4. Veriyi böl ve kaydet
        split_and_save_data(processed_sentences)
        
        # 5. Doğrulama
        validate_output_files()
        
        # 6. Örnek çıktı
        print_sample_output()
        
        print(f"\n🎉 Veri ön işleme başarıyla tamamlandı!")
        print(f"Sonraki adım: python scripts/02_train_model.py")
        
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 