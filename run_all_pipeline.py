#!/usr/bin/env python3
"""
HMM POS Tagger - Tüm Pipeline Script

Bu script, veri ön işleme, model eğitimi, değerlendirme ve interaktif demo adımlarını
sırasıyla çalıştırır.

Usage:
    python run_all_pipeline.py
"""

import os
import sys
import time
import subprocess

# Import için gerekli path ayarı
sys.path.append(os.path.abspath('.'))

# Script dizinleri
SCRIPTS_DIR = 'scripts'

def run_script(script_name):
    """
    Belirtilen scripti çalıştırır ve çıktıyı ekrana yazdırır.
    
    Args:
        script_name (str): Çalıştırılacak script dosyası
    """
    print(f"\n{'='*60}")
    print(f"🚀 {script_name} çalıştırılıyor...")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    result = subprocess.run(['python', os.path.join(SCRIPTS_DIR, script_name)], 
                            capture_output=True, text=True)
    end_time = time.time()
    
    print(result.stdout)
    if result.stderr:
        print(f"⚠️  Hata çıktısı:\n{result.stderr}")
    
    print(f"\n⏱️  Süre: {end_time - start_time:.2f} saniye")
    print(f"{'='*60}\n")
    
    return result.returncode == 0

def main():
    """
    Ana pipeline fonksiyonu.
    """
    print("🔬 HMM POS Tagger - Tüm Pipeline Başlıyor")
    print("=" * 60)
    
    # 1. Veri ön işleme
    if not run_script('01_preprocess_data.py'):
        print("❌ Veri ön işleme başarısız oldu. Pipeline durduruluyor.")
        return
    
    # 2. Model eğitimi
    if not run_script('02_train_model.py'):
        print("❌ Model eğitimi başarısız oldu. Pipeline durduruluyor.")
        return
    
    # 3. Model değerlendirmesi
    if not run_script('04_evaluate.py'):
        print("❌ Model değerlendirmesi başarısız oldu. Pipeline durduruluyor.")
        return
    
    # 4. İnteraktif demo (opsiyonel)
    demo = input("\n🎮 İnteraktif demo çalıştırılsın mı? (E/H): ").strip().upper()
    if demo == 'E':
        run_script('05_interactive_demo.py')
    
    print("\n🎉 Pipeline başarıyla tamamlandı!")

if __name__ == '__main__':
    main() 