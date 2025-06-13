"""
Hidden Markov Model Implementation

Bu modül HMM tabanlı POS tagger'ın ana model sınıfını içerir.
Model eğitimi, kaydetme, yükleme ve tahmin fonksiyonları sağlar.
"""

import pickle
import os
from .counts import HMMCounts


class HMMModel:
    """
    Hidden Markov Model ana sınıfı.
    
    Bu sınıf HMM'in tüm parametrelerini (transition ve emission probabilities)
    tutar ve model eğitimi, kaydetme/yükleme işlemlerini yapar.
    """
    
    def __init__(self):
        """
        HMMModel constructor. Model parametrelerini initialize eder.
        """
        self.counts = None          # HMMCounts instance
        self.tags = set()          # Tüm POS tag'leri
        self.vocab = set()         # Tüm kelimeler
        self.smoothing = 0.1       # Smoothing parametresi
        self.is_trained = False    # Model eğitildi mi?
        
        # Model meta bilgileri
        self.model_info = {
            'version': '1.0',
            'training_sentences': 0,
            'training_tokens': 0,
            'tag_count': 0,
            'vocab_size': 0
        }
        
        # Metadata alias for compatibility
        self.metadata = self.model_info
    
    def train(self, train_sentences, smoothing=0.1):
        """
        HMM modelini verilen cümlelerle eğitir.
        
        Args:
            train_sentences (list): Eğitim cümleleri listesi.
                                  Her cümle token dict'leri içerir.
            smoothing (float): Add-k smoothing parametresi
        
        Example:
            >>> model = HMMModel()
            >>> sentences = [[{'form': 'Bu', 'upos': 'PRON'}, {'form': 'kitap', 'upos': 'NOUN'}]]
            >>> model.train(sentences, smoothing=0.1)
        """
        if not train_sentences:
            raise ValueError("Eğitim cümleleri boş olamaz!")
        
        print(f"🚀 HMM Model Eğitimi Başlıyor...")
        print(f"Smoothing parametresi: {smoothing}")
        
        # HMM sayımlarını hesapla
        self.counts = HMMCounts()
        self.counts.count_from_sentences(train_sentences)
        
        # Model parametrelerini ayarla
        self.smoothing = smoothing
        self.tags = set(self.counts.tag_counts.keys())
        self.tags.discard("<START>")  # START tag'ini çıkar
        self.tags.discard("<END>")    # END tag'ini çıkar
        self.vocab = self.counts.vocab.copy()
        
        # Model bilgilerini güncelle
        self.model_info.update({
            'training_sentences': self.counts.total_sentences,
            'training_tokens': self.counts.total_tokens,
            'tag_count': len(self.tags),
            'vocab_size': len(self.vocab)
        })
        
        self.is_trained = True
        
        self._print_training_summary()
    
    def save(self, filepath):
        """
        Eğitilmiş modeli dosyaya kaydeder.
        
        Args:
            filepath (str): Model dosyasının kaydedileceği yol
        
        Raises:
            ValueError: Model eğitilmemişse
            IOError: Dosya yazma hatası
        """
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmemiş! Önce train() metodunu çağırın.")
        
        # Dosya dizinini oluştur
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        try:
            model_data = {
                'counts': self.counts,
                'tags': self.tags,
                'vocab': self.vocab,
                'smoothing': self.smoothing,
                'model_info': self.model_info,
                'is_trained': self.is_trained
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"✅ Model başarıyla kaydedildi: {filepath}")
            self._print_file_size(filepath)
            
        except Exception as e:
            raise IOError(f"Model kaydedilemedi: {e}")
    
    @classmethod
    def load(cls, filepath):
        """
        Kaydedilmiş modeli dosyadan yükler.
        
        Args:
            filepath (str): Yüklenecek model dosyasının yolu
        
        Raises:
            FileNotFoundError: Dosya bulunamadıysa
            ValueError: Model formatı hatalıysa
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Yeni model instance oluştur
            model = cls()
            
            # Model verilerini yükle
            model.counts = model_data['counts']
            model.tags = model_data['tags']
            model.vocab = model_data['vocab']
            model.smoothing = model_data['smoothing']
            model.model_info = model_data.get('model_info', {})
            model.metadata = model.model_info  # Alias
            model.is_trained = model_data.get('is_trained', True)
            
            print(f"✅ Model başarıyla yüklendi: {filepath}")
            model._print_model_info()
            
            return model
            
        except Exception as e:
            raise ValueError(f"Model yüklenemedi: {e}")
    
    def get_transition_prob(self, prev_tag, curr_tag):
        """
        İki tag arasındaki geçiş olasılığını döndürür.
        
        Args:
            prev_tag (str): Önceki tag
            curr_tag (str): Şimdiki tag
        
        Returns:
            float: Log geçiş olasılığı
        """
        if not self.is_trained:
            raise ValueError("Model eğitilmemiş!")
        
        return self.counts.get_transition_prob(prev_tag, curr_tag, self.smoothing)
    
    def get_emission_prob(self, tag, word):
        """
        Tag'den kelime emisyon olasılığını döndürür.
        
        Args:
            tag (str): POS tag
            word (str): Kelime
        
        Returns:
            float: Log emisyon olasılığı
        """
        if not self.is_trained:
            raise ValueError("Model eğitilmemiş!")
        
        return self.counts.get_emission_prob(tag, word, self.smoothing)
    
    def viterbi_decode(self, words):
        """
        Viterbi algoritması ile cümledeki kelimeler için en olası tag sequence'ini bulur.
        
        Args:
            words (list): Kelime listesi
        
        Returns:
            list: En olası POS tag sequence'i
        """
        if not self.is_trained:
            raise ValueError("Model eğitilmemiş!")
        
        from .viterbi import ViterbiDecoder
        decoder = ViterbiDecoder(self)
        return decoder.decode(words)
    
    def predict_tag(self, word):
        """
        Tek bir kelime için en olası tag'i tahmin eder.
        
        Args:
            word (str): Tahmin edilecek kelime
        
        Returns:
            str: En olası POS tag
        """
        if not self.is_trained:
            raise ValueError("Model eğitilmemiş!")
        
        best_tag = None
        best_prob = float('-inf')
        
        for tag in self.tags:
            prob = self.get_emission_prob(tag, word)
            if prob > best_prob:
                best_prob = prob
                best_tag = tag
        
        return best_tag if best_tag else 'NOUN'  # Fallback
    
    def get_model_statistics(self):
        """
        Model istatistiklerini döndürür.
        
        Returns:
            dict: Model istatistikleri
        """
        if not self.is_trained:
            return {"error": "Model eğitilmemiş"}
        
        stats = self.model_info.copy()
        stats.update({
            'smoothing_parameter': self.smoothing,
            'transition_pairs': sum(len(v) for v in self.counts.transition_counts.values()),
            'emission_pairs': sum(len(v) for v in self.counts.emission_counts.values()),
            'most_common_tags': self.counts.tag_counts.most_common(5)
        })
        
        return stats
    
    def _print_training_summary(self):
        """Eğitim özetini yazdırır."""
        print(f"\n🎯 Model Eğitimi Tamamlandı!")
        print(f"{'='*40}")
        print(f"Eğitim cümleleri: {self.model_info['training_sentences']:,}")
        print(f"Eğitim token'ları: {self.model_info['training_tokens']:,}")
        print(f"POS tag sayısı: {self.model_info['tag_count']}")
        print(f"Vocabulary boyutu: {self.model_info['vocab_size']:,}")
        print(f"Smoothing: {self.smoothing}")
        
        print(f"\nPOS Tag'leri: {', '.join(sorted(self.tags))}")
    
    def _print_model_info(self):
        """Yüklenen model bilgilerini yazdırır."""
        print(f"\n📋 Model Bilgileri:")
        print(f"{'='*30}")
        for key, value in self.model_info.items():
            if isinstance(value, (int, float)):
                if isinstance(value, int) and value > 1000:
                    print(f"{key}: {value:,}")
                else:
                    print(f"{key}: {value}")
            else:
                print(f"{key}: {value}")
    
    def _print_file_size(self, filepath):
        """Dosya boyutunu yazdırır."""
        try:
            size_bytes = os.path.getsize(filepath)
            if size_bytes > 1024 * 1024:  # MB
                size_str = f"{size_bytes / (1024*1024):.1f} MB"
            elif size_bytes > 1024:  # KB
                size_str = f"{size_bytes / 1024:.1f} KB"
            else:
                size_str = f"{size_bytes} bytes"
            
            print(f"📁 Dosya boyutu: {size_str}")
        except:
            pass
    
    def validate_model(self):
        """
        Model tutarlılığını kontrol eder.
        
        Returns:
            tuple: (is_valid, error_messages)
        """
        errors = []
        
        if not self.is_trained:
            errors.append("Model eğitilmemiş")
            return False, errors
        
        if not self.counts:
            errors.append("HMM counts bulunamadı")
        
        if not self.tags:
            errors.append("POS tag seti boş")
        
        if not self.vocab:
            errors.append("Vocabulary boş")
        
        if len(self.tags) < 2:
            errors.append("En az 2 POS tag gerekli")
        
        return len(errors) == 0, errors 