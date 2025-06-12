"""
HMM N-gram Counts ve Probability Calculator

Bu modül Hidden Markov Model için gerekli geçiş ve emisyon olasılıklarını hesaplar.
Smoothing teknikleri ve OOV (Out-of-Vocabulary) handling içerir.
"""

from collections import defaultdict, Counter
import math


class HMMCounts:
    """
    HMM için n-gram sayımlarını tutan ve olasılık hesaplayan sınıf.
    
    Bu sınıf şu olasılıkları hesaplar:
    - Transition probabilities: P(tag_i | tag_{i-1})
    - Emission probabilities: P(word | tag)
    """
    
    def __init__(self):
        """
        HMMCounts constructor. Tüm sayaçları initialize eder.
        """
        # P(tag_i | tag_{i-1}) için sayımlar
        self.transition_counts = defaultdict(Counter)
        
        # P(word | tag) için sayımlar
        self.emission_counts = defaultdict(Counter)
        
        # Her tag'in toplam sayısı
        self.tag_counts = Counter()
        
        # Vocabulary (tüm unique kelimeler)
        self.vocab = set()
        
        # İstatistikler
        self.total_tokens = 0
        self.total_sentences = 0
    
    def count_from_sentences(self, sentences):
        """
        Cümlelerden n-gram sayımlarını çıkarır.
        
        Args:
            sentences (list): Her cümle için token listesi içeren liste.
                            Token format: {'form': str, 'upos': str}
        
        Example:
            >>> counts = HMMCounts()
            >>> sentences = [[{'form': 'Bu', 'upos': 'PRON'}, {'form': 'kitap', 'upos': 'NOUN'}]]
            >>> counts.count_from_sentences(sentences)
        """
        print("📊 N-gram sayımları hesaplanıyor...")
        
        for sentence in sentences:
            if not sentence:  # Boş cümleleri atla
                continue
                
            self.total_sentences += 1
            
            # Cümle başlangıcı için özel tag
            prev_tag = "<START>"
            self.tag_counts[prev_tag] += 1
            
            for token in sentence:
                word = token['form'].lower()  # Kelimeleri küçük harfe çevir
                tag = token['upos']
                
                # Sayımları güncelle
                self.transition_counts[prev_tag][tag] += 1
                self.emission_counts[tag][word] += 1
                self.tag_counts[tag] += 1
                self.vocab.add(word)
                
                self.total_tokens += 1
                prev_tag = tag
            
            # Cümle sonu için özel tag
            self.transition_counts[prev_tag]["<END>"] += 1
        
        self._print_count_statistics()
    
    def get_transition_prob(self, prev_tag, curr_tag, smoothing=0.1):
        """
        İki tag arasındaki geçiş olasılığını hesaplar.
        
        P(curr_tag | prev_tag) = (count(prev_tag, curr_tag) + smoothing) / 
                                 (count(prev_tag) + smoothing * |tagset|)
        
        Args:
            prev_tag (str): Önceki tag
            curr_tag (str): Şimdiki tag
            smoothing (float): Add-k smoothing parametresi
        
        Returns:
            float: Log olasılık değeri
        """
        numerator = self.transition_counts[prev_tag][curr_tag] + smoothing
        denominator = sum(self.transition_counts[prev_tag].values()) + smoothing * len(self.tag_counts)
        
        # Sıfır bölme kontrolü
        if denominator == 0:
            return float('-inf')
        
        return math.log(numerator / denominator)
    
    def get_emission_prob(self, tag, word, smoothing=0.1):
        """
        Tag'den kelime çıkış olasılığını hesaplar.
        
        P(word | tag) = (count(tag, word) + smoothing) / 
                        (count(tag) + smoothing * |vocab|)
        
        Args:
            tag (str): POS tag
            word (str): Kelime
            smoothing (float): Add-k smoothing parametresi
        
        Returns:
            float: Log olasılık değeri
        """
        word = word.lower()
        
        if word in self.vocab:
            # Bilinen kelime
            numerator = self.emission_counts[tag][word] + smoothing
            denominator = self.tag_counts[tag] + smoothing * len(self.vocab)
        else:
            # Bilinmeyen kelime (OOV)
            unk_prob = self._get_unk_prob(tag, word)
            numerator = unk_prob + smoothing
            denominator = self.tag_counts[tag] + smoothing * len(self.vocab)
        
        # Sıfır bölme kontrolü
        if denominator == 0:
            return float('-inf')
        
        return math.log(numerator / denominator)
    
    def _get_unk_prob(self, tag, word):
        """
        Bilinmeyen kelimeler için olasılık hesaplar.
        Türkçe morfolojik analiz kullanır.
        
        Args:
            tag (str): POS tag
            word (str): Bilinmeyen kelime
        
        Returns:
            float: OOV olasılık skoru (0-1 arası)
        """
        # Türkçe suffix analizi tabanlı OOV handling
        suffix_scores = {
            'VERB': self._check_verb_suffixes(word),
            'NOUN': self._check_noun_suffixes(word),
            'ADJ': self._check_adj_suffixes(word),
            'ADV': self._check_adv_suffixes(word),
            'PROPN': self._check_propn_features(word),
            'NUM': self._check_num_features(word),
        }
        
        # İlgili tag için score, yoksa default
        return suffix_scores.get(tag, 0.05)
    
    def _check_verb_suffixes(self, word):
        """Fiil suffixlerini kontrol eder."""
        verb_suffixes = ['mak', 'mek', 'yor', 'dı', 'di', 'du', 'dü', 
                        'tı', 'ti', 'tu', 'tü', 'r', 'ır', 'ir', 'ur', 'ür',
                        'miş', 'muş', 'müş', 'mış']
        
        for suffix in verb_suffixes:
            if word.endswith(suffix):
                return 0.7
        return 0.1
    
    def _check_noun_suffixes(self, word):
        """İsim suffixlerini kontrol eder."""
        noun_suffixes = ['lar', 'ler', 'lık', 'lik', 'luk', 'lük',
                        'da', 'de', 'ta', 'te', 'dan', 'den', 'tan', 'ten',
                        'nın', 'nin', 'nun', 'nün', 'ın', 'in', 'un', 'ün']
        
        for suffix in noun_suffixes:
            if word.endswith(suffix):
                return 0.6
        return 0.2
    
    def _check_adj_suffixes(self, word):
        """Sıfat suffixlerini kontrol eder."""
        adj_suffixes = ['li', 'lı', 'lu', 'lü', 'sız', 'siz', 'suz', 'süz',
                       'sal', 'sel', 'ki']
        
        for suffix in adj_suffixes:
            if word.endswith(suffix):
                return 0.5
        return 0.1
    
    def _check_adv_suffixes(self, word):
        """Zarf suffixlerini kontrol eder."""
        adv_suffixes = ['ca', 'ce', 'ça', 'çe', 'arak', 'erek']
        
        for suffix in adv_suffixes:
            if word.endswith(suffix):
                return 0.6
        return 0.1
    
    def _check_propn_features(self, word):
        """Özel isim özelliklerini kontrol eder."""
        # İlk harf büyük mü?
        if word and word[0].isupper():
            return 0.8
        return 0.1
    
    def _check_num_features(self, word):
        """Sayı özelliklerini kontrol eder."""
        # Rakam içeriyor mu?
        if any(c.isdigit() for c in word):
            return 0.9
        
        # Sayı kelimeleri
        number_words = ['bir', 'iki', 'üç', 'dört', 'beş', 'altı', 'yedi', 
                       'sekiz', 'dokuz', 'on', 'yüz', 'bin', 'milyon']
        if word.lower() in number_words:
            return 0.8
            
        return 0.1
    
    def _print_count_statistics(self):
        """Sayım istatistiklerini yazdırır."""
        print(f"\n📈 HMM Sayım İstatistikleri:")
        print(f"{'='*40}")
        print(f"Toplam cümle: {self.total_sentences:,}")
        print(f"Toplam token: {self.total_tokens:,}")
        print(f"Unique kelime: {len(self.vocab):,}")
        print(f"POS tag sayısı: {len(self.tag_counts):,}")
        print(f"Transition pairs: {sum(len(v) for v in self.transition_counts.values()):,}")
        print(f"Emission pairs: {sum(len(v) for v in self.emission_counts.values()):,}")
        
        # En sık tag'leri göster
        most_common_tags = self.tag_counts.most_common(5)
        print(f"\nEn sık POS tag'leri:")
        for tag, count in most_common_tags:
            percentage = (count / self.total_tokens) * 100
            print(f"  {tag}: {count:,} (%{percentage:.1f})")
    
    def get_tag_vocabulary(self, tag):
        """
        Belirli bir tag için kelime dağarcığını döndürür.
        
        Args:
            tag (str): POS tag
        
        Returns:
            set: Bu tag ile görülen unique kelimeler
        """
        return set(self.emission_counts[tag].keys())
    
    def get_most_likely_tags(self, word, top_k=3):
        """
        Bir kelime için en olası tag'leri döndürür.
        
        Args:
            word (str): Kelime
            top_k (int): Kaç tane tag döndürülecek
        
        Returns:
            list: (tag, olasılık) tuple'ları
        """
        word = word.lower()
        tag_probs = []
        
        for tag in self.tag_counts:
            prob = self.get_emission_prob(tag, word)
            tag_probs.append((tag, prob))
        
        # En yüksek olasılıklı tag'leri döndür
        tag_probs.sort(key=lambda x: x[1], reverse=True)
        return tag_probs[:top_k] 