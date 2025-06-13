"""
CoNLL-U Format Corpus Reader

Bu modül CoNLL-U formatındaki dosyaları okumak ve parse etmek için kullanılır.
CoNLL-U formatı Universal Dependencies projesi tarafından kullanılan standart formattır.
"""

class CoNLLUReader:
    """
    CoNLL-U formatındaki dosyaları okuyup parse eden sınıf.
    
    CoNLL-U formatı şu sütunları içerir:
    ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC
    """
    
    def __init__(self, filepath):
        """
        CoNLLUReader constructor.
        
        Args:
            filepath (str): Okunacak CoNLL-U dosyasının yolu
        """
        self.filepath = filepath
        self._sentences = None  # Cache for sentences
    
    def __iter__(self):
        """Iterator interface for sentence iteration."""
        if self._sentences is None:
            self._sentences = self.read_sentences()
        return iter(self._sentences)
    
    def read_sentences(self):
        """
        CoNLL-U dosyasından tüm cümleleri okur ve parse eder.
        
        Returns:
            list: Her cümle için token listesi içeren liste.
                  Her token bir dict'tir: {'form': str, 'upos': str}
        
        Example:
            >>> reader = CoNLLUReader('data/train.conllu')
            >>> sentences = reader.read_sentences()
            >>> print(sentences[0])
            [{'form': 'Bu', 'upos': 'PRON'}, {'form': 'kitap', 'upos': 'NOUN'}]
        """
        sentences = []
        current_sentence = []
        
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Boş satır veya yorum satırı - cümle bitişi
                    if line.startswith('#') or line == '':
                        if current_sentence:
                            sentences.append(current_sentence)
                            current_sentence = []
                        continue
                    
                    # Token satırını parse et
                    token = self._parse_token_line(line, line_num)
                    if token:
                        current_sentence.append(token)
                
                # Dosya sonunda kalan cümle varsa ekle
                if current_sentence:
                    sentences.append(current_sentence)
        
        except FileNotFoundError:
            raise FileNotFoundError(f"CoNLL-U dosyası bulunamadı: {self.filepath}")
        except UnicodeDecodeError:
            raise UnicodeDecodeError(f"Dosya encoding hatası: {self.filepath}. UTF-8 encoding gerekli.")
        
        print(f"✅ {len(sentences)} cümle başarıyla okundu: {self.filepath}")
        self._sentences = sentences  # Cache sentences
        return sentences
    
    def _parse_token_line(self, line, line_num):
        """
        Tek bir token satırını parse eder.
        
        Args:
            line (str): Parse edilecek satır
            line_num (int): Satır numarası (hata raporlama için)
        
        Returns:
            dict or None: Token bilgilerini içeren dict veya hatalı satır için None
        """
        parts = line.split('\t')
        
        # CoNLL-U formatında en az 10 sütun olmalı
        if len(parts) < 10:
            print(f"⚠️  Hatalı satır format (satır {line_num}): {line[:50]}...")
            return None
        
        try:
            # Sadece basit token ID'leri al (1, 2, 3...), range'leri atla (1-2, 3-4...)
            token_id = parts[0]
            if '-' in token_id or '.' in token_id:
                return None
            
            form = parts[1]      # Kelimenin surface formu
            lemma = parts[2]     # Kelimenin lemması
            upos = parts[3]      # Universal POS tag
            
            # Boş alanları kontrol et
            if not form or not upos or form == '_' or upos == '_':
                print(f"⚠️  Eksik bilgi (satır {line_num}): form='{form}', upos='{upos}'")
                return None
            
            return {
                'form': form,
                'lemma': lemma if lemma != '_' else form,
                'upos': upos
            }
            
        except (IndexError, ValueError) as e:
            print(f"⚠️  Token parse hatası (satır {line_num}): {e}")
            return None
    
    def get_vocabulary(self, sentences=None):
        """
        Corpus'taki tüm unique kelimeleri döndürür.
        
        Args:
            sentences (list, optional): Cümle listesi. None ise dosyadan okur.
        
        Returns:
            set: Unique kelimeler seti (küçük harfe çevrilmiş)
        """
        if sentences is None:
            sentences = self.read_sentences()
        
        vocabulary = set()
        for sentence in sentences:
            for token in sentence:
                vocabulary.add(token['form'].lower())
        
        return vocabulary
    
    def get_tag_set(self, sentences=None):
        """
        Corpus'taki tüm unique POS etiketlerini döndürür.
        
        Args:
            sentences (list, optional): Cümle listesi. None ise dosyadan okur.
        
        Returns:
            set: Unique POS etiketleri seti
        """
        if sentences is None:
            sentences = self.read_sentences()
        
        tags = set()
        for sentence in sentences:
            for token in sentence:
                tags.add(token['upos'])
        
        return tags
    
    def print_statistics(self, sentences=None):
        """
        Corpus istatistiklerini yazdırır.
        
        Args:
            sentences (list, optional): Cümle listesi. None ise dosyadan okur.
        """
        if sentences is None:
            sentences = self.read_sentences()
        
        total_tokens = sum(len(sent) for sent in sentences)
        vocabulary = self.get_vocabulary(sentences)
        tags = self.get_tag_set(sentences)
        
        avg_sent_length = total_tokens / len(sentences) if sentences else 0
        
        print(f"\n📊 Corpus İstatistikleri - {self.filepath}")
        print(f"{'='*50}")
        print(f"Toplam cümle sayısı: {len(sentences):,}")
        print(f"Toplam token sayısı: {total_tokens:,}")
        print(f"Ortalama cümle uzunluğu: {avg_sent_length:.1f}")
        print(f"Unique kelime sayısı: {len(vocabulary):,}")
        print(f"POS etiket sayısı: {len(tags)}")
        print(f"POS etiketleri: {', '.join(sorted(tags))}")
    
    def get_sentence_count(self):
        """
        Corpus'taki toplam cümle sayısını döndürür.
        
        Returns:
            int: Cümle sayısı
        """
        if self._sentences is None:
            self._sentences = self.read_sentences()
        return len(self._sentences)
    
    def get_token_count(self):
        """
        Corpus'taki toplam token sayısını döndürür.
        
        Returns:
            int: Token sayısı
        """
        if self._sentences is None:
            self._sentences = self.read_sentences()
        return sum(len(sent) for sent in self._sentences)
    
    def get_tag_statistics(self):
        """
        POS tag'ların frekans istatistiklerini döndürür.
        
        Returns:
            dict: {tag: count} formatında istatistikler
        """
        if self._sentences is None:
            self._sentences = self.read_sentences()
        
        tag_counts = {}
        for sentence in self._sentences:
            for token in sentence:
                tag = token['upos']
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        return tag_counts
    
    def get_word_statistics(self):
        """
        Kelimelerin frekans istatistiklerini döndürür.
        
        Returns:
            dict: {word: count} formatında istatistikler
        """
        if self._sentences is None:
            self._sentences = self.read_sentences()
        
        word_counts = {}
        for sentence in self._sentences:
            for token in sentence:
                word = token['form'].lower()
                word_counts[word] = word_counts.get(word, 0) + 1
        
        return word_counts 