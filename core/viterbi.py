"""
Viterbi Algorithm Implementation

Bu modül Viterbi algoritması ile HMM'den en olası state sequence'ını (POS tag dizisi) 
bulur. Dynamic programming tabanlı optimal decoding yapar.
"""

import math
from collections import defaultdict


class ViterbiDecoder:
    """
    Viterbi algoritması ile POS tag sequence decoding yapan sınıf.
    
    Viterbi algoritması, HMM'de verilen observation sequence için
    en olası hidden state sequence'ını (POS tag dizisi) bulur.
    """
    
    def __init__(self, hmm_model):
        """
        ViterbiDecoder constructor.
        
        Args:
            hmm_model: Eğitilmiş HMMModel instance
        """
        self.model = hmm_model
        
        if not self.model.is_trained:
            raise ValueError("HMM model henüz eğitilmemiş!")
    
    def decode(self, words):
        """
        Viterbi algoritması ile kelime dizisi için en olası POS tag dizisini bulur.
        
        Algorithm:
        1. Initialize: İlk kelime için tüm tag'lerin initial probabilities
        2. Forward pass: Her kelime için tüm tag'leri optimize et  
        3. Backtrack: En iyi path'i geriye doğru izle
        
        Args:
            words (list): Etiketlenecek kelimeler listesi
        
        Returns:
            list: En olası POS tag dizisi
        
        Example:
            >>> decoder = ViterbiDecoder(trained_model)
            >>> tags = decoder.decode(['Bu', 'kitap', 'güzel'])
            >>> print(tags)  # ['PRON', 'NOUN', 'ADJ']
        """
        if not words:
            return []
        
        if len(words) == 1:
            # Tek kelime için basit tahmin
            return [self.model.predict_tag(words[0])]
        
        n = len(words)
        tags = list(self.model.tags)
        
        if not tags:
            raise ValueError("Model'de hiç POS tag yok!")
        
        # Dynamic Programming tabloları
        # viterbi[t][tag] = t zamanındaki tag için en iyi log probability
        viterbi = defaultdict(lambda: defaultdict(lambda: float('-inf')))
        
        # backpointer[t][tag] = t zamanındaki tag için en iyi önceki tag
        backpointer = defaultdict(dict)
        
        # Step 1: Initialize (t=0)
        self._initialize_viterbi(viterbi, backpointer, tags, words[0])
        
        # Step 2: Forward pass (t=1...n-1)
        for t in range(1, n):
            self._forward_step(viterbi, backpointer, tags, words[t], t)
        
        # Step 3: Find best final tag
        best_final_tag = self._find_best_final_tag(viterbi, tags, n-1)
        
        # Step 4: Backtrack to get best path
        best_path = self._backtrack_path(backpointer, best_final_tag, n-1)
        
        return best_path
    
    def _initialize_viterbi(self, viterbi, backpointer, tags, first_word):
        """
        Viterbi tablosunu ilk kelime için initialize eder.
        
        Args:
            viterbi: Viterbi DP tablosu
            backpointer: Backpointer tablosu
            tags: POS tag listesi
            first_word: İlk kelime
        """
        for tag in tags:
            # P(tag | <START>) * P(word | tag)
            trans_prob = self.model.get_transition_prob("<START>", tag)
            emis_prob = self.model.get_emission_prob(tag, first_word)
            
            viterbi[0][tag] = trans_prob + emis_prob
            backpointer[0][tag] = "<START>"
    
    def _forward_step(self, viterbi, backpointer, tags, word, t):
        """
        Viterbi forward step - t zamanındaki tüm tag'ler için optimize eder.
        
        Args:
            viterbi: Viterbi DP tablosu
            backpointer: Backpointer tablosu  
            tags: POS tag listesi
            word: t zamanındaki kelime
            t: Zaman adımı
        """
        for curr_tag in tags:
            max_prob = float('-inf')
            best_prev_tag = None
            
            # Tüm önceki tag'lerden geçişleri dene
            for prev_tag in tags:
                # P(curr_tag | prev_tag) * P(word | curr_tag) * viterbi[t-1][prev_tag]
                trans_prob = self.model.get_transition_prob(prev_tag, curr_tag)
                emis_prob = self.model.get_emission_prob(curr_tag, word)
                
                # Log space'te toplama
                total_prob = viterbi[t-1][prev_tag] + trans_prob + emis_prob
                
                if total_prob > max_prob:
                    max_prob = total_prob
                    best_prev_tag = prev_tag
            
            viterbi[t][curr_tag] = max_prob
            backpointer[t][curr_tag] = best_prev_tag
    
    def _find_best_final_tag(self, viterbi, tags, final_time):
        """
        Final time step'te en iyi tag'i bulur.
        
        Args:
            viterbi: Viterbi DP tablosu
            tags: POS tag listesi
            final_time: Son zaman adımı
        
        Returns:
            str: En iyi final tag
        """
        best_final_tag = None
        best_final_prob = float('-inf')
        
        for tag in tags:
            # Cümle sonu geçiş olasılığını da dahil et
            final_prob = viterbi[final_time][tag]
            end_prob = self.model.get_transition_prob(tag, "<END>")
            total_prob = final_prob + end_prob
            
            if total_prob > best_final_prob:
                best_final_prob = total_prob
                best_final_tag = tag
        
        return best_final_tag if best_final_tag else tags[0]
    
    def _backtrack_path(self, backpointer, best_final_tag, final_time):
        """
        En iyi path'i geriye doğru izleyerek tag sequence'ını bulur.
        
        Args:
            backpointer: Backpointer tablosu
            best_final_tag: En iyi final tag
            final_time: Son zaman adımı
        
        Returns:
            list: En iyi tag dizisi
        """
        path = [best_final_tag]
        
        # Geriye doğru en iyi path'i izle
        for t in range(final_time, 0, -1):
            prev_tag = backpointer[t][path[-1]]
            path.append(prev_tag)
        
        # Path'i ters çevir (START tag'ini çıkar)
        path.reverse()
        return path[1:]  # <START> tag'ini çıkar
    
    def decode_with_probabilities(self, words):
        """
        Kelime dizisi için tag sequence'ı ve olasılıklarını döndürür.
        
        Args:
            words (list): Etiketlenecek kelimeler
        
        Returns:
            tuple: (tag_sequence, log_probability)
        """
        if not words:
            return [], 0.0
        
        tags = self.decode(words)
        
        # Total log probability hesapla
        log_prob = 0.0
        prev_tag = "<START>"
        
        for word, tag in zip(words, tags):
            trans_prob = self.model.get_transition_prob(prev_tag, tag)
            emis_prob = self.model.get_emission_prob(tag, word)
            log_prob += trans_prob + emis_prob
            prev_tag = tag
        
        # Cümle sonu olasılığını ekle
        end_prob = self.model.get_transition_prob(prev_tag, "<END>")
        log_prob += end_prob
        
        return tags, log_prob
    
    def get_top_k_sequences(self, words, k=3):
        """
        En olası k tag sequence'ını döndürür.
        
        Not: Bu implementasyon basit bir yaklaşım kullanır.
        Gerçek k-best Viterbi daha karmaşıktır.
        
        Args:
            words (list): Etiketlenecek kelimeler
            k (int): Kaç sequence döndürülecek
        
        Returns:
            list: (tag_sequence, log_probability) tuple'ları
        """
        if not words or k <= 0:
            return []
        
        # İlk olarak en iyi sequence'ı bul
        best_tags, best_prob = self.decode_with_probabilities(words)
        results = [(best_tags, best_prob)]
        
        # Basit yaklaşım: Her pozisyonda 2. en iyi tag'i dene
        if k > 1 and len(words) > 1:
            for i in range(len(words)):
                # i. pozisyondaki tag'i değiştir
                alternative_tags = best_tags.copy()
                
                # Bu kelime için 2. en iyi tag'i bul
                word = words[i]
                tag_probs = []
                for tag in self.model.tags:
                    prob = self.model.get_emission_prob(tag, word)
                    tag_probs.append((tag, prob))
                
                tag_probs.sort(key=lambda x: x[1], reverse=True)
                
                if len(tag_probs) > 1:
                    second_best_tag = tag_probs[1][0]
                    alternative_tags[i] = second_best_tag
                    
                    # Bu sequence'ın olasılığını hesapla
                    alt_prob = self._calculate_sequence_probability(words, alternative_tags)
                    results.append((alternative_tags, alt_prob))
        
        # Olasılığa göre sırala ve k tane döndür
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def _calculate_sequence_probability(self, words, tags):
        """
        Verilen word-tag sequence'ının log probability'sini hesaplar.
        
        Args:
            words (list): Kelimeler
            tags (list): Tag'ler
        
        Returns:
            float: Log probability
        """
        if len(words) != len(tags):
            return float('-inf')
        
        log_prob = 0.0
        prev_tag = "<START>"
        
        for word, tag in zip(words, tags):
            trans_prob = self.model.get_transition_prob(prev_tag, tag)
            emis_prob = self.model.get_emission_prob(tag, word)
            log_prob += trans_prob + emis_prob
            prev_tag = tag
        
        # Cümle sonu
        end_prob = self.model.get_transition_prob(prev_tag, "<END>")
        log_prob += end_prob
        
        return log_prob
    
    def print_decoding_details(self, words, show_probabilities=True):
        """
        Decoding detaylarını yazdırır (debugging için).
        
        Args:
            words (list): Kelimeler
            show_probabilities (bool): Olasılıkları göster
        """
        if not words:
            print("Boş kelime listesi")
            return
        
        tags, total_prob = self.decode_with_probabilities(words)
        
        print(f"\n🔍 Viterbi Decoding Detayları:")
        print(f"{'='*50}")
        print(f"Kelime sayısı: {len(words)}")
        print(f"Toplam log probability: {total_prob:.4f}")
        print(f"Normalized probability: {math.exp(total_prob):.2e}")
        
        print(f"\nKelime → Tag eşleştirmesi:")
        for i, (word, tag) in enumerate(zip(words, tags)):
            if show_probabilities:
                emis_prob = self.model.get_emission_prob(tag, word)
                print(f"{i+1:2d}. {word:15s} → {tag:8s} (log_prob: {emis_prob:.3f})")
            else:
                print(f"{i+1:2d}. {word:15s} → {tag}")
        
        if show_probabilities:
            print(f"\nTransition probabilities:")
            prev_tag = "<START>"
            for i, tag in enumerate(tags):
                trans_prob = self.model.get_transition_prob(prev_tag, tag)
                print(f"{prev_tag:8s} → {tag:8s}: {trans_prob:.3f}")
                prev_tag = tag
            
            end_prob = self.model.get_transition_prob(prev_tag, "<END>")
            print(f"{prev_tag:8s} → <END>    : {end_prob:.3f}") 