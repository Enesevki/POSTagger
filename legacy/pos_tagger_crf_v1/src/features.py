# src/features.py

import re
from typing import List, Tuple, Dict, Any

def get_word_shape(word: str) -> str:
    """
    Compute a simplified shape of the word:
     - Uppercase letters → 'X'
     - Lowercase letters → 'x'
     - Digits           → 'd'
     - Other chars remain as-is
    Then compress repeated runs (e.g. 'Xxxxddd' → 'Xxd').
    """
    shape_chars = []
    for ch in word:
        if ch.isupper():
            shape_chars.append('X')
        elif ch.islower():
            shape_chars.append('x')
        elif ch.isdigit():
            shape_chars.append('d')
        else:
            shape_chars.append(ch)
    shape = ''.join(shape_chars)
    # Compress runs
    return re.sub(r'(.)\1+', r'\1', shape)

def token2features(
    sent: List[Tuple[Any, ...]],
    i: int
) -> Dict[str, Any]:
    """
    Extract a feature dict for the token at position i in the sentence.
    
    sent: List of tuples; each tuple is at least (word, tag), optionally
          (word, tag, morph_feats_str).
    """
    word = sent[i][0]
    features: Dict[str, Any] = {
        'bias': 1.0,
        'word.lower': word.lower(),
        'word[-3:]': word[-3:],
        'word[:3]': word[:3],
        'word.isupper': word.isupper(),
        'word.istitle': word.istitle(),
        'word.isdigit': word.isdigit(),
        'word.shape': get_word_shape(word),
        'word.isalpha': word.isalpha(),
        'word.has_hyphen': '-' in word,
    }

    # If morphological features are provided, add them as boolean flags
    if len(sent[i]) > 2:
        morph_feats = sent[i][2]
        for feat in morph_feats.split('|'):
            features[f'morph_{feat}'] = True

    # Position features
    if i == 0:
        features['BOS'] = True
    if i == len(sent) - 1:
        features['EOS'] = True

    # Previous token features
    if i > 0:
        prev_word = sent[i-1][0]
        features.update({
            '-1:word.lower': prev_word.lower(),
            '-1:word.istitle': prev_word.istitle(),
            '-1:word.isupper': prev_word.isupper(),
        })
    else:
        features['BOS'] = True

    # Next token features
    if i < len(sent) - 1:
        next_word = sent[i+1][0]
        features.update({
            '+1:word.lower': next_word.lower(),
            '+1:word.istitle': next_word.istitle(),
            '+1:word.isupper': next_word.isupper(),
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent: List[Tuple[Any, ...]]) -> List[Dict[str, Any]]:
    """
    Convert a sentence (list of token-tuples) into a list of feature dicts.
    """
    return [token2features(sent, i) for i in range(len(sent))]

def sent2labels(sent: List[Tuple[str, str]]) -> List[str]:
    """
    Extract the sequence of gold labels from a sentence.
    """
    return [label for (_token, label, *rest) in sent]

def sent2tokens(sent: List[Tuple[Any, ...]]) -> List[str]:
    """
    Extract just the words from a sentence.
    """
    return [token for (token, *rest) in sent]
