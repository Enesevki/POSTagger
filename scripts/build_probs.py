import json
import math
from collections import defaultdict

def build_transition_probs(transition_counts_path, tag_counts_path, transition_probs_path):
    with open(transition_counts_path, "r", encoding="utf-8") as f:
        transition_counts = json.load(f)
    with open(tag_counts_path, "r", encoding="utf-8") as f:
        tag_counts = json.load(f)
    all_tags = list(tag_counts.keys())
    tag_set_size = len(all_tags)
    transition_probs = defaultdict(dict)
    for prev_tag in all_tags:
        total = sum(transition_counts.get(prev_tag, {}).values()) + tag_set_size
        for next_tag in all_tags:
            count = transition_counts.get(prev_tag, {}).get(next_tag, 0) + 1  # Laplace smoothing
            transition_probs[prev_tag][next_tag] = math.log(count / total)
    with open(transition_probs_path, "w", encoding="utf-8") as f:
        json.dump(transition_probs, f, ensure_ascii=False, indent=2)

def build_emission_probs(tagged_path, emission_counts_path, emission_probs_path, tag_counts_path):
    emission_counts = defaultdict(lambda: defaultdict(int))
    tag_counts = defaultdict(int)
    with open(tagged_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                word, tag = line.split("\t")
                emission_counts[tag][word] += 1
                tag_counts[tag] += 1
    with open(emission_counts_path, "w", encoding="utf-8") as f:
        json.dump({tag: dict(words) for tag, words in emission_counts.items()}, f, ensure_ascii=False, indent=2)
    with open(tag_counts_path, "w", encoding="utf-8") as f:
        json.dump(dict(tag_counts), f, ensure_ascii=False, indent=2)
    vocab = set()
    for words in emission_counts.values():
        vocab.update(words.keys())
    vocab_size = len(vocab)
    emission_probs = defaultdict(dict)
    for tag, words in emission_counts.items():
        total = tag_counts[tag] + vocab_size + 1
        for word in vocab:
            count = words.get(word, 0) + 1
            emission_probs[tag][word] = math.log(count / total)
        emission_probs[tag]["<UNK>"] = math.log(1 / total)
    with open(emission_probs_path, "w", encoding="utf-8") as f:
        json.dump(emission_probs, f, ensure_ascii=False, indent=2)
