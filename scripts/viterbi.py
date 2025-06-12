import json
import math

LOG_ZERO = -1e10

def load_probs(transition_probs_path, emission_probs_path, tag_counts_path):
    with open(transition_probs_path, "r", encoding="utf-8") as f:
        transition_probs = json.load(f)
    with open(emission_probs_path, "r", encoding="utf-8") as f:
        emission_probs = json.load(f)
    with open(tag_counts_path, "r", encoding="utf-8") as f:
        tag_counts = json.load(f)
    return transition_probs, emission_probs, list(tag_counts.keys())

def viterbi_tag(sentence, transition_probs, emission_probs, tag_set):
    words = sentence.strip().split()
    V = [{}]
    path = {}
    for tag in tag_set:
        emit = emission_probs[tag].get(words[0], emission_probs[tag].get("<UNK>", LOG_ZERO))
        trans = transition_probs.get("START", {}).get(tag, LOG_ZERO)
        V[0][tag] = trans + emit
        path[tag] = [tag]
    for t in range(1, len(words)):
        V.append({})
        new_path = {}
        for curr_tag in tag_set:
            max_prob, prev_tag_best = max(
                ((V[t-1][prev_tag] +
                  transition_probs.get(prev_tag, {}).get(curr_tag, LOG_ZERO) +
                  emission_probs[curr_tag].get(words[t], emission_probs[curr_tag].get("<UNK>", LOG_ZERO)), prev_tag)
                 for prev_tag in tag_set),
                key=lambda x: x[0]
            )
            V[t][curr_tag] = max_prob
            new_path[curr_tag] = path[prev_tag_best] + [curr_tag]
        path = new_path
    max_final_tag = max(V[-1], key=lambda tag: V[-1][tag])
    return path[max_final_tag]

def run_viterbi(sentences_path, output_path, transition_probs_path, emission_probs_path, tag_counts_path):
    transition_probs, emission_probs, tag_set = load_probs(transition_probs_path, emission_probs_path, tag_counts_path)
    with open(sentences_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            sentence = line.strip()
            if not sentence:
                continue
            tags = viterbi_tag(sentence, transition_probs, emission_probs, tag_set)
            fout.write(f"{sentence}\t{' '.join(tags)}\n")
