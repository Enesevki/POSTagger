from flask import Flask, request, jsonify
import json
import math

# Model dosyalarını yükle
with open("data/final/transition_probs.json", "r", encoding="utf-8") as f:
    transition_probs = json.load(f)
with open("data/final/emission_probs.json", "r", encoding="utf-8") as f:
    emission_probs = json.load(f)
with open("data/final/tag_counts.json", "r", encoding="utf-8") as f:
    tag_counts = json.load(f)
tag_set = list(tag_counts.keys())

LOG_ZERO = -1e10

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
    return list(zip(words, path[max_final_tag]))

# Flask app
app = Flask(__name__)

@app.route("/tag", methods=["POST"])
def tag():
    data = request.get_json()
    sentence = data.get("sentence", "")
    result = viterbi_tag(sentence, transition_probs, emission_probs, tag_set)
    return jsonify({"result": result})

# Test endpoint
@app.route("/", methods=["GET"])
def home():
    return "Türkçe POS Tagger API (Viterbi tabanlı)"

if __name__ == "__main__":
    app.run(debug=True)
