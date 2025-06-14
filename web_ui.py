import streamlit as st
import json
import math

POS_TR = {
    "NOUN": "İsim",
    "VERB": "Fiil",
    "ADJ": "Sıfat",
    "ADV": "Zarf",
    "PRON": "Zamir",
    "PROPN": "Özel İsim",
    "AUX": "Yardımcı Fiil",
    "ADP": "İlgeç (Edat/Prepozisyon)",
    "DET": "Belirteç (Tanımlık)",
    "NUM": "Sayı",
    "CCONJ": "Bağlaç (Eşdizim)",
    "SCONJ": "Bağlaç (Alt cümle)",
    "PART": "Partikül",
    "INTJ": "Ünlem",
    "PUNCT": "Noktalama",
    "SYM": "Sembol",
    "X": "Bilinmeyen",
}

@st.cache_resource
def load_model():
    with open("data/final/transition_probs.json", "r", encoding="utf-8") as f:
        transition_probs = json.load(f)
    with open("data/final/emission_probs.json", "r", encoding="utf-8") as f:
        emission_probs = json.load(f)
    with open("data/final/tag_counts.json", "r", encoding="utf-8") as f:
        tag_counts = json.load(f)
    tag_set = list(tag_counts.keys())
    return transition_probs, emission_probs, tag_set

LOG_ZERO = -1e10

def viterbi_tag(sentence, transition_probs, emission_probs, tag_set):
    words = sentence.strip().split()
    if not words:
        return []
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

# --- Streamlit UI ---
st.set_page_config(page_title="Türkçe POS Tagger", page_icon="🤖", layout="centered")
st.title("Türkçe POS Tagger Demo 🤖")
st.write("Herhangi bir Türkçe cümle girin, kelimelerin hangi tür (POS) olduğunu görün.")

sentence = st.text_input("Cümlenizi girin:", "")

if st.button("POS Tagle!") or (sentence and st.session_state.get("already_tagged") != sentence):
    st.session_state["already_tagged"] = sentence
    transition_probs, emission_probs, tag_set = load_model()
    result = viterbi_tag(sentence, transition_probs, emission_probs, tag_set)
    if result:
        st.write("### Sonuç:")
        # Tablo için list of dict (daha şık görünür)
        display = [
            {"Kelime": w, "Tag": t, "Türkçesi": POS_TR.get(t, "-")}
            for w, t in result
        ]
        st.table(display)
    else:
        st.info("Lütfen bir cümle girin.")
