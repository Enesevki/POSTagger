# app.py

import streamlit as st
from src.model import load_crf_model, predict_tags
from src.features import sent2features
import os

POS_TR = {
    "NOUN": "İsim",
    "VERB": "Fiil",
    "ADJ": "Sıfat",
    "ADV": "Zarf",
    "PRON": "Zamir",
    "PROPN": "Özel İsim",
    "AUX": "Yardımcı Fiil",
    "ADP": "İlgeç",
    "DET": "Belirteç",
    "NUM": "Sayı",
    "CCONJ": "Bağlaç",
    "SCONJ": "Bağlaç (Alt cümle)",
    "PART": "Partikül",
    "INTJ": "Ünlem",
    "PUNCT": "Noktalama",
    "SYM": "Sembol",
    "X": "Bilinmeyen",
    "O": "Diğer"
}

# --- Streamlit UI Ayarları ---
st.set_page_config(page_title="Türkçe POS Tagger", page_icon="🤖", layout="centered")
st.title("Türkçe POS Tagger Demo 🤖")
st.write("Herhangi bir Türkçe cümle girin, kelimelerin hangi tür (POS) olduğunu görün.")

@st.cache_resource
def load_model():
    model_path = "crf_final.pkl"
    if not os.path.exists(model_path):
        st.error(f"Model dosyası bulunamadı: {model_path}")
        st.stop()
    return load_crf_model(model_path)

# --- Kullanıcı Girişi ---
sentence = st.text_input("Cümlenizi girin:", "")

if st.button("POS Tagle!") or (sentence and st.session_state.get("already_tagged") != sentence):
    st.session_state["already_tagged"] = sentence

    model = load_model()

    words = sentence.strip().split()
    if not words:
        st.info("Lütfen bir cümle girin.")
    else:
        # "O" etiketi dummy olarak kullanılabilir
        dummy_tagged = [(word, "O") for word in words]
        features = [sent2features(dummy_tagged)]
        predictions = predict_tags(model, features)[0]

        display = [
            {"Kelime": w, "Etiket": tag, "Türkçesi": POS_TR.get(tag, "-")}
            for w, tag in zip(words, predictions)
        ]

        st.write("### Sonuç:")
        st.table(display)
