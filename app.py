# app.py

import streamlit as st
from src.model import load_crf_model, predict_tags
from src.features import sent2features
import os

# Türkçe POS etiket açıklamaları
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

# UI başlığı ve ayarlar
st.set_page_config(page_title="Türkçe POS Tagger", page_icon="🤖", layout="centered")
st.title("Türkçe POS Tagger Demo 🤖")
st.write("Cümleyi olduğu gibi veriyoruz. Token yok, split yok. Sihirli bir test :)")

@st.cache_resource
def load_model():
    model_path = "outputs/models/crf_final.pkl"
    if not os.path.exists(model_path):
        st.error(f"Model dosyası bulunamadı: {model_path}")
        st.stop()
    return load_crf_model(model_path)

# Giriş
sentence = st.text_input("Cümlenizi girin:", "")

if st.button("POS Tagle!") or (sentence and st.session_state.get("already_tagged") != sentence):
    st.session_state["already_tagged"] = sentence

    model = load_model()

    if not sentence.strip():
        st.info("Lütfen bir cümle girin.")
    else:
        try:
            # Token yok: Tüm cümle tek 'kelime' olarak veriliyor
            tokens = [(sentence.strip(), "O")]
            features = [sent2features(tokens)]
            predictions = predict_tags(model, features)[0]

            display = [{
                "Cümle": sentence,
                "Etiket": predictions[0],
                "Türkçesi": POS_TR.get(predictions[0], "-")
            }]

            st.write("### Sonuç:")
            st.table(display)
        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")
