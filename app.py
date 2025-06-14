import streamlit as st
from src.model import load_crf_model, predict_tags
from src.features import sent2features
from src.data_loader import read_conll
from io import StringIO

st.set_page_config(page_title="Türkçe POS Tagger", layout="centered")

st.title(" Türkçe CRF POS Tagger")
st.write("Bir cümle girin ve modelin tahmin ettiği sözcük türlerini görün.")

# Modeli yükle
@st.cache_resource
def load_model():
    return load_crf_model("outputs/models/crf_final.pkl")

model = load_model()

# Kullanıcıdan cümle al
user_input = st.text_area("Cümle giriniz:", "Ben seni seviyorum.")

if st.button("Etiketle"):
    if not user_input.strip():
        st.warning("Lütfen bir cümle giriniz.")
    else:
        # Tokenizasyon ve özellik çıkarımı
        tokens = [(word, "O") for word in user_input.strip().split()]
        features = [sent2features(tokens)]
        predicted = predict_tags(model, features)[0]

        st.markdown("###  Etiketlenmiş Çıktı")
        for word, tag in zip(user_input.strip().split(), predicted):
            st.write(f"`{word}` → **{tag}**")
