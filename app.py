import streamlit as st
from src.model import load_crf_model, predict_tags
from src.features import sent2features
import stanza
import os

POS_TR = {
    "NOUN": "İsim", "VERB": "Fiil", "ADJ": "Sıfat", "ADV": "Zarf", "PRON": "Zamir",
    "PROPN": "Özel İsim", "AUX": "Yardımcı Fiil", "ADP": "İlgeç", "DET": "Belirteç",
    "NUM": "Sayı", "CCONJ": "Bağlaç", "SCONJ": "Alt Bağlaç", "PART": "Partikül",
    "INTJ": "Ünlem", "PUNCT": "Noktalama", "SYM": "Sembol", "X": "Bilinmeyen", "O": "Diğer"
}

st.set_page_config(page_title="Türkçe POS Tagger", page_icon="🤖", layout="centered")
st.title("Türkçe POS Tagger Demo 🤖")
st.write("Stanza ile doğru tokenizasyon ile çalışan POS Tagger.")

@st.cache_resource
def load_model():
    model_path = "outputs/models/crf_final.pkl"
    if not os.path.exists(model_path):
        st.error(f"Model dosyası bulunamadı: {model_path}")
        st.stop()
    return load_crf_model(model_path)

@st.cache_resource
def load_stanza():
    stanza.download("tr")  # sadece ilk sefer
    return stanza.Pipeline("tr", processors="tokenize", tokenize_no_ssplit=True)

# Kullanıcı girişi
sentence = st.text_input("Cümlenizi girin:", "")

if st.button("POS Tagle!") or (sentence and st.session_state.get("already_tagged") != sentence):
    st.session_state["already_tagged"] = sentence

    model = load_model()
    nlp = load_stanza()

    if not sentence.strip():
        st.info("Lütfen bir cümle girin.")
    else:
        doc = nlp(sentence.strip())
        tokens = [(word.text, "O") for sent in doc.sentences for word in sent.words]
        
        if not tokens:
            st.warning("Token bulunamadı.")
        else:
            features = [sent2features(tokens)]
            predictions = predict_tags(model, features)[0]

            display = [
                {"Kelime": word, "Etiket": tag, "Türkçesi": POS_TR.get(tag, "-")}
                for (word, _), tag in zip(tokens, predictions)
            ]

            st.write("### Sonuç:")
            st.table(display)
