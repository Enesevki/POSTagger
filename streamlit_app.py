import streamlit as st
import time
from core import HMMModel
import stanza

# Model ve tokenizer yükle
@st.cache_resource
def load_model():
    return HMMModel.load("models/hmm_model.pkl")

@st.cache_resource
def load_tokenizer():
    return stanza.Pipeline(lang='tr', processors='tokenize', use_gpu=False, verbose=False)

def tokenize(sentence, tokenizer):
    doc = tokenizer(sentence)
    return [word.text for sent in doc.sentences for word in sent.words]

def tag_sentence(model, tokenizer, sentence):
    words = tokenize(sentence, tokenizer)
    tags = model.viterbi_decode(words)
    return list(zip(words, tags))

# UI Başlıyor
st.set_page_config(page_title="Türkçe POS Tagger", layout="centered")
st.title(" Türkçe POS Etiketleyici (HMM + Viterbi)")
st.markdown("Bir cümle girin, sistem her kelimenin türünü tahmin etsin.")

# Input
user_input = st.text_input(" Cümle girin", placeholder="Örnek: Bugün hava çok güzel.")

# Model yükleme
model = load_model()
tokenizer = load_tokenizer()

# Tahmin
if user_input:
    with st.spinner("Etiketleniyor..."):
        try:
            result = tag_sentence(model, tokenizer, user_input)
            st.success("✅ Etiketleme tamamlandı!")
            st.markdown("### 📊 Sonuçlar:")
            st.table([{"Kelime": w, "POS Tag": t} for w, t in result])
        except Exception as e:
            st.error(f" Hata: {e}")
