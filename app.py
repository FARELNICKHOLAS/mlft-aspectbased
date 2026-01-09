import streamlit as st
import numpy as np
import pickle
import re
import json

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

MAX_LEN = 180

st.set_page_config(
    page_title="Aspect-Based Sentiment Analysis",
    layout="centered"
)

st.title("Aspect-Based Sentiment Analysis")
st.write("Analisis sentimen dan aspek pada ulasan aplikasi menggunakan BiLSTM + FastText")

@st.cache_resource
def load_models():
    sentiment_model = load_model("bilstm_sentiment.h5")
    aspect_model = load_model("bilstm_aspect.h5")
    return sentiment_model, aspect_model

@st.cache_resource
def load_tokenizers():
    with open("tokenizer_sentiment.pkl", "rb") as f:
        tok_sent = pickle.load(f)
    with open("tokenizer_aspect.pkl", "rb") as f:
        tok_asp = pickle.load(f)
    with open("label_encoder_aspect.pkl", "rb") as f:
        le_asp = pickle.load(f)
    return tok_sent, tok_asp, le_asp

@st.cache_resource
def load_preprocess_resources():
    with open("merged_slang_dict.json", "r", encoding="utf-8") as f:
        slang_dict = json.load(f)

    with open("stopwords-id.txt", "r", encoding="utf-8") as f:
        stopwords = f.read().splitlines()

    if "nya" not in stopwords:
        stopwords.append("nya")

    stemmer = StemmerFactory().create_stemmer()

    return slang_dict, set(stopwords), stemmer

sentiment_model, aspect_model = load_models()
tok_sent, tok_asp, le_asp = load_tokenizers()
slang_dict, stopwords, stemmer = load_preprocess_resources()

def cleaning(text: str) -> str:
    text = text.lower()
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"[^a-z]", " ", text)
    text = re.sub(r"\t|\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    return text.strip()


def normalize_slang(text: str) -> str:
    tokens = re.findall(r"\w+|\S", text)
    normalized = [slang_dict.get(tok, tok) for tok in tokens]
    return " ".join(normalized)


def remove_stopwords(text: str) -> str:
    return " ".join([w for w in text.split() if w not in stopwords])


def stemming(text: str) -> str:
    return stemmer.stem(text)


def full_preprocess(text: str) -> str:
    text = cleaning(text)
    text = normalize_slang(text)
    text = remove_stopwords(text)
    text = stemming(text)
    return text


def tokenize_and_pad(text: str, tokenizer):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    return pad

st.subheader("Masukkan Teks Ulasan")
user_input = st.text_area(
    "Contoh: aplikasinya sering error dan pembayaran gagal",
    height=120
)


if st.button("Analisis"):
    if user_input.strip() == "":
        st.warning("Teks tidak boleh kosong.")
    else:
        processed_text = full_preprocess(user_input)

        st.caption(f"Preprocessed text: `{processed_text}`")


        X_sent = tokenize_and_pad(processed_text, tok_sent)
        sent_prob = sentiment_model.predict(X_sent, verbose=0)[0][0]
        sent_label = "Positif" if sent_prob >= 0.5 else "Negatif"


        X_asp = tokenize_and_pad(processed_text, tok_asp)
        asp_probs = aspect_model.predict(X_asp, verbose=0)[0]
        asp_idx = np.argmax(asp_probs)
        asp_label = le_asp.inverse_transform([asp_idx])[0]
        asp_conf = asp_probs[asp_idx]

 
        st.subheader("Hasil Analisis")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Sentimen",
                value=sent_label,
                delta=f"{sent_prob:.2f}"
            )

        with col2:
            st.metric(
                label="Aspek",
                value=asp_label,
                delta=f"{asp_conf:.2f}"
            )

        st.progress(float(min(sent_prob, 1.0)))

    
