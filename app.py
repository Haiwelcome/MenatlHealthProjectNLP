import streamlit as st
import joblib
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

st.set_page_config(page_title="Mental Health Intensity Predictor")

st.title("🧠 Mental Health Intensity Prediction")

@st.cache_resource
def load_nltk():
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")

load_nltk()

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

@st.cache_resource
def load_models():
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    model = joblib.load("logreg_model.pkl")
    return tfidf, model

tfidf, model = load_models()

text = st.text_area("Enter text")

if st.button("Predict"):
    clean = clean_text(text)
    vec = tfidf.transform([clean])
    pred = model.predict(vec)[0]
    conf = max(model.predict_proba(vec)[0])

    st.write("Prediction:", pred)
    st.write("Confidence:", round(conf, 2))
