import os

# Must run before transformers / tensorflow are imported
os.environ["USE_TF"] = "0"  # stop transformers from importing TensorFlow/Keras 3
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import html
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import gensim.downloader as gensim_api
from gensim.models import Word2Vec
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from transformers import AutoModel, AutoTokenizer

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"

st.set_page_config(page_title="RailSahayak AI", page_icon="🚆", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif}
.stApp {background:#f5f7fb}
.block-container {padding-top:1.7rem;padding-bottom:3rem;max-width:1440px}
h1,h2,h3 {font-family:'Manrope',sans-serif;color:#172640;letter-spacing:-.025em}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#101d35,#1b3458)}
[data-testid="stSidebar"] * {color:#edf4ff}
[data-testid="stSidebar"] hr {border-color:#3a4b68}
.hero {background:linear-gradient(120deg,#142543,#204765 65%,#147c78);border-radius:20px;padding:28px 32px;color:white;margin-bottom:22px;box-shadow:0 12px 30px #14254318}
.hero .tag {color:#a9f0e3;font-size:.74rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase}
.hero h1 {color:white;font-size:2.15rem;margin:8px 0}
.hero p {color:#dce8f5;margin:0;font-size:1rem}
.card,.metric {background:white;border:1px solid #e3e9f2;border-radius:15px;padding:19px 21px;box-shadow:0 4px 16px #1c2d4b08}
.metric {min-height:105px}
.metric-label {font-size:.77rem;color:#718097;font-weight:700;margin-bottom:8px}
.metric-value {font-family:'Manrope',sans-serif;color:#172640;font-size:1.3rem;font-weight:800;overflow-wrap:anywhere}
.metric-sub {font-size:.75rem;color:#12847a;margin-top:6px}
.eyebrow {color:#12847a;font-size:.75rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase}
.pill {display:inline-block;padding:5px 10px;border-radius:999px;background:#e8f7f4;color:#11776e;font-size:.76rem;font-weight:700}
.muted {color:#718097;font-size:.9rem}
div.stButton>button[kind="primary"] {background:#087f75;border:0;border-radius:10px;font-weight:700;min-height:44px}
div.stButton>button[kind="primary"]:hover {background:#066a62}
div[data-testid="stTextArea"] textarea {border-radius:11px;border-color:#d9e1ed}
[data-testid="stDataFrame"] {border:1px solid #e3e9f2;border-radius:12px;overflow:hidden}
.footer {color:#7a879b;font-size:.78rem;padding-top:18px;border-top:1px solid #e3e9f2;margin-top:28px}
</style>
""", unsafe_allow_html=True)


def clean_text(text):
    text = html.unescape(str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^A-Za-z0-9'!?., ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def average_word_vectors(texts, word_vectors, vector_size):
    rows = []
    for text in texts:
        vectors = [word_vectors[word] for word in text.split() if word in word_vectors]
        rows.append(np.mean(vectors, axis=0) if vectors else np.zeros(vector_size))
    return np.asarray(rows)


@st.cache_resource(show_spinner="Loading saved models…")
def load_assets():
    required = [
        "tfidf_model.pkl", "word2vec_fixed.model", "w2v_classifier.pkl",
        "glove_classifier.pkl", "lstm_model.keras",
        "tokenizer.json", "label_encoder.pkl",
        "bert_classifier.pkl", "type_model.pkl"
    ]
    missing = [x for x in required if not (MODEL_DIR / x).exists()]
    if missing:
        return None, missing

    a = {}
    a["tfidf"] = joblib.load(MODEL_DIR / "tfidf_model.pkl")
    a["w2v"] = Word2Vec.load(str(MODEL_DIR / "word2vec_fixed.model"))
    a["w2v_clf"] = joblib.load(MODEL_DIR / "w2v_classifier.pkl")
    a["glove"] = gensim_api.load("glove-twitter-25")
    a["glove_clf"] = joblib.load(MODEL_DIR / "glove_classifier.pkl")
    a["lstm"] = load_model(MODEL_DIR / "lstm_model.keras")
    a["tokenizer"] = tokenizer_from_json((MODEL_DIR / "tokenizer.json").read_text(encoding="utf-8"))
    a["label_encoder"] = joblib.load(MODEL_DIR / "label_encoder.pkl")
    a["bert_tokenizer"] = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    a["bert_encoder"] = AutoModel.from_pretrained("distilbert-base-uncased")
    a["bert_encoder"].eval()
    a["bert_clf"] = joblib.load(MODEL_DIR / "bert_classifier.pkl")
    a["type"] = joblib.load(MODEL_DIR / "type_model.pkl")
    a["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a["bert_encoder"].to(a["device"])
    return a, []


def bert_embedding(text, a):
    tokens = a["bert_tokenizer"]([text], padding=True, truncation=True, max_length=96, return_tensors="pt")
    tokens = {k: v.to(a["device"]) for k, v in tokens.items()}
    with torch.no_grad():
        out = a["bert_encoder"](**tokens)
        mask = tokens["attention_mask"].unsqueeze(-1)
        emb = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return emb.cpu().numpy()


def predict_all(text, a):
    cleaned = clean_text(text)
    p = {}
    p["TF-IDF + Logistic Regression"] = str(a["tfidf"].predict([cleaned])[0])
    p["Word2Vec + Logistic Regression"] = str(a["w2v_clf"].predict(average_word_vectors([cleaned], a["w2v"].wv, 100))[0])
    p["GloVe + Logistic Regression"] = str(a["glove_clf"].predict(average_word_vectors([cleaned], a["glove"], 25))[0])
    seq = pad_sequences(a["tokenizer"].texts_to_sequences([cleaned]), maxlen=80, padding="post", truncating="post")
    probs = a["lstm"].predict(seq, verbose=0)
    p["LSTM"] = str(a["label_encoder"].inverse_transform(np.argmax(probs, axis=1))[0])
    p["DistilBERT + Logistic Regression"] = str(a["bert_clf"].predict(bert_embedding(cleaned, a))[0])
    issue = str(a["type"].predict([cleaned])[0])
    return p, issue


def hero(title, subtitle, tag="PASSENGER EXPERIENCE INTELLIGENCE"):
    st.markdown(f'<div class="hero"><div class="tag">{tag}</div><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def metric(label, value, note):
    st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-sub">{note}</div></div>', unsafe_allow_html=True)


with st.sidebar:
    st.markdown('<div style="font-size:2rem">🚆</div><div style="font-family:Manrope,sans-serif;font-size:1.35rem;font-weight:800">RailSahayak AI</div><div style="font-size:.8rem;color:#bdcce2">Passenger complaint intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("NAVIGATION", ["Overview", "Complaint Analyzer", "Model Comparison", "About"])
    st.markdown("---")
    st.markdown("**SYSTEM**")
    st.markdown('<span class="pill">Saved models · Local inference</span>', unsafe_allow_html=True)
    st.caption("No model retraining when the dashboard opens.")
    st.markdown("---")
    st.caption("Academic project · Demonstration only")

try:
    assets, missing = load_assets()
except Exception as exc:
    st.error("A saved model could not be loaded.")
    st.exception(exc)
    st.stop()

if assets is None:
    hero("Passenger Complaint Intelligence", "Analyze passenger messages using five NLP approaches.")
    st.error("Some required model files are missing.")
    for item in missing:
        st.write(f"- `{item}`")
    st.info("Keep these files inside the `models` folder next to app.py.")
    st.stop()

if page == "Overview":
    hero("Passenger Complaint Intelligence", "A clear workspace to analyze complaints and compare NLP predictions.")
    st.markdown('<div class="eyebrow">WORKSPACE OVERVIEW</div>', unsafe_allow_html=True)
    st.markdown("## Welcome to your analysis dashboard")
    st.markdown('<p class="muted">Explore model predictions, compare classification approaches, and review the estimated issue type.</p>', unsafe_allow_html=True)
    cols = st.columns(4)
    for col, title, value, note in zip(cols, ["CLASSIFICATION MODELS", "ISSUE-TYPE MODEL", "INFERENCE", "INPUT"], ["05", "01", "Local", "Text"], ["Five NLP approaches", "Separate classifier", "Saved model files", "Passenger complaint"]):
        with col:
            metric(title, value, note)
    st.write("")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### What you can do")
        st.markdown("- **Analyze a complaint:** enter a passenger message and generate predictions.\n- **Compare models:** review the outputs side by side.\n- **Estimate issue type:** use the separate issue-type classifier.\n- **Review results:** inspect predictions in a clear table.")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Models in this workspace")
        st.markdown('<span class="pill">TF-IDF</span> Logistic Regression<br><span class="pill">Word2Vec</span> Logistic Regression<br><span class="pill">GloVe</span> Logistic Regression<br><span class="pill">LSTM</span> Sequence classifier<br><span class="pill">DistilBERT</span> Embeddings + Logistic Regression', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.write("")
    st.info("Choose **Complaint Analyzer** in the sidebar to begin.")

elif page == "Complaint Analyzer":
    hero("Complaint Analyzer", "Run a passenger message through all five saved NLP models.", "ANALYSIS WORKSPACE")
    left, right = st.columns([1.15, .85], gap="large")
    with left:
        st.markdown("### Passenger message")
        message = st.text_area("Complaint text", placeholder="Example: My train is delayed by three hours and there is no announcement at the station.", height=165, label_visibility="collapsed")
        st.caption("Include the main problem and relevant details for a useful test.")
        run = st.button("Analyze complaint  →", type="primary", use_container_width=True)
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### How it works")
        st.markdown("**01 · Normalize**  \nClean the complaint text.\n\n**02 · Classify**  \nGenerate predictions with five models.\n\n**03 · Summarize**  \nDisplay model outputs and issue-type estimate.")
        st.markdown("</div>", unsafe_allow_html=True)
    if run:
        if not message.strip():
            st.warning("Please enter a passenger message first.")
        else:
            with st.spinner("Analyzing with saved models…"):
                try:
                    p, issue = predict_all(message, assets)
                    st.session_state["last_message"] = message
                    st.session_state["last_predictions"] = p
                    st.session_state["last_issue"] = issue
                    st.session_state["last_model"] = list(p.keys())[0]
                except Exception as exc:
                    st.error("Prediction failed. Technical details:")
                    st.exception(exc)
    if "last_predictions" in st.session_state:
        p = st.session_state["last_predictions"]
        issue = st.session_state["last_issue"]
        model = st.selectbox("Highlight a model", list(p.keys()), index=list(p.keys()).index(st.session_state.get("last_model", list(p.keys())[0])))
        st.session_state["last_model"] = model
        st.markdown("### Analysis results")
        c1, c2, c3 = st.columns(3)
        with c1:
            metric("SELECTED MODEL", model.split(" + ")[0], "Highlighted prediction")
        with c2:
            metric("PREDICTED LABEL", p[model], "Classification output")
        with c3:
            metric("ESTIMATED ISSUE TYPE", issue, "Separate classifier")
        st.markdown("#### Model predictions")
        st.dataframe(pd.DataFrame([{"Model": k, "Predicted label": v} for k, v in p.items()]), use_container_width=True, hide_index=True)
        with st.expander("View analyzed message"):
            st.write(st.session_state["last_message"])
        st.caption("These are model estimates for demonstration, not official railway decisions.")

elif page == "Model Comparison":
    hero("Model Comparison", "Review the five model outputs for your latest analyzed complaint.", "MODEL INSIGHTS")
    if "last_predictions" not in st.session_state:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### No comparison yet")
        st.write("Open Complaint Analyzer, enter a message, and click Analyze complaint.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        df = pd.DataFrame([{"Model": k, "Predicted label": v} for k, v in st.session_state["last_predictions"].items()])
        counts = df["Predicted label"].value_counts()
        common = str(counts.index[0])
        agreement = int(counts.iloc[0])
        c1, c2, c3 = st.columns(3)
        with c1:
            metric("MODELS COMPARED", str(len(df)), "All classifiers")
        with c2:
            metric("MOST COMMON LABEL", common, "Most frequent output")
        with c3:
            metric("MODELS WITH THAT LABEL", f"{agreement}/{len(df)}", "Simple agreement count")
        st.markdown("### Prediction distribution")
        st.bar_chart(counts)
        st.markdown("### Detailed comparison")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption("Agreement is descriptive only; it is not a calibrated confidence score or proof of correctness.")

elif page == "About":
    hero("About RailSahayak AI", "A student project exploring NLP-based analysis of Indian Railways passenger complaints.", "PROJECT INFORMATION")
    left, right = st.columns([1.1, .9], gap="large")
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Project objective")
        st.write("Demonstrate how natural language processing can analyze passenger complaint messages, compare classification approaches, and estimate an issue type.")
        st.markdown("### Methodology")
        st.write("The dashboard combines TF-IDF, Word2Vec, GloVe, LSTM, and frozen DistilBERT embeddings with classifiers. A separate issue-type classifier provides an additional output.")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Limitations")
        st.write("Predictions depend on training data and preprocessing. Model agreement does not prove correctness. Outputs are not official railway decisions or emergency guidance.")
        st.markdown("### Local inference")
        st.write("Saved model artifacts are loaded from the local `models` folder; the dashboard does not retrain models at startup.")
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="footer"><b>RailSahayak AI</b> · Passenger complaint intelligence demo <span style="float:right">Streamlit · NLP model comparison</span></div>', unsafe_allow_html=True)
