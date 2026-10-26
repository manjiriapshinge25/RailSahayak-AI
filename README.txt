# 🚆 RailSahayak AI
### NLP-Based Railway Complaint Detection and Issue Classification

RailSahayak AI is a Natural Language Processing (NLP) project that analyzes railway-related passenger messages to identify complaints and estimate the type of issue reported. It combines five NLP-based classification approaches in an interactive Streamlit dashboard, allowing users to compare model predictions for the same message.

## 🌐 Live Demo

**[Launch RailSahayak AI Dashboard](https://railsahayak-ai-gczvvjdpocwybtvfdwz2ee.streamlit.app/)**

**[GitHub Repository](https://github.com/manjiriapshinge25/RailSahayak-AI)** · **[Dataset](https://huggingface.co/datasets/faizmubeen/indian_railways_complaint_tweets)**

## ✨ Features

- **Complaint Detection:** Classifies passenger messages as complaints or non-complaints.
- **Five NLP Models:** Compares predictions from different text-representation and classification approaches.
- **Issue-Type Classification:** Estimates the category of a reported railway issue.
- **Model Comparison:** Displays predictions side by side and summarizes agreement between models.
- **Interactive Dashboard:** Provides Overview, Complaint Analyzer, Model Comparison, and About sections.
- **Saved Model Inference:** Loads trained model artifacts for predictions without retraining them every time the dashboard opens.

## 🧠 Models Used

| Model | Approach |
|---|---|
| TF-IDF + Logistic Regression | Word and phrase-based text features |
| Word2Vec + Logistic Regression | Average word embeddings |
| GloVe Twitter + Logistic Regression | Pretrained social-media word vectors |
| LSTM | Sequence-based neural classification |
| DistilBERT + Logistic Regression | Frozen transformer embeddings with a classifier |

A separate classifier estimates the issue type associated with a passenger message.

## 📊 Model Performance

The following results were reported for complaint detection on the held-out test set.

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| DistilBERT embeddings + Logistic Regression | 0.909 | 0.907 | 0.909 |
| LSTM | 0.902 | 0.899 | 0.902 |
| TF-IDF + Logistic Regression | 0.900 | 0.897 | 0.900 |
| Word2Vec + Logistic Regression | 0.893 | 0.890 | 0.893 |
| GloVe Twitter + Logistic Regression | 0.818 | 0.812 | 0.817 |

DistilBERT achieved the highest reported macro-F1 score. The comparison has limitations because the models differed in training setup and computational requirements.

The separate issue-type classification experiment reported an accuracy of 0.443 and macro-F1 of 0.363, indicating scope for further improvement.

## 🗂️ Dataset

The project uses the public **Indian Railways Complaint Tweets** dataset hosted on Hugging Face.

- **Source:** [Hugging Face Dataset](https://huggingface.co/datasets/faizmubeen/indian_railways_complaint_tweets)
- **Original dataset size:** 4,535 records
- **Records after duplicate removal:** 4,504
- **Complaint records:** 2,595
- **Non-complaint records:** 1,909
- **Train/test split:** 3,603 training records and 901 test records

The dataset contains historical social-media messages. It is not an official Rail Madad database or a live railway complaint feed.

## ⚙️ Technology Stack

- **Language:** Python
- **Dashboard:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Deep Learning:** TensorFlow/Keras, PyTorch
- **NLP:** Gensim, Hugging Face Transformers
- **Model Persistence:** Joblib and saved model artifacts
- **Deployment:** Streamlit Community Cloud
- **Version Control:** Git and GitHub

## 📁 Project Structure

```text
RailSahayak-AI/
├── app.py
├── requirements.txt
├── NLP Project.ipynb
├── SAVE_MODELS_CELL.py
├── models/
│   ├── tfidf_model.pkl
│   ├── word2vec_fixed.model
│   ├── w2v_classifier.pkl
│   ├── glove_classifier.pkl
│   ├── lstm_model.keras
│   ├── tokenizer.json
│   ├── label_encoder.pkl
│   ├── bert_classifier.pkl
│   └── type_model.pkl
└── README.md
```

The structure above describes the main application files and expected model artifacts; additional files may be present in the repository.

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/manjiriapshinge25/RailSahayak-AI.git
cd RailSahayak-AI
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Check the model files

Ensure the required saved artifacts are present inside the `models/` folder alongside `app.py`.

### 5. Launch the dashboard

```bash
streamlit run app.py
```

Streamlit will provide a local URL in the terminal, usually `http://localhost:8501`.

**Note:** The application requires the saved model artifacts. Installing the dependencies alone does not recreate trained models. The GloVe and DistilBERT resources may also need to be downloaded on first use.

## 📓 Jupyter Notebook

The Jupyter Notebook contains the project workflow for data preparation, model development, and evaluation.

**Notebook:** `NLP Project.ipynb`

Open the notebook from the repository's file list on GitHub to view the code and outputs.

## ⚠️ Limitations

- Predictions depend on the training data, preprocessing, and model quality.
- Issue-type classification requires further improvement.
- Model agreement does not guarantee that a prediction is correct.
- The dashboard analyzes user-entered text; it does not collect live social-media posts.
- It does not submit or track official grievances or connect to Rail Madad/CRIS.
- Outputs are for educational demonstration and human review, not official railway decisions.

## 🔮 Future Scope

- Improve issue-type classification and evaluate additional models.
- Add confusion matrices, per-class precision/recall, and error analysis.
- Test performance on newer, external datasets.
- Explore authorized integration with railway complaint systems and real-time data sources.

## 👩‍💻 Author

**Manjiri Apshinge**


---

*RailSahayak AI demonstrates how NLP can support the analysis of railway passenger complaints and comparison of text-classification models.*
