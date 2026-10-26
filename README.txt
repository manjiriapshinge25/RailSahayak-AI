RailSahayak AI — Streamlit app starter

1. Keep app.py, requirements.txt, SAVE_MODELS_CELL.py, and the models/ folder together.
2. Open NLP Project.ipynb in Jupyter.
3. The current notebook kernel must contain trained model objects before saving. If the kernel is empty, run the notebook's prerequisite and training cells once.
4. Copy the code from SAVE_MODELS_CELL.py into a new cell at the END of the notebook and run it.
   The code saves to ./RailSahayak_AI_App/models relative to the notebook's current working directory.
   If that creates a different RailSahayak_AI_App folder than the one containing app.py, copy the resulting models folder into the same folder as app.py.
5. Install dependencies in a terminal opened in this folder:
   pip install -r requirements.txt
6. Start the app:
   streamlit run app.py

Important:
- There were no saved model artifacts in the provided notebook, so trained objects cannot be recovered from displayed notebook outputs alone.
- First-time setup may require downloading the pretrained GloVe and DistilBERT resources during training.
- Do not share the models folder publicly if it contains anything other than intended project artifacts.
- Predictions are educational/demo outputs, not official railway decisions.
