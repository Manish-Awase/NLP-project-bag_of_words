import spacy
from sklearn.feature_extraction.text import CountVectorizer
from joblib import load
import os
import numpy as np

nlp = spacy.load("en_core_web_sm")


model_path = './artifact/model/trained_model.joblib'
vectorizer_path = './artifact/vectorizer/vectorizer.joblib'
# Load model and vectorizer safely
def load_artifacts(model_path, vectorizer_path):
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Model or vectorizer file not found.")
    model = load(model_path)
    vectorizer = load(vectorizer_path)
    return model, vectorizer
# /////////////////////////////////
# Load the model back into memory
model, vectorizer=load_artifacts(model_path, vectorizer_path)
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]
def predict(email:str):
    email_v = vectorizer.transform([email])
    y_predict = model.predict(email_v)
    return y_predict

