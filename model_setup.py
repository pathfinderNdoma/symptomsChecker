# model_setup.py
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load precomputed objects
df = pd.read_csv('data/symptoms_disease.csv')
vectorizer = joblib.load('vectorizer.joblib')
context_vectors = joblib.load('context_vectors.joblib')

def get_relevant_context(question, top_n=5):
    question_vec = vectorizer.transform([question])
    cosine_similarities = np.dot(context_vectors.toarray(), question_vec.T.toarray()).flatten()
    relevant_indices = cosine_similarities.argsort()[-top_n:][::-1]
    return df.iloc[relevant_indices]
