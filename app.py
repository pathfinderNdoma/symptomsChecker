from fastapi import FastAPI
from pydantic import BaseModel
from model_setup import get_relevant_context
import pandas as pd
import numpy as np
import sklearn
import joblib
import sys
import re  

from greetings import GREETING_WORDS

app = FastAPI()

class SymptomRequest(BaseModel):
    symptoms: str  # Comma-separated symptoms string



@app.post("/predict/")
def predict_disease(request: SymptomRequest):
    raw_text = request.symptoms.strip().lower()

    # Check if it's a greeting message using word boundary matching
    if any(re.search(rf"\b{greet}\b", raw_text) for greet in GREETING_WORDS):
        return greetings()

    # Extract and validate symptoms
    raw_symptoms = [s.strip().lower() for s in raw_text.split(",") if s.strip()]

    if len(raw_symptoms) < 4:
        return {"error": "Not enough symptoms provided to make a decision. Please provide at least 4 symptoms."}

    # Proceed with disease prediction using all provided symptoms
    user_query = "I have " + ", ".join(raw_symptoms)
    contexts = get_relevant_context(user_query)
    diseases = contexts['disease'].tolist()

    return {
        "input_symptoms": raw_symptoms,
        "predicted_diseases": diseases,
        "disclaimer": "This is an AI generated prediction and could be wrng, kindly have further engagements with our medical team. Always consult a medical professional."
    }


# GREETINGS ENDPOINT
@app.get("/greetings")
def greetings():
    return {
        "message": "Welcome, please kindly enter your symptoms to use the symptom checker"
    }

# LIBRARY VERSIONS ENDPOINT
@app.get("/libversions")
def get_library_versions():
    # Hardcoded Colab versions
    colab_versions = {
        "pandas": "2.2.2",
        "numpy": "2.0.2",
        "scikit_learn": "1.6.1",
        "transformers": "4.52.2",
        "joblib": "1.5.0"
    }

    # Current local environment versions
    local_versions = {
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__
    }

    return {
        "colab_versions (The libraries versions before the vectorizers were downloaded using joblib)": colab_versions,
        "local_env_versions": local_versions
    }
