# huggingface_qa.py
import requests
import os

API_URL = "https://api-inference.huggingface.co/models/dmis-lab/biobert-large-cased-v1.1-squad"
HF_TOKEN = os.getenv("HF_TOKEN")  # Set this securely in your environment

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

def query_huggingface(question: str, context: str):
    payload = {
        "inputs": {
            "question": question,
            "context": context
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["answer"]
