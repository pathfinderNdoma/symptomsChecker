# symptoms_loader.py
import csv

def load_symptoms(file_path: str) -> set:
    symptoms_set = set()
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            symptom = row[0].strip().lower()
            if symptom:
                symptoms_set.add(symptom)
    return symptoms_set

# Global variable (only loads once)
SYMPTOMS_LIST = load_symptoms("data/symptoms.csv")
