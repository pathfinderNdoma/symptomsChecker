from fastapi import FastAPI, Request
from typing import List
import numpy as np
import json
import pickle
from contextlib import asynccontextmanager
import ast
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware



#function to load the model
from joblib import load
loaded_model = load("aimodel.pkl")  # Works better for scikit-learn objects
  


 #Initialize an instance of FastAPI app   
app =FastAPI()

@app.get("/")
async def main():
    return {"message": "Hello World"}

origins = [
    "http://localhost:5173",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#Function to process user input
columns = [
'itching',
 'skin_rash',
 'nodal_skin_eruptions',
 'continuous_sneezing',
 'shivering',
 'chills',
 'joint_pain',
 'stomach_pain',
 'acidity',
 'ulcers_on_tongue',
 'muscle_wasting',
 'vomiting',
 'burning_micturition',
 'spotting_ urination',
 'fatigue',
 'weight_gain',
 'anxiety',
 'cold_hands_and_feets',
 'mood_swings',
 'weight_loss',
 'restlessness',
 'lethargy',
 'patches_in_throat',
 'irregular_sugar_level',
 'cough',
 'high_fever',
 'sunken_eyes',
 'breathlessness',
 'sweating',
 'dehydration',
 'indigestion',
 'headache',
 'yellowish_skin',
 'dark_urine',
 'nausea',
 'loss_of_appetite',
 'pain_behind_the_eyes',
 'back_pain',
 'constipation',
 'abdominal_pain',
 'diarrhoea',
 'mild_fever',
 'yellow_urine',
 'yellowing_of_eyes',
 'acute_liver_failure',
 'fluid_overload',
 'swelling_of_stomach',
 'swelled_lymph_nodes',
 'malaise',
 'blurred_and_distorted_vision',
 'phlegm',
 'throat_irritation',
 'redness_of_eyes',
 'sinus_pressure',
 'runny_nose',
 'congestion',
 'chest_pain',
 'weakness_in_limbs',
 'fast_heart_rate',
 'pain_during_bowel_movements',
 'pain_in_anal_region',
 'bloody_stool',
 'irritation_in_anus',
 'neck_pain',
 'dizziness',
 'cramps',
 'bruising',
 'obesity',
 'swollen_legs',
 'swollen_blood_vessels',
 'puffy_face_and_eyes',
 'enlarged_thyroid',
 'brittle_nails',
 'swollen_extremeties',
 'excessive_hunger',
 'extra_marital_contacts',
 'drying_and_tingling_lips',
 'slurred_speech',
 'knee_pain',
 'hip_joint_pain',
 'muscle_weakness',
 'stiff_neck',
 'swelling_joints',
 'movement_stiffness',
 'spinning_movements',
 'loss_of_balance',
 'unsteadiness',
 'weakness_of_one_body_side',
 'loss_of_smell',
 'bladder_discomfort',
 'foul_smell_of urine',
 'continuous_feel_of_urine',
 'passage_of_gases',
 'internal_itching',
 'toxic_look_(typhos)',
 'depression',
 'irritability',
 'muscle_pain',
 'altered_sensorium',
 'red_spots_over_body',
 'belly_pain',
 'abnormal_menstruation',
 'dischromic _patches',
 'watering_from_eyes',
 'increased_appetite',
 'polyuria',
 'family_history',
 'mucoid_sputum',
 'rusty_sputum',
 'lack_of_concentration',
 'visual_disturbances',
 'receiving_blood_transfusion',
 'receiving_unsterile_injections',
 'coma',
 'stomach_bleeding',
 'distention_of_abdomen',
 'history_of_alcohol_consumption',
 'fluid_overload.1',
 'blood_in_sputum',
 'prominent_veins_on_calf',
 'palpitations',
 'painful_walking',
 'pus_filled_pimples',
 'blackheads',
 'scurring',
 'skin_peeling',
 'silver_like_dusting',
 'small_dents_in_nails',
 'inflammatory_nails',
 'blister',
 'red_sore_around_nose',
 'yellow_crust_ooze'
]
#existing_symptoms = ['yellow_crust_ooze', 'excessive_hunger', 'red_sore_around_nose', 'weight_loss', 'high_fever', 'yellowish_skin', 'dark_urine']

def prepare_input(existing_symptoms, columns):
    
    symptom_array = []
    for column in columns:
        if column in existing_symptoms:
            symptom_array.append(1)  # Symptom is present
        else:
            symptom_array.append(0)  # Symptom is absent
            
    symptom_array = np.array(symptom_array).reshape(1, -1)  # Convert to NumPy array and reshape
    result_array = symptom_array.flatten().tolist()  # Flatten the array and convert it to a nested Python list
    return json.dumps(result_array)  # Convert Python list to JSON string


#Define a route to predict disease
@app.post("/predict_disease/")
async def predict_disease(request: Request):
    payload = await request.json()
    existing_symptoms = payload.get('symptoms', [])
    #symptoms_list = ast.literal_eval(existing_symptoms)
    
    a = prepare_input(existing_symptoms, columns)

    # Decode the output string to a list
    a_list = ast.literal_eval(a)
    my_string = ''.join(map(str, a_list))
    
    # Convert the list to a NumPy array and reshape it
    output_array = np.array(a_list).reshape(1, -1)
   
    # Convert the NumPy array to a string representation of a nested list with double square brackets
    formatted_output_str = output_array.flatten().tolist()
    #return  formatted_output_str
    predicted_disease = loaded_model.predict(output_array)
    return {"There is a possibility you are most likely having"+ " " + predicted_disease[0]}


@app.get("/home/{message}")
async def home(message):
    return {"message": message}
