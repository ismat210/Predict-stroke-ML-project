# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import pickle
import os

# -------------------------------
# Load the full pipeline
# -------------------------------
pipeline_path = os.path.join("artifacts", "stroke_pipeline.pkl")
with open(pipeline_path, "rb") as f:
    pipeline = pickle.load(f)

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="Stroke Prediction API", version="1.0")

# -------------------------------
# Define input schema
# -------------------------------
class PatientData(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str

# -------------------------------
# Batch prediction endpoint
# -------------------------------
@app.post("/predict_batch")
def predict_batch(data: List[PatientData]):
    # Convert list of patients to DataFrame
    df = pd.DataFrame([d.dict() for d in data])
    
    # Predict class (0 = no stroke, 1 = stroke)
    predictions = pipeline.predict(df)
    
    # Predict probability of stroke
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(df)[:, 1]  # probability of class 1
    else:
        probabilities = [None] * len(predictions)
    
    # Format output
    results = [
        {"prediction": int(p), "stroke_probability": float(prob) if prob is not None else None}
        for p, prob in zip(predictions, probabilities)
    ]
    
    return {"results": results}

# -------------------------------
# Root endpoint
# -------------------------------
@app.get("/")
def root():
    return {"message": "Welcome to the Stroke Prediction API!"}