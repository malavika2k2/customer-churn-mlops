from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import numpy as np
from typing import List

# Create FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using machine learning",
    version="1.0.0"
)

# Define request model
class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    SeniorCitizen: int

class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    prediction_class: str

# Load the model at startup
model = None

@app.on_event("startup")
async def load_model():
    """Load the trained model when the application starts"""
    global model
    try:
        model_path = "models/random_forest_model.joblib"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(" Model loaded successfully")
        else:
            print(" Model file not found")
    except Exception as e:
        print(f" Error loading model: {e}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    """Predict customer churn probability"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([[
            customer.tenure,
            customer.MonthlyCharges,
            customer.TotalCharges,
            customer.SeniorCitizen
        ]], columns=['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen'])
        
        # Make prediction
        probability = model.predict_proba(input_data)[0][1]  # Probability of churn (class 1)
        prediction = probability > 0.5  # Convert to boolean prediction
        
        return PredictionResponse(
            churn_probability=round(probability, 4),
            churn_prediction=bool(prediction),
            prediction_class="Churn" if prediction else "No Churn"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch_predict")
async def batch_predict(customers: List[CustomerData]):
    """Predict churn for multiple customers at once"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert inputs to DataFrame
        input_data = pd.DataFrame([[
            customer.tenure,
            customer.MonthlyCharges,
            customer.TotalCharges,
            customer.SeniorCitizen
        ] for customer in customers], columns=['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen'])
        
        # Make predictions
        probabilities = model.predict_proba(input_data)[:, 1]
        predictions = probabilities > 0.5
        
        results = []
        for i, (prob, pred) in enumerate(zip(probabilities, predictions)):
            results.append({
                "customer_id": i,
                "churn_probability": round(prob, 4),
                "churn_prediction": bool(pred),
                "prediction_class": "Churn" if pred else "No Churn"
            })
        
        return {
            "predictions": results,
            "total_customers": len(customers)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)