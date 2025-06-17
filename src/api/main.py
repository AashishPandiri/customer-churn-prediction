from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from src.api.schemas import CustomerData, ChurnPrediction, BatchPredictionRequest, BatchPredictionResponse
from src.data.data_preprocessing import DataPreprocessor
from src.models.ensemble_model import EnsembleModel
import joblib

from src.utils.logger import setup_logger
logger = setup_logger(__name__)

app = FastAPI(
    title="Churn Prediction API",
    description="API for predicting customer churn using machine learning models.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
preprocessor_encoder = None
preprocessor_scaler = None

@app.on_event("startup")
async def load_model():
    global model, preprocessor_encoder, preprocessor_scaler
    try:
        model = EnsembleModel()
        model.load_model('models/churn_model.pkl')
        
        preprocessor_encoder = joblib.load('models/preprocessing_encoders.pkl')
        preprocessor_scaler = joblib.load('models/preprocessing_scaler.pkl')
        
        logger.info("Model and preprocessors loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model or preprocessors: {e}")
        raise HTTPException(status_code=500, detail="Model loading failed.")
    
def get_risk_level(probability: float) -> str:
    if probability >= 0.7:
        return "High Risk"
    elif probability >= 0.3:
        return "Medium Risk"
    else:
        return "Low Risk"
    
def preprocess_customer_data(customer_data: CustomerData) -> pd.DataFrame:
    data_dict = customer_data.dict()
    df = pd.DataFrame([data_dict])
    
    df = preprocessor_encoder.encode_features(df, fit=False)
    df = preprocessor_scaler.scale_features(df, fit=False)

    return df

@app.get("/")
async def root():
    return {"message": "Customer Churn Prediction API is running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_encoder_loaded": preprocessor_encoder is not None,
        "preprocessor_scaler_loaded": preprocessor_scaler is not None
    }
    
@app.post("/predict", response_model=ChurnPrediction)
async def predict_churn(request: BatchPredictionRequest):
    try:
        predictions = []
        
        for i, customer_data in enumerate(request.customers):
            preprocessed_data = preprocess_customer_data(customer_data)
            
            churn_probability = model.predict_proba(preprocessed_data)[0]
            churn_prediction = model.predict(preprocessed_data)[0]
            risk_level = get_risk_level(churn_probability)
            
            predictions.append(ChurnPrediction(
                customer_id=f"customer_{i+1}",
                churn_probability=float(churn_probability),
                churn_prediction=int(churn_prediction),
                risk_level=risk_level
            ))
            
        return BatchPredictionResponse(predictions=predictions)
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction failed.")
    
@app.get("/model/info")
async def get_model_info():
    return {
        "model_type": "Ensemble (Random Forest + XGBoost)",
        "version": "1.0.0",
        "features": model.get_feature_importance(feature_names=model.rf_model.feature_names_in_).to_dict(orient='records'),
    }