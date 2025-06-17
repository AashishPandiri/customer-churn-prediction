from pydantic import BaseModel
from typing import List, Optional

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

class ChurnPrediction(BaseModel):
    customer_id: Optional[str] = None
    churn_probability: float
    churn_prediction: int
    risk_level: str
    
class BatchPredictionRequest(BaseModel):
    customers: List[CustomerData]
    
class BatchPredictionResponse(BaseModel):
    predictions: List[ChurnPrediction]