from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

router = APIRouter(prefix="/predict", tags=["Prediction"])

# Load models globally
preprocessor = None
model = None

@router.on_event("startup")
def load_models():
    global preprocessor, model
    try:
        if os.path.exists('artifacts/data_transformation/preprocessor.joblib'):
            preprocessor = joblib.load('artifacts/data_transformation/preprocessor.joblib')
        if os.path.exists('artifacts/model_trainer/model.joblib'):
            model = joblib.load('artifacts/model_trainer/model.joblib')
    except Exception as e:
        print(f"Warning: Could not load models. {e}")

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
    id: int = 999999

@router.post("/")
def predict_churn(data: CustomerData):
    if not model or not preprocessor:
        raise HTTPException(status_code=500, detail="Models are not trained/loaded yet.")
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.model_dump() if hasattr(data, 'model_dump') else data.dict()])
    
    try:
        # Transform and Predict
        X_transformed = preprocessor.transform(input_df)
        prob = model.predict_proba(X_transformed)[0][1] * 100
        
        return {
            "status": "success",
            "churn_probability": prob
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
