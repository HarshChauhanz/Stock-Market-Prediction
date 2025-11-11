from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import datetime as dt
# Adjust this import if your 'src' folder is one level up
import sys
sys.path.append("..") 
from src.features import create_date_features, get_feature_columns

app = FastAPI() # <--- UVICORN LOOKS FOR THIS LINE

# Add CORS so frontend can talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- UPDATE THIS PATH TO WHERE YOUR MODELS ACTUALLY ARE ---
# If models are in D:\stock_prediction_project\models, use ".." to go up one level
MODELS_DIR = "../models/" 

class PredictionRequest(BaseModel):
    bank_name: str
    date: str
    period: str

@app.get("/")
def home():
    return {"message": "Stock Prediction API is running"}

@app.post("/predict")
def predict_stock(request: PredictionRequest):
    model_path = os.path.join(MODELS_DIR, f"{request.bank_name}_model.pkl")
    
    if not os.path.exists(model_path):
        # Try looking in local backend folder just in case user put them there
        model_path = os.path.join("models/", f"{request.bank_name}_model.pkl")
        if not os.path.exists(model_path):
             raise HTTPException(status_code=404, detail=f"Model not found at {model_path}")

    try:
        model = joblib.load(model_path)
        target_date = pd.to_datetime(request.date)

        if request.period == "day":
            start_date = target_date - dt.timedelta(days=7)
            end_date = target_date + dt.timedelta(days=7)
        elif request.period == "month":
            start_date = target_date.replace(day=1)
            end_date = (start_date + pd.DateOffset(months=1)) - dt.timedelta(days=1)
        elif request.period == "year":
            start_date = target_date.replace(month=1, day=1)
            end_date = target_date.replace(month=12, day=31)

        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        df_predict = pd.DataFrame({'Date': date_range})
        df_features = create_date_features(df_predict)
        X_pred = df_features[get_feature_columns()]
        df_predict['Predicted_Close'] = model.predict(X_pred)

        target_prediction = df_predict[df_predict['Date'] == target_date]['Predicted_Close'].values
        target_val = target_prediction[0] if len(target_prediction) > 0 else 0

        return {
            "bank": request.bank_name,
            "target_date": request.date,
            "period": request.period,
            "dates": df_predict['Date'].dt.strftime('%Y-%m-%d').tolist(),
            "prices": df_predict['Predicted_Close'].tolist(),
            "target_prediction": target_val
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))