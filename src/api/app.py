from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from pathlib import Path
from src.models.predict import PM25Predictor

# Initialize app and model
app = FastAPI(title="AQI Predictor API", 
              description="Predicts PM2.5 levels using XGBoost")

# Load predictor once at startup
predictor = PM25Predictor()

# Request/Response schemas
class PredictionRequest(BaseModel):
    hour: int
    day_of_week: int
    temp_c: float
    wind_speed_ms: float
    pm25_24h_avg: float
    pm25_lag1h: float

class PredictionResponse(BaseModel):
    pm25_pred: float
    units: str = "µg/m³"

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict PM2.5 concentration given weather and historical air quality data.
    
    Example Input:
    ```json
    {
        "hour": 14,
        "day_of_week": 2,
        "temp_c": 22.5,
        "wind_speed_ms": 3.2,
        "pm25_24h_avg": 12.7,
        "pm25_lag1h": 15.3
    }
    ```
    """
    try:
        # Convert request to DataFrame (matches training format)
        input_df = pd.DataFrame([request.dict()])
        
        # Predict
        prediction = predictor.make_predictions(input_df)[0]
        
        return {
            "pm25_pred": round(float(prediction), 2),
            "units": "µg/m³"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}