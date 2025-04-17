import xgboost as xgb
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

class PM25Predictor:
    def __init__(self):
        # Load assets - now only XGBoost
        self.model = self._load_xgboost_model('models/xgboost_pm25.json')
        self.scaler = joblib.load('models/scaler.pkl')  # Keep if you still want scaling
    
    def _load_xgboost_model(self, path):
        """Loads XGBoost model from JSON"""
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file {path} not found. Train model first.")
        
        model = xgb.Booster()
        model.load_model(path)
        return model
    
    def make_predictions(self, new_data):
        """
        Predict PM2.5 levels from new input data.
        
        Args:
            new_data (DataFrame/dict): Must contain these features:
                ['hour', 'day_of_week', 'temp_c', 'wind_speed_ms', 'pm25_24h_avg', 'pm25_lag1h']
        
        Returns:
            numpy.ndarray: Predicted PM2.5 values
        """
        # Handle both DataFrame and API dict input
        if isinstance(new_data, dict):
            input_df = pd.DataFrame([new_data])
        else:
            input_df = new_data.copy()

        # Validate input
        required_cols = [
            'hour', 'day_of_week', 'month', 'day', 
            'temp_c', 'wind_speed_ms', 'humidity_pct', 
            'pm25_24h_avg', 'pm25_6h_avg', 'pm25_6h_std', 
            'pm25_lag1h', 'pm25_lag2h', 'pm25_lag3h'
        ]

        if not all(col in input_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in input_df.columns]
            raise ValueError(f"Missing required columns: {missing}")
        
        print(f"Input data columns: {input_df.columns}")

        # Preprocess (XGBoost doesn't strictly need scaling, but keeping for consistency)
        X = self.scaler.transform(input_df[required_cols])  # Scaling if needed
        print(f"Preprocessed data (first row): {X[0]}")

        # Convert to DMatrix (optimal for XGBoost)
        dmatrix = xgb.DMatrix(X)
        
        # Predict
        predictions = self.model.predict(dmatrix)
        print(f"Predictions: {predictions}")
        
        return predictions