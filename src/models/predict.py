import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from pathlib import Path

class PM25Predictor:
    def __init__(self):
        # Load assets
        self.models = {
            'linear': self._load_model('models/linear_pm25.pkl'),
            'lstm': self._load_model('models/lstm_pm25.keras')
        }
        self.scaler = joblib.load('models/scaler.pkl')
    
    def _load_model(self, path):
        """Handles loading different model types"""
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file {path} not found. Train models first.")
        
        if path.endswith('.pkl'):
            return joblib.load(path)
        else:
            return load_model(path)
    
    def make_predictions(self, new_data, model_type='linear'):
        """
        Predict PM2.5 levels from new input data.
        
        Args:
            new_data (DataFrame): Must contain these columns:
                ['hour', 'day_of_week', 'temp_c', 'wind_speed_ms', 'pm25_24h_avg', 'pm25_lag1h']
            model_type (str): 'linear' or 'lstm'
        
        Returns:
            numpy.ndarray: Predicted PM2.5 values
        """
        # Validate input
        required_cols = ['hour', 'day_of_week', 'temp_c', 'wind_speed_ms', 'pm25_24h_avg', 'pm25_lag1h']
        if not all(col in new_data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in new_data.columns]
            raise ValueError(f"Missing required columns: {missing}")
        
        # Preprocess
        X = self.scaler.transform(new_data[required_cols])
        
        # Model-specific processing
        if model_type == 'lstm':
            X = X.reshape((X.shape[0], 1, X.shape[1]))
        
        # Predict
        return self.models[model_type].predict(X).flatten()