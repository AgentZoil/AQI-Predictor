import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error
from pathlib import Path

class PM25ModelTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def prepare_data(self, df, target_col='pm25_ugm3', test_size=0.2):
        """Prepares features and splits data"""
        features = ['hour', 'day_of_week', 'temp_c', 'wind_speed_ms', 'pm25_24h_avg', 'pm25_lag1h']
        X = df[features].dropna()
        y = df[target_col].loc[X.index]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Time-series split (no shuffling)
        split_idx = int(len(X) * (1 - test_size))
        return (
            X_scaled[:split_idx], 
            X_scaled[split_idx:],
            y[:split_idx],
            y[split_idx:]
        )
    
    def train_linear_model(self, X_train, y_train):
        """Trains a regularized linear model"""
        model = Ridge(alpha=0.1)
        model.fit(X_train, y_train)
        return model
    
    def train_lstm_model(self, X_train, y_train):
        """Trains an LSTM model"""
        # Reshape for LSTM [samples, timesteps, features]
        X_3d = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        
        model = Sequential([
            LSTM(64, input_shape=(1, X_train.shape[1])),
            Dense(1)
        ])
        model.compile(loss='mae', optimizer='adam')
        model.fit(X_3d, y_train, epochs=20, batch_size=32, verbose=0)
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluates model performance"""
        if 'keras' in str(type(model)):
            X_test_3d = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
            y_pred = model.predict(X_test_3d).flatten()
        else:
            y_pred = model.predict(X_test)
        
        return {
            'mae': mean_absolute_error(y_test, y_pred)
        }
    
    def save_model(self, model, model_type):
        """Saves model to disk"""
        Path("models").mkdir(exist_ok=True)
        if model_type == 'linear':
            joblib.dump(model, 'models/linear_pm25.pkl')
        else:
            model.save('models/lstm_pm25.keras')
        joblib.dump(self.scaler, 'models/scaler.pkl')