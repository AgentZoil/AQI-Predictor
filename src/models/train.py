import xgboost as xgb
import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import numpy as np

class PM25ModelTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.features = [
            'hour', 'day_of_week', 'month', 'day',
            'temp_c', 'wind_speed_ms', 'humidity_pct',
            'pm25_24h_avg', 'pm25_6h_avg', 'pm25_6h_std',
            'pm25_lag1h', 'pm25_lag2h', 'pm25_lag3h'
        ]

    def prepare_data(self, df, target_col='pm25_ugm3', test_size=0.2, val_size=0.15):
        X = df[self.features].dropna()
        y = df[target_col].loc[X.index]
        
        # Optional: Log-transform target
        y = np.log1p(y)

        X_scaled = self.scaler.fit_transform(X)
        
        test_idx = int(len(X) * (1 - test_size))
        val_idx = int(test_idx * (1 - val_size))
        
        return (
            X_scaled[:val_idx], X_scaled[val_idx:test_idx], X_scaled[test_idx:],
            y[:val_idx], y[val_idx:test_idx], y[test_idx:]
        )

    def train_xgboost_model(self, X_train, y_train, X_val=None, y_val=None):
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': 3000,
            'max_depth': 8,
            'learning_rate': 0.03,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'eval_metric': 'mae'
        }

        if X_val is not None and y_val is not None:
            model = xgb.XGBRegressor(**params, early_stopping_rounds=100)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
        else:
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
        
        return model

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        # Undo log transformation
        y_pred = np.expm1(y_pred)
        y_true = np.expm1(y_test)
        return {'mae': mean_absolute_error(y_true, y_pred)}

    def save_model(self, model):
        Path("models").mkdir(exist_ok=True)
        model.save_model('models/xgboost_pm25.json')
        joblib.dump(self.scaler, 'models/scaler.pkl')
