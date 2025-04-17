from pathlib import Path
import pandas as pd
from data.loader import DataLoader
from visualization.plots import AQIVisualizer
from analysis.time_series import TimeSeriesAnalyzer
from models.train import PM25ModelTrainer
from models.predict import PM25Predictor
import numpy as np

def main():
    # Initialize components
    data_loader = DataLoader("data/raw/Wollongong_09042022_10042025.xlsx")
    visualizer = AQIVisualizer()
    analyzer = TimeSeriesAnalyzer()
    model_trainer = PM25ModelTrainer()
    
    # Data pipeline
    print("Loading and cleaning data...")
    df = data_loader.load_and_clean()
    
    # Show summary
    print("\n--- Summary Info ---")
    print(f"Data covers: {df.index.min()} to {df.index.max()}")
    print(f"Total entries: {len(df)}")
    print(df.describe())
    print(df.head())  # Check if 'datetime' is being parsed correctly
    print(df.index)  # Ensure it's a DatetimeIndex
    print(df.columns)


    
    # Visualization pipeline
    print("\nGenerating visualizations...")
    visualizer.plot_time_series(df)
    visualizer.plot_correlation_heatmap(df)
    visualizer.plot_monthly_boxplots(df)
    
    # Analysis pipeline
    print("\nPerforming time series analysis...")
    analyzer.seasonal_decomposition(df)
    
    # ===== XGBOOST MODELING SECTION =====
    print("\nStarting XGBoost training...")
    
    # 1. Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = model_trainer.prepare_data(df)

    # 2. Train XGBoost model
    print("\nTraining XGBoost model...")
    xgb_model = model_trainer.train_xgboost_model(
        X_train, y_train,
        X_val, y_val
    )

    # 3. Evaluate
    xgb_metrics = model_trainer.evaluate_model(xgb_model, X_test, y_test)
    print(f"\nXGBoost MAE: {xgb_metrics['mae']:.2f} µg/m³")
    
    # 4. Save model
    model_trainer.save_model(xgb_model)
    
    # 5. Plot predictions
    y_pred = xgb_model.predict(X_test)
    visualizer.plot_predictions(
        y_true=y_test,
        y_pred=y_pred,
        model_name="XGBoost"
    )
    
    # ===== EXAMPLE PREDICTIONS =====
    print("\nGenerating example predictions...")
    predictor = PM25Predictor()

    # Example 1: Using the most recent data (what you already have)
    example_input = pd.DataFrame(
        X_test[-24:],
        columns=['hour', 'day_of_week', 'month', 'day', 'temp_c', 'wind_speed_ms', 
                'humidity_pct', 'pm25_24h_avg', 'pm25_6h_avg', 'pm25_6h_std', 
                'pm25_lag1h', 'pm25_lag2h', 'pm25_lag3h']
    )
    prediction = np.expm1(predictor.make_predictions(example_input)[0])
    print(f"\n--- Example 1: Most Recent Data ---")
    print(f"XGBoost predicts next PM2.5: {prediction:.2f} µg/m³")
    print(f"Actual last value was: {df['pm25_ugm3'].iloc[-1]:.2f} µg/m³")

    # Add more examples from different time periods
    # Example 2: From beginning of test set
    early_example = pd.DataFrame(
        X_test[:24],
        columns=['hour', 'day_of_week', 'month', 'day', 'temp_c', 'wind_speed_ms', 
                'humidity_pct', 'pm25_24h_avg', 'pm25_6h_avg', 'pm25_6h_std', 
                'pm25_lag1h', 'pm25_lag2h', 'pm25_lag3h']
    )
    early_prediction = np.expm1(predictor.make_predictions(early_example)[0])
    test_start_index = len(df) - len(X_test)  # Calculate where test set begins in original df
    print(f"\n--- Example 2: Early Test Data ---")
    print(f"Time period: {df.index[test_start_index]}")
    print(f"XGBoost predicts PM2.5: {early_prediction:.2f} µg/m³")
    print(f"Actual value was: {df['pm25_ugm3'].iloc[test_start_index]:.2f} µg/m³")

    # Example 3: From middle of test set
    mid_idx = len(X_test) // 2
    mid_example = pd.DataFrame(
        X_test[mid_idx:mid_idx+24],
        columns=['hour', 'day_of_week', 'month', 'day', 'temp_c', 'wind_speed_ms', 
                'humidity_pct', 'pm25_24h_avg', 'pm25_6h_avg', 'pm25_6h_std', 
                'pm25_lag1h', 'pm25_lag2h', 'pm25_lag3h']
    )
    mid_prediction = np.expm1(predictor.make_predictions(mid_example)[0])
    mid_df_idx = test_start_index + mid_idx
    print(f"\n--- Example 3: Mid-Test Data ---")
    print(f"Time period: {df.index[mid_df_idx]}")
    print(f"XGBoost predicts PM2.5: {mid_prediction:.2f} µg/m³")
    print(f"Actual value was: {df['pm25_ugm3'].iloc[mid_df_idx]:.2f} µg/m³")

if __name__ == "__main__":
    main()
