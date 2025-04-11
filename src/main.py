from pathlib import Path
import pandas as pd
from data.loader import DataLoader
from visualization.plots import AQIVisualizer
from analysis.time_series import TimeSeriesAnalyzer
from models.train import PM25ModelTrainer
from models.predict import PM25Predictor

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
    
    # Visualization pipeline
    print("\nGenerating visualizations...")
    visualizer.plot_time_series(df)
    visualizer.plot_correlation_heatmap(df)
    visualizer.plot_monthly_boxplots(df)
    visualizer.plot_year_over_year(df)
    
    # Analysis pipeline
    print("\nPerforming time series analysis...")
    analyzer.seasonal_decomposition(df)
    
    # ===== MODELING SECTION =====
    print("\nStarting model training...")
    
    # 1. Prepare data
    X_train, X_test, y_train, y_test = model_trainer.prepare_data(df)
    
    # 2. Train models
    print("\nTraining Linear Regression model...")
    linear_model = model_trainer.train_linear_model(X_train, y_train)
    
    print("\nTraining LSTM model...")
    lstm_model = model_trainer.train_lstm_model(X_train, y_train)
    
    # 3. Evaluate models
    print("\nEvaluating models...")
    linear_metrics = model_trainer.evaluate_model(linear_model, X_test, y_test)
    lstm_metrics = model_trainer.evaluate_model(lstm_model, X_test, y_test)
    
    print(f"\nLinear Model MAE: {linear_metrics['mae']:.2f} µg/m³")
    print(f"LSTM Model MAE: {lstm_metrics['mae']:.2f} µg/m³")
    
    # 4. Save models using .keras format (recommended)
    model_trainer.save_model(linear_model, 'linear')  # will save as models/linear_pm25.keras
    model_trainer.save_model(lstm_model, 'lstm')      # will save as models/lstm_pm25.keras

    # Report best model
    if linear_metrics['mae'] < lstm_metrics['mae']:
        print("\nBest model: Linear Regression")
    else:
        print("\nBest model: LSTM")

    # 5. Plot prediction results for Linear Regression
    visualizer.plot_predictions(
        y_true=y_test,
        y_pred=linear_model.predict(X_test),
        model_name="Linear Regression"
    )
    
    # 6. Plot prediction results for LSTM
    # Reshape X_test for LSTM model: it should be 3D with shape (samples, time steps, features)
    X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])  # Reshape to (samples, 1, features)
    
    visualizer.plot_predictions(
        y_true=y_test,
        y_pred=lstm_model.predict(X_test_lstm),
        model_name="LSTM"
    )
    
    # ===== EXAMPLE PREDICTIONS SECTION =====
    print("\nGenerating example predictions...")
    
    # Initialize predictor
    predictor = PM25Predictor()  # loads models from .keras files
    
    # Create input: last 24 hours from test set
    example_input = X_test[-24:].copy()
    example_input = pd.DataFrame(
        example_input,
        columns=['hour', 'day_of_week', 'temp_c', 'wind_speed_ms', 'pm25_24h_avg', 'pm25_lag1h']
    )
    
    # Make predictions
    linear_preds = predictor.make_predictions(example_input, model_type='linear')
    lstm_preds = predictor.make_predictions(example_input, model_type='lstm')
    
    # Output comparison
    print("\n--- Prediction Example ---")
    print(f"Linear model predicts next PM2.5: {linear_preds[-1]:.2f} µg/m³")
    print(f"LSTM model predicts next PM2.5: {lstm_preds[-1]:.2f} µg/m³")
    print(f"Actual value was: {y_test.iloc[-1]:.2f} µg/m³")
    
    print("\nAll outputs saved to docs/images/ and models/")

if __name__ == "__main__":
    main()
