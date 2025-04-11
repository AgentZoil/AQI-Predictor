import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import os
import numpy as np


plt.switch_backend('Agg')

class AQIVisualizer:
    def __init__(self, output_dir: str = "docs/images"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)  # Ensure the directory exists
    
    def plot_time_series(self, df: pd.DataFrame):
        """Step 5: PM2.5 Time Series Plots"""
        plt.figure(figsize=(16, 8))
        plt.subplot(2, 1, 1)
        df['pm25_ugm3'].resample('D').mean().plot(title='Daily Average PM2.5')
        plt.ylabel('PM2.5 (µg/m³)')

        plt.subplot(2, 1, 2)
        df['pm25_ugm3'].resample('MS').mean().plot(title='Monthly Average PM2.5')
        plt.ylabel('PM2.5 (µg/m³)')
        plt.tight_layout()
        self._save_matplotlib_plot('pm25_time_series.png')
    
    def plot_correlation_heatmap(self, df: pd.DataFrame):
        """Step 6: Correlation Heatmap"""
        plt.figure(figsize=(12, 8))
        rolling_df = df.rolling(window=24*7, min_periods=1).mean()
        sns.heatmap(rolling_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Feature Correlations (7-day Rolling Averages)')
        self._save_matplotlib_plot('feature_correlations_rolling.png')

    def plot_monthly_boxplots(self, df: pd.DataFrame):
        """Step 8: Monthly Boxplots"""
        df['month'] = df.index.month
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='month', y='pm25_ugm3', data=df)
        plt.title('Monthly Distribution of PM2.5')
        plt.xlabel('Month')
        plt.ylabel('PM2.5 (µg/m³)')
        self._save_matplotlib_plot('monthly_boxplots.png')
    
    def plot_year_over_year(self, df: pd.DataFrame):
        """Step 9: Year-over-Year Comparison"""
        df['year'] = df.index.year
        plt.figure(figsize=(14, 6))
        for year in sorted(df['year'].unique()):
            year_data = df[df['year'] == year].copy()
            plt.plot(year_data.index, year_data['pm25_ugm3'].rolling(24*7).mean(), label=str(year))
        plt.title('Year-over-Year PM2.5 Comparison')
        plt.xlabel('Date')
        plt.ylabel('PM2.5 (µg/m³)')
        plt.legend()
        plt.tight_layout()
        self._save_matplotlib_plot('year_over_year_comparison.png')

    def plot_predictions(self, y_true, y_pred, model_name):
        # Ensure y_true is a pandas Series
        if not isinstance(y_true, pd.Series):
            y_true = pd.Series(y_true)

        # Flatten y_pred (if it has multiple dimensions) and ensure it's a pandas Series with the same index as y_true
        if isinstance(y_pred, np.ndarray):
            y_pred = y_pred.flatten()  # Flatten the ndarray to 1D

        if not isinstance(y_pred, pd.Series):
            y_pred = pd.Series(y_pred, index=y_true.index)

        # Find the most recent date in the true data
        most_recent_date = y_true.index[-1]
        
        # Calculate the date 1 month ago (30 days)
        one_month_ago = most_recent_date - pd.DateOffset(days=30)
        
        # Filter the data to the last month for better comparison
        last_month_data = y_true[(y_true.index >= one_month_ago) & 
                                (y_true.index <= most_recent_date)]
        last_month_predictions = y_pred[(y_pred.index >= one_month_ago) & 
                                        (y_pred.index <= most_recent_date)]

        # Create a figure with Plotly
        fig = go.Figure()

        # Plot the true values (observed PM2.5) for the last month
        fig.add_trace(go.Scatter(
            x=last_month_data.index,
            y=last_month_data,
            mode='lines',
            name='True PM2.5',
            line=dict(color='blue', width=2),
        ))

        # Plot the predicted values for the given model for the last month
        fig.add_trace(go.Scatter(
            x=last_month_predictions.index,
            y=last_month_predictions,
            mode='lines',
            name=f'{model_name} Predictions',
            line=dict(color='orange', width=2, dash='dash'),
        ))

        # Customize the layout with title, labels, and legend
        fig.update_layout(
            title=f'PM2.5 Prediction vs Reality for the Last Month ({model_name})',
            xaxis_title='Time',
            yaxis_title='PM2.5 (µg/m³)',
            template='plotly_dark',
            legend=dict(x=0, y=1, traceorder='normal'),
            hovermode='x unified'
        )

        # Enable zooming and panning
        fig.update_xaxes(rangeslider_visible=True)  # Add range slider
        fig.update_yaxes(scaleanchor="x")  # Make the y-axis scale consistent with the x-axis

        # Show the plot
        fig.show()

        # Save the plot as an image
        fig.write_image(f'docs/images/predictions_{model_name.lower().replace(" ", "_")}_last_month.png')

        
    def _save_matplotlib_plot(self, filename: str):
        """Helper to save matplotlib plots consistently"""
        plt.savefig(Path(self.output_dir) / filename)
        plt.close()

