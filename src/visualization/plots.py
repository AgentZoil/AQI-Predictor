import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
from pathlib import Path

plt.switch_backend('Agg')

class AQIVisualizer:
    def __init__(self, output_dir: str = "docs/images"):
        self.output_dir = output_dir
    
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
        self._save_plot('pm25_time_series.png')
    
    def plot_correlation_heatmap(self, df: pd.DataFrame):
        """Step 6: Correlation Heatmap"""
        plt.figure(figsize=(12, 8))
        rolling_df = df.rolling(window=24*7, min_periods=1).mean()
        sns.heatmap(rolling_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Feature Correlations (7-day Rolling Averages)')
        self._save_plot('feature_correlations_rolling.png')
    
    def plot_monthly_boxplots(self, df: pd.DataFrame):
        """Step 8: Monthly Boxplots"""
        df['month'] = df.index.month
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='month', y='pm25_ugm3', data=df)
        plt.title('Monthly Distribution of PM2.5')
        plt.xlabel('Month')
        plt.ylabel('PM2.5 (µg/m³)')
        self._save_plot('monthly_boxplots.png')
    
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
        self._save_plot('year_over_year_comparison.png')
    
    def _save_plot(self, filename: str):
        """Helper to save plots consistently"""
        plt.savefig(Path(self.output_dir) / filename)
        plt.close()