import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
from pathlib import Path

class TimeSeriesAnalyzer:
    def __init__(self, output_dir: str = "docs/images"):
        self.output_dir = output_dir
    
    def seasonal_decomposition(self, df: pd.DataFrame, target_col: str = 'pm25_ugm3'):
        """Step 7: Seasonal Decomposition"""
        self._weekly_decomposition(df, target_col)
        self._annual_decomposition(df, target_col)
    
    def _weekly_decomposition(self, df: pd.DataFrame, target_col: str):
        """Weekly decomposition"""
        result = seasonal_decompose(df[target_col].dropna(), period=24*7)
        fig, axs = plt.subplots(4, 1, figsize=(14, 10))
        result.observed.plot(ax=axs[0]); axs[0].set_ylabel("Observed")
        result.trend.plot(ax=axs[1]); axs[1].set_ylabel("Trend")
        result.seasonal.plot(ax=axs[2]); axs[2].set_ylabel("Weekly Seasonal")
        result.resid.plot(ax=axs[3]); axs[3].set_ylabel("Residual")
        plt.suptitle('Weekly Seasonality Decomposition')
        plt.tight_layout()
        self._save_plot('seasonal_decomposition_weekly.png')
    
    def _annual_decomposition(self, df: pd.DataFrame, target_col: str):
        """Annual decomposition"""
        monthly_data = df[target_col].resample('MS').mean()
        result = seasonal_decompose(monthly_data.dropna(), period=12)
        fig, axs = plt.subplots(4, 1, figsize=(14, 10))
        result.observed.plot(ax=axs[0]); axs[0].set_ylabel("Observed")
        result.trend.plot(ax=axs[1]); axs[1].set_ylabel("Trend")
        result.seasonal.plot(ax=axs[2]); axs[2].set_ylabel("Annual Seasonal")
        result.resid.plot(ax=axs[3]); axs[3].set_ylabel("Residual")
        plt.suptitle('Annual Seasonality Decomposition')
        plt.tight_layout()
        self._save_plot('seasonal_decomposition_annual.png')
    
    def _save_plot(self, filename: str):
        """Helper to save plots"""
        plt.savefig(Path(self.output_dir) / filename)
        plt.close()
