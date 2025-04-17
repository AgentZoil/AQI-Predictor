import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import os
import numpy as np
import xgboost as xgb
from typing import List, Dict

plt.switch_backend('Agg')

class AQIVisualizer:
    def __init__(self, output_dir: str = "docs/images"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_style("darkgrid")
        self.color_palette = sns.color_palette("husl", 8)
    
    def _save_matplotlib_plot(self, filename: str, dpi: int = 300):
        """Helper to save matplotlib plots"""
        path = Path(self.output_dir) / filename
        plt.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {path}")

    def _save_plotly_plot(self, fig: go.Figure, filename: str):
        """Helper to save plotly plots"""
        path = Path(self.output_dir) / filename
        fig.write_image(path)
        print(f"Saved plot to {path}")

    def plot_monthly_boxplots(self, df: pd.DataFrame):
        """Monthly distribution analysis using boxplots"""
        plt.figure(figsize=(12, 6))
        
        # Create month column if not exists
        if 'month' not in df.columns:
            df['month'] = df.index.month
        
        sns.boxplot(
            x='month',
            y='pm25_ugm3',
            data=df,
            palette=self.color_palette,
            showfliers=False  # Hide outliers for cleaner visualization
        )
        
        plt.title('Monthly PM2.5 Distribution')
        plt.xlabel('Month')
        plt.ylabel('PM2.5 (µg/m³)')
        plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        self._save_matplotlib_plot('monthly_boxplots.png')

    def plot_time_series(self, df: pd.DataFrame):
        """PM2.5 time series analysis"""
        plt.figure(figsize=(16, 10))
        
        # Daily and weekly rolling averages
        plt.subplot(2, 1, 1)
        df['pm25_ugm3'].rolling(24).mean().plot(
            color=self.color_palette[0],
            label='24h Rolling Avg'
        )
        df['pm25_ugm3'].rolling(24*7).mean().plot(
            color=self.color_palette[1],
            label='Weekly Rolling Avg'
        )
        plt.title('PM2.5 Concentration Trends')
        plt.ylabel('PM2.5 (µg/m³)')
        plt.legend()
        
        # Monthly resampled
        plt.subplot(2, 1, 2)
        monthly = df['pm25_ugm3'].resample('MS').mean()
        monthly.plot(kind='bar', color=self.color_palette[2])
        plt.title('Monthly Average PM2.5')
        plt.ylabel('PM2.5 (µg/m³)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        self._save_matplotlib_plot('pm25_time_series.png')

    def plot_correlation_heatmap(self, df: pd.DataFrame):
        """Feature correlation analysis"""
        plt.figure(figsize=(12, 8))
        corr_matrix = df.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            cmap='icefire',
            center=0,
            fmt='.2f',
            linewidths=.5
        )
        plt.title('Feature Correlation Matrix')
        self._save_matplotlib_plot('feature_correlations.png')

    def plot_feature_distributions(self, df: pd.DataFrame):
        """Feature distributions and outliers"""
        numeric_cols = ['temp_c', 'wind_speed_ms', 'pm25_ugm3', 'humidity_pct']
        plt.figure(figsize=(15, 10))
        
        for i, col in enumerate(numeric_cols, 1):
            plt.subplot(2, 2, i)
            sns.histplot(df[col], kde=True, color=self.color_palette[i+2])
            plt.title(f'{col} Distribution')
        
        plt.tight_layout()
        self._save_matplotlib_plot('feature_distributions.png')

    def plot_xgboost_feature_importance(self, model: xgb.Booster, feature_names: List[str]):
        """XGBoost feature importance visualization"""
        importance_types = ['weight', 'gain', 'cover']
        fig = go.Figure()
        
        for imp_type in importance_types:
            importance = model.get_score(importance_type=imp_type)
            sorted_idx = np.argsort(list(importance.values()))
            
            fig.add_trace(go.Bar(
                y=[feature_names[i] for i in sorted_idx],
                x=[list(importance.values())[i] for i in sorted_idx],
                name=imp_type.capitalize(),
                orientation='h'
            ))
        
        fig.update_layout(
            title='XGBoost Feature Importance',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            barmode='group',
            template='plotly_dark',
            height=600
        )
        self._save_plotly_plot(fig, 'xgboost_feature_importance.png')

    def plot_predictions(self, y_true: pd.Series, y_pred: np.ndarray, model_name: str):
        """Interactive prediction vs actual visualization"""
        if not isinstance(y_true, pd.Series):
            y_true = pd.Series(y_true)
        
        y_pred = pd.Series(y_pred.flatten(), index=y_true.index)
        
        # Create 30-day rolling metrics
        metrics = pd.DataFrame({
            'true': y_true,
            'pred': y_pred
        }).rolling(24*30).agg({
            'true': ['mean', 'std'],
            'pred': ['mean', 'std']
        })
        
        fig = go.Figure()
        
        # Actual values
        fig.add_trace(go.Scatter(
            x=y_true.index,
            y=y_true,
            name='Actual PM2.5',
            mode='lines',
            line=dict(color='#1f77b4', width=1),
            opacity=0.7
        ))
        
        # Predictions
        fig.add_trace(go.Scatter(
            x=y_pred.index,
            y=y_pred,
            name=f'{model_name} Predictions',
            mode='lines',
            line=dict(color='#ff7f0e', width=2),
            opacity=0.9
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=metrics.index,
            y=metrics[('pred', 'mean')] + metrics[('pred', 'std')],
            fill=None,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=metrics.index,
            y=metrics[('pred', 'mean')] - metrics[('pred', 'std')],
            fill='tonexty',
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(255, 127, 14, 0.2)',
            name='±1 Std Dev'
        ))
        
        fig.update_layout(
            title=f'{model_name} Predictions vs Actual',
            xaxis_title='Date',
            yaxis_title='PM2.5 (µg/m³)',
            hovermode='x unified',
            template='plotly_dark',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add MAE annotation
        mae = np.mean(np.abs(y_true - y_pred))
        fig.add_annotation(
            x=0.05,
            y=0.95,
            xref='paper',
            yref='paper',
            text=f'MAE: {mae:.2f} µg/m³',
            showarrow=False,
            bgcolor='white',
            font=dict(size=12)
        )
        
        self._save_plotly_plot(fig, f'predictions_{model_name.lower()}.png')

    def plot_residuals(self, y_true: pd.Series, y_pred: np.ndarray, model_name: str):
        """Residual analysis plot"""
        residuals = y_true - y_pred.flatten()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Residuals vs Predicted
        sns.scatterplot(
            x=y_pred,
            y=residuals,
            alpha=0.5,
            color=self.color_palette[4],
            ax=ax1
        )
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_title('Residuals vs Predicted')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        
        # Residual distribution
        sns.histplot(
            residuals,
            kde=True,
            color=self.color_palette[5],
            ax=ax2
        )
        ax2.set_title('Residual Distribution')
        ax2.set_xlabel('Residuals')
        
        plt.suptitle(f'{model_name} Residual Analysis')
        self._save_matplotlib_plot(f'residuals_{model_name.lower()}.png')

    def plot_lagged_autocorrelation(self, df: pd.DataFrame, max_lags: int = 48):
        """Autocorrelation analysis for PM2.5"""
        from statsmodels.graphics.tsaplots import plot_acf
        
        plt.figure(figsize=(12, 6))
        plot_acf(
            df['pm25_ugm3'],
            lags=max_lags,
            title='PM2.5 Autocorrelation',
            color=self.color_palette[6]
        )
        plt.xlabel('Lag (hours)')
        plt.ylabel('Correlation')
        self._save_matplotlib_plot('pm25_autocorrelation.png')