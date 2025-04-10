import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np

# Prevent GUI display of plots
plt.switch_backend('Agg')

# --------------------------
# Step 1: Load and Clean Data
# --------------------------
try:
    df = pd.read_excel("Wollongong_09042022_10042025.xlsx", skiprows=1, header=0)
    df.columns = ['date', 'time', 'temp_c', 'wind_speed_ms', 'pm25_ugm3', 'humidity_pct']
    df = df[df['date'] != "Date"]
    numeric_cols = ['temp_c', 'wind_speed_ms', 'pm25_ugm3', 'humidity_pct']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    print("Data loaded and cleaned successfully.")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# --------------------------
# Step 2: Handle PM2.5 Data
# --------------------------
print("\n--- Checking for negative PM2.5 values ---")
negative_pm25 = df[df['pm25_ugm3'] < 0]
print(f"Found {len(negative_pm25)} rows with negative PM2.5 values")
df = df[df['pm25_ugm3'] >= 0]

# --------------------------
# Step 3: Fix Datetime Issues
# --------------------------
print("\n--- Fixing Datetime Conversion ---")
def fix_datetime(row):
    try:
        if row['time'] == '24:00':
            new_date = pd.to_datetime(row['date'], format='%d/%m/%Y') + pd.Timedelta(days=1)
            return new_date.strftime('%d/%m/%Y') + ' 00:00'
        return row['date'] + ' ' + row['time']
    except:
        return None

df['datetime_str'] = df.apply(fix_datetime, axis=1)
df['datetime'] = pd.to_datetime(df['datetime_str'], format='%d/%m/%Y %H:%M', errors='coerce')
df = df[df['datetime'].notna()]
df = df.set_index('datetime').drop(columns=['date', 'time', 'datetime_str'])

# --------------------------
# Step 4: Summary Information
# --------------------------
print("\n--- Summary Info ---")
print(f"Data covers: {df.index.min()} to {df.index.max()}")
print(f"Total entries: {len(df)}")
print(df.describe())

# --------------------------
# Step 5: PM2.5 Time Series Plots
# --------------------------
plt.figure(figsize=(16, 8))
plt.subplot(2, 1, 1)
df['pm25_ugm3'].resample('D').mean().plot(title='Daily Average PM2.5 (April 2022 - April 2025)')
plt.ylabel('PM2.5 (µg/m³)')

plt.subplot(2, 1, 2)
df['pm25_ugm3'].resample('MS').mean().plot(title='Monthly Average PM2.5')
plt.ylabel('PM2.5 (µg/m³)')
plt.tight_layout()
plt.savefig('pm25_time_series.png')
plt.close()

# --------------------------
# Step 6: Correlation Heatmap
# --------------------------
plt.figure(figsize=(12, 8))
rolling_df = df.rolling(window=24*7, min_periods=1).mean()
sns.heatmap(rolling_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlations (7-day Rolling Averages)')
plt.savefig('feature_correlations_rolling.png')
plt.close()

# --------------------------
# Step 7: Seasonal Decomposition
# --------------------------
# Weekly
weekly_result = seasonal_decompose(df['pm25_ugm3'].dropna(), period=24*7)
fig, axs = plt.subplots(4, 1, figsize=(14, 10))
weekly_result.observed.plot(ax=axs[0]); axs[0].set_ylabel("Observed")
weekly_result.trend.plot(ax=axs[1]); axs[1].set_ylabel("Trend")
weekly_result.seasonal.plot(ax=axs[2]); axs[2].set_ylabel("Weekly Seasonal")
weekly_result.resid.plot(ax=axs[3]); axs[3].set_ylabel("Residual")
plt.suptitle('Weekly Seasonality Decomposition')
plt.tight_layout()
plt.savefig('seasonal_decomposition_weekly.png')
plt.close()

# Annual
monthly_pm25 = df['pm25_ugm3'].resample('MS').mean()
annual_result = seasonal_decompose(monthly_pm25.dropna(), period=12)
fig, axs = plt.subplots(4, 1, figsize=(14, 10))
annual_result.observed.plot(ax=axs[0]); axs[0].set_ylabel("Observed")
annual_result.trend.plot(ax=axs[1]); axs[1].set_ylabel("Trend")
annual_result.seasonal.plot(ax=axs[2]); axs[2].set_ylabel("Annual Seasonal")
annual_result.resid.plot(ax=axs[3]); axs[3].set_ylabel("Residual")
plt.suptitle('Annual Seasonality Decomposition')
plt.tight_layout()
plt.savefig('seasonal_decomposition_annual.png')
plt.close()

# --------------------------
# Step 8: Monthly Boxplots
# --------------------------
df['month'] = df.index.month
plt.figure(figsize=(12, 6))
sns.boxplot(x='month', y='pm25_ugm3', data=df)
plt.title('Monthly Distribution of PM2.5 (3 Years)')
plt.xlabel('Month')
plt.ylabel('PM2.5 (µg/m³)')
plt.savefig('monthly_boxplots.png')
plt.close()

# --------------------------
# Step 9: Year-over-Year Comparison
# --------------------------
df['year'] = df.index.year
df['month_day'] = df.index.strftime('%m-%d')
plt.figure(figsize=(14, 6))
for year in sorted(df['year'].unique()):
    year_data = df[df['year'] == year].copy()
    # Use index to plot instead of parsing month_day
    plt.plot(year_data.index, year_data['pm25_ugm3'].rolling(24*7).mean(), label=str(year))
plt.title('Year-over-Year PM2.5 Comparison (7-day Rolling Average)')
plt.xlabel('Date')
plt.ylabel('PM2.5 (µg/m³)')
plt.legend()
plt.tight_layout()
plt.savefig('year_over_year_comparison.png')
plt.close()

# --------------------------
# Step 10: Done
# --------------------------
print("\nPlots saved to current directory:")
print("- pm25_time_series.png")
print("- feature_correlations_rolling.png")
print("- seasonal_decomposition_weekly.png")
print("- seasonal_decomposition_annual.png")
print("- monthly_boxplots.png")
print("- year_over_year_comparison.png")
