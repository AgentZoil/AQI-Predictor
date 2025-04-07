import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# 1. Load data
df = pd.read_excel("Wollongong.xls")

# 2. Clean column names
df.columns = ['date', 'time', 'temp_c', 'wind_speed_ms', 'pm25_ugm3', 'humidity_pct']

# 3. Create datetime index (FIXED VERSION)
try:
    df['datetime'] = pd.to_datetime(
        df['date'].astype(str) + ' ' + df['time'].astype(str),
        format='%d/%m/%Y %H:%M'  # For "01/03/2025 01:00" format
    )
except ValueError as e:
    print("Date parsing error. Actual date format:")
    print(df[['date', 'time']].head())
    raise e

df = df.set_index('datetime')

# 4. Create features
df['pm25_prev_hour'] = df['pm25_ugm3'].shift(1)
df = df.dropna()  # Remove rows with missing previous hour data

# 5. Prepare model inputs
features = ['temp_c', 'wind_speed_ms', 'humidity_pct', 'pm25_prev_hour']
X = df[features]
y = df['pm25_ugm3']

# 6. Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = XGBRegressor()
model.fit(X_train, y_train)

print("Model trained successfully!")