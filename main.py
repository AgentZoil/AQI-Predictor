import pandas as pd

# Load the data skipping first row (metadata) and use second row as header
try:
    df = pd.read_excel("Wollongong.xls", skiprows=1, header=0)
    
    # Clean column names
    df.columns = ['date', 'time', 'temp_c', 'wind_speed_ms', 'pm25_ugm3', 'humidity_pct']
    
    # Remove any remaining metadata rows where date is "Date"
    df = df[df['date'] != "Date"]
    
    # Convert numeric columns
    numeric_cols = ['temp_c', 'wind_speed_ms', 'pm25_ugm3', 'humidity_pct']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # Display cleaned data
    print("Cleaned Data (first 10 rows):")
    print(df.head(10))
    
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nBasic Statistics:")
    print(df.describe())
    
except Exception as e:
    print(f"Error loading file: {e}")

# Check for negative PM2.5 values (physically impossible)
print("\n--- Checking for negative PM2.5 values ---")
negative_pm25 = df[df['pm25_ugm3'] < 0]
print(f"Found {len(negative_pm25)} rows with negative PM2.5 values")
if not negative_pm25.empty:
    print("\nSample negative PM2.5 rows:")
    print(negative_pm25.head())

# Optional: Remove negative PM2.5 values
print("\n--- Handling negative values ---")
initial_count = len(df)
df = df[df['pm25_ugm3'] >= 0]
final_count = len(df)
print(f"Removed {initial_count - final_count} rows with negative PM2.5 values")
print(f"Remaining data: {final_count} rows")

print("\n--- Fixing Datetime Conversion ---")

# First identify problematic time values
invalid_times = df[~df['time'].astype(str).str.match(r'^\d{2}:\d{2}$')]
print(f"Found {len(invalid_times)} rows with non-standard time formats")
if not invalid_times.empty:
    print("\nSample invalid times:")
    print(invalid_times[['date', 'time']].head())

# Fix the 24:00 issue by:
# 1. Converting "24:00" to "00:00" of next day
# 2. Then adjusting the date accordingly

def fix_datetime(row):
    try:
        if row['time'] == '24:00':
            # Convert to next day at 00:00
            new_date = pd.to_datetime(row['date'], format='%d/%m/%Y') + pd.Timedelta(days=1)
            return new_date.strftime('%d/%m/%Y') + ' ' + '00:00'
        return row['date'] + ' ' + row['time']
    except:
        return None

# Apply the fix
df['datetime_str'] = df.apply(fix_datetime, axis=1)

# Check for any remaining issues
print("\nRows that couldn't be converted:")
print(df[df['datetime_str'].isna()][['date', 'time']].head())

# Now convert to datetime
df['datetime'] = pd.to_datetime(
    df['datetime_str'],
    format='%d/%m/%Y %H:%M',
    errors='coerce'
)

# Verify conversion
print("\nConversion results:")
print(f"Successfully converted: {df['datetime'].notna().sum()}")
print(f"Failed conversions: {df['datetime'].isna().sum()}")
print("\nSample converted datetimes:")
print(df[~df['datetime'].isna()][['date', 'time', 'datetime']].head(3))

# Drop rows that couldn't be converted
initial_count = len(df)
df = df[df['datetime'].notna()]
final_count = len(df)
print(f"\nRemoved {initial_count - final_count} rows with invalid datetime")
print(f"Remaining data: {final_count} rows")

# Set datetime index
df = df.set_index('datetime')
print("\nFinal datetime index:")
print(df.index[[0, 1, -2, -1]])  # Show first two and last two timestamps