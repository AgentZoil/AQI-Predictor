# Wollongong Air Quality Data Processor

A Python script that processes air quality monitoring data from Wollongong, cleaning and transforming the data for analysis.

## Overview

This program loads environmental monitoring data from an Excel file containing air quality measurements from Wollongong. It performs multiple data cleaning and transformation steps to prepare the data for analysis, including:

- Handling metadata rows
- Cleaning column names
- Converting data types
- Detecting and removing invalid values (negative PM2.5 readings)
- Fixing time format issues (e.g., 24:00 timestamps)
- Creating a proper datetime index

## Features

- **Data Cleaning**: Removes metadata and converts columns to appropriate data types
- **Data Validation**: Detects and handles physically impossible values (negative PM2.5 readings)
- **Time Normalization**: Fixes time format issues (24:00 → 00:00 the next day)
- **Basic Analysis**: Provides descriptive statistics of the cleaned dataset

## Requirements

- Python 3.x
- pandas
- openpyxl (for Excel file handling)

## Installation

1. Clone this repository or download the script
2. Create a virtual environment:
   ```
   python -m venv .venv
   ```
3. Activate the virtual environment:
   - On macOS/Linux:
     ```
     source .venv/bin/activate
     ```
   - On Windows:
     ```
     .venv\Scripts\activate
     ```
4. Install the required dependencies:
   ```
   pip install pandas openpyxl
   ```

## Usage

1. Place your `Wollongong.xls` file in the same directory as the script
2. Run the script:
   ```
   python main.py
   ```

## Input Data Format

The program expects an Excel file with the following characteristics:
- First row contains metadata (skipped during import)
- Second row contains headers
- Data columns include date, time, temperature, wind speed, PM2.5 measurements, and humidity

## Code Explanation

### Data Loading and Initial Cleaning

```python
# Load the data skipping first row (metadata) and use second row as header
df = pd.read_excel("Wollongong.xls", skiprows=1, header=0)
    
# Clean column names
df.columns = ['date', 'time', 'temp_c', 'wind_speed_ms', 'pm25_ugm3', 'humidity_pct']
```

This section loads the Excel file while skipping the metadata row, then renames columns to standardized names.

### Data Type Conversion and Validation

```python
# Convert numeric columns
numeric_cols = ['temp_c', 'wind_speed_ms', 'pm25_ugm3', 'humidity_pct']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
```

Converts appropriate columns to numeric data types, using `coerce` to handle non-numeric values.

### Negative PM2.5 Value Detection and Removal

```python
# Check for negative PM2.5 values (physically impossible)
negative_pm25 = df[df['pm25_ugm3'] < 0]

# Remove negative PM2.5 values
df = df[df['pm25_ugm3'] >= 0]
```

Identifies and removes rows with physically impossible negative PM2.5 values.

### Datetime Processing

```python
def fix_datetime(row):
    if row['time'] == '24:00':
        # Convert to next day at 00:00
        new_date = pd.to_datetime(row['date'], format='%d/%m/%Y') + pd.Timedelta(days=1)
        return new_date.strftime('%d/%m/%Y') + ' ' + '00:00'
    return row['date'] + ' ' + row['time']

# Apply the fix
df['datetime_str'] = df.apply(fix_datetime, axis=1)

# Convert to datetime
df['datetime'] = pd.to_datetime(
    df['datetime_str'],
    format='%d/%m/%Y %H:%M',
    errors='coerce'
)
```

Handles time format issues, particularly the "24:00" time notation by converting it to "00:00" of the next day, then creates a proper datetime column.

### Setting the Datetime Index

```python
# Set datetime index
df = df.set_index('datetime')
```

Creates a datetime index for easier time series analysis.

## Output

The program provides:
1. A preview of the cleaned data
2. Data types of each column
3. Basic descriptive statistics
4. Counts of problematic values (negative PM2.5)
5. Confirmation of datetime processing
6. Final dataset size and datetime index

## Notes

The program detected an OLE2 inconsistency warning during Excel file reading, but this doesn't affect the data processing.

## License

[Your License Here]