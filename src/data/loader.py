import pandas as pd
from pathlib import Path

class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.numeric_cols = ['temp_c', 'wind_speed_ms', 'pm25_ugm3', 'humidity_pct']
    
    def load_and_clean(self) -> pd.DataFrame:
        """Steps 1-3 from your original code"""
        try:
            df = pd.read_excel(self.data_path, skiprows=1, header=0)
            df.columns = ['date', 'time', 'temp_c', 'wind_speed_ms', 'pm25_ugm3', 'humidity_pct']
            df = df[df['date'] != "Date"]
            df[self.numeric_cols] = df[self.numeric_cols].apply(pd.to_numeric, errors='coerce')
            df = self._handle_pm25(df)
            df = self._fix_datetime(df)
            return df
        except Exception as e:
            raise ValueError(f"Error loading file: {e}")

    def _handle_pm25(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 2: Handle negative PM2.5 values"""
        negative_pm25 = df[df['pm25_ugm3'] < 0]
        print(f"Found {len(negative_pm25)} rows with negative PM2.5 values")
        return df[df['pm25_ugm3'] >= 0]

    def _fix_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 3: Fix datetime issues"""
        df['datetime_str'] = df.apply(self._fix_datetime_row, axis=1)
        df['datetime'] = pd.to_datetime(df['datetime_str'], format='%d/%m/%Y %H:%M', errors='coerce')
        df = df[df['datetime'].notna()]
        return df.set_index('datetime').drop(columns=['date', 'time', 'datetime_str'])

    def _fix_datetime_row(self, row):
        """Helper for datetime conversion"""
        try:
            if row['time'] == '24:00':
                new_date = pd.to_datetime(row['date'], format='%d/%m/%Y') + pd.Timedelta(days=1)
                return new_date.strftime('%d/%m/%Y') + ' 00:00'
            return row['date'] + ' ' + row['time']
        except:
            return None