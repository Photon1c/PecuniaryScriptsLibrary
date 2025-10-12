# flight_path.py
import pandas as pd

def compute_altitude_series(prices: pd.Series) -> list:
    open_price = prices.iloc[0]
    return ((prices - open_price) / open_price * 100).round(2).tolist()
