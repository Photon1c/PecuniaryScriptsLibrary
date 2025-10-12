# fuel_gauge.py
import pandas as pd

def compute_fuel_levels(volumes: pd.Series) -> list:
    total = volumes.sum()
    fuel = ((total - volumes.cumsum()) / total * 100).round(1).tolist()
    return fuel

# intraday_fuel_model.py
def generate_intraday_fuel_curve(steps=5) -> list:
    return [round(100 * (1 - i / (steps - 1)), 1) for i in range(steps)]
