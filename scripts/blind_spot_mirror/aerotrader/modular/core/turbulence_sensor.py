# turbulence_sensor.py
import pandas as pd


def detect_iv_turbulence(iv_series: pd.Series, candle_df: pd.DataFrame, wick_threshold=0.3) -> list:
    """
    Classifies turbulence for each step based on IV delta and candle shape.
    Returns a list of 'Calm', 'Moderate', or 'Heavy' for each row in candle_df.
    """
    turbulence_levels = []
    iv_std = iv_series.std()
    iv_mean = iv_series.mean()
    prev_iv = None
    for idx, row in candle_df.iterrows():
        current_iv = iv_series.iloc[idx]
        # Calculate IV delta
        if prev_iv is None:
            iv_delta = 0
        else:
            iv_delta = abs(current_iv - prev_iv)
        prev_iv = current_iv

        # Candle shape analysis
        open_ = row['Open']
        close = row['Close/Last']
        high = row['High']
        low = row['Low']
        body = abs(close - open_)
        range_ = high - low if high != low else 1e-9
        wick_top = high - max(open_, close)
        wick_bot = min(open_, close) - low
        body_ratio = body / range_
        wick_top_ratio = wick_top / range_
        wick_bot_ratio = wick_bot / range_

        # Turbulence logic
        if iv_delta > iv_std or (wick_top_ratio > wick_threshold and wick_bot_ratio > wick_threshold):
            turbulence = "Heavy"
        elif iv_delta > 0.5 * iv_std or (wick_top_ratio > wick_threshold or wick_bot_ratio > wick_threshold):
            turbulence = "Moderate"
        else:
            turbulence = "Calm"
        turbulence_levels.append(turbulence)
    return turbulence_levels
