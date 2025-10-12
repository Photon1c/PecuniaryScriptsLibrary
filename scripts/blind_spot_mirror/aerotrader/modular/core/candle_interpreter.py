# candle_interpreter.py
"""
Analyzes daily candles and labels them with flight behavior metaphors:
- 'Thrust': strong directional candle (full-body)
- 'Stall': large upper wick, rejection from altitude
- 'Go-around': large lower wick, aborted descent
- 'Hover': indecision or EMA drag zone
"""

import pandas as pd

def interpret_daily_candle(row, body_threshold=0.6, wick_threshold=0.3):
    open_ = row['Open']
    close = row['Close/Last']
    high = row['High']
    low = row['Low']

    body = abs(close - open_)
    range_ = high - low if high != low else 1e-9  # avoid zero division
    wick_top = high - max(open_, close)
    wick_bot = min(open_, close) - low

    # Proportions
    body_ratio = body / range_
    wick_top_ratio = wick_top / range_
    wick_bot_ratio = wick_bot / range_

    # Interpret flight behavior
    if body_ratio > body_threshold:
        return "Thrust"
    elif wick_top_ratio > wick_threshold:
        return "Stall"
    elif wick_bot_ratio > wick_threshold:
        return "Go-around"
    else:
        return "Hover"

def apply_interpretation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Flight Phase'] = df.apply(interpret_daily_candle, axis=1)
    return df[['Date', 'Close/Last', 'Flight Phase']]

