# stall_detector.py
import pandas as pd


def detect_stalls(prices: pd.Series, candle_df: pd.DataFrame, iv_series: pd.Series, ema_span=20, threshold=0.5, wick_threshold=0.3) -> list:
    """
    Detects stall risk using EMA drag, candle shape, and IV delta.
    Returns a list of booleans (True if stall risk, else False) for each step.
    """
    ema = prices.ewm(span=ema_span).mean()
    iv_std = iv_series.std()
    prev_iv = None
    stalls = []
    for idx, row in candle_df.iterrows():
        price = prices.iloc[idx]
        current_iv = iv_series.iloc[idx] if idx < len(iv_series) else iv_series.iloc[-1]
        # EMA drag
        ema_drag = abs(price - ema.iloc[idx]) < threshold
        # IV delta
        if prev_iv is None:
            iv_delta = 0
        else:
            iv_delta = abs(current_iv - prev_iv)
        prev_iv = current_iv
        iv_spike = iv_delta > iv_std
        # Candle shape
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
        doji = body_ratio < 0.2 and (wick_top_ratio > wick_threshold and wick_bot_ratio > wick_threshold)
        upper_wick_stall = wick_top_ratio > wick_threshold and close < open_  # upper wick after uptrend
        # Stall risk if any condition is met
        stall = ema_drag or iv_spike or doji or upper_wick_stall
        stalls.append(stall)
    return stalls
