import pandas as pd
from pathlib import Path

def load_stock_data(ticker, base_dir="F:/inputs/stocks"):
    ticker_upper = ticker.upper()
    filepath = Path(base_dir) / f"{ticker_upper}.csv"
    if filepath.exists():
        df = pd.read_csv(filepath, parse_dates=["Date"])
        # Sort by date to ensure proper ordering
        if "Date" in df.columns:
            df = df.sort_values("Date", ascending=True)
        return df
    else:
        raise FileNotFoundError(f"Stock file not found: {filepath}")

def get_latest_price(ticker, base_dir="F:/inputs/stocks"):
    """
    Get the most recent closing price for a ticker.
    Returns the price from the row with the latest date.
    """
    df = load_stock_data(ticker, base_dir=base_dir)
    
    # Ensure we have data
    if len(df) == 0:
        raise ValueError(f"No data found for {ticker}")
    
    # Sort by date to get most recent (should already be sorted by load_stock_data, but be safe)
    if "Date" in df.columns:
        df = df.sort_values("Date", ascending=True)
    
    # Get the last (most recent) row
    latest_row = df.iloc[-1]
    
    # Extract price from appropriate column
    if "Close/Last" in df.columns:
        price_raw = latest_row["Close/Last"]
        price = float(str(price_raw).replace("$", "").replace(",", "").strip())
    elif "Close" in df.columns:
        price_raw = latest_row["Close"]
        price = float(str(price_raw).replace("$", "").replace(",", "").strip()) if isinstance(price_raw, str) else float(price_raw)
    else:
        raise ValueError(f"No price column found in {ticker} data. Available: {df.columns.tolist()}")
    
    # Get the date for logging
    if "Date" in df.columns:
        latest_date = latest_row["Date"]
        print(f"Latest {ticker} price: ${price:.2f} (as of {latest_date})")
    else:
        print(f"Latest {ticker} price: ${price:.2f}")
    
    return price

def get_most_recent_option_date(ticker, base_dir="F:/inputs/options/log"):
    ticker_lower = ticker.lower()
    ticker_dir = Path(base_dir) / ticker_lower

    date_dirs = [d for d in ticker_dir.iterdir() if d.is_dir()]
    if not date_dirs:
        raise FileNotFoundError(f"No date directories found for {ticker_lower}.")

    most_recent_dir = max(date_dirs, key=lambda d: d.stat().st_mtime)
    most_recent_date = most_recent_dir.name
    print(f"Most recent date for {ticker_lower}: {most_recent_date}")
    return most_recent_date

def load_option_chain_data(ticker, date=None, base_dir="F:/inputs/options/log"):
    ticker_lower = ticker.lower()
    ticker_dir = Path(base_dir) / ticker_lower

    if date:
        date_dir = ticker_dir / date
    else:
        # Auto-discover the most recent date directory
        date_dirs = [d for d in ticker_dir.iterdir() if d.is_dir()]
        if not date_dirs:
            raise FileNotFoundError(f"No date directories found for {ticker_lower}.")
        date_dir = max(date_dirs, key=lambda d: d.stat().st_mtime)  # most recently modified

    filepath = date_dir / f"{ticker_lower}_quotedata.csv"
    if filepath.exists():
        return pd.read_csv(filepath, skiprows=3)
    else:
        raise FileNotFoundError(f"Option chain file not found: {filepath}")
        
def load_option_data(ticker, date=None, base_dir="F:/inputs/options/log"):
    ticker_lower = ticker.lower()
    ticker_dir = Path(base_dir) / ticker_lower

    if date:
        date_dir = ticker_dir / date
    else:
        # Auto-discover the most recent date directory
        date_dirs = [d for d in ticker_dir.iterdir() if d.is_dir()]
        if not date_dirs:
            raise FileNotFoundError(f"No date directories found for {ticker_lower}.")
        date_dir = max(date_dirs, key=lambda d: d.stat().st_mtime)  # most recently modified

    filepath = date_dir / f"{ticker_lower}_quotedata.csv"
    if filepath.exists():
        return pd.read_csv(filepath, skiprows=3)
    else:
        raise FileNotFoundError(f"Option chain file not found: {filepath}")

# Example usage:
stock_df = load_stock_data("SPY")
option_df = load_option_chain_data("spy")  # auto-discover most recent date
most_recent_date = get_most_recent_option_date("spy")  # just print the date
