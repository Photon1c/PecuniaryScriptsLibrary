import pandas as pd
from pathlib import Path

def load_stock_data(ticker, base_dir="F:/inputs/stocks"):
    ticker_upper = ticker.upper()
    filepath = Path(base_dir) / f"{ticker_upper}.csv"
    if filepath.exists():
        return pd.read_csv(filepath, parse_dates=["Date"])
    else:
        raise FileNotFoundError(f"Stock file not found: {filepath}")

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
