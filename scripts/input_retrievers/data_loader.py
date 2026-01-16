import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

def load_stock_data(ticker, base_dir="F:/inputs/stocks", validate_dates=True):
    """
    Load stock data from CSV with date validation.
    
    Args:
        ticker: Stock ticker symbol
        base_dir: Directory containing stock CSV files
        validate_dates: If True, check for date parsing issues and warn about stale data
    
    Returns:
        DataFrame sorted by Date ascending
    """
    ticker_upper = ticker.upper()
    filepath = Path(base_dir) / f"{ticker_upper}.csv"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Stock file not found: {filepath}")
    
    # Read CSV with date parsing
    df = pd.read_csv(filepath, parse_dates=["Date"], dayfirst=False)
    
    if len(df) == 0:
        raise ValueError(f"Empty CSV file: {filepath}")
    
    # Validate Date column
    if "Date" not in df.columns:
        raise ValueError(f"No 'Date' column found in {filepath}")
    
    # Check for date parsing failures
    if df["Date"].isna().any():
        na_count = df["Date"].isna().sum()
        print(f"⚠️  WARNING: {na_count} rows in {ticker} have unparseable dates - removing them")
        df = df.dropna(subset=["Date"])
    
    # Sort by date to ensure proper ordering (most important!)
    df = df.sort_values("Date", ascending=True).reset_index(drop=True)
    
    # Validate date order and detect issues
    if validate_dates and len(df) > 0:
        earliest = df["Date"].min()
        latest = df["Date"].max()
        
        # Check if dates are actually datetime objects
        if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            print(f"⚠️  WARNING: Date column in {ticker} is not datetime type: {df['Date'].dtype}")
        
        # Check for future dates (suggests wrong parsing)
        today = pd.Timestamp.now().normalize()
        future_dates = df[df["Date"] > today]
        if len(future_dates) > 0:
            print(f"⚠️  WARNING: {ticker} has {len(future_dates)} dates in the future!")
            print(f"   First future date: {future_dates.iloc[0]['Date']}")
            print(f"   This suggests date format issues in the CSV!")
        
        # Check for reversed/unsorted dates in source
        if df["Date"].iloc[0] > df["Date"].iloc[-1]:
            print(f"ℹ️  NOTE: {ticker} CSV was in reverse chronological order (now corrected)")
    
    return df

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

def diagnose_csv_dates(ticker, base_dir="F:/inputs/stocks", show_sample=True):
    """
    Diagnostic tool to check what dates are actually in a CSV file.
    Useful for debugging date parsing issues.
    
    Args:
        ticker: Stock ticker symbol
        base_dir: Directory containing stock CSV files
        show_sample: If True, show first and last 5 date entries
    """
    ticker_upper = ticker.upper()
    filepath = Path(base_dir) / f"{ticker_upper}.csv"
    
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return
    
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC: {ticker} CSV Date Analysis")
    print(f"{'='*70}")
    print(f"File: {filepath}")
    
    # Read without date parsing to see raw values
    df_raw = pd.read_csv(filepath)
    print(f"\nTotal rows: {len(df_raw)}")
    print(f"Columns: {', '.join(df_raw.columns.tolist())}")
    
    if "Date" in df_raw.columns:
        print(f"\n📅 Raw Date column (first 5):")
        print(df_raw["Date"].head().to_string())
        print(f"\n📅 Raw Date column (last 5):")
        print(df_raw["Date"].tail().to_string())
        
        # Now try parsing
        df_parsed = pd.read_csv(filepath, parse_dates=["Date"])
        df_parsed = df_parsed.sort_values("Date", ascending=True)
        
        print(f"\n✅ Parsed & Sorted Dates:")
        print(f"   Earliest: {df_parsed['Date'].min()}")
        print(f"   Latest:   {df_parsed['Date'].max()}")
        print(f"   Date type: {df_parsed['Date'].dtype}")
        
        # Check age
        today = pd.Timestamp.now().normalize()
        latest = df_parsed['Date'].max()
        days_old = (today - latest).days
        print(f"   Age: {days_old} days old")
        
        if show_sample:
            print(f"\n📊 Last 10 dates in file:")
            print(df_parsed[["Date"]].tail(10).to_string())
    else:
        print("\n❌ No 'Date' column found!")
    
    print(f"{'='*70}\n")

# Example usage (commented out - import functions as needed):
# stock_df = load_stock_data("SPY")
# option_df = load_option_chain_data("spy")  # auto-discover most recent date
# most_recent_date = get_most_recent_option_date("spy")  # just print the date
# diagnose_csv_dates("SPY")  # Debug date issues
