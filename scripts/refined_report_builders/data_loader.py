import pandas as pd
from pathlib import Path

def find_latest_stock_file(ticker, base_dir="F:/inputs/stocks"):
    """
    Finds the latest stock CSV for a ticker based on max(Date) within the files.
    Searches for files matching ticker*.csv (case-insensitive).
    """
    base_path = Path(base_dir)
    ticker_upper = ticker.upper()
    
    # Search for candidate files: ticker*.csv
    # We use glob with case-insensitivity if possible, or just filter ourselves
    candidates = [f for f in base_path.glob("*.csv") if f.name.upper().startswith(ticker_upper)]
    
    if not candidates:
        return None
        
    latest_file = None
    max_date = None
    
    print(f"DEBUG: Searching for latest {ticker_upper} file in {base_dir}...")
    
    for f in candidates:
        try:
            # Read only Date column to speed up check
            # We don't know the exact column name yet, usually "Date"
            temp_df = pd.read_csv(f, nrows=100) # Check first 100 rows
            date_col = next((c for c in temp_df.columns if "Date" in c), None)
            
            if date_col:
                # Read all dates from this file
                df_dates = pd.read_csv(f, usecols=[date_col])
                df_dates[date_col] = pd.to_datetime(df_dates[date_col], errors='coerce')
                current_max = df_dates[date_col].max()
                
                print(f"DEBUG: Found {f.name} with max date {current_max}")
                
                if max_date is None or (current_max is not None and current_max > max_date):
                    max_date = current_max
                    latest_file = f
            else:
                print(f"DEBUG: No 'Date' column in {f.name}, skipping for date-based check.")
        except Exception as e:
            print(f"DEBUG: Error reading {f.name}: {e}")
            continue
            
    if latest_file:
        print(f"DEBUG: Chose {latest_file.name} as latest file for {ticker_upper} (max date: {max_date})")
        return latest_file
    
    # Fallback to mtime if date parsing failed for all
    latest_file = max(candidates, key=lambda f: f.stat().st_mtime)
    print(f"DEBUG: Fallback to mtime for {ticker_upper}: {latest_file.name}")
    return latest_file

def load_stock_data(ticker, base_dir="F:/inputs/stocks"):
    ticker_upper = ticker.upper()
    filepath = find_latest_stock_file(ticker, base_dir)
    
    if filepath and filepath.exists():
        df = pd.read_csv(filepath)
        
        # Robust Date parsing
        date_col = next((c for c in df.columns if "Date" in c), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])
            df = df.sort_values(date_col, ascending=True)
            if date_col != "Date":
                df = df.rename(columns={date_col: "Date"})
        
        # Robust Close/Last parsing (remove $ and ,)
        for col in ["Close/Last", "Close"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df
    else:
        raise FileNotFoundError(f"Stock file not found for ticker: {ticker_upper} in {base_dir}")

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

if __name__ == "__main__":
    # Example usage:
    stock_df = load_stock_data("SPY")
    option_df = load_option_chain_data("spy")  # auto-discover most recent date
    most_recent_date = get_most_recent_option_date("spy")  # just print the date
