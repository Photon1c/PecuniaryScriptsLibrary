#EMA Analyzer Silver Hawk
#Generate table analyzing list of stocks and note their EMA divergence.
import yfinance as yf
import pandas as pd
from tabulate import tabulate

# List of stock tickers
#tickers = ['AAPL', 'MSFT', 'GOOGL']  # Add your desired tickers here
tickers_raw = pd.read_csv(r"F:\inputs\masterlists\holdings.csv")
tickers = tickers_raw["symbol"].to_list()
# Create a list to store results as dictionaries
results_list = []

# Fetch data for all tickers
data = yf.download(tickers, period='30d', group_by='ticker', progress=False)

# Debug: Print raw columns
#print(f"Raw columns: {data.columns}")

# Process each ticker
# Process each ticker
for ticker in tickers:
    print(f"Processing ticker: {ticker}")

    # Extract data for the specific ticker and make a copy to avoid SettingWithCopyWarning
    if isinstance(data.columns, pd.MultiIndex):
        ticker_data = data[ticker].copy()

    # Debug: Print processed columns
    #print(f"Processed columns for {ticker}: {ticker_data.columns}")

    if not ticker_data.empty and 'Close' in ticker_data.columns:
        # Calculate the 7-day EMA
        ticker_data['7 Day EMA'] = ticker_data['Close'].ewm(span=7, adjust=False).mean()

        # Validate data
        if ticker_data['Close'].notna().any() and ticker_data['7 Day EMA'].notna().any():
            spot_price = ticker_data['Close'].iloc[-1]
            ema = ticker_data['7 Day EMA'].iloc[-1]
            divergence = spot_price - ema

            # Append to results
            results_list.append({
                'Ticker': ticker,
                'Spot Price': spot_price,
                '7 Day EMA': ema,
                'Divergence': divergence
            })
        else:
            print(f"Data validation failed for {ticker}: 'Close' or '7 Day EMA' is missing or all NaN.")
    else:
        print(f"No valid data retrieved for {ticker}.")
    print("-" * 50)


# Convert results to DataFrame
results = pd.DataFrame(results_list)

# Display results if available
if not results.empty:
    print(tabulate(results, headers='keys', tablefmt='grid'))
else:
    print("No valid data available for the selected tickers.")
