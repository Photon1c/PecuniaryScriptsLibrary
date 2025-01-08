# Ripple Strategy For Option Chain Screening
# Apply critical rupture events' ripple effect theory to ranking list of stock
# tickers by expected projected returns.

import yfinance as yf
import pandas as pd
import numpy as np

def load_tickers(file_path):
    """Load tickers from a CSV file."""
    return pd.read_csv(file_path)["symbol"].tolist()

def fetch_stock_data(tickers, start_date, end_date):
    """Fetch historical stock data."""
    data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        data[ticker] = stock.history(start=start_date, end=end_date)
    return data

def calculate_weekly_returns(data):
    """Calculate weekly returns for each stock."""
    weekly_returns = {}
    for ticker, df in data.items():
        df['Weekly Return'] = df['Close'].pct_change(5)  # Weekly percentage change
        weekly_returns[ticker] = df['Weekly Return']
    return weekly_returns

def detect_ripples(data):
    """Detect precursor ripples based on daily percentage changes."""
    ripples = {}
    for ticker, df in data.items():
        df['Daily Change'] = df['Close'].pct_change()
        df['Ripple Score'] = abs(df['Daily Change'])  # Example metric for ripple strength
        ripples[ticker] = df['Ripple Score'].rolling(window=5).mean()  # 5-day moving average
    return ripples

def rank_stocks_by_ripples(ripples, weekly_returns):
    """Rank stocks based on their ripples and expected 1-week returns."""
    ranked_stocks = []
    for ticker in ripples.keys():
        avg_ripple = ripples[ticker].iloc[-1]  # Latest ripple score
        next_week_return = weekly_returns[ticker].iloc[-1]
        ranked_stocks.append((ticker, avg_ripple, next_week_return))
    ranked_stocks.sort(key=lambda x: (-x[1], -x[2]))  # Sort by ripple score, then by return
    return ranked_stocks

# Main execution
file_path = r"Path:\To\Tickers.csv"  # Path to CSV containing stock tickers
start_date = "2024-12-01" # Adjust as desired, automatic datetime function suggested
end_date = "2025-01-06"
tickers = load_tickers(file_path)
stock_data = fetch_stock_data(tickers, start_date, end_date)
weekly_returns = calculate_weekly_returns(stock_data)
ripples = detect_ripples(stock_data)
ranked_stocks = rank_stocks_by_ripples(ripples, weekly_returns)

# Display top-ranked stocks
ranked_df = pd.DataFrame(ranked_stocks, columns=["Ticker", "Ripple Score", "Expected Weekly Return"])
display(ranked_df)  # For Jupyter notebooks
ranked_df.to_csv(r"Path:\To\Output\Report.csv")
