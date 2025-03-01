# Compute Sharpe Ratio from CSV file from Nasdaq.com
import pandas as pd
import numpy as np

# Load the CSV file (replace 'stock_data.csv' with your actual filename)
file_path = stock_data.csv"
stock = pd.read_csv(file_path, parse_dates=['Date'])

# Ensure the data is sorted by date
stock = stock.sort_values('Date')

# Use 'Adj Close' if available; otherwise, use 'Close/Last'
price_col = 'Adj Close' if 'Adj Close' in stock.columns else 'Close/Last'

# Calculate daily returns
stock['Daily Return'] = stock[price_col].pct_change()

# Define risk-free rate (e.g., 1% annualized -> daily rate)
risk_free_daily = 0.01 / 252

# Calculate excess returns
stock['Excess Return'] = stock['Daily Return'] - risk_free_daily

# Compute Sharpe Ratio components
mean_excess = stock['Excess Return'].mean()
std_excess = stock['Excess Return'].std()

# Annualize the Sharpe Ratio
sharpe_ratio = (mean_excess / std_excess) * np.sqrt(252)

print("Annualized Sharpe Ratio for {ticker}:", sharpe_ratio)
