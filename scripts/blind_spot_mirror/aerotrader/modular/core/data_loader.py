# data_loader.py
# Loads stock and option chain data from CSV files based on settings.json

import json
import os
import pandas as pd
from datetime import datetime

# Load settings
with open("settings.json", "r") as f:
    settings = json.load(f)

STOCK_PATH = settings.get("stock_data_path")
OPTION_PATH = settings.get("option_data_path")
TICKERS = settings.get("tickers", [])

if not STOCK_PATH or not OPTION_PATH:
    raise ValueError("Stock or Option data paths missing in settings.json. Please fix.")


def load_stock_data(ticker):
    ticker = ticker.upper()
    filepath = os.path.join(STOCK_PATH, f"{ticker}.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Stock CSV not found: {filepath}")
    return pd.read_csv(filepath)


def load_option_data(ticker, max_files=6):
    ticker_lower = ticker.lower()
    base_dir = os.path.join(OPTION_PATH, ticker_lower)

    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Options directory not found: {base_dir}")

    # List all date folders
    date_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    # Parse dates and sort
    def parse_date(d):
        try:
            return datetime.strptime(d, "%m_%d_%Y")
        except ValueError:
            return None

    date_folders = [(d, parse_date(d)) for d in date_folders]
    date_folders = [d for d in date_folders if d[1] is not None]
    date_folders.sort(key=lambda x: x[1], reverse=True)

    selected_folders = date_folders[:max_files]

    dataframes = []
    for folder, _ in selected_folders:
        filepath = os.path.join(base_dir, folder, f"{ticker_lower}_quotedata.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, skiprows=3)
            dataframes.append(df)
        else:
            print(f"Warning: Missing quotedata file in {folder}")

    if not dataframes:
        raise FileNotFoundError(f"No quotedata CSVs found for {ticker}")

    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df


def load_all_stock_data():
    data = {}
    for ticker in TICKERS:
        try:
            data[ticker] = load_stock_data(ticker)
        except FileNotFoundError as e:
            print(e)
    return data


def load_all_option_data():
    data = {}
    for ticker in TICKERS:
        try:
            data[ticker] = load_option_data(ticker)
        except FileNotFoundError as e:
            print(e)
    return data


if __name__ == "__main__":
    stocks = load_all_stock_data()
    options = load_all_option_data()
    print("Stocks loaded:", list(stocks.keys()))
    print("Options loaded:", list(options.keys()))
