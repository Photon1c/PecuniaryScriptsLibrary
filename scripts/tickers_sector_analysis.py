# Sector/Industry Best and Worst Performer Analysis
# Return Analysis from input CSV file
import yfinance as yf
import pandas as pd

def get_stock_performance():
    # Load tickers and their sectors
    tickers_file = 'ticker_and_industry.csv'  # Ensure this file has columns: symbol, industry
    try:
        tickers_df = pd.read_csv(tickers_file)
    except FileNotFoundError:
        print("Error: 'ticker_and_sectors.csv' not found.")
        return

    # Ensure the column names match
    if 'symbol' not in tickers_df.columns or 'industry' not in tickers_df.columns:
        print("Error: 'tickers_and_sectors.csv' must have 'symbol' and 'Sector' columns.")
        return

    tickers = tickers_df['symbol'].tolist()

    # Retrieve today's stock data
    try:
        data = yf.download(tickers, period="1d", interval="1d", progress=False)
    except Exception as e:
        print(f"Error retrieving data: {e}")
        return

    # Check if data is retrieved correctly
    if data.empty or 'Close' not in data.columns or 'Open' not in data.columns:
        print("No valid data retrieved. Ensure the tickers are correct and try again.")
        return

    # Calculate percentage change for the day
    try:
        performance = (data['Close'] - data['Open']) / data['Open'] * 100
        performance = performance.reset_index().melt(id_vars="Date", var_name="symbol", value_name="Change (%)")
    except KeyError as e:
        print(f"Error calculating performance: {e}")
        return

    # Merge with sectors
    try:
        performance_with_sectors = pd.merge(performance, tickers_df, on="symbol", how="left")
    except Exception as e:
        print(f"Error merging data: {e}")
        return

    # Get top and worst performers
    try:
        top_performers = performance_with_sectors.nlargest(5, 'Change (%)')
        worst_performers = performance_with_sectors.nsmallest(5, 'Change (%)')
    except KeyError as e:
        print(f"Error finding top/worst performers: {e}")
        return

    # Combine results for display
    combined = pd.concat([top_performers, worst_performers])

    # Display results
    print("\nTop and Worst Performing Stocks:")
    print(combined[['symbol', 'industry', 'Change (%)']])

    # Save results to CSV
    try:
        combined.to_csv('top_and_worst_performers.csv', index=False)
        print("\nResults saved to 'top_and_worst_performers.csv'.")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    get_stock_performance()
