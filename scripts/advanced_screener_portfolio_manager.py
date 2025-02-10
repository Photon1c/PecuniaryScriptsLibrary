# Portfolio Sorter
import pandas as pd

# Load the CSV file
file_path = r'https://owlspruce.cheddarbutler.com/datavault/holdings/option_holdings.csv'
data = pd.read_csv(file_path)

# Convert relevant columns to numeric, coercing errors to NaN
data['Strike'] = pd.to_numeric(data['Strike'], errors='coerce')
data['Stock Price'] = pd.to_numeric(data['Stock Price'].str.replace('$', '', regex=False), errors='coerce')
data['Cost Basis'] = pd.to_numeric(data['Cost Basis'], errors='coerce')
data['Market Value'] = pd.to_numeric(data['Market Value'], errors='coerce')
data['dtE'] = pd.to_numeric(data['dtE'], errors='coerce')

# Initialize categories
gains_keep = []
gains_close = []
losses_keep = []
losses_close = []

# Iterate through each row in the DataFrame
for index, row in data.iterrows():
    symbol = row['symbol']
    option_type = row['Type']
    strike_price = row['Strike']
    current_price = row['Stock Price']
    cost_basis = row['Cost Basis']
    market_value = row['Market Value']
    days_to_expiration = row['dtE']

    # Skip rows with missing or invalid data
    if pd.isna(strike_price) or pd.isna(current_price) or pd.isna(cost_basis) or pd.isna(market_value) or pd.isna(days_to_expiration):
        continue

    # Determine if the option is deep in the money (ITM)
    if option_type == 'call':
        itm = current_price > strike_price  # Call is ITM if current price > strike price
    elif option_type == 'put':
        itm = current_price < strike_price  # Put is ITM if current price < strike price
    else:
        itm = False  # Invalid option type

    # Categorize the ticker
    if market_value >= 1.5 * cost_basis:  # Gains_close condition
        gains_close.append(symbol)
    elif itm and days_to_expiration > 30:  # Gains_keep condition
        gains_keep.append(symbol)
    elif market_value < cost_basis and days_to_expiration > 0:  # Losses_keep condition
        losses_keep.append(symbol)
    elif days_to_expiration == 0:  # Losses_close condition
        losses_close.append(symbol)

# Print the results
print("Gains_keep:", gains_keep)
print("Gains_close:", gains_close)
print("Losses_keep:", losses_keep)
print("Losses_close:", losses_close)

# Optionally, save the results to a new CSV file
results = {
    'Gains_keep': gains_keep,
    'Gains_close': gains_close,
    'Losses_keep': losses_keep,
    'Losses_close': losses_close
}
results_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results.items()]))
results_df.to_csv('categorized_portfolio.csv', index=False)
