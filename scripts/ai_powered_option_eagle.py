# Using a user-fed description, use GPT-4o to generate option payoff diagram and heat map.
# Import Modules
from openai import OpenAI
import json
import numpy as np
import matplotlib.pyplot as plt
import mplcyberpunk  # Added cyberpunk styling
from dotenv import load_dotenv
import os
import torch

# Load API key stored in .env file
load_dotenv()

client = OpenAI()

import re  # Added for removing dollar sign

def calculate_payoff(strategy, stock_prices):
    """
    Calculate the total payoff for a strategy across a range of stock prices.
    """
    payoff = np.zeros_like(stock_prices)

    for opt in strategy:
        strike = opt["strike"]
        quantity = opt["quantity"]
        premium = opt["premium"]
        opt_type = opt["type"].lower()

        if opt_type == "call":
            payoff += quantity * (np.maximum(stock_prices - strike, 0) - premium)
        elif opt_type == "put":
            payoff += quantity * (np.maximum(strike - stock_prices, 0) - premium)

    return payoff


def parse_option_strategy(user_input):
    # Remove dollar signs from input
    cleaned_input = re.sub(r'\$', '', user_input)

    prompt = f"""You are an expert in options trading. A user describes an option strategy. Return the structured data in JSON format with:
    - 'ticker' (the ticker symbol mentioned by the user)
    - 'type' (call/put)
    - 'strike' (strike price)
    - 'quantity' (positive for long, negative for short)
    - 'premium' (price paid or received for each option)
    
    Example:
    User: "I bought 2 AAPL calls at a $100 strike for $5 and sold 1 PG call at $110 strike for $3"
    Response: 
    [
        {{"ticker": "AAPL", "type": "call", "strike": 100, "quantity": 2, "premium": 5}},
        {{"ticker": "PG", "type": "call", "strike": 110, "quantity": -1, "premium": 3}}
    ]
    
    User: "{cleaned_input}"
    Response:
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert options strategy parser."},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract response text
    response_text = response.choices[0].message.content.strip()

    # Debugging: Print raw response
    print("Response Text:", response_text)

    try:
        # Remove backticks and extract JSON content
        if response_text.startswith("```json") and response_text.endswith("```"):
            response_text = response_text[7:-3].strip()  # Strip markdown formatting

        # Parse the response as JSON
        strategy = json.loads(response_text)
        return strategy
    except json.JSONDecodeError as e:
        print("Failed to decode JSON:", e)
        return []  # Return an empty list to handle errors gracefully

# Payoff calculation
def plot_payoff(strategy):
    if not strategy:
        print("No valid strategy to plot.")
        return

    # Extract ticker from the first option in the strategy (assuming all have the same ticker)
    ticker = strategy[0].get("ticker", "Unknown")  # Use "Unknown" if no ticker is provided

    # Extract strike prices
    strikes = [option["strike"] for option in strategy]  

    # Define stock price range dynamically (80%-120% of median strike)
    atm_strike = np.median(strikes)
    low, high = atm_strike * 0.8, atm_strike * 1.2
    stock_prices = np.linspace(low, high, 500)

    payoff = calculate_payoff(strategy, stock_prices)

    # Apply cyberpunk style
    plt.style.use("cyberpunk")

    plt.figure(figsize=(10, 6))
    plt.plot(stock_prices, payoff, label="Payoff", color='cyan', linewidth=2)

    # Ensure X-axis focuses only on plotted data
    plt.xlim(min(stock_prices), max(stock_prices))

    # Add grid, zero lines
    plt.axhline(0, color='white', linestyle='--', linewidth=0.8)
    plt.axvline(atm_strike, color='white', linestyle='--', linewidth=0.8)

    # Labels and title
    # Extract all strikes and premiums for display
    strike_premium_info = ", ".join(
        [f"Strike: {opt['strike']}, Premium: {opt['premium']}" for opt in strategy]
    )
    
    plt.title(f"Option Strategy Payoff Diagram for {ticker.upper()}\n{strike_premium_info}",
              fontsize=14, fontweight='bold')

    plt.xlabel("Stock Price at Expiration", fontsize=12)
    plt.ylabel("Net Payoff ($)", fontsize=12)
    plt.legend()

    # Apply cyberpunk glow effect
    mplcyberpunk.add_glow_effects()
    plt.savefig("test.png")
    plt.show()


def plot_payoff_heatmap(strategy):
    if not strategy:
        print("No strategy provided.")
        return

    strikes = [opt["strike"] for opt in strategy]
    atm_strike = np.median(strikes)

    # Torch range setup
    stock_prices = torch.linspace(atm_strike * 0.8, atm_strike * 1.2, 100)
    premium_adjust = torch.linspace(-2.0, 2.0, 100)  # +/- $2 change in premium

    # Create 2D grid
    S, P = torch.meshgrid(stock_prices, premium_adjust, indexing="ij")
    payoff_surface = torch.zeros_like(S)

    for opt in strategy:
        strike = opt["strike"]
        quantity = opt["quantity"]
        premium = opt["premium"]
        opt_type = opt["type"].lower()

        if opt_type == "call":
            intrinsic = torch.maximum(S - strike, torch.tensor(0.0))
        elif opt_type == "put":
            intrinsic = torch.maximum(strike - S, torch.tensor(0.0))
        else:
            continue

        payoff_surface += quantity * (intrinsic - (premium + P))

    # Plot with matplotlib
    plt.figure(figsize=(12, 8))
    cp = plt.contourf(S.numpy(), P.numpy(), payoff_surface.numpy(), levels=50, cmap='coolwarm')
    plt.colorbar(cp, label="Net Payoff ($)")
    plt.title("PnL Heatmap: Stock Price vs Premium Adjustment")
    plt.xlabel("Stock Price at Expiration")
    plt.ylabel("Premium Adjustment ($)")
    plt.axhline(0, color='white', linestyle='--', linewidth=1)
    plt.axvline(atm_strike, color='white', linestyle='--', linewidth=1)
    plt.show()


# Test the code
stock_prices = np.linspace(50, 150, 500)  # Example range of stock prices
user_input = "I bought 1 PLTR put at a 92 strike for 0.14"
strategy = parse_option_strategy(user_input)
print("Parsed strategy:", strategy)  # Debugging step to check parsed strategy
plot_payoff(strategy)
plot_payoff_heatmap(strategy)
