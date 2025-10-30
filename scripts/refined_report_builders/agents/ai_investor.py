"" ! caution ! 

FOR EDUCATION PURPOSES ONLY, NOT REAL INVESTMENT ADVICE

Version 1.0

Standalone script

""

import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

def generate_option_investment_plan(portfolio_stats, risk_tolerance, market_outlook):
    """
    Generates an investment plan using option contracts.
    
    Parameters:
    - portfolio_stats (dict): Contains capital, current holdings, volatility, etc.
    - risk_tolerance (str): 'low', 'medium', or 'high'
    - market_outlook (str): 'bullish', 'bearish', or 'neutral'
    
    Returns:
    - dict: Recommended option strategies and rationale
    """

    # Step 1: Prepare prompt for OpenAI
    prompt = f"""
    You are a financial strategist. Based on the following portfolio stats:
    {portfolio_stats}
    
    Risk tolerance: {risk_tolerance}
    Market outlook: {market_outlook}
    
    Design a solid, airtight investment plan using option contracts. 
    Include:
    - Recommended strategies (e.g., covered calls, protective puts, spreads)
    - Strike price and expiration guidelines
    - How to anticipate and avoid market maker traps (e.g., IV crush, pinning)
    - Position sizing and capital allocation
    - Exit strategies and risk management
    - Keep it within 200 tokens for testing purposes.
    """

    
    # Step 2: Call OpenAI API
    response = client.responses.create(
        model="gpt-5",
        input=prompt
    )

    # Step 3: Parse and return the plan
    plan = response.output_text
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "plan": plan
    }

# Example usage
portfolio = {
    "capital": 100000,
    "holdings": ["AAPL", "TSLA"],
    "volatility": "moderate",
    "time_horizon": "6 months"
}

plan = generate_option_investment_plan(portfolio, "medium", "bullish")
print(plan["plan"])
