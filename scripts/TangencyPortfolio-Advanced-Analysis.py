#Generate comprehensive report with advanced metrics and tangency portfolio assessment
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from datetime import datetime, timedelta

# Set aesthetic parameters for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

def analyze_portfolio(tickers, period='2mo', risk_free_rate=0.0, n_portfolios=10000, save_report=True):
    """
    Performs comprehensive portfolio analysis including efficient frontier calculation
    and visualization with improved reporting.
    
    Parameters:
    -----------
    tickers : list
        List of stock ticker symbols
    period : str
        Time period for historical data (e.g. '2mo', '1y')
    risk_free_rate : float
        Annual risk-free rate as decimal
    n_portfolios : int
        Number of random portfolios to generate
    save_report : bool
        Whether to save the report to files
    
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    print(f"⏳ Starting portfolio analysis for {len(tickers)} assets: {', '.join(tickers)}")
    print(f"📅 Analysis period: {period}")
    
    # Calculate end date and start date for better labeling
    end_date = datetime.now()
    if period.endswith('mo'):
        months = int(period[:-2])
        start_date = end_date - timedelta(days=30*months)
    elif period.endswith('y'):
        years = int(period[:-1])
        start_date = end_date - timedelta(days=365*years)
    else:
        start_date = "historical start"
    
    date_range = f"{start_date.strftime('%Y-%m-%d') if isinstance(start_date, datetime) else start_date} to {end_date.strftime('%Y-%m-%d')}"
    print(f"📈 Downloading historical data from {date_range}")
    
    # Download the historical data for the tickers with progress indicator
    prices = yf.download(tickers, period=period, progress=False)['Adj Close']
    
    # Check for missing data
    missing_data = prices.isna().sum()
    if missing_data.sum() > 0:
        print("⚠️ Warning: Missing data detected")
        print(missing_data[missing_data > 0])
        # Forward fill missing values
        prices = prices.fillna(method='ffill').fillna(method='bfill')
    
    # Calculate daily and cumulative returns
    daily_returns = prices.pct_change().dropna()
    cumulative_returns = (1 + daily_returns).cumprod() - 1
    
    # Calculate the annualized statistics
    trading_days_per_year = 252
    annualized_returns = daily_returns.mean() * trading_days_per_year
    annualized_volatility = daily_returns.std() * np.sqrt(trading_days_per_year)
    
    # Display asset performance summary
    performance_summary = pd.DataFrame({
        'Annualized Return': annualized_returns * 100,
        'Annualized Volatility': annualized_volatility * 100,
        'Sharpe Ratio': (annualized_returns - risk_free_rate) / annualized_volatility,
        'Cumulative Return': cumulative_returns.iloc[-1] * 100,
        'Max Drawdown': (prices / prices.cummax() - 1).min() * 100
    })
    
    print("\n📊 Asset Performance Summary:")
    print(performance_summary.round(2))
    
    # Calculate correlation matrix
    correlation = daily_returns.corr()
    
    # Calculate the mean and covariance of returns
    mu = daily_returns.mean().values * trading_days_per_year
    cov = daily_returns.cov() * trading_days_per_year
    
    print("\n🔄 Generating random portfolios and calculating efficient frontier...")
    
    # Generate random portfolio weights
    weights = np.random.random(size=(n_portfolios, len(tickers)))
    weights /= np.sum(weights, axis=1, keepdims=True)
    
    # Calculate portfolio metrics
    portfolio_annual_returns = np.sum(mu * weights, axis=1)
    portfolio_annual_volatility = np.sqrt(np.sum(np.dot(weights, cov) * weights, axis=1))
    sharpe_ratios = (portfolio_annual_returns - risk_free_rate) / portfolio_annual_volatility
    
    # Find the tangency (max Sharpe ratio) portfolio
    tangency_idx = np.argmax(sharpe_ratios)
    tangency_weights = weights[tangency_idx, :]
    tangency_return = portfolio_annual_returns[tangency_idx]
    tangency_volatility = portfolio_annual_volatility[tangency_idx]
    tangency_sharpe = sharpe_ratios[tangency_idx]
    
    # Find minimum volatility portfolio
    min_vol_idx = np.argmin(portfolio_annual_volatility)
    min_vol_weights = weights[min_vol_idx, :]
    min_vol_return = portfolio_annual_returns[min_vol_idx]
    min_vol_volatility = portfolio_annual_volatility[min_vol_idx]
    min_vol_sharpe = sharpe_ratios[min_vol_idx]
    
    # Calculate the efficient frontier
    target_returns = np.linspace(min_vol_return, mu.max() * 1.1, 100)
    efficient_volatilities = []
    efficient_weights = []
    
    print("\n📉 Optimizing efficient frontier...")
    
    for target_return in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'eq', 'fun': lambda w: np.sum(mu * w) - target_return}
        )
        bounds = tuple((0.0, 1.0) for _ in range(len(tickers)))
        
        result = minimize(
            lambda w: np.sqrt(np.dot(w.T, np.dot(cov, w))),
            np.ones(len(tickers)) / len(tickers),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            efficient_volatilities.append(np.sqrt(np.dot(result.x.T, np.dot(cov, result.x))))
            efficient_weights.append(result.x)
        else:
            print(f"⚠️ Optimization failed for target return {target_return:.4f}")
    
    # Print the results for key portfolios
    print("\n🌟 Optimal Portfolio Allocations:")
    print("\nTangency Portfolio (Maximum Sharpe Ratio):")
    print(f"Expected Annual Return: {tangency_return*100:.2f}%")
    print(f"Expected Annual Volatility: {tangency_volatility*100:.2f}%")
    print(f"Sharpe Ratio: {tangency_sharpe:.4f}")
    print("Asset Allocation:")
    for i, ticker in enumerate(tickers):
        print(f"  {ticker}: {tangency_weights[i]*100:.2f}%")
    
    print("\nMinimum Volatility Portfolio:")
    print(f"Expected Annual Return: {min_vol_return*100:.2f}%")
    print(f"Expected Annual Volatility: {min_vol_volatility*100:.2f}%")
    print(f"Sharpe Ratio: {min_vol_sharpe:.4f}")
    print("Asset Allocation:")
    for i, ticker in enumerate(tickers):
        print(f"  {ticker}: {min_vol_weights[i]*100:.2f}%")
    
    # Create visualizations
    create_visualizations(
        tickers, prices, daily_returns, cumulative_returns, correlation,
        portfolio_annual_volatility, portfolio_annual_returns, sharpe_ratios,
        tangency_volatility, tangency_return, min_vol_volatility, min_vol_return,
        efficient_volatilities, target_returns, tangency_weights, min_vol_weights,
        save_report
    )
    
    # Return results as dictionary
    results = {
        'tickers': tickers,
        'period': period,
        'date_range': date_range,
        'prices': prices,
        'daily_returns': daily_returns,
        'cumulative_returns': cumulative_returns,
        'performance_summary': performance_summary,
        'correlation': correlation,
        'efficient_frontier': {
            'returns': target_returns,
            'volatilities': efficient_volatilities,
            'weights': efficient_weights
        },
        'tangency_portfolio': {
            'return': tangency_return,
            'volatility': tangency_volatility,
            'sharpe': tangency_sharpe,
            'weights': tangency_weights
        },
        'min_vol_portfolio': {
            'return': min_vol_return,
            'volatility': min_vol_volatility,
            'sharpe': min_vol_sharpe,
            'weights': min_vol_weights
        }
    }
    
    return results

def create_visualizations(
    tickers, prices, daily_returns, cumulative_returns, correlation,
    portfolio_volatility, portfolio_returns, sharpe_ratios,
    tangency_volatility, tangency_return, min_vol_volatility, min_vol_return,
    efficient_volatilities, efficient_returns, tangency_weights, min_vol_weights,
    save_figures=False
):
    """
    Create comprehensive visualizations for portfolio analysis
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # 1. Price History
    plt.figure(figsize=(12, 6))
    for ticker in tickers:
        plt.plot(prices[ticker] / prices[ticker].iloc[0], label=ticker)
    plt.title('Normalized Price History', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Normalized Price (Start=1)', fontsize=12)
    plt.legend(frameon=True, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_figures:
        plt.savefig(f'price_history_{timestamp}.png', dpi=300)
    plt.show()
    
    # 2. Cumulative Returns
    plt.figure(figsize=(12, 6))
    for ticker in tickers:
        plt.plot(cumulative_returns[ticker], label=ticker)
    plt.title('Cumulative Returns', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return (%)', fontsize=12)
    plt.legend(frameon=True, loc='best')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_figures:
        plt.savefig(f'cumulative_returns_{timestamp}.png', dpi=300)
    plt.show()
    
    # 3. Correlation Matrix Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Asset Correlation Matrix', fontsize=14)
    plt.tight_layout()
    if save_figures:
        plt.savefig(f'correlation_matrix_{timestamp}.png', dpi=300)
    plt.show()
    
    # 4. Efficient Frontier Plot
    plt.figure(figsize=(14, 10))
    
    # Create a scatter plot of random portfolios
    scatter = plt.scatter(portfolio_volatility * 100, portfolio_returns * 100, 
               c=sharpe_ratios, cmap='viridis', alpha=0.6, s=10)
    
    # Plot efficient frontier
    plt.plot(np.array(efficient_volatilities) * 100, efficient_returns * 100, 
             'r--', linewidth=2, label='Efficient Frontier')
    
    # Highlight key portfolios
    plt.scatter(tangency_volatility * 100, tangency_return * 100, 
               s=300, c='lime', edgecolors='black', marker='*', 
               label='Tangency Portfolio (Max Sharpe)')
    
    plt.scatter(min_vol_volatility * 100, min_vol_return * 100, 
               s=300, c='red', edgecolors='black', marker='P', 
               label='Minimum Volatility Portfolio')
    
    # Add individual assets
    for i, ticker in enumerate(tickers):
        annual_return = daily_returns.mean()[i] * 252 * 100
        annual_vol = daily_returns.std()[i] * np.sqrt(252) * 100
        plt.scatter(annual_vol, annual_return, s=200, marker='D',
                   label=f'{ticker}')

    # Add labels and title    
    plt.colorbar(scatter, label='Sharpe Ratio')
    plt.xlabel('Annual Volatility (%)', fontsize=14)
    plt.ylabel('Annual Expected Return (%)', fontsize=14)
    plt.title('Portfolio Optimization: Efficient Frontier', fontsize=16)
    plt.legend(loc='upper left', frameon=True, fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add risk-free rate line if applicable
    risk_free_rate = 0.0  # Could be parameterized
    if risk_free_rate > 0:
        max_sharpe_slope = (tangency_return - risk_free_rate) / tangency_volatility
        x_range = np.linspace(0, max(portfolio_volatility) * 1.2, 100)
        plt.plot(x_range * 100, (risk_free_rate + max_sharpe_slope * x_range) * 100, 
                'g-', label='Capital Market Line')
        plt.scatter(0, risk_free_rate * 100, c='black', s=80, label='Risk-Free Asset')
    
    plt.tight_layout()
    if save_figures:
        plt.savefig(f'efficient_frontier_{timestamp}.png', dpi=300)
    plt.show()
    
    # 5. Optimal Portfolio Allocation Pie Charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Tangency Portfolio
    wedges, texts, autotexts = ax1.pie(
        tangency_weights * 100, 
        labels=tickers, 
        autopct='%1.1f%%',
        textprops={'fontsize': 12}, 
        colors=sns.color_palette("viridis", len(tickers)),
        shadow=True, 
        startangle=90
    )
    ax1.set_title('Tangency Portfolio Allocation (Max Sharpe Ratio)', fontsize=14)
    
    # Minimum Volatility Portfolio
    wedges, texts, autotexts = ax2.pie(
        min_vol_weights * 100, 
        labels=tickers, 
        autopct='%1.1f%%',
        textprops={'fontsize': 12}, 
        colors=sns.color_palette("viridis", len(tickers)),
        shadow=True, 
        startangle=90
    )
    ax2.set_title('Minimum Volatility Portfolio Allocation', fontsize=14)
    
    plt.tight_layout()
    if save_figures:
        plt.savefig(f'portfolio_allocations_{timestamp}.png', dpi=300)
    plt.show()

# Execute the analysis
if __name__ == "__main__":
    # Define the list of stock tickers
    tickers = ['AAPL', 'AMZN', 'GOOG', 'META', 'MSFT']
    
    # Run the analysis
    results = analyze_portfolio(
        tickers=tickers,
        period='2mo',
        risk_free_rate=0.02,  # 2% risk-free rate
        n_portfolios=10000,
        save_report=True
    )
    
    print("\n✅ Analysis complete!")
