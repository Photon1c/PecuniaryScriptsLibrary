"""
Efficient Frontier Analysis
Computes and visualizes the efficient frontier for a portfolio of stocks.

Uses shared tickers.json for ticker list and data_loader for consistent data access.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import data_loader


# -----------------------------
# Configuration
# -----------------------------
SCRIPT_DIR = Path(__file__).parent
TICKERS_JSON = SCRIPT_DIR.parent.parent / "tickers.json"
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_tickers() -> List[str]:
    """Load ticker symbols from shared JSON file."""
    if not TICKERS_JSON.exists():
        raise FileNotFoundError(f"Tickers file not found: {TICKERS_JSON}")
    
    with open(TICKERS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Extract just the ticker symbols (first element of each [symbol, type] pair)
    return [ticker[0] for ticker in data["tickers"]]


def load_price_data(tickers: List[str], min_periods: int = 60) -> pd.DataFrame:
    """
    Load historical price data for tickers using data_loader.
    
    Args:
        tickers: List of ticker symbols
        min_periods: Minimum number of periods required
    
    Returns:
        DataFrame with dates as index and tickers as columns
    """
    price_data = {}
    valid_tickers = []
    
    for ticker in tickers:
        try:
            df = data_loader.load_stock_data(ticker)
            
            # Get close price column
            if "Close/Last" in df.columns:
                price_col = "Close/Last"
            elif "Close" in df.columns:
                price_col = "Close"
            else:
                print(f"WARNING: No price column found for {ticker}, skipping")
                continue
            
            # Ensure Date column exists
            if "Date" not in df.columns:
                print(f"WARNING: No Date column found for {ticker}, skipping")
                continue
            
            # Set Date as index and extract prices
            df = df.set_index("Date")
            prices = pd.to_numeric(df[price_col], errors='coerce').dropna()
            
            if len(prices) >= min_periods:
                price_data[ticker] = prices
                valid_tickers.append(ticker)
                print(f"✓ Loaded {ticker}: {len(prices)} periods")
            else:
                print(f"WARNING: {ticker} has only {len(prices)} periods (need {min_periods}), skipping")
        except Exception as e:
            print(f"ERROR loading {ticker}: {e}")
            continue
    
    if not price_data:
        raise ValueError("No valid price data loaded for any ticker")
    
    # Combine into single DataFrame, aligning dates
    prices_df = pd.DataFrame(price_data)
    prices_df = prices_df.sort_index()
    
    # Forward fill missing values (same ticker across dates)
    prices_df = prices_df.ffill()
    
    # Drop rows where any ticker is still missing
    prices_df = prices_df.dropna()
    
    print(f"\nLoaded {len(valid_tickers)} tickers with {len(prices_df)} common periods")
    print(f"Date range: {prices_df.index.min()} to {prices_df.index.max()}")
    
    return prices_df


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily returns from prices."""
    return prices.pct_change().dropna()


def calculate_portfolio_stats(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> Tuple[float, float]:
    """
    Calculate portfolio expected return and standard deviation.
    
    Args:
        weights: Portfolio weights (must sum to 1)
        mu: Expected returns vector
        cov: Covariance matrix
    
    Returns:
        (expected_return, standard_deviation)
    """
    portfolio_return = np.dot(weights, mu)
    portfolio_std = np.sqrt(np.dot(weights, np.dot(cov, weights)))
    return portfolio_return, portfolio_std


def generate_random_portfolios(
    mu: np.ndarray,
    cov: np.ndarray,
    n_portfolios: int = 10000,
    risk_free_rate: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate random portfolios and calculate their statistics.
    
    Returns:
        (weights, returns, stds, sharpes)
    """
    n_assets = len(mu)
    weights = np.random.random(size=(n_portfolios, n_assets))
    weights /= np.sum(weights, axis=1, keepdims=True)
    
    returns = np.zeros(n_portfolios)
    stds = np.zeros(n_portfolios)
    
    for i in range(n_portfolios):
        returns[i], stds[i] = calculate_portfolio_stats(weights[i], mu, cov)
    
    sharpes = (returns - risk_free_rate) / stds
    
    return weights, returns, stds, sharpes


def find_tangency_portfolio(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_free_rate: float = 0.0
) -> Tuple[np.ndarray, float, float, float]:
    """
    Find the tangency portfolio (maximum Sharpe ratio).
    
    Returns:
        (weights, return, std, sharpe)
    """
    n_assets = len(mu)
    
    def negative_sharpe(weights):
        port_return, port_std = calculate_portfolio_stats(weights, mu, cov)
        if port_std == 0:
            return 1e10
        return -(port_return - risk_free_rate) / port_std
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * n_assets
    
    # Start from equal weights
    initial_weights = np.ones(n_assets) / n_assets
    
    result = minimize(
        negative_sharpe,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    if not result.success:
        print(f"WARNING: Optimization did not converge: {result.message}")
    
    tangency_weights = result.x
    tangency_return, tangency_std = calculate_portfolio_stats(tangency_weights, mu, cov)
    tangency_sharpe = (tangency_return - risk_free_rate) / tangency_std
    
    return tangency_weights, tangency_return, tangency_std, tangency_sharpe


def calculate_efficient_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the efficient frontier.
    
    Returns:
        (frontier_returns, frontier_stds)
    """
    def portfolio_variance(weights):
        return np.dot(weights, np.dot(cov, weights))
    
    n_assets = len(mu)
    frontier_returns = np.linspace(mu.min(), mu.max(), n_points)
    frontier_stds = []
    
    bounds = [(0.0, 1.0)] * n_assets
    
    for target_return in frontier_returns:
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'eq', 'fun': lambda w: np.dot(w, mu) - target_return}
        ]
        
        initial_weights = np.ones(n_assets) / n_assets
        
        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            frontier_std = np.sqrt(result.fun)
            frontier_stds.append(frontier_std)
        else:
            # If optimization fails, use a large value
            frontier_stds.append(np.nan)
    
    frontier_stds = np.array(frontier_stds)
    
    # Filter out invalid points
    valid = np.isfinite(frontier_stds)
    frontier_returns = frontier_returns[valid]
    frontier_stds = frontier_stds[valid]
    
    return frontier_returns, frontier_stds


def plot_efficient_frontier(
    portfolio_stds: np.ndarray,
    portfolio_returns: np.ndarray,
    sharpes: np.ndarray,
    frontier_stds: np.ndarray,
    frontier_returns: np.ndarray,
    tangency_std: float,
    tangency_return: float,
    tangency_weights: np.ndarray,
    tickers: List[str],
    output_path: Path,
    risk_free_rate: float = 0.0
) -> None:
    """Create visualization of efficient frontier."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot random portfolios with color by Sharpe ratio
    scatter = ax.scatter(
        portfolio_stds, portfolio_returns,
        c=sharpes, cmap='viridis',
        alpha=0.6, s=20, edgecolors='none',
        zorder=1
    )
    
    # Plot efficient frontier
    ax.plot(
        frontier_stds, frontier_returns,
        'r--', linewidth=2.5, label='Efficient Frontier',
        zorder=3
    )
    
    # Plot tangency portfolio
    ax.scatter(
        tangency_std, tangency_return,
        s=400, marker='*', color='gold',
        edgecolors='black', linewidths=2,
        label='Tangency Portfolio (Max Sharpe)',
        zorder=5
    )
    
    # Add risk-free rate line (if non-zero)
    if risk_free_rate != 0:
        x_max = max(portfolio_stds.max(), frontier_stds.max())
        ax.plot(
            [0, x_max], [risk_free_rate, tangency_return],
            'g:', linewidth=2, alpha=0.7,
            label=f'Capital Market Line (rf={risk_free_rate:.2%})',
            zorder=2
        )
    
    # Formatting
    ax.set_xlabel('Risk (Standard Deviation)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Expected Return', fontsize=13, fontweight='bold')
    ax.set_title('Efficient Frontier and Tangency Portfolio', fontsize=15, fontweight='bold', pad=20)
    
    # Add colorbar for Sharpe ratio
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sharpe Ratio', fontsize=11, rotation=270, labelpad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # Add text box with tangency portfolio weights
    weight_text = "Tangency Portfolio Weights:\n"
    for ticker, weight in zip(tickers, tangency_weights):
        weight_text += f"{ticker}: {weight:.2%}\n"
    weight_text += f"\nSharpe Ratio: {tangency_return / tangency_std:.3f}"
    weight_text += f"\nReturn: {tangency_return:.2%}"
    weight_text += f"\nRisk: {tangency_std:.2%}"
    
    ax.text(
        0.02, 0.98, weight_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        family='monospace'
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chart saved to {output_path}")


def save_results(
    tangency_weights: np.ndarray,
    tangency_return: float,
    tangency_std: float,
    tangency_sharpe: float,
    tickers: List[str],
    frontier_returns: np.ndarray,
    frontier_stds: np.ndarray,
    output_path: Path
) -> None:
    """Save results to JSON file."""
    results = {
        "tangency_portfolio": {
            "weights": {ticker: float(weight) for ticker, weight in zip(tickers, tangency_weights)},
            "expected_return": float(tangency_return),
            "standard_deviation": float(tangency_std),
            "sharpe_ratio": float(tangency_sharpe)
        },
        "efficient_frontier": {
            "returns": frontier_returns.tolist(),
            "standard_deviations": frontier_stds.tolist()
        },
        "tickers": tickers,
        "n_frontier_points": len(frontier_returns)
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")


def print_summary(
    tickers: List[str],
    mu: np.ndarray,
    tangency_weights: np.ndarray,
    tangency_return: float,
    tangency_std: float,
    tangency_sharpe: float
) -> None:
    """Print summary statistics."""
    print("\n" + "="*70)
    print("EFFICIENT FRONTIER ANALYSIS SUMMARY")
    print("="*70)
    print(f"\nTickers analyzed: {', '.join(tickers)}")
    print(f"\nIndividual Asset Statistics:")
    print("-" * 70)
    for i, ticker in enumerate(tickers):
        print(f"  {ticker:6s}: Return={mu[i]:7.2%}, Weight={tangency_weights[i]:6.2%}")
    
    print(f"\nTangency Portfolio (Maximum Sharpe Ratio):")
    print("-" * 70)
    print(f"  Expected Return:  {tangency_return:7.2%}")
    print(f"  Standard Dev:     {tangency_std:7.2%}")
    print(f"  Sharpe Ratio:     {tangency_sharpe:7.3f}")
    print("="*70 + "\n")


def main(
    tickers: Optional[List[str]] = None,
    n_portfolios: int = 10000,
    n_frontier_points: int = 100,
    risk_free_rate: float = 0.0,
    min_periods: int = 60,
    output_dir: Optional[Path] = None
) -> None:
    """Main analysis function."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    # Load tickers
    if tickers is None:
        tickers = load_tickers()
    
    print(f"Analyzing {len(tickers)} tickers...")
    
    # Load price data
    prices = load_price_data(tickers, min_periods=min_periods)
    
    # Use only tickers that were successfully loaded
    tickers = list(prices.columns)
    
    # Calculate returns
    returns = calculate_returns(prices)
    
    # Calculate statistics
    mu = returns.mean().values * 252  # Annualized
    cov = returns.cov().values * 252   # Annualized
    
    print(f"\nCalculating efficient frontier with {n_frontier_points} points...")
    frontier_returns, frontier_stds = calculate_efficient_frontier(mu, cov, n_points=n_frontier_points)
    
    print(f"Generating {n_portfolios} random portfolios...")
    weights, portfolio_returns, portfolio_stds, sharpes = generate_random_portfolios(
        mu, cov, n_portfolios=n_portfolios, risk_free_rate=risk_free_rate
    )
    
    print("Finding tangency portfolio...")
    tangency_weights, tangency_return, tangency_std, tangency_sharpe = find_tangency_portfolio(
        mu, cov, risk_free_rate=risk_free_rate
    )
    
    # Print summary
    print_summary(tickers, mu, tangency_weights, tangency_return, tangency_std, tangency_sharpe)
    
    # Create visualization
    chart_path = output_dir / "efficient_frontier.png"
    plot_efficient_frontier(
        portfolio_stds, portfolio_returns, sharpes,
        frontier_stds, frontier_returns,
        tangency_std, tangency_return, tangency_weights,
        tickers, chart_path, risk_free_rate=risk_free_rate
    )
    
    # Save results
    results_path = output_dir / "efficient_frontier_results.json"
    save_results(
        tangency_weights, tangency_return, tangency_std, tangency_sharpe,
        tickers, frontier_returns, frontier_stds, results_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate and visualize efficient frontier for portfolio of stocks"
    )
    parser.add_argument(
        "--tickers", nargs="+",
        help="Specific tickers to analyze (default: all from tickers.json)"
    )
    parser.add_argument(
        "--n-portfolios", type=int, default=10000,
        help="Number of random portfolios to generate (default: 10000)"
    )
    parser.add_argument(
        "--n-frontier-points", type=int, default=100,
        help="Number of points on efficient frontier (default: 100)"
    )
    parser.add_argument(
        "--risk-free-rate", type=float, default=0.0,
        help="Risk-free rate for Sharpe ratio calculation (default: 0.0)"
    )
    parser.add_argument(
        "--min-periods", type=int, default=60,
        help="Minimum number of periods required per ticker (default: 60)"
    )
    parser.add_argument(
        "--output-dir", type=str,
        help="Output directory (default: formula/fineagle/output)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    main(
        tickers=args.tickers,
        n_portfolios=args.n_portfolios,
        n_frontier_points=args.n_frontier_points,
        risk_free_rate=args.risk_free_rate,
        min_periods=args.min_periods,
        output_dir=output_dir
    )
