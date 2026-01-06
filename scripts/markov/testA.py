#!/usr/bin/env python3
"""
Markov Blanket Analysis for Option Pricing Prediction

This script implements a Bayesian network approach to option pricing using Markov blanket discovery.
The Markov blanket of the "option premium" node identifies the minimal set of variables that
render the option premium conditionally independent of all other variables in the network.

Based on the theoretical framework described in the outline, this implementation:
1. Defines the Bayesian network structure for option pricing
2. Computes the Markov blanket for the option premium node
3. Loads and analyzes real option data
4. Provides visualizations and analysis

Nodes in the network:
0: Spot_Price
1: Volatility
2: Interest_Rate
3: Time_to_Expiration
4: Strike_Price
5: Market_Sentiment
6: Economic_Indicators
7: Option_Premium (target node)
8: Trading_Volume
9: News
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplcyberpunk
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from pathlib import Path
import networkx as nx
import os
from datetime import datetime
from data_loader import load_option_chain_data, get_latest_price, load_stock_data, get_most_recent_option_date

# Initialize rich console for beautiful output
console = Console()

class MarkovBlanketAnalyzer:
    """
    Implements Markov blanket discovery for Bayesian networks in option pricing.
    """

    def __init__(self):
        # Define node names based on the outline
        self.node_names = [
            "Spot_Price", "Volatility", "Interest_Rate", "Time_to_Expiration",
            "Strike_Price", "Market_Sentiment", "Economic_Indicators",
            "Option_Premium", "Trading_Volume", "News"
        ]

        # Adjacency matrix from the outline (transposed for column-major convention)
        self.adjacency_matrix = np.array([
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # Spot_Price -> Option_Premium
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # Volatility -> Option_Premium
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # Interest_Rate -> Option_Premium
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # Time_to_Expiration -> Option_Premium
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # Strike_Price -> Option_Premium
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Market_Sentiment -> Volatility
            [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # Economic_Indicators -> Interest_Rate, Market_Sentiment
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Option_Premium -> Trading_Volume
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Trading_Volume (no outgoing edges)
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # News -> Trading_Volume
        ])

        self.target_node = 7  # Option_Premium

    def get_parents(self, node_idx):
        """Get direct parents of a node."""
        return np.where(self.adjacency_matrix[:, node_idx] == 1)[0].tolist()

    def get_children(self, node_idx):
        """Get direct children of a node."""
        return np.where(self.adjacency_matrix[node_idx, :] == 1)[0].tolist()

    def get_spouses(self, node_idx):
        """Get spouses (co-parents of children) of a node."""
        children = self.get_children(node_idx)
        spouses = set()

        for child in children:
            # Get all parents of this child except the target node
            child_parents = self.get_parents(child)
            spouses.update([p for p in child_parents if p != node_idx])

        return list(spouses)

    def compute_markov_blanket(self, node_idx):
        """
        Compute the Markov blanket for a given node.

        MB(node) = Parents(node) ∪ Children(node) ∪ Spouses(node)
        """
        parents = self.get_parents(node_idx)
        children = self.get_children(node_idx)
        spouses = self.get_spouses(node_idx)

        # Combine all sets
        markov_blanket = set(parents + children + spouses)

        return {
            'parents': parents,
            'children': children,
            'spouses': spouses,
            'markov_blanket': sorted(list(markov_blanket))
        }

    def create_network_graph(self):
        """Create a NetworkX graph from the adjacency matrix."""
        G = nx.DiGraph()

        # Add nodes
        for i, name in enumerate(self.node_names):
            G.add_node(i, label=name)

        # Add edges
        rows, cols = self.adjacency_matrix.shape
        for i in range(rows):
            for j in range(cols):
                if self.adjacency_matrix[i, j] == 1:
                    G.add_edge(i, j)

        return G

    def visualize_network(self, highlight_blanket=True):
        """Visualize the Bayesian network with optional Markov blanket highlighting."""
        plt.style.use("cyberpunk")
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        G = self.create_network_graph()
        pos = nx.spring_layout(G, seed=42)

        # Draw all nodes
        node_colors = ['lightblue'] * len(self.node_names)
        node_sizes = [800] * len(self.node_names)

        # Highlight target node and Markov blanket
        if highlight_blanket:
            mb_info = self.compute_markov_blanket(self.target_node)
            blanket_nodes = mb_info['markov_blanket']

            for node_idx in blanket_nodes:
                node_colors[node_idx] = 'orange'
                node_sizes[node_idx] = 1000

            # Highlight target node
            node_colors[self.target_node] = 'red'
            node_sizes[self.target_node] = 1200

        # Draw the graph
        nx.draw(G, pos, with_labels=True,
                labels={i: name for i, name in enumerate(self.node_names)},
                node_color=node_colors, node_size=node_sizes,
                font_size=8, font_weight='bold', ax=ax,
                edge_color='gray', arrows=True, arrowsize=20)

        # Add title and legend
        title_text = f"Bayesian Network for Option Pricing\nMarkov Blanket of {self.node_names[self.target_node]}"
        ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)

        if highlight_blanket:
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=12, label=f'Target: {self.node_names[self.target_node]}'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Markov Blanket'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=8, label='Other Variables')
            ]
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))

        plt.tight_layout()
        mplcyberpunk.add_glow_effects()
        return fig

    def analyze_option_data(self, ticker="SPY"):
        """Load and analyze option data to demonstrate the Markov blanket concept."""
        try:
            # Get most recent option date
            recent_date = get_most_recent_option_date(ticker.lower())

            # Load option data
            console.print(f"\n[bold cyan]Loading option data for {ticker}...[/bold cyan]")
            option_df = load_option_chain_data(ticker.lower())

            # Load stock data for additional variables
            try:
                stock_df = load_stock_data(ticker)
                latest_price = get_latest_price(ticker)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load stock data: {e}[/yellow]")
                latest_price = None

            # Display data summary
            console.print(f"[green]Successfully loaded {len(option_df)} option contracts[/green]")

            if len(option_df) > 0:
                # Show option chain for strikes near current price
                self.display_option_chain_table(option_df, ticker, latest_price, recent_date)

            return option_df, latest_price

        except Exception as e:
            console.print(f"[red]Error loading data: {e}[/red]")
            return None, None

    def display_option_chain_table(self, option_df, ticker, latest_price, recent_date):
        """Display option chain for strikes near current price and save to files."""
        if latest_price is None:
            console.print("[red]No stock price available for strike analysis[/red]")
            return

        # Create output directory
        output_dir = Path("../output")
        output_dir.mkdir(exist_ok=True)

        # Find strikes closest to current price (4 above and below)
        strikes = sorted(option_df['Strike'].unique())

        # Find the strike closest to current price
        closest_strike = min(strikes, key=lambda x: abs(x - latest_price))

        # Get strikes around this closest strike (4 below, 4 above)
        closest_idx = strikes.index(closest_strike)
        start_idx = max(0, closest_idx - 4)
        end_idx = min(len(strikes), closest_idx + 5)  # +5 to include the closest one

        target_strikes = strikes[start_idx:end_idx]

        # Filter dataframe to only these strikes
        filtered_df = option_df[option_df['Strike'].isin(target_strikes)].copy()

        # Prepare data for CSV export
        csv_data = []

        # Create table for terminal display (brief summary)
        summary_table = Table(title=f"[bold blue]{ticker} Option Chain Summary[/bold blue]")
        summary_table.add_column("Strike", justify="right", no_wrap=True)
        summary_table.add_column("Call Bid/Ask", justify="center", no_wrap=True)
        summary_table.add_column("Put Bid/Ask", justify="center", no_wrap=True)
        summary_table.add_column("Moneyness", justify="center", no_wrap=True)
        summary_table.add_column("Volume", justify="right", no_wrap=True)

        for strike in target_strikes:
            strike_data = filtered_df[filtered_df['Strike'] == strike]

            if len(strike_data) == 0:
                continue

            row = strike_data.iloc[0]

            # Call option data
            call_bid = row.get('Bid', 0)
            call_ask = row.get('Ask', 0)
            call_volume = row.get('Volume', 0)

            # Put option data
            put_bid = row.get('Bid.1', 0)
            put_ask = row.get('Ask.1', 0)
            put_volume = row.get('Volume.1', 0)

            # Determine moneyness
            if abs(strike - latest_price) < 1:
                moneyness = "ATM"
            elif strike < latest_price:
                moneyness = "ITM"
            else:
                moneyness = "OTM"

            # Format for terminal summary
            call_str = f"${call_bid:.2f}/${call_ask:.2f}" if call_bid > 0 or call_ask > 0 else "$0.00/$0.00"
            put_str = f"${put_bid:.2f}/${put_ask:.2f}" if put_bid > 0 or put_ask > 0 else "$0.00/$0.00"
            total_volume = call_volume + put_volume

            summary_table.add_row(
                f"${strike:.0f}",
                call_str,
                put_str,
                moneyness,
                f"{total_volume:,}"
            )

            # Add to CSV data
            csv_data.append({
                'Strike': strike,
                'Call_Bid': call_bid,
                'Call_Ask': call_ask,
                'Call_Volume': call_volume,
                'Put_Bid': put_bid,
                'Put_Ask': put_ask,
                'Put_Volume': put_volume,
                'Moneyness': moneyness,
                'Total_Volume': total_volume,
                'Stock_Price': latest_price
            })

        # Display brief summary in terminal
        console.print(summary_table)

        # Save detailed data to files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{ticker}_option_chain_{timestamp}"

        # Save as CSV
        csv_df = pd.DataFrame(csv_data)
        csv_path = output_dir / f"{base_filename}.csv"
        csv_df.to_csv(csv_path, index=False)

        # Save as formatted text file
        txt_path = output_dir / f"{base_filename}.txt"
        with open(txt_path, 'w') as f:
            f.write(f"{ticker} Option Chain Analysis\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Stock Price: ${latest_price:.2f}\n")
            f.write(f"Data Date: {recent_date}/2025\n")
            f.write(f"Strikes shown: {len(target_strikes)}\n\n")

            # Write detailed table
            f.write("Detailed Option Chain:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Strike':<8} {'Type':<6} {'Bid':>8} {'Ask':>8} {'Spread':>8} {'Moneyness':<10} {'Volume':>10}\n")
            f.write("-" * 80 + "\n")

            for strike in target_strikes:
                strike_data = [d for d in csv_data if d['Strike'] == strike]
                if not strike_data:
                    continue
                data = strike_data[0]

                # Call data
                call_spread = data['Call_Ask'] - data['Call_Bid'] if data['Call_Ask'] > 0 and data['Call_Bid'] > 0 else 0
                f.write(f"${strike:<7.0f} CALL   ${data['Call_Bid']:>7.2f} ${data['Call_Ask']:>7.2f} ${call_spread:>7.2f} {data['Moneyness']:<10} {data['Call_Volume']:>10,}\n")

                # Put data
                put_spread = data['Put_Ask'] - data['Put_Bid'] if data['Put_Ask'] > 0 and data['Put_Bid'] > 0 else 0
                f.write(f"${strike:<7.0f} PUT    ${data['Put_Bid']:>7.2f} ${data['Put_Ask']:>7.2f} ${put_spread:>7.2f} {data['Moneyness']:<10} {data['Put_Volume']:>10,}\n")
                f.write("\n")

            # Summary statistics
            total_volume = sum(d['Total_Volume'] for d in csv_data)
            call_bids = [d['Call_Bid'] for d in csv_data if d['Call_Bid'] > 0]
            call_asks = [d['Call_Ask'] for d in csv_data if d['Call_Ask'] > 0]
            put_bids = [d['Put_Bid'] for d in csv_data if d['Put_Bid'] > 0]
            put_asks = [d['Put_Ask'] for d in csv_data if d['Put_Ask'] > 0]

            avg_call_spread = (sum(call_asks) - sum(call_bids)) / len(call_asks) if call_asks else 0
            avg_put_spread = (sum(put_asks) - sum(put_bids)) / len(put_asks) if put_asks else 0

            f.write("\nSummary Statistics:\n")
            f.write(f"Total Volume (shown strikes): {total_volume:,}\n")
            f.write(f"Average Call Spread: ${avg_call_spread:.2f}\n")
            f.write(f"Average Put Spread: ${avg_put_spread:.2f}\n")
            f.write(f"Contracts analyzed: {len(csv_data)}\n")
            f.write(f"Full chain contracts: {len(option_df)}\n")

        console.print(f"\n[green]SUCCESS: Option chain data saved to:[/green]")
        console.print(f"  [cyan]CSV:[/cyan] {csv_path}")
        console.print(f"  [cyan]Text:[/cyan] {txt_path}")

        # Add summary info
        total_volume = sum(d['Total_Volume'] for d in csv_data)
        console.print(f"\n[dim]Stock Price: ${latest_price:.2f} | Total Volume: {total_volume:,} | Strikes: {len(target_strikes)} | Date: {recent_date}/2025[/dim]")

    def display_markov_blanket_analysis(self):
        """Display comprehensive Markov blanket analysis."""
        console.print(Panel.fit(
        "[bold cyan]Markov Blanket Analysis for Option Pricing[/bold cyan]\n\n"
        "This analysis identifies the minimal set of variables that render the option premium\n"
        "conditionally independent of all other variables in the Bayesian network.",
        title="[bold]Analysis Overview[/bold]"
    ))

        # Compute Markov blanket
        mb_info = self.compute_markov_blanket(self.target_node)
        target_name = self.node_names[self.target_node]

        # Create results table
        table = Table(title=f"Markov Blanket Components for {target_name}")
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Variables", style="green")
        table.add_column("Count", style="magenta", justify="right")

        # Parents
        parent_names = [self.node_names[i] for i in mb_info['parents']]
        table.add_row("Parents (Direct Causes)", ", ".join(parent_names), str(len(parent_names)))

        # Children
        child_names = [self.node_names[i] for i in mb_info['children']]
        table.add_row("Children (Direct Effects)", ", ".join(child_names), str(len(child_names)))

        # Spouses
        spouse_names = [self.node_names[i] for i in mb_info['spouses']]
        table.add_row("Spouses (Co-parents)", ", ".join(spouse_names), str(len(spouse_names)))

        # Markov Blanket
        blanket_names = [self.node_names[i] for i in mb_info['markov_blanket']]
        table.add_row("[bold]Markov Blanket[/bold]", "[bold]" + ", ".join(blanket_names) + "[/bold]", f"[bold]{len(blanket_names)}[/bold]")

        console.print(table)

        # Analysis insights
        insights = [
            f"[bold blue]DATA:[/bold blue] The Markov blanket contains {len(mb_info['markov_blanket'])} variables that fully determine {target_name}",
            f"[bold red]TARGET:[/bold red] Traditional Black-Scholes uses: Spot Price, Strike Price, Time to Expiration, Interest Rate, Volatility",
            f"[bold yellow]ENHANCED:[/bold yellow] Markov blanket adds: {', '.join([name for name in blanket_names if name not in ['Spot_Price', 'Strike_Price', 'Time_to_Expiration', 'Interest_Rate', 'Volatility']])}",
            f"[bold green]INSIGHT:[/bold green] These additional variables can capture market inefficiencies, sentiment, and feedback loops"
        ]

        insights_panel = Panel(
            "\n".join(f"• {insight}" for insight in insights),
            title="[bold]Key Insights[/bold]",
            border_style="green"
        )
        console.print(insights_panel)


def main():
    """Main execution function."""
    console.print("[bold magenta]>>> Starting Markov Blanket Analysis for Option Pricing[/bold magenta]\n")

    # Initialize analyzer
    analyzer = MarkovBlanketAnalyzer()

    # Display network analysis
    analyzer.display_markov_blanket_analysis()

    # Load and analyze real data
    console.print("\n" + "="*80)
    console.print("[bold yellow]>>> Loading Real Option Data for Validation[/bold yellow]")

    # Try different tickers in case some data is missing
    tickers_to_try = ["SPY", "AAPL", "TSLA", "NVDA"]

    option_data = None
    latest_price = None

    for ticker in tickers_to_try:
        console.print(f"[dim]Attempting to load data for {ticker}...[/dim]")
        option_data, latest_price = analyzer.analyze_option_data(ticker)
        if option_data is not None:
            break

    if option_data is None:
        console.print("[red]ERROR: Could not load option data for any ticker. Please check data paths.[/red]")
        return

    # Create visualization
    console.print("\n" + "="*80)
    console.print("[bold yellow]>>> Creating Network Visualization[/bold yellow]")

    try:
        fig = analyzer.visualize_network()
        plt.savefig('markov_blanket_network.png', dpi=300, bbox_inches='tight',
                   facecolor='lightgray', edgecolor='none')
        console.print("[green]SUCCESS: Network visualization saved as 'markov_blanket_network.png'[/green]")

        # Show plot if running interactively
        plt.show(block=False)
        plt.pause(0.1)  # Brief pause to allow rendering

    except Exception as e:
        console.print(f"[red]ERROR: Error creating visualization: {e}[/red]")

    # Final summary
    console.print("\n" + "="*80)
    summary_panel = Panel(
        "[bold green]ANALYSIS COMPLETE![/bold green]\n\n"
        "The Markov blanket approach provides a data-driven framework for option pricing\n"
        "that goes beyond traditional models by identifying all relevant variables\n"
        "through causal relationships in the Bayesian network.\n\n"
        "[dim]This implementation demonstrates how graph theory and probabilistic\n"
        "reasoning can enhance financial modeling and prediction.[/dim]",
        title="[bold]Summary[/bold]"
    )
    console.print(summary_panel)


if __name__ == "__main__":
    main()