"""
Aggregate Markov Analysis Reports

This script reads all ticker reports from output/markov/{TICKER}/ directories
and generates summary tables grouped by regime, structure, gate state, etc.
"""

import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from datetime import datetime
import json

console = Console()

# Configuration
SCRIPT_DIR = Path(__file__).parent.resolve()
# Reports are in metascripts/output/markov (same level as markov directory)
OUTPUT_DIR = SCRIPT_DIR.parent / "output" / "markov"
OUTPUT_DIR = OUTPUT_DIR.resolve()  # Resolve to absolute path
AGGREGATE_OUTPUT_DIR = OUTPUT_DIR / "aggregated"
AGGREGATE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_reports() -> pd.DataFrame:
    """
    Load all ticker reports from subdirectories.
    
    Returns:
        DataFrame with all ticker reports combined
    """
    all_data = []
    
    console.print(f"[dim]Looking for reports in: {OUTPUT_DIR}[/dim]")
    
    if not OUTPUT_DIR.exists():
        console.print(f"[red]Output directory not found: {OUTPUT_DIR}[/red]")
        console.print(f"[yellow]Please ensure reports are in: {OUTPUT_DIR}[/yellow]")
        return pd.DataFrame()
    
    # Find all ticker subdirectories (exclude 'aggregated' directory)
    ticker_dirs = [d for d in OUTPUT_DIR.iterdir() 
                   if d.is_dir() and not d.name.startswith('.') and d.name.upper() != 'AGGREGATED']
    
    if not ticker_dirs:
        console.print(f"[yellow]No ticker subdirectories found in {OUTPUT_DIR}[/yellow]")
        return pd.DataFrame()
    
    console.print(f"[cyan]Found {len(ticker_dirs)} ticker directories[/cyan]")
    
    for ticker_dir in ticker_dirs:
        ticker = ticker_dir.name.upper()
        csv_path = ticker_dir / f"{ticker}_report.csv"
        
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                all_data.append(df)
                console.print(f"  [green]✓[/green] Loaded {ticker}")
            except Exception as e:
                console.print(f"  [red]✗[/red] Error loading {ticker}: {e}")
        else:
            console.print(f"  [yellow]⚠[/yellow] No report found for {ticker}")
    
    if not all_data:
        console.print("[red]No reports loaded[/red]")
        return pd.DataFrame()
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    console.print(f"\n[green]Successfully loaded {len(combined_df)} ticker reports[/green]")
    
    return combined_df


def generate_summary_tables(df: pd.DataFrame) -> dict:
    """
    Generate summary tables grouped by various dimensions.
    
    Returns:
        Dictionary of summary DataFrames
    """
    summaries = {}
    
    if df.empty:
        return summaries
    
    # 1. By Regime
    if 'Regime' in df.columns:
        regime_summary = df.groupby('Regime').agg({
            'Ticker': 'count',
            'Kelly_Fractional': 'mean',
            'Gate_State': lambda x: x.value_counts().to_dict(),
            'Market_State': lambda x: x.value_counts().to_dict(),
        }).rename(columns={'Ticker': 'Count'})
        regime_summary['Tickers'] = df.groupby('Regime')['Ticker'].apply(lambda x: ', '.join(x))
        summaries['by_regime'] = regime_summary
    
    # 2. By Structure Family
    if 'Structure_Family' in df.columns:
        structure_summary = df.groupby('Structure_Family').agg({
            'Ticker': 'count',
            'Kelly_Fractional': 'mean',
            'Regime': lambda x: x.value_counts().to_dict(),
            'Gate_State': lambda x: x.value_counts().to_dict(),
        }).rename(columns={'Ticker': 'Count'})
        structure_summary['Tickers'] = df.groupby('Structure_Family')['Ticker'].apply(lambda x: ', '.join(x))
        summaries['by_structure'] = structure_summary
    
    # 3. By Gate State
    if 'Gate_State' in df.columns:
        gate_summary = df.groupby('Gate_State').agg({
            'Ticker': 'count',
            'Kelly_Fractional': 'mean',
            'Regime': lambda x: x.value_counts().to_dict(),
            'Structure_Family': lambda x: x.value_counts().to_dict(),
        }).rename(columns={'Ticker': 'Count'})
        gate_summary['Tickers'] = df.groupby('Gate_State')['Ticker'].apply(lambda x: ', '.join(x))
        summaries['by_gate_state'] = gate_summary
    
    # 4. By Market State
    if 'Market_State' in df.columns:
        market_summary = df.groupby('Market_State').agg({
            'Ticker': 'count',
            'Kelly_Fractional': 'mean',
            'Regime': lambda x: x.value_counts().to_dict(),
            'Gate_State': lambda x: x.value_counts().to_dict(),
        }).rename(columns={'Ticker': 'Count'})
        market_summary['Tickers'] = df.groupby('Market_State')['Ticker'].apply(lambda x: ', '.join(x))
        summaries['by_market_state'] = market_summary
    
    # 5. Combined: Regime + Structure
    if 'Regime' in df.columns and 'Structure_Family' in df.columns:
        regime_structure = df.groupby(['Regime', 'Structure_Family']).agg({
            'Ticker': 'count',
            'Kelly_Fractional': 'mean',
        }).rename(columns={'Ticker': 'Count'})
        regime_structure['Tickers'] = df.groupby(['Regime', 'Structure_Family'])['Ticker'].apply(lambda x: ', '.join(x))
        summaries['by_regime_structure'] = regime_structure
    
    # 6. Combined: Regime + Gate State
    if 'Regime' in df.columns and 'Gate_State' in df.columns:
        regime_gate = df.groupby(['Regime', 'Gate_State']).agg({
            'Ticker': 'count',
            'Kelly_Fractional': 'mean',
        }).rename(columns={'Ticker': 'Count'})
        regime_gate['Tickers'] = df.groupby(['Regime', 'Gate_State'])['Ticker'].apply(lambda x: ', '.join(x))
        summaries['by_regime_gate'] = regime_gate
    
    # 7. Full detail table (sorted by regime, then structure)
    if 'Regime' in df.columns:
        detail_table = df[['Ticker', 'Regime', 'Structure_Family', 'Gate_State', 'Market_State',
                           'Kelly_Fractional', 'Kelly_Adjusted', 'P', 'B', 'Multiplier',
                           'Residual_R2', 'Skew_R2', 'Reflexive_Sleeve_Status']].copy()
        detail_table = detail_table.sort_values(['Regime', 'Structure_Family', 'Gate_State'])
        summaries['detail'] = detail_table
    
    return summaries


def print_rich_tables(summaries: dict):
    """Print summary tables using Rich console."""
    
    # By Regime
    if 'by_regime' in summaries:
        table = Table(title="Summary by Regime", show_header=True, header_style="bold magenta")
        table.add_column("Regime", style="cyan")
        table.add_column("Count", justify="right", style="green")
        table.add_column("Avg Kelly Fractional", justify="right", style="yellow")
        table.add_column("Tickers", style="dim")
        
        for regime, row in summaries['by_regime'].iterrows():
            table.add_row(
                str(regime),
                str(int(row['Count'])),
                f"{row['Kelly_Fractional']:.4f}",
                row['Tickers']
            )
        console.print(table)
        console.print()
    
    # By Structure Family
    if 'by_structure' in summaries:
        table = Table(title="Summary by Structure Family", show_header=True, header_style="bold magenta")
        table.add_column("Structure", style="cyan")
        table.add_column("Count", justify="right", style="green")
        table.add_column("Avg Kelly Fractional", justify="right", style="yellow")
        table.add_column("Tickers", style="dim")
        
        for structure, row in summaries['by_structure'].iterrows():
            table.add_row(
                str(structure),
                str(int(row['Count'])),
                f"{row['Kelly_Fractional']:.4f}",
                row['Tickers']
            )
        console.print(table)
        console.print()
    
    # By Gate State
    if 'by_gate_state' in summaries:
        table = Table(title="Summary by Gate State", show_header=True, header_style="bold magenta")
        table.add_column("Gate State", style="cyan")
        table.add_column("Count", justify="right", style="green")
        table.add_column("Avg Kelly Fractional", justify="right", style="yellow")
        table.add_column("Tickers", style="dim")
        
        for gate_state, row in summaries['by_gate_state'].iterrows():
            table.add_row(
                str(gate_state),
                str(int(row['Count'])),
                f"{row['Kelly_Fractional']:.4f}",
                row['Tickers']
            )
        console.print(table)
        console.print()
    
    # By Market State
    if 'by_market_state' in summaries:
        table = Table(title="Summary by Market State", show_header=True, header_style="bold magenta")
        table.add_column("Market State", style="cyan")
        table.add_column("Count", justify="right", style="green")
        table.add_column("Avg Kelly Fractional", justify="right", style="yellow")
        table.add_column("Tickers", style="dim")
        
        for market_state, row in summaries['by_market_state'].iterrows():
            table.add_row(
                str(market_state),
                str(int(row['Count'])),
                f"{row['Kelly_Fractional']:.4f}",
                row['Tickers']
            )
        console.print(table)
        console.print()
    
    # Combined: Regime + Structure
    if 'by_regime_structure' in summaries:
        table = Table(title="Summary by Regime + Structure", show_header=True, header_style="bold magenta")
        table.add_column("Regime", style="cyan")
        table.add_column("Structure", style="cyan")
        table.add_column("Count", justify="right", style="green")
        table.add_column("Avg Kelly Fractional", justify="right", style="yellow")
        table.add_column("Tickers", style="dim")
        
        for (regime, structure), row in summaries['by_regime_structure'].iterrows():
            table.add_row(
                str(regime),
                str(structure),
                str(int(row['Count'])),
                f"{row['Kelly_Fractional']:.4f}",
                row['Tickers']
            )
        console.print(table)
        console.print()
    
    # Combined: Regime + Gate State
    if 'by_regime_gate' in summaries:
        table = Table(title="Summary by Regime + Gate State", show_header=True, header_style="bold magenta")
        table.add_column("Regime", style="cyan")
        table.add_column("Gate State", style="cyan")
        table.add_column("Count", justify="right", style="green")
        table.add_column("Avg Kelly Fractional", justify="right", style="yellow")
        table.add_column("Tickers", style="dim")
        
        for (regime, gate_state), row in summaries['by_regime_gate'].iterrows():
            table.add_row(
                str(regime),
                str(gate_state),
                str(int(row['Count'])),
                f"{row['Kelly_Fractional']:.4f}",
                row['Tickers']
            )
        console.print(table)
        console.print()
    
    # Detail table
    if 'detail' in summaries:
        console.print("[bold cyan]Full Detail Table (sorted by Regime, Structure, Gate State)[/bold cyan]")
        detail_df = summaries['detail']
        
        # Print in chunks to avoid overwhelming the console
        chunk_size = 20
        for i in range(0, len(detail_df), chunk_size):
            chunk = detail_df.iloc[i:i+chunk_size]
            table = Table(show_header=True, header_style="bold")
            for col in chunk.columns:
                table.add_column(col, style="cyan" if col == "Ticker" else "white")
            
            for _, row in chunk.iterrows():
                table.add_row(*[str(val) for val in row])
            
            console.print(table)
            if i + chunk_size < len(detail_df):
                console.print(f"[dim]... showing {i+1}-{min(i+chunk_size, len(detail_df))} of {len(detail_df)}[/dim]\n")


def save_summary_files(summaries: dict, df: pd.DataFrame):
    """Save summary tables to CSV and markdown files."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save full combined data
    combined_csv = AGGREGATE_OUTPUT_DIR / f"all_tickers_combined_{timestamp}.csv"
    df.to_csv(combined_csv, index=False)
    console.print(f"[green]Saved combined data: {combined_csv}[/green]")
    
    # Save each summary table
    for name, summary_df in summaries.items():
        if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
            csv_path = AGGREGATE_OUTPUT_DIR / f"summary_{name}_{timestamp}.csv"
            summary_df.to_csv(csv_path)
            console.print(f"[green]Saved {name} summary: {csv_path}[/green]")
    
    # Create markdown summary report
    md_path = AGGREGATE_OUTPUT_DIR / f"summary_report_{timestamp}.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Markov Analysis Summary Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Tickers Analyzed:** {len(df)}\n\n")
        
        # By Regime
        if 'by_regime' in summaries:
            f.write("## Summary by Regime\n\n")
            f.write("| Regime | Count | Avg Kelly Fractional | Tickers |\n")
            f.write("|--------|-------|---------------------|----------|\n")
            for regime, row in summaries['by_regime'].iterrows():
                f.write(f"| {regime} | {int(row['Count'])} | {row['Kelly_Fractional']:.4f} | {row['Tickers']} |\n")
            f.write("\n")
        
        # By Structure
        if 'by_structure' in summaries:
            f.write("## Summary by Structure Family\n\n")
            f.write("| Structure | Count | Avg Kelly Fractional | Tickers |\n")
            f.write("|-----------|-------|---------------------|----------|\n")
            for structure, row in summaries['by_structure'].iterrows():
                f.write(f"| {structure} | {int(row['Count'])} | {row['Kelly_Fractional']:.4f} | {row['Tickers']} |\n")
            f.write("\n")
        
        # By Gate State
        if 'by_gate_state' in summaries:
            f.write("## Summary by Gate State\n\n")
            f.write("| Gate State | Count | Avg Kelly Fractional | Tickers |\n")
            f.write("|------------|-------|---------------------|----------|\n")
            for gate_state, row in summaries['by_gate_state'].iterrows():
                f.write(f"| {gate_state} | {int(row['Count'])} | {row['Kelly_Fractional']:.4f} | {row['Tickers']} |\n")
            f.write("\n")
        
        # By Market State
        if 'by_market_state' in summaries:
            f.write("## Summary by Market State\n\n")
            f.write("| Market State | Count | Avg Kelly Fractional | Tickers |\n")
            f.write("|--------------|-------|---------------------|----------|\n")
            for market_state, row in summaries['by_market_state'].iterrows():
                f.write(f"| {market_state} | {int(row['Count'])} | {row['Kelly_Fractional']:.4f} | {row['Tickers']} |\n")
            f.write("\n")
        
        # Combined tables
        if 'by_regime_structure' in summaries:
            f.write("## Summary by Regime + Structure\n\n")
            f.write("| Regime | Structure | Count | Avg Kelly Fractional | Tickers |\n")
            f.write("|--------|-----------|-------|---------------------|----------|\n")
            for (regime, structure), row in summaries['by_regime_structure'].iterrows():
                f.write(f"| {regime} | {structure} | {int(row['Count'])} | {row['Kelly_Fractional']:.4f} | {row['Tickers']} |\n")
            f.write("\n")
        
        if 'by_regime_gate' in summaries:
            f.write("## Summary by Regime + Gate State\n\n")
            f.write("| Regime | Gate State | Count | Avg Kelly Fractional | Tickers |\n")
            f.write("|--------|------------|-------|---------------------|----------|\n")
            for (regime, gate_state), row in summaries['by_regime_gate'].iterrows():
                f.write(f"| {regime} | {gate_state} | {int(row['Count'])} | {row['Kelly_Fractional']:.4f} | {row['Tickers']} |\n")
            f.write("\n")
        
        # Full detail table
        if 'detail' in summaries:
            f.write("## Full Detail Table\n\n")
            detail_df = summaries['detail']
            # Convert to markdown table manually for compatibility
            cols = detail_df.columns.tolist()
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("|" + "|".join(["---" for _ in cols]) + "|\n")
            for _, row in detail_df.iterrows():
                f.write("| " + " | ".join([str(val) for val in row]) + " |\n")
            f.write("\n\n")
    
    console.print(f"[green]Saved markdown report: {md_path}[/green]")


def main():
    """Main function to aggregate and display reports."""
    console.print("[bold magenta]>>> Markov Analysis Report Aggregator[/bold magenta]\n")
    
    # Load all reports
    console.print("[bold yellow]Loading ticker reports...[/bold yellow]")
    df = load_all_reports()
    
    if df.empty:
        console.print("[red]No data to aggregate. Exiting.[/red]")
        return
    
    # Generate summaries
    console.print("\n[bold yellow]Generating summary tables...[/bold yellow]")
    summaries = generate_summary_tables(df)
    
    # Print to console
    console.print("\n[bold cyan]Summary Tables:[/bold cyan]\n")
    print_rich_tables(summaries)
    
    # Save to files
    console.print("\n[bold yellow]Saving summary files...[/bold yellow]")
    save_summary_files(summaries, df)
    
    console.print(f"\n[bold green]✓ Aggregation complete![/bold green]")
    console.print(f"[dim]Files saved to: {AGGREGATE_OUTPUT_DIR}[/dim]")


if __name__ == "__main__":
    main()
