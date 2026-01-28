#!/usr/bin/env python3
"""
Markov Blanket-Driven Option Pricing: Predictive Model Implementation (UPGRADED v10 - Markov Engine Integration)

UPGRADES v10 (Markov Engine):
- Integrated explicit Markov chain engine for regime/gate state analysis
- Discrete-time transition matrix P and stationary distribution π
- Continuous-time generator Q and Kolmogorov forward equation dp/dt = pQ
- Markov risk flags and Kelly modifiers based on π and flow diagnostics
- New report sections: "Markov Engine" and "Kolmogorov Flow"
- Enhanced narrative generation with Markov diagnostics references

This version is better organized, optimized, and modularized compared to testB.py.

KEY FEATURES:
1. Markov Engine Module (markov_engine.py):
   - Estimates transition matrix P from historical state sequences
   - Computes stationary distribution π solving π = πP
   - Estimates continuous-time generator Q and evolves Kolmogorov flow
   - Computes Markov risk flags (STABLE_PIN, DRIFTING_EXPRESSIVE, RUPTURE_DRIFT, TRANSIENT_UNCERTAIN)
   - Provides Kelly fraction modifiers based on π and flow diagnostics

2. KellyGateMarkov Class:
   - Extends KellyGate with Markov engine integration
   - Automatically builds state history from regime observations
   - Applies Markov risk flags to gate state decisions
   - Modifies Kelly fractions based on stationary distribution and flow

3. Enhanced Reporting:
   - New "Markov Engine" section showing P, π, occupation frequencies, KL divergence
   - New "Kolmogorov Flow" section showing p(0) and p(t) for multiple horizons
   - Enhanced Teixiptla narrative with references to "memory without collapse" and Markov diagnostics

USAGE:
  # Single ticker (requires state history for full Markov analysis)
  python testC.py --ticker SPY --markov-history-file history.json

  # Universal mode (builds state history incrementally)
  python testC.py --universal --markov-horizon 5.0 --markov-rate-scale 1.0

  # With custom Kolmogorov flow horizon
  python testC.py --ticker SPY --markov-horizon 10.0

NOTES:
- Markov analysis requires at least 3 historical state observations
- State history is built incrementally in universal mode
- Single-ticker mode can use --markov-history-file to provide historical states
- Markov diagnostics appear in reports when sufficient history exists
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

# Import core classes from testB.py
from testB import (
    BlackScholesPricer, DataPreparator, OptionPricingPredictor, KellyGate,
    load_tickers, setup_logging, should_build_reflexive_sleeve,
    save_markdown_report, save_csv_report, save_reflexive_plan,
    process_single_ticker, DEFAULT_CAPITAL, REFLEXIVE_EXP_CAP_FRAC,
    REFLEXIVE_STOP_FRAC, REFLEXIVE_DTE_INITIAL, REFLEXIVE_MAX_LEGS,
    REFLEXIVE_INITIAL_DIR, console
)

# Import Markov engine
from markov_engine import (
    MarkovEngine, MarkovRiskFlag, RegimeState, GateState,
    kl_divergence, compute_mixing_time
)

# Import other dependencies
from data_loader import load_option_chain_data, get_latest_price, get_most_recent_option_date
from testA import MarkovBlanketAnalyzer
from markov_mask import MarkovMaskComputer
from reflexive_bifurcation import generate_reflexive_plan, LegPlan
from state_machine import MarketState, MarketSignals, compute_market_state, describe_actions
from rich.panel import Panel
from rich.table import Table


class KellyGateMarkov(KellyGate):
    """
    Extended Kelly Gate with Markov engine integration.
    
    Adds Markov chain analysis to the existing Kelly Gate logic:
    - Estimates transition matrix P from historical state sequences
    - Computes stationary distribution π
    - Evolves Kolmogorov flow for short-horizon regime forecasts
    - Applies Markov risk flags and Kelly modifiers
    """
    
    def __init__(self, logger=None, fractional_kelly=0.25, max_position_pct=0.05, debug=False,
                 markov_engine: Optional[MarkovEngine] = None,
                 historical_states: Optional[List[str]] = None):
        """
        Initialize Kelly Gate with Markov engine.
        
        Args:
            logger: Logger instance
            fractional_kelly: Fractional Kelly multiplier (default: 0.25)
            max_position_pct: Maximum position size as fraction of capital (default: 0.05)
            debug: Enable debug logging
            markov_engine: Optional MarkovEngine instance (creates new if None)
            historical_states: Optional historical state sequence for transition estimation
        """
        super().__init__(logger, fractional_kelly, max_position_pct, debug)
        
        # Initialize Markov engine
        self.markov_engine = markov_engine or MarkovEngine()
        self.historical_states = historical_states or []
        
        # Store Markov analysis results
        self.markov_results = None
    
    def add_historical_state(self, regime: str, gate: Optional[str] = None):
        """Add a state observation to historical sequence."""
        self.historical_states.append(regime)
    
    def compute_gate_with_markov(
        self,
        option_df,
        stock_price,
        current_date,
        call_data,
        put_data,
        decomposition_results,
        skew_results,
        training_results,
        rate_scale: float = 1.0,
        horizon: float = 5.0
    ) -> Dict:
        """
        Compute Kelly Gate with Markov engine integration.
        
        Args:
            option_df: Option chain DataFrame
            stock_price: Current stock price
            current_date: Current date
            call_data: Call option data
            put_data: Put option data
            decomposition_results: Premium decomposition results
            skew_results: Skew analysis results
            training_results: Model training results
            rate_scale: Time scale for generator (default: 1.0)
            horizon: Short-horizon time for Kolmogorov flow (default: 5.0)
        
        Returns:
            Extended gate_results dictionary with Markov diagnostics
        """
        # Compute base gate (from parent class)
        gate_results = self.compute_gate(
            option_df, stock_price, current_date, call_data, put_data,
            decomposition_results, skew_results, training_results
        )
        
        # Add current state to historical sequence
        current_regime = gate_results['regime']
        current_gate = gate_results.get('gate_state')
        self.add_historical_state(current_regime, current_gate)
        
        # Run Markov analysis if we have enough historical data
        if len(self.historical_states) >= 3:
            try:
                markov_analysis = self.markov_engine.analyze_state_sequence(
                    self.historical_states,
                    current_regime,
                    current_gate,
                    rate_scale=rate_scale,
                    horizon=horizon
                )
                
                # Apply Markov risk flag and Kelly modifier
                risk_flag = markov_analysis['risk_flag']
                kelly_modifier = markov_analysis['kelly_modifier']
                
                # Integrate Markov modifier into Kelly fraction
                kelly_fractional_original = gate_results['kelly_fractional']
                kelly_fractional_markov = kelly_fractional_original * kelly_modifier
                
                # Update gate state based on Markov diagnostics
                # If STABLE_PIN, force BLOCK or THROTTLED
                if risk_flag == MarkovRiskFlag.STABLE_PIN:
                    if gate_results['gate_state'] == 'OPEN':
                        gate_results['gate_state'] = 'THROTTLED'
                    elif gate_results['gate_state'] == 'DEPLOY':
                        gate_results['gate_state'] = 'BLOCK'
                
                # Update Kelly fractions
                gate_results['kelly_fractional_markov'] = kelly_fractional_markov
                gate_results['kelly_adjusted_markov'] = gate_results['kelly_adjusted'] * kelly_modifier
                
                # Store Markov results
                gate_results['markov_analysis'] = markov_analysis
                gate_results['markov_risk_flag'] = risk_flag.value
                gate_results['markov_kelly_modifier'] = kelly_modifier
                
                self.markov_results = markov_analysis
                
                if self.debug:
                    self.logger.info(
                        f"Markov analysis: risk_flag={risk_flag.value}, "
                        f"kelly_modifier={kelly_modifier:.3f}, "
                        f"kelly_fractional: {kelly_fractional_original:.4f} -> {kelly_fractional_markov:.4f}"
                    )
            except Exception as e:
                self.logger.warning(f"Error in Markov analysis: {e}")
                # Continue with base gate results
                gate_results['markov_analysis'] = None
                gate_results['markov_risk_flag'] = 'TRANSIENT_UNCERTAIN'
                gate_results['markov_kelly_modifier'] = 0.3  # Conservative default
        else:
            # Not enough historical data
            gate_results['markov_analysis'] = None
            gate_results['markov_risk_flag'] = 'TRANSIENT_UNCERTAIN'
            gate_results['markov_kelly_modifier'] = 0.3  # Conservative default
            if self.debug:
                self.logger.info(f"Insufficient historical states ({len(self.historical_states)} < 3) for Markov analysis")
        
        return gate_results


def format_markov_engine_section(markov_analysis: Dict) -> str:
    """
    Format Markov Engine section for report.
    
    Args:
        markov_analysis: Dictionary from MarkovEngine.analyze_state_sequence()
    
    Returns:
        Formatted ASCII text for Markov Engine section
    """
    if markov_analysis is None:
        return "Markov Engine: Insufficient historical data for analysis.\n"
    
    P = markov_analysis['P']
    pi = markov_analysis['pi']
    occupation = markov_analysis['occupation']
    risk_flag = markov_analysis['risk_flag']
    state_names = markov_analysis['state_names']
    state_to_idx = markov_analysis['state_to_idx']
    
    # Build state list
    state_list = ", ".join([f"{idx}: {name}" for idx, name in enumerate(state_names)])
    
    # Format transition matrix (compact, rounded to 2 decimals)
    P_str = "Transition matrix P (row = from, col = to):\n"
    for i, row in enumerate(P):
        row_str = " ".join([f"{val:5.2f}" for val in row])
        P_str += f"    [{row_str} ]\n"
    
    # Format stationary distribution
    pi_str = "Stationary distribution π:\n"
    for state_name in state_names:
        idx = state_to_idx.get(state_name, -1)
        if idx >= 0 and idx < len(pi):
            pi_str += f"    {state_name}: {pi[idx]*100:.1f}%\n"
    
    # Format empirical occupation vs π
    occupation_str = "Empirical occupation vs π:\n"
    for state_name in state_names:
        idx = state_to_idx.get(state_name, -1)
        if idx >= 0 and idx < len(pi):
            emp_pct = occupation[idx] * 100 if idx < len(occupation) else 0.0
            pi_pct = pi[idx] * 100
            occupation_str += f"    {state_name}: {emp_pct:.1f}% (emp) vs {pi_pct:.1f}% (π)\n"
    
    # Compute KL divergence
    kl = kl_divergence(occupation, pi)
    
    # Build full section
    section = (
        f"─────────────────────────── Markov Engine ───────────────────────────\n"
        f"States: [{state_list}]\n\n"
        f"{P_str}\n"
        f"{pi_str}\n"
        f"{occupation_str}\n"
        f"KL divergence (empirical || π): {kl:.4f} bits\n"
        f"Markov risk flag: {risk_flag.value}\n"
    )
    
    return section


def format_kolmogorov_flow_section(markov_analysis: Dict, horizons: List[float] = None) -> str:
    """
    Format Kolmogorov Flow section for report.
    
    Args:
        markov_analysis: Dictionary from MarkovEngine.analyze_state_sequence()
        horizons: List of time horizons to show (default: [1.0, 5.0])
    
    Returns:
        Formatted ASCII text for Kolmogorov Flow section
    """
    if markov_analysis is None:
        return "Kolmogorov Flow: Insufficient historical data for analysis.\n"
    
    if horizons is None:
        horizons = [1.0, 5.0]
    
    p0 = markov_analysis['p0']
    Q = markov_analysis['Q']
    state_names = markov_analysis['state_names']
    state_to_idx = markov_analysis['state_to_idx']
    engine = MarkovEngine(regime_states=state_names)
    
    # Format initial distribution
    p0_str = "Initial distribution p(0):\n"
    for state_name in state_names:
        idx = state_to_idx.get(state_name, -1)
        if idx >= 0 and idx < len(p0):
            p0_str += f"    {state_name}: {p0[idx]*100:.1f}%, "
    p0_str = p0_str.rstrip(", ") + "\n"
    
    # Evolve and format for each horizon
    flow_str = ""
    for horizon in horizons:
        p_t = engine.evolve_distribution(p0, Q, horizon)
        flow_str += f"\nProjected p(t={horizon}):\n"
        for state_name in state_names:
            idx = state_to_idx.get(state_name, -1)
            if idx >= 0 and idx < len(p_t):
                flow_str += f"    {state_name}: {p_t[idx]*100:.1f}%, "
        flow_str = flow_str.rstrip(", ") + "\n"
    
    # Generate interpretation
    interpretation = generate_flow_interpretation(markov_analysis, p_t)
    
    section = (
        f"───────────────────────── Kolmogorov Flow ─────────────────────────\n"
        f"{p0_str}"
        f"{flow_str}\n"
        f"Interpretation:\n"
        f"    {interpretation}\n"
    )
    
    return section


def generate_flow_interpretation(markov_analysis: Dict, p_t: np.ndarray) -> str:
    """
    Generate one-line interpretation of Kolmogorov flow.
    
    Args:
        markov_analysis: Dictionary from MarkovEngine.analyze_state_sequence()
        p_t: Final distribution p(t) from flow evolution
    
    Returns:
        Interpretation string
    """
    if markov_analysis is None:
        return "Insufficient data for flow interpretation."
    
    state_names = markov_analysis['state_names']
    state_to_idx = markov_analysis['state_to_idx']
    p0 = markov_analysis['p0']
    
    # Get indices
    pin_idx = state_to_idx.get("PIN", 0)
    expressive_idx = state_to_idx.get("EXPRESSIVE", 1)
    rupture_idx = state_to_idx.get("RUPTURE_CANDIDATE", 2)
    
    # Extract probabilities
    p0_pin = p0[pin_idx] if pin_idx < len(p0) else 0.0
    p_t_pin = p_t[pin_idx] if pin_idx < len(p_t) else 0.0
    p_t_expressive = p_t[expressive_idx] if expressive_idx < len(p_t) else 0.0
    p_t_rupture = p_t[rupture_idx] if rupture_idx < len(p_t) else 0.0
    
    # Generate interpretation
    if p_t_pin > 0.8:
        return "Regime probability remains PIN-dominant with minimal leakage, consistent with a stable market that jitters but does not rupture."
    elif p_t_pin > 0.5 and p_t_expressive + p_t_rupture < 0.2:
        return "Regime probability remains PIN-dominant with slow leakage into EXPRESSIVE, consistent with a market that jitters but does not yet rupture."
    elif p_t_expressive + p_t_rupture > 0.3:
        return "Regime probability shows significant flow toward EXPRESSIVE/RUPTURE states, indicating potential convexity permission and elevated market stress."
    elif p_t_pin < p0_pin and p_t_expressive > 0.1:
        return "Probability mass is flowing away from PIN toward EXPRESSIVE, suggesting the mask's stability is degrading and agency transfer may be imminent."
    else:
        return "Regime probability shows moderate flow with no dominant pattern, indicating transient uncertainty in market structure."


def display_markov_sections(markov_analysis: Optional[Dict], horizons: List[float] = None):
    """
    Display Markov Engine and Kolmogorov Flow sections in Rich console.
    
    Args:
        markov_analysis: Optional Markov analysis results
        horizons: List of time horizons for flow (default: [1.0, 5.0])
    """
    if markov_analysis is None:
        console.print("[yellow]Markov Engine: Insufficient historical data for analysis.[/yellow]")
        return
    
    if horizons is None:
        horizons = [1.0, 5.0]
    
    P = markov_analysis['P']
    pi = markov_analysis['pi']
    occupation = markov_analysis['occupation']
    risk_flag = markov_analysis['risk_flag']
    state_names = markov_analysis['state_names']
    state_to_idx = markov_analysis['state_to_idx']
    engine = MarkovEngine(regime_states=state_names)
    
    # Markov Engine Panel
    console.print("\n[bold yellow]>>> Markov Engine[/bold yellow]")
    
    # State list
    state_list = ", ".join([f"{idx}: {name}" for idx, name in enumerate(state_names)])
    console.print(f"[dim]States: [{state_list}][/dim]\n")
    
    # Transition matrix table
    P_table = Table(title="Transition Matrix P (row = from, col = to)", show_header=True)
    P_table.add_column("From →", style="cyan")
    for i, state_name in enumerate(state_names):
        P_table.add_column(state_name[:8], justify="right", style="green")
    
    for i, row in enumerate(P):
        row_values = [f"{val:.2f}" for val in row]
        P_table.add_row(state_names[i][:8], *row_values)
    console.print(P_table)
    
    # Stationary distribution
    pi_table = Table(title="Stationary Distribution π", show_header=True)
    pi_table.add_column("State", style="cyan")
    pi_table.add_column("Probability", justify="right", style="green")
    for state_name in state_names:
        idx = state_to_idx.get(state_name, -1)
        if idx >= 0 and idx < len(pi):
            pi_table.add_row(state_name, f"{pi[idx]*100:.1f}%")
    console.print(pi_table)
    
    # Empirical occupation vs π
    occupation_table = Table(title="Empirical Occupation vs π", show_header=True)
    occupation_table.add_column("State", style="cyan")
    occupation_table.add_column("Empirical", justify="right", style="yellow")
    occupation_table.add_column("Stationary (π)", justify="right", style="green")
    for state_name in state_names:
        idx = state_to_idx.get(state_name, -1)
        if idx >= 0 and idx < len(pi):
            emp_pct = occupation[idx] * 100 if idx < len(occupation) else 0.0
            pi_pct = pi[idx] * 100
            occupation_table.add_row(state_name, f"{emp_pct:.1f}%", f"{pi_pct:.1f}%")
    console.print(occupation_table)
    
    # KL divergence
    kl = kl_divergence(occupation, pi)
    console.print(f"[dim]KL divergence (empirical || π): {kl:.4f} bits[/dim]")
    console.print(f"[bold]Markov risk flag: {risk_flag.value}[/bold]\n")
    
    # Kolmogorov Flow Panel
    console.print("[bold yellow]>>> Kolmogorov Flow[/bold yellow]")
    
    p0 = markov_analysis['p0']
    Q = markov_analysis['Q']
    
    # Initial distribution
    p0_table = Table(title="Initial Distribution p(0)", show_header=True)
    p0_table.add_column("State", style="cyan")
    p0_table.add_column("Probability", justify="right", style="green")
    for state_name in state_names:
        idx = state_to_idx.get(state_name, -1)
        if idx >= 0 and idx < len(p0):
            p0_table.add_row(state_name, f"{p0[idx]*100:.1f}%")
    console.print(p0_table)
    
    # Evolved distributions
    for horizon in horizons:
        p_t = engine.evolve_distribution(p0, Q, horizon)
        p_t_table = Table(title=f"Projected p(t={horizon})", show_header=True)
        p_t_table.add_column("State", style="cyan")
        p_t_table.add_column("Probability", justify="right", style="green")
        for state_name in state_names:
            idx = state_to_idx.get(state_name, -1)
            if idx >= 0 and idx < len(p_t):
                p_t_table.add_row(state_name, f"{p_t[idx]*100:.1f}%")
        console.print(p_t_table)
    
    # Interpretation
    p_t_final = engine.evolve_distribution(p0, Q, horizons[-1])
    interpretation = generate_flow_interpretation(markov_analysis, p_t_final)
    interpretation_panel = Panel(interpretation, title="[bold cyan]Interpretation[/bold cyan]", border_style="cyan")
    console.print(interpretation_panel)


def generate_teixiptla_narrative_markov(
    gate_results: Dict,
    markov_analysis: Optional[Dict],
    df_masks: Optional[pd.DataFrame] = None
) -> str:
    """
    Generate Teixiptla narrative with Markov diagnostics references.
    
    Args:
        gate_results: Kelly Gate results dictionary
        markov_analysis: Optional Markov analysis results
        df_masks: Optional DataFrame with mask states
    
    Returns:
        Narrative text string
    """
    regime = gate_results.get('regime', 'UNKNOWN')
    gate_state = gate_results.get('gate_state', 'UNKNOWN')
    risk_flag_str = gate_results.get('markov_risk_flag', 'TRANSIENT_UNCERTAIN')
    
    # Get mask state information if available
    top_mask_state = 'UNKNOWN'
    if df_masks is not None and len(df_masks) > 0 and 'mask_state' in df_masks.columns:
        mask_state_counts = df_masks['mask_state'].value_counts().to_dict()
        non_dormant = {k: v for k, v in mask_state_counts.items() if k != 'DORMANT'}
        if non_dormant:
            top_mask_state = max(non_dormant.items(), key=lambda x: x[1])[0]
    
    # Build narrative with Markov references
    if risk_flag_str == 'STABLE_PIN':
        narrative = (
            "ATM contracts show elevated sensitivity without convexity permission. "
            "The Markov mask remains intact; memory without collapse traps the chain in PIN. "
            "Ergodic occupation frequencies confirm stable PIN dominance, with minimal probability "
            "leakage into expressive states. No EXPRESSIVE or RUPTURE_CANDIDATE states detected, "
            "indicating the system is operating within PIN constraints."
        )
    elif risk_flag_str == 'DRIFTING_EXPRESSIVE':
        narrative = (
            "Occupation measure is rotating toward expressive states; ergodic averages are "
            "re-weighting away from PIN. The Kolmogorov flow shows probability mass leaking from "
            "PIN into EXPRESSIVE, suggesting the mask's stability is degrading. "
            "Convexity permission may be emerging, but regime constraints still limit deployment."
        )
    elif risk_flag_str == 'RUPTURE_DRIFT':
        narrative = (
            "Kolmogorov flow shows increasing probability through rupture corridors; the mask's "
            "stability is degrading. Stationary distribution π indicates growing mass in "
            "RUPTURE_CANDIDATE states, with short-horizon flow confirming probability drift. "
            "The system is transitioning from PIN toward active rupture, requiring heightened "
            "risk management."
        )
    elif regime == "PIN" and top_mask_state == "SENSITIVE":
        narrative = (
            "ATM contracts show elevated sensitivity without convexity permission. "
            "The Markov mask remains intact; agency is present but suppressed. "
            "No EXPRESSIVE or RUPTURE_CANDIDATE states detected, indicating the system "
            "is operating within PIN constraints."
        )
    else:
        narrative = (
            f"Regime: {regime}, Markov risk flag: {risk_flag_str}. "
            "The Markov mask system is operational, with contract-level agency "
            "encoded in mask states and expressivity scores. Transition dynamics show "
            "transient uncertainty, requiring continued monitoring."
        )
    
    return narrative


def save_markdown_report_markov(
    ticker: str,
    training_results: dict,
    decomposition_results: dict,
    skew_results: dict,
    gate_results: dict,
    market_state: MarketState,
    signals: MarketSignals,
    reflexive_plan: list[LegPlan],
    capital_value: float,
    output_dir: Path
) -> Path:
    """
    Save markdown report with Markov Engine and Kolmogorov Flow sections.
    
    Extends the base save_markdown_report function with new sections.
    """
    # Call base function first
    md_path = save_markdown_report(
        ticker, training_results, decomposition_results, skew_results,
        gate_results, market_state, signals, reflexive_plan, capital_value, output_dir
    )
    
    # Append Markov sections
    markov_analysis = gate_results.get('markov_analysis')
    if markov_analysis is not None:
        with open(md_path, 'a', encoding='utf-8') as f:
            f.write("\n## Markov Engine\n\n")
            f.write(format_markov_engine_section(markov_analysis))
            
            f.write("\n## Kolmogorov Flow\n\n")
            f.write(format_kolmogorov_flow_section(markov_analysis))
    
    return md_path


def process_single_ticker_markov(
    ticker: str,
    args,
    output_dir: Path,
    logger: logging.Logger,
    analyzer: MarkovBlanketAnalyzer,
    predictor: OptionPricingPredictor,
    kelly_gate_markov: Optional[KellyGateMarkov] = None
) -> dict:
    """
    Process a single ticker with Markov engine integration.
    
    Similar to process_single_ticker but uses KellyGateMarkov.
    """
    # Import base function and adapt it
    from testB import process_single_ticker
    
    # For now, we'll call the base function and then enhance results
    # In a full refactor, we'd replace the KellyGate instantiation
    results = process_single_ticker(ticker, args, output_dir, logger, analyzer, predictor)
    
    # If we have a KellyGateMarkov instance, re-run gate computation with Markov
    if kelly_gate_markov is not None and results.get('gate_results'):
        try:
            # Re-compute gate with Markov (would need to reload data, but for now just enhance)
            # In practice, we'd modify process_single_ticker to accept a gate class
            markov_analysis = results['gate_results'].get('markov_analysis')
            if markov_analysis:
                # Results already have Markov analysis
                pass
        except Exception as e:
            logger.warning(f"Error enhancing results with Markov: {e}")
    
    return results


def main():
    """
    Main execution function with full Markov engine integration.
    
    This is an enhanced version that properly wires the Markov engine into the pipeline.
    """
    parser = argparse.ArgumentParser(
        description='Markov Blanket-Driven Option Pricing Model (v10 - Markov Engine)'
    )
    parser.add_argument('--ticker', type=str, default='SPY', help='Stock ticker symbol (default: SPY)')
    parser.add_argument('--date', type=str, default=None, help='Option chain date (YYYY-MM-DD), defaults to most recent')
    parser.add_argument('--model', type=str, default='lightgbm', choices=['lightgbm'], help='Model type (default: lightgbm)')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds (default: 5)')
    parser.add_argument('--output', type=str, default=None, help='Output directory (default: dated folder)')
    parser.add_argument('--skip-shap', action='store_true', help='Skip SHAP plots (faster execution)')
    parser.add_argument('--no-log-target', dest='log_target', action='store_false', default=True, help='Disable log(premium+1) target transformation')
    parser.add_argument('--debug', action='store_true', help='Print debug information for Kelly Gate and Markov Engine')
    parser.add_argument('--capital', type=float, default=None, help=f'Total portfolio capital K (default: {DEFAULT_CAPITAL:.2f})')
    parser.add_argument('--force-reflexive', action='store_true', help='Force reflexive sleeve generation')
    parser.add_argument('--universal', action='store_true', help='Process all tickers from tickers.json')
    parser.add_argument('--markov-horizon', type=float, default=5.0, help='Kolmogorov flow horizon (default: 5.0)')
    parser.add_argument('--markov-rate-scale', type=float, default=1.0, help='Markov generator rate scale (default: 1.0)')
    parser.add_argument('--markov-history-file', type=str, default=None, help='Path to JSON file with historical state sequence')
    
    args = parser.parse_args()
    
    console.print("[bold magenta]>>> Markov Blanket-Driven Option Pricing: v10 - Markov Engine Integration[/bold magenta]\n")
    
    # Load historical states if provided
    historical_states = []
    if args.markov_history_file and Path(args.markov_history_file).exists():
        try:
            with open(args.markov_history_file, 'r') as f:
                data = json.load(f)
                historical_states = data.get('states', [])
            console.print(f"[green]Loaded {len(historical_states)} historical states from {args.markov_history_file}[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load historical states: {e}[/yellow]")
    
    # Import necessary components from testB
    from testB import (
        load_option_chain_data, get_latest_price, get_most_recent_option_date
    )
    from testB import process_single_ticker as process_single_ticker_base
    
    # Handle universal mode
    if args.universal:
        console.print("[yellow]Universal mode: Processing all tickers with Markov engine integration[/yellow]")
        console.print("[yellow]Note: Markov analysis requires state history - will build incrementally[/yellow]\n")
        
        # Load all tickers
        try:
            tickers = load_tickers()
        except FileNotFoundError as e:
            console.print(f"[red]Error: {e}[/red]")
            return
        
        # Setup universal output directory
        universal_output_dir = Path("../output") / "markov" / f"universal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        universal_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logger = setup_logging(universal_output_dir)
        logger.info(f"Starting universal mode with {len(tickers)} tickers")
        
        # Shared Markov engine across tickers (builds state history)
        shared_markov_engine = MarkovEngine()
        shared_historical_states = historical_states.copy()
        
        successful = []
        failed = []
        
        for ticker in tickers:
            try:
                ticker_output_dir = universal_output_dir / ticker.upper()
                ticker_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Initialize components
                analyzer = MarkovBlanketAnalyzer()
                predictor = OptionPricingPredictor(analyzer, logger, model_type=args.model)
                
                # Create Markov-enhanced Kelly Gate with shared history
                kelly_gate_markov = KellyGateMarkov(
                    logger=logger,
                    debug=args.debug,
                    markov_engine=shared_markov_engine,
                    historical_states=shared_historical_states.copy()
                )
                
                # Process ticker (we'll need to modify process_single_ticker to accept kelly_gate_markov)
                # For now, use base function and enhance results
                results = process_single_ticker_base(
                    ticker, args, ticker_output_dir, logger, analyzer, predictor
                )
                
                # Enhance with Markov if we have gate results
                if results.get('gate_results') and results.get('success'):
                    try:
                        # Re-compute gate with Markov engine
                        # This is a simplified approach - full version would modify process_single_ticker
                        current_regime = results['gate_results']['regime']
                        shared_historical_states.append(current_regime)
                        
                        # If we have enough history, run Markov analysis
                        if len(shared_historical_states) >= 3:
                            markov_analysis = shared_markov_engine.analyze_state_sequence(
                                shared_historical_states,
                                current_regime,
                                results['gate_results'].get('gate_state'),
                                rate_scale=args.markov_rate_scale,
                                horizon=args.markov_horizon
                            )
                            results['gate_results']['markov_analysis'] = markov_analysis
                            results['gate_results']['markov_risk_flag'] = markov_analysis['risk_flag'].value
                            results['gate_results']['markov_kelly_modifier'] = markov_analysis['kelly_modifier']
                    except Exception as e:
                        logger.warning(f"Error enhancing {ticker} with Markov: {e}")
                
                # Save enhanced reports
                if results.get('success'):
                    try:
                        save_markdown_report_markov(
                            ticker, results.get('training_results'), results.get('decomposition_results'),
                            results.get('skew_results', {'r2': 0.0}), results['gate_results'],
                            results['market_state'], results['signals'],
                            results.get('reflexive_plan', []), results.get('capital_value', DEFAULT_CAPITAL),
                            ticker_output_dir
                        )
                        successful.append(ticker)
                    except Exception as e:
                        logger.error(f"Error saving reports for {ticker}: {e}")
                        failed.append(ticker)
                else:
                    failed.append(ticker)
                    
            except Exception as e:
                console.print(f"[red]Error processing {ticker}: {e}[/red]")
                logger.error(f"Error processing {ticker}: {e}", exc_info=True)
                failed.append(ticker)
        
        # Save final state history
        history_path = universal_output_dir / "markov_state_history.json"
        with open(history_path, 'w') as f:
            json.dump({'states': shared_historical_states}, f, indent=2)
        console.print(f"[green]Saved state history to {history_path}[/green]")
        
        # Print summary
        console.print(f"\n{'='*80}")
        console.print("[bold green]UNIVERSAL MODE SUMMARY[/bold green]")
        console.print('='*80)
        console.print(f"Successfully processed: {len(successful)}/{len(tickers)} tickers")
        if successful:
            console.print(f"\n[green]Successful:[/green] {', '.join(successful)}")
        if failed:
            console.print(f"\n[red]Failed:[/red] {', '.join(failed)}")
        console.print(f"\n[dim]Reports saved to: {universal_output_dir}[/dim]")
        console.print('='*80)
        return
    
    # Single ticker mode with full Markov integration
    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path("../output") / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting Markov-enhanced analysis for {args.ticker}")
    
    # Import necessary components from testB
    from testB import (
        DataPreparator, load_option_chain_data, get_latest_price, get_most_recent_option_date
    )
    
    # Initialize analyzer and predictor
    analyzer = MarkovBlanketAnalyzer()
    predictor = OptionPricingPredictor(analyzer, logger, model_type=args.model)
    
    # Initialize Markov engine with historical states
    markov_engine = MarkovEngine()
    kelly_gate_markov = KellyGateMarkov(
        logger=logger,
        debug=args.debug,
        markov_engine=markov_engine,
        historical_states=historical_states.copy()
    )
    
    # Load data
    console.print("[bold yellow]>>> Loading Option Data[/bold yellow]")
    ticker = args.ticker.upper()
    
    try:
        if args.date:
            option_df = load_option_chain_data(ticker.lower(), date=args.date)
        else:
            recent_date = get_most_recent_option_date(ticker.lower())
            option_df = load_option_chain_data(ticker.lower())
        
        stock_price = get_latest_price(ticker)
        current_date = datetime.now()
        
        console.print(f"[green]Loaded {len(option_df)} option contracts for {ticker}[/green]")
        console.print(f"[green]Stock price: ${stock_price:.2f}[/green]")
        logger.info(f"Loaded {len(option_df)} contracts, stock price: ${stock_price:.2f}")
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        logger.error(f"Error loading data: {e}", exc_info=True)
        return
    
    # Prepare features
    console.print("\n[bold yellow]>>> Preparing Features[/bold yellow]")
    try:
        call_data, put_data = DataPreparator.prepare_features(
            option_df, stock_price, current_date=current_date, use_log_target=args.log_target
        )
        logger.info(f"Prepared features: {len(call_data['targets'])} call options")
    except Exception as e:
        console.print(f"[red]Error preparing features: {e}[/red]")
        logger.error(f"Error preparing features: {e}", exc_info=True)
        return
    
    # Plot feature correlation
    if not args.universal:  # Skip in universal mode
        console.print("\n[bold yellow]>>> Feature Correlation Analysis[/bold yellow]")
        predictor.plot_feature_correlation(call_data, output_dir)
    
    # Train models
    console.print("\n[bold yellow]>>> Training Models[/bold yellow]")
    training_results = predictor.train_models(call_data, put_data, n_folds=args.folds)
    
    if training_results is None:
        console.print("[red]Failed to train models[/red]")
        logger.error("Failed to train models")
        return
    
    # Decompose premium
    console.print("\n[bold yellow]>>> Premium Decomposition[/bold yellow]")
    decomposition_results = predictor.decompose_premium(call_data, put_data, training_results)
    
    # Analyze skew
    console.print("\n[bold yellow]>>> Skew Attribution Analysis[/bold yellow]")
    skew_results = predictor.analyze_skew_attribution(call_data, put_data)
    
    # Kelly Gate with Markov
    console.print("\n[bold yellow]>>> Teixiptla-Garage-Markov Kelly Gate (with Markov Engine)[/bold yellow]")
    reflexive_plan: list[LegPlan] = []
    market_state: MarketState = MarketState.PIN
    capital_value = args.capital if args.capital is not None else DEFAULT_CAPITAL
    
    try:
        # Use Markov-enhanced gate computation
        gate_results = kelly_gate_markov.compute_gate_with_markov(
            option_df, stock_price, current_date, call_data, put_data,
            decomposition_results, skew_results, training_results,
            rate_scale=args.markov_rate_scale,
            horizon=args.markov_horizon
        )
        
        # State Machine
        execution_quality = gate_results.get('execution_quality', {})
        skew_features = gate_results.get('skew_features', {})
        
        signals = MarketSignals(
            regime=gate_results['regime'],
            kelly_fraction=gate_results.get('kelly_fractional_markov', gate_results['kelly_fractional']),
            gate_state=gate_results['gate_state'],
            spread=execution_quality.get('bid_ask_spread_pct', 0.0) / 100.0 if execution_quality.get('bid_ask_spread_pct') else 0.0,
            quality=execution_quality.get('quality_score', 0.0),
            skew_slope=skew_features.get('skew_slope_puts', 0.0),
            curvature=skew_features.get('smile_curvature', 0.0),
        )
        
        market_state = compute_market_state(signals)
        
        # Reflexive sleeve
        if should_build_reflexive_sleeve(gate_results, market_state, force=args.force_reflexive):
            try:
                reflexive_plan = generate_reflexive_plan(
                    K=capital_value,
                    kelly_fraction=gate_results.get('kelly_fractional_markov', gate_results['kelly_fractional']),
                    exp_cap_frac=REFLEXIVE_EXP_CAP_FRAC,
                    stop_frac=REFLEXIVE_STOP_FRAC,
                    dte_initial=REFLEXIVE_DTE_INITIAL,
                    initial_direction=REFLEXIVE_INITIAL_DIR,
                    max_legs=REFLEXIVE_MAX_LEGS,
                )
            except Exception as e:
                logger.warning(f"Error generating reflexive sleeve: {e}")
                reflexive_plan = []
        
        # Display Markov sections if available
        markov_analysis = gate_results.get('markov_analysis')
        if markov_analysis:
            display_markov_sections(markov_analysis, horizons=[1.0, args.markov_horizon])
        else:
            console.print(f"[yellow]Markov analysis: Insufficient historical states ({len(kelly_gate_markov.historical_states)} < 3)[/yellow]")
            console.print("[yellow]Run multiple analyses or provide --markov-history-file to build state history[/yellow]\n")
        
    except Exception as e:
        console.print(f"[yellow]Error computing Kelly Gate: {e}[/yellow]")
        logger.warning(f"Error computing Kelly Gate: {e}", exc_info=True)
        gate_results = {
            'regime': 'PIN',
            'structure_family': 'PROBE_ONLY',
            'kelly_raw': 0.0,
            'kelly_fractional': 0.0,
            'kelly_adjusted': 0.0,
            'gate_state': 'BLOCK',
            'p': 0.5,
            'b': 1.0,
            'multiplier': 0.0,
            'markov_analysis': None,
            'markov_risk_flag': 'TRANSIENT_UNCERTAIN',
            'markov_kelly_modifier': 0.3
        }
        signals = MarketSignals(
            regime='PIN',
            kelly_fraction=0.0,
            gate_state='BLOCK',
            spread=0.0,
            quality=0.5,
            skew_slope=0.0,
            curvature=0.0,
        )
        market_state = compute_market_state(signals)
    
    # Visualize and save
    console.print("\n[bold yellow]>>> Generating Visualizations[/bold yellow]")
    
    # Enhance plot_feature_correlation to add explicit axis labels
    original_plot_corr = predictor.plot_feature_correlation
    def enhanced_plot_correlation(call_data, output_dir):
        """Enhanced version with explicit axis labels."""
        import matplotlib.pyplot as plt
        import mplcyberpunk
        from testB import SEABORN_AVAILABLE
        try:
            import seaborn as sns
        except ImportError:
            sns = None
        
        predictor.logger.info("Generating feature correlation heatmap...")
        
        X_full = call_data['full_features']
        valid = call_data['targets'] > 0.01
        X_full_valid = X_full[valid]
        
        df_features = pd.DataFrame(X_full_valid, columns=predictor.feature_names_full)
        corr_matrix = df_features.corr()
        
        plt.style.use("cyberpunk")
        fig, ax = plt.subplots(figsize=(12, 10))
        if SEABORN_AVAILABLE and sns:
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        else:
            im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            ax.set_xticks(np.arange(len(predictor.feature_names_full)))
            ax.set_yticks(np.arange(len(predictor.feature_names_full)))
            ax.set_xticklabels(predictor.feature_names_full, rotation=45, ha='right')
            ax.set_yticklabels(predictor.feature_names_full)
            for i in range(len(predictor.feature_names_full)):
                for j in range(len(predictor.feature_names_full)):
                    text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Add explicit axis labels
        ax.set_xlabel('Features', fontsize=12, fontweight='bold')
        ax.set_ylabel('Features', fontsize=12, fontweight='bold')
        ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        mplcyberpunk.add_glow_effects()
        
        output_path = output_dir / f"feature_correlation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='lightgray', edgecolor='none')
        console.print(f"[green]Correlation heatmap saved: {output_path}[/green]")
        predictor.logger.info(f"Correlation heatmap saved: {output_path}")
        plt.close()
    
    # Temporarily replace method
    predictor.plot_feature_correlation = enhanced_plot_correlation
    
    # Generate visualizations (all plots in visualize_results already have axis labels)
    predictor.visualize_results(decomposition_results, skew_results, training_results, output_dir, args)
    
    # Restore original
    predictor.plot_feature_correlation = original_plot_corr
    
    console.print("\n[bold yellow]>>> Saving Models[/bold yellow]")
    predictor.save_models(output_dir)
    
    # Save reports with Markov sections
    console.print("\n[bold yellow]>>> Saving Reports[/bold yellow]")
    save_markdown_report_markov(
        ticker, training_results, decomposition_results, skew_results,
        gate_results, market_state, signals, reflexive_plan, capital_value, output_dir
    )
    
    save_reflexive_plan(reflexive_plan, output_dir)
    
    # Save state history for future runs
    if len(kelly_gate_markov.historical_states) > 0:
        history_path = output_dir / "markov_state_history.json"
        with open(history_path, 'w') as f:
            json.dump({'states': kelly_gate_markov.historical_states}, f, indent=2)
        console.print(f"[green]Saved state history to {history_path}[/green]")
    
    # Summary (like testB.py)
    console.print("\n" + "="*80)
    improvement_pct = ((training_results['classical_cv_mae'].mean() - training_results['full_cv_mae'].mean()) / 
                       training_results['classical_cv_mae'].mean() * 100)
    extra_pct = (np.mean(decomposition_results['extra']) / np.mean(decomposition_results['actual']) * 100)
    residual_r2 = decomposition_results.get('residual_r2', 0.0)
    
    # Build summary with Kelly Gate and Markov diagnostics
    summary_text = (
        "[bold green]ANALYSIS COMPLETE![/bold green]\n\n"
        "The Markov blanket-driven approach has been successfully applied to option pricing.\n"
        "Key findings:\n"
        f"  • Baseline model MAE: ${training_results.get('baseline_mae', 0):.2f}\n"
        f"  • Classical model MAE: ${training_results['classical_mae']:.2f}\n"
        f"  • Combined model MAE: ${training_results['combined_mae']:.2f}\n"
        f"  • Improvement: {improvement_pct:.1f}% (CV MAE reduction)\n"
        f"  • Extra component (V, N) accounts for {extra_pct:.1f}% of premium\n"
        f"  • Extra variance explained (residual R²): {residual_r2:.3f} ({residual_r2*100:.1f}%)\n"
        f"  • Skew model R²: {skew_results['r2']:.3f} ({skew_results['r2']*100:.1f}% of variance)\n\n"
        "[bold cyan]Kelly Gate:[/bold cyan]\n"
        f"  • Regime: {gate_results['regime']}\n"
        f"  • Suggested structure: {gate_results['structure_family']}\n"
        f"  • Kelly (raw): {gate_results['kelly_raw']:.4f}\n"
        f"  • Kelly (fractional): {gate_results.get('kelly_fractional_markov', gate_results['kelly_fractional']):.4f}\n"
        f"  • Gate state: {gate_results['gate_state']}\n"
    )
    
    # Add Markov diagnostics if available
    markov_analysis = gate_results.get('markov_analysis')
    if markov_analysis:
        risk_flag = gate_results.get('markov_risk_flag', 'UNKNOWN')
        kelly_modifier = gate_results.get('markov_kelly_modifier', 1.0)
        summary_text += (
            f"  • Markov risk flag: {risk_flag}\n"
            f"  • Markov Kelly modifier: {kelly_modifier:.3f}\n"
        )
    
    summary_text += (
        f"  • Skew: put-call diff={gate_results['skew_features']['put_call_iv_diff']:.4f}, "
        f"slope={gate_results['skew_features']['skew_slope_puts']:.4f}, "
        f"curvature={gate_results['skew_features']['smile_curvature']:.4f}\n"
        f"  • Term: front={gate_results['term_features']['front_iv']:.4f}, "
        f"back={gate_results['term_features']['back_iv']:.4f}, "
        f"slope={gate_results['term_features']['term_slope']:.6f}, "
        f"inverted={gate_results['term_features']['is_inverted']}\n"
        f"  • Execution: spread={gate_results['execution_quality']['bid_ask_spread_pct']:.2f}%, "
        f"quality={gate_results['execution_quality']['quality_score']:.3f}\n\n"
    )
    
    # Add Reflexive Sleeve section
    if not reflexive_plan:
        reflexive_section = (
            "───────────────────── Reflexive Sleeve ─────────────────────\n"
            "Gate: BLOCKED\n"
            f"Reason: Kelly Gate or Teixiptla regime does not permit reflexive nesting.\n"
        )
    else:
        E0 = REFLEXIVE_EXP_CAP_FRAC * capital_value
        kelly_frac = gate_results.get('kelly_fractional_markov', gate_results['kelly_fractional'])
        E0_eff = min(E0, kelly_frac * capital_value)
        
        leg_lines = ["Leg  Dir   DTE(d)   Sleeve   Stop", "---  ----  ------   ------   ----"]
        for lp in reflexive_plan:
            leg_lines.append(
                f" {lp.leg}   {lp.direction.upper():4s}  {lp.dte:6.1f}  ${lp.sleeve_entry:7.2f}  ${lp.stop_loss:7.2f}"
            )
        leg_table = "\n".join(leg_lines)
        
        reflexive_section = (
            "───────────────────── Reflexive Sleeve ─────────────────────\n"
            f"Max sleeve cap: {REFLEXIVE_EXP_CAP_FRAC:.0%} of K (K=${capital_value:.2f})\n"
            f"Effective sleeve (Kelly-limited): ${E0_eff:.2f}\n\n"
            f"{leg_table}\n"
        )
    
    summary_text = summary_text + "\n" + reflexive_section + (
        f"\n[dim]Results saved to: {output_dir}[/dim]\n"
        "[dim]This demonstrates how Markov blanket features capture market inefficiencies[/dim]\n"
        "[dim]beyond traditional Black-Scholes assumptions.[/dim]"
    )
    
    summary_panel = Panel(summary_text, title="[bold]Summary[/bold]")
    console.print(summary_panel)
    
    # Kelly Gate panel (separate)
    gate_panel_text = (
        f"[bold]Regime:[/bold] {gate_results['regime']}\n"
        f"[bold]Structure:[/bold] {gate_results['structure_family']}\n"
        f"[bold]Kelly (raw):[/bold] {gate_results['kelly_raw']:.4f}\n"
        f"[bold]Kelly (fractional):[/bold] {gate_results.get('kelly_fractional_markov', gate_results['kelly_fractional']):.4f}\n"
        f"[bold]Kelly (adjusted):[/bold] {gate_results.get('kelly_adjusted_markov', gate_results['kelly_adjusted']):.4f}\n"
        f"[bold]Gate state:[/bold] {gate_results['gate_state']}\n"
        f"[bold]p:[/bold] {gate_results['p']:.3f}, [bold]b:[/bold] {gate_results['b']:.3f}\n"
        f"[bold]Multiplier:[/bold] {gate_results['multiplier']:.3f}"
    )
    if markov_analysis:
        gate_panel_text += (
            f"\n[bold]Markov risk flag:[/bold] {gate_results.get('markov_risk_flag', 'N/A')}\n"
            f"[bold]Markov Kelly modifier:[/bold] {gate_results.get('markov_kelly_modifier', 1.0):.3f}"
        )
    gate_panel = Panel(gate_panel_text, title="[bold cyan]Kelly Gate[/bold cyan]", border_style="cyan")
    console.print(gate_panel)
    
    # State Machine panel
    actions_text = describe_actions(market_state)
    state_machine_text = (
        f"[bold]Current state:[/bold] {market_state.value}\n"
        f"[bold]Actions:[/bold] {actions_text}\n"
        f"[bold]Notes:[/bold] Derived from regime={signals.regime}, gate={signals.gate_state}, "
        f"Kelly={signals.kelly_fraction:.4f}"
    )
    state_machine_panel = Panel(state_machine_text, title="[bold cyan]State Machine[/bold cyan]", border_style="cyan")
    console.print(state_machine_panel)
    
    # ASCII-style box
    console.print("\n╭──────────────────────────── State Machine ─────────────────────────────╮")
    state_line = f"│ Current state: {market_state.value:<57}│"
    console.print(state_line)
    if len(actions_text) > 63:
        words = actions_text.split()
        lines = []
        current_line = ""
        for word in words:
            test_line = current_line + (" " + word if current_line else word)
            if len(test_line) <= 63:
                current_line = test_line
            else:
                if current_line:
                    lines.append(f"│ Actions: {current_line:<63}│")
                current_line = word
        if current_line:
            lines.append(f"│ Actions: {current_line:<63}│")
        for line in lines:
            console.print(line)
    else:
        console.print(f"│ Actions: {actions_text:<63}│")
    notes_line = f"│ Notes: Derived from regime={signals.regime}, gate={signals.gate_state}, Kelly={signals.kelly_fraction:.4f}"
    max_notes_len = 75 - len("│ Notes: ")
    if len(notes_line) > 75:
        notes_line = notes_line[:72] + "..."
    notes_line = notes_line.ljust(75) + "│"
    console.print(notes_line)
    console.print("╰────────────────────────────────────────────────────────────────────────╯")
    
    # Generate and display narrative
    console.print("\n[bold yellow]>>> Narrative Summary[/bold yellow]")
    try:
        # Try to load masks if available
        masks_path = output_dir / "markov_masks.csv"
        df_masks = None
        if masks_path.exists():
            try:
                df_masks = pd.read_csv(masks_path)
            except:
                pass
        
        narrative = generate_teixiptla_narrative_markov(
            gate_results, gate_results.get('markov_analysis'), df_masks
        )
        narrative_panel = Panel(narrative, title="[bold cyan]Teixiptla Narrative[/bold cyan]", border_style="cyan")
        console.print(narrative_panel)
        logger.info(f"Narrative summary: {narrative}")
    except Exception as e:
        logger.warning(f"Error generating narrative: {e}")
        console.print("[dim]Could not generate narrative summary.[/dim]")
    
    logger.info("Analysis complete")


if __name__ == "__main__":
    main()
