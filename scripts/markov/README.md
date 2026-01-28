# Markov Blanket Analysis for Option Pricing

A comprehensive Python implementation of Markov blanket discovery for Bayesian networks applied to option pricing prediction. This tool identifies the minimal set of variables that render the option premium conditionally independent of all other variables, then uses machine learning to build predictive models with Kelly Gate permission layers and state machine-driven trading decisions.

## Overview

This project implements a Bayesian network approach to option pricing that goes beyond traditional models like Black-Scholes by identifying causal relationships between market variables. The Markov blanket of the "option premium" node provides an optimal feature set for prediction, capturing market inefficiencies, sentiment-driven effects, and feedback loops that traditional models miss.

The system includes:
- **Markov Blanket Discovery**: Optimal feature selection via Bayesian networks
- **Machine Learning Models**: LightGBM-based predictive models with residual decomposition
- **Kelly Gate**: Permission layer for option trading with regime inference
- **Markov Engine** (v10): Explicit Markov chain analysis for regime/gate state transitions
- **State Machine**: Explicit market state classification (PIN, RANGE, TREND, RUPTURE, etc.)
- **Reflexive Bifurcation Sleeve**: Nested leg planning for option strategies
- **Markov Masks**: Contract-level agency encoding
- **Batch Processing**: Universal mode for processing multiple tickers

## What is a Markov Blanket?

In a Bayesian network, the **Markov blanket** of a target node is the minimal set of nodes that renders the target conditionally independent of all other nodes in the network. For a target node $P$ (option premium), the Markov blanket consists of:

- **Parents** (direct causes): Variables that directly influence $P$
- **Children** (direct effects): Variables that $P$ directly influences  
- **Spouses** (co-parents): Other parents of $P$'s children

This set shields $P$ from irrelevant variables and serves as an optimal feature set for prediction, helping avoid overfitting while capturing essential dependencies.

## Features

### Core Analysis
- ✅ **Markov Blanket Discovery**: Automatically computes the Markov blanket for the option premium node
- ✅ **Bayesian Network Visualization**: Generates network graphs with highlighted Markov blanket nodes
- ✅ **Real Option Data Analysis**: Loads and analyzes actual option chain data
- ✅ **Premium Decomposition**: Separates classical (Black-Scholes) from extra (volume/news) components
- ✅ **Skew Attribution**: Analyzes put-call IV differences and volatility skew

### Machine Learning
- ✅ **LightGBM Models**: Classical, full, and residual models with cross-validation
- ✅ **Feature Engineering**: Moneyness, log transforms, interaction terms
- ✅ **SHAP Integration**: Model interpretability with SHAP plots (optional)
- ✅ **Model Persistence**: Save/load trained models

### Trading Decision Framework
- ✅ **Kelly Gate**: Regime inference (PIN/EXPRESSIVE/RUPTURE_CANDIDATE), structure family recommendation, Kelly sizing
- ✅ **Markov Engine** (v10): Discrete-time transition matrix P, stationary distribution π, continuous-time generator Q, Kolmogorov flow evolution
- ✅ **State Machine**: Explicit market state classification with action policies
- ✅ **Reflexive Bifurcation Sleeve**: Nested leg planning for option strategies
- ✅ **Markov Masks**: Contract-level agency encoding (SENSITIVE/EXPRESSIVE/DORMANT)

### Batch Processing & Reporting
- ✅ **Universal Mode**: Process all tickers from `tickers.json` in batch
- ✅ **Report Generation**: Automatic markdown and CSV reports per ticker
- ✅ **Aggregate Reports**: Summary tables grouped by regime, structure, gate state, etc.

## Installation

See [SETUP.md](SETUP.md) for detailed installation instructions.

### Quick Install

```bash
cd metascripts/markov
pip install -r requirements.txt
```

### Required Dependencies

- `numpy`, `pandas`, `matplotlib`, `scipy`
- `scikit-learn`, `lightgbm`
- `rich`, `mplcyberpunk`
- `networkx` (for testA.py)

### Optional Dependencies

- `shap` - For SHAP plots and model interpretability
- `seaborn` - Enhanced visualizations

## Usage

### Basic Usage (testA.py - Network Analysis)

```bash
cd metascripts/markov
python testA.py
```

The script will:
1. Display the Markov blanket analysis for option pricing
2. Load the most recent option chain data
3. Show a summary table of strikes near the current price
4. Save detailed option chain data to `../output/` directory
5. Generate a network visualization (`markov_blanket_network.png`)

### Advanced Usage (testB.py - Full Analysis)

#### Single Ticker Analysis

```bash
# Basic run
python testB.py --ticker SPY

# With specific date
python testB.py --ticker SPY --date 2025-01-19

# With custom capital for reflexive sleeve
python testB.py --ticker SPY --capital 50000

# Skip SHAP plots for faster execution
python testB.py --ticker SPY --skip-shap

# Debug mode (detailed Kelly Gate output)
python testB.py --ticker SPY --debug
```

### Markov Engine Integration (testC.py - v10)

**testC.py** is an enhanced, better-organized version of testB.py with explicit Markov chain engine integration:

#### Key Features
- **Discrete-time Markov chain**: Transition matrix P estimation from historical state sequences
- **Stationary distribution π**: Computes ergodic occupation frequencies
- **Continuous-time generator Q**: Kolmogorov forward equation dp/dt = pQ
- **Markov risk flags**: STABLE_PIN, DRIFTING_EXPRESSIVE, RUPTURE_DRIFT, TRANSIENT_UNCERTAIN
- **Kelly modifiers**: Adjusts Kelly fractions based on π and flow diagnostics
- **Enhanced reporting**: New "Markov Engine" and "Kolmogorov Flow" sections

#### Single Ticker with Markov Engine

```bash
# Basic run (will note insufficient history if < 3 states)
python testC.py --ticker SPY

# With historical state file
python testC.py --ticker SPY --markov-history-file history.json

# Custom Kolmogorov flow horizon
python testC.py --ticker SPY --markov-horizon 10.0

# Custom rate scale for generator
python testC.py --ticker SPY --markov-rate-scale 1.5
```

#### Universal Mode with Markov Engine

```bash
# Builds state history incrementally across tickers
python testC.py --universal --markov-horizon 5.0

# With custom rate scale
python testC.py --universal --markov-horizon 5.0 --markov-rate-scale 1.0
```

**Markov Engine Arguments:**
- `--markov-horizon FLOAT` - Kolmogorov flow horizon (default: 5.0)
- `--markov-rate-scale FLOAT` - Generator rate scale (default: 1.0)
- `--markov-history-file PATH` - JSON file with historical state sequence

**State History Format:**
```json
{
  "states": ["PIN", "PIN", "EXPRESSIVE", "PIN", "RUPTURE_CANDIDATE"]
}
```

The system automatically saves state history to `markov_state_history.json` after each run.

#### Universal Mode (Batch Processing)

Process all tickers from `tickers.json`:

```bash
python testB.py --universal
```

This will:
- Load all tickers from `tickers.json`
- Process each ticker individually
- Save reports to `output/markov/{TICKER}/` subdirectories
- Generate `{TICKER}_report.md` and `{TICKER}_report.csv` for each ticker
- Display progress and summary statistics

**Universal Mode Options:**
```bash
# Universal mode with custom capital
python testB.py --universal --capital 100000

# Universal mode with force reflexive (override gates)
python testB.py --universal --force-reflexive
```

#### Command-Line Arguments

**Basic Options:**
- `--ticker SPY` - Stock ticker symbol (default: SPY)
- `--date YYYY-MM-DD` - Option chain date (default: most recent)
- `--folds 5` - Number of cross-validation folds (default: 5)
- `--output DIR` - Custom output directory (default: dated folder)

**Analysis Options:**
- `--skip-shap` - Skip SHAP plots for faster execution
- `--use-pca` - Use PCA on classical features (experimental)
- `--no-log-target` or `--raw-premium` - Use raw premium instead of log(premium+1)
- `--debug` - Print debug information for Kelly Gate

**Testing Options:**
- `--sanity-test-masks` - Run sanity test: temporarily set MASK_MAX_DTE=7
- `--test-expressive` - Test EXPRESSIVE masks: reduce thresholds or bypass PIN damping

**Trading Options:**
- `--capital FLOAT` - Total portfolio capital K (default: 10000.0)
- `--force-reflexive` - Force reflexive sleeve generation even if gate is BLOCK

**Batch Processing:**
- `--universal` - Process all tickers from tickers.json

### Aggregate Reports (aggregate_reports.py)

Generate summary tables from all ticker reports:

```bash
python aggregate_reports.py
```

This will:
- Load all `{TICKER}_report.csv` files from `output/markov/{TICKER}/`
- Generate summary tables grouped by:
  - Regime
  - Structure Family
  - Gate State
  - Market State
  - Regime + Structure (combined)
  - Regime + Gate State (combined)
- Save aggregated reports to `output/markov/aggregated/`
- Display Rich tables in console

**Output Files:**
- `all_tickers_combined_{timestamp}.csv` - All ticker data combined
- `summary_{name}_{timestamp}.csv` - Individual summary tables
- `summary_report_{timestamp}.md` - Markdown report with all summaries

## What testB.py Does

The full analysis pipeline includes:

1. **Data Loading**: Loads option chain data and stock prices
2. **Feature Preparation**: Extracts classical (Black-Scholes) and full (Markov blanket) features
3. **Model Training**: 
   - Classical model (5 features: S, σ, r, τ, K)
   - Full model (8+ features: adds V, N, interactions)
   - Residual model (trained on residuals to isolate volume/news effects)
4. **Premium Decomposition**: Separates premium into classical vs. extra components
5. **Skew Attribution**: Analyzes put-call IV differences using Ridge regression
6. **Kelly Gate**: 
   - Regime inference (PIN/EXPRESSIVE/RUPTURE_CANDIDATE)
   - Structure family recommendation (PROBE_ONLY/NORMAL/EXPLOIT)
   - Kelly sizing with multipliers
   - Gate state determination (BLOCK/OPEN/PARTIAL)
7. **State Machine**: Maps signals to explicit market states with action policies
8. **Reflexive Sleeve**: Generates nested leg plans when permitted
9. **Markov Masks**: Contract-level agency encoding
10. **Visualizations**: SHAP plots, correlation heatmaps, analysis charts
11. **Report Generation**: Saves models, JSON, CSV, and markdown reports

## What testC.py Does (v10 - Markov Engine)

**testC.py** extends testB.py with explicit Markov chain analysis:

1. **All testB.py features** (steps 1-11 above)
2. **Markov Engine Integration**:
   - Estimates transition matrix P from historical regime/gate state sequences
   - Computes stationary distribution π solving π = πP
   - Estimates continuous-time generator Q from P
   - Evolves Kolmogorov flow: dp/dt = pQ for short-horizon regime forecasts
   - Computes Markov risk flags based on π and flow diagnostics
   - Applies Kelly modifiers based on Markov diagnostics
3. **Enhanced Reporting**:
   - New "Markov Engine" section: P, π, occupation frequencies, KL divergence
   - New "Kolmogorov Flow" section: p(0), p(t) for multiple horizons, flow interpretation
   - Enhanced narrative with references to "memory without collapse" and regime stability
4. **State History Management**:
   - Automatically builds state history from regime observations
   - Saves/loads state history from JSON files
   - Incremental history building in universal mode

## File Structure

```
markov/
├── README.md                    # This file
├── SETUP.md                     # Installation guide
├── requirements.txt             # Python dependencies
├── testA.py                     # Network analysis (Markov blanket discovery)
├── testB.py                     # Full predictive model (ML + Kelly Gate + State Machine)
├── testC.py                     # Enhanced version with Markov Engine (v10)
├── markov_engine.py             # Markov chain engine module
├── aggregate_reports.py          # Batch report aggregator
├── markov_mask.py               # Markov Masks module (contract-level agency)
├── reflexive_bifurcation.py     # Reflexive sleeve planning
├── state_machine.py             # Market state classification
├── data_loader.py               # Data loading utilities
├── test_viz.py                  # Visualization test script
├── dev/                         # Development documentation
│   ├── KELLY_GATE_README.md
│   ├── MARKOV_MASKS_README.md
│   ├── outlineA.md
│   └── outlineB.md
└── wiki.md                      # Theoretical background
```

## Output Files

### Single Ticker Mode

Results are saved to `../output/analysis_{TIMESTAMP}/`:

- **Models**: `classical_model.pkl`, `full_model.pkl`, `residual_model.pkl`, `skew_model.pkl`
- **Data**: `contract_scores.csv`, `markov_masks.csv`, `markov_masks.json`
- **Gate Results**: `gate.json`, `reflexive_plan.json`
- **Markov Engine** (testC.py only): `markov_state_history.json`
- **Visualizations**: 
  - `feature_correlation_{timestamp}.png`
  - `markov_blanket_analysis_{timestamp}.png`
  - `shap_*.png` (if SHAP is installed)
- **Logs**: `analysis_{timestamp}.log`

### Universal Mode

Results are saved to `../output/markov/{TICKER}/`:

- `{TICKER}_report.md` - Markdown report with full analysis
- `{TICKER}_report.csv` - CSV with key metrics
- `reflexive_plan.json` - Reflexive sleeve plan (if applicable)

### Aggregate Reports

Results are saved to `../output/markov/aggregated/`:

- `all_tickers_combined_{timestamp}.csv` - All ticker data
- `summary_{name}_{timestamp}.csv` - Summary tables
- `summary_report_{timestamp}.md` - Markdown summary report

## Bayesian Network Structure

The implementation uses a 10-node Bayesian network for option pricing:

| Node | Variable | Description |
|------|----------|-------------|
| 0 | Spot_Price | Current stock price |
| 1 | Volatility | Market volatility |
| 2 | Interest_Rate | Risk-free interest rate |
| 3 | Time_to_Expiration | Time until option expiry |
| 4 | Strike_Price | Option strike price |
| 5 | Market_Sentiment | Market sentiment indicators |
| 6 | Economic_Indicators | Economic data |
| 7 | **Option_Premium** | **Target variable** |
| 8 | Trading_Volume | Option trading volume |
| 9 | News | News events |

### Markov Blanket for Option Premium

The Markov blanket of the Option_Premium node (node 7) includes:

- **Parents (5)**: Spot_Price, Volatility, Interest_Rate, Time_to_Expiration, Strike_Price
- **Children (1)**: Trading_Volume
- **Spouses (1)**: News

**Total: 7 variables** that fully determine the option premium.

### Comparison to Black-Scholes

Traditional Black-Scholes uses only the 5 parent variables. The Markov blanket approach adds:
- **Trading_Volume** (child) - Captures market activity and liquidity effects
- **News** (spouse) - Captures sentiment-driven volatility and external shocks

These additional variables help capture market inefficiencies that Black-Scholes misses.

## State Machine

The state machine maps current market signals into explicit states:

- **PIN**: Hard block, no directional trades, probes only
- **RANGE**: Pinned but with some permission, light probes allowed
- **TREND**: Directional entries allowed with modest reflexive sleeve
- **RUPTURE_PREP**: Small seed positions and staged reflexive sleeve
- **RUPTURE_ACTIVE**: Full reflexive sleeve allowed within Kelly and risk caps
- **COOLDOWN**: Stand down, no new risk

States are derived from regime, Kelly fraction, and gate state.

## Markov Engine (v10 - testC.py)

The Markov engine treats regime/gate states as a Markov chain, providing both discrete-time and continuous-time analysis:

### Discrete-Time Lens
- **Transition Matrix P**: Estimated from historical state sequences
- **Stationary Distribution π**: Solves π = πP using power iteration
- **Ergodic Occupation**: Empirical occupation frequencies vs. π
- **Mixing Time**: Convergence rate of Pⁿ to π

### Continuous-Time Lens
- **Generator Q**: Estimated from P via Q = rate_scale × (P - I)
- **Kolmogorov Flow**: Numerically integrates dp/dt = pQ
- **Short-Horizon Forecasts**: p(t) at multiple horizons (e.g., t=1, t=5)

### Risk Flags
- **STABLE_PIN**: π[PIN] > 0.7 and flow remains PIN-dominant → very conservative
- **DRIFTING_EXPRESSIVE**: Probability leaking from PIN to EXPRESSIVE → moderate
- **RUPTURE_DRIFT**: Significant rupture probability in flow → allow higher Kelly but still gated
- **TRANSIENT_UNCERTAIN**: Early or unstable regime → conservative default

### Integration with Kelly Gate
- Markov risk flags feed into gate state decisions
- Kelly modifiers (0.1-0.8) multiply existing Kelly fractions
- STABLE_PIN forces gate to BLOCK/THROTTLED
- Flow diagnostics inform regime stability assessment

## Reflexive Bifurcation Sleeve

When market conditions permit (TREND, RUPTURE_PREP, or RUPTURE_ACTIVE states), the system can generate nested leg plans:

- **Leg 1**: Initial direction (call/put) with base sizing
- **Leg 2+**: Flipped direction with progressive sizing based on previous leg's stop loss
- **Sizing**: Kelly-limited with configurable cap (default: 20% of capital)
- **Stop Loss**: Per-leg fractional stops (default: 9.7%)

## Data Requirements

The script expects data in the following structure:

**Stock Data:**
- Location: `F:/inputs/stocks/`
- Format: CSV files with columns: `Date`, `Close/Last`, `Volume`, `Open`, `High`, `Low`
- Example: `F:/inputs/stocks/SPY.csv`

**Option Chain Data:**
- Location: `F:/inputs/options/log/{ticker}/`
- Format: CSV files named `{ticker}_quotedata.csv` in date directories
- Example: `F:/inputs/options/log/spy/12_22_2025/spy_quotedata.csv`

You can modify the base directories in `data_loader.py` if your data is stored elsewhere.

## Example Output

### Terminal Summary

```
>>> Markov Blanket-Driven Option Pricing: Predictive Model

>>> Loading Option Data
Loaded 1247 option contracts for SPY
Stock price: $684.83

>>> Training Models
Classical model - CV MAE: $0.33, CV R²: -0.280
Full model - CV MAE: $0.33, CV R²: -0.280
Combined (classical + residual) - Test MAE: $0.20, R²: 0.170

>>> Teixiptla-Garage-Markov Kelly Gate
Regime: PIN
Structure: PROBE_ONLY
Kelly (fractional): 0.0000
Gate state: BLOCK

>>> State Machine
Current state: PIN
Actions: No directional trades; probes only, no reflexive sleeve.

>>> Reflexive Bifurcation Sleeve
Gate: BLOCKED
Reason: Kelly Gate or Teixiptla regime does not permit reflexive nesting.
```

### Universal Mode Summary

```
UNIVERSAL MODE SUMMARY
================================================================================
Successfully processed: 3/39 tickers

Successful: SPY, QQQ, IWM

Failed: DIA, GLD, SLV, ...

Reports saved to: ..\output\markov
```

## Theoretical Background

### Markov Condition

The concept of a Markov blanket is rooted in the **Markov condition**, which states that in a probabilistic graphical model, each variable is conditionally independent of its non-descendants given its parents. This condition implies the existence of a minimal separating set—the Markov blanket—that shields a variable from the rest of the network.

### Why This Approach?

1. **Data-Driven**: Unlike Black-Scholes which relies on ideal assumptions, this approach learns from actual market data
2. **Feature Selection**: The Markov blanket provides an optimal feature set, avoiding irrelevant variables
3. **Causal Understanding**: Reveals causal relationships between market variables
4. **Market Inefficiencies**: Can capture effects like volatility smiles, fat tails, and sentiment-driven movements
5. **Permission Layers**: Kelly Gate and State Machine provide explicit trading decision frameworks

### References

- Pearl, Judea (1988). *Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference*
- Statnikov, Alexander et al. (2013). "Algorithms for discovery of multiple Markov boundaries"
- See `wiki.md` and `dev/outlineA.md` for more detailed theoretical background

## Customization

### Changing the Network Structure

Edit the `adjacency_matrix` in `MarkovBlanketAnalyzer.__init__()` to modify the Bayesian network structure.

### Adjusting Minimum Data Requirements

The minimum number of samples required for training is set to 40 (configurable in `testB.py` line 417).

### Output Directories

- Single ticker: `../output/analysis_{TIMESTAMP}/`
- Universal mode: `../output/markov/{TICKER}/`
- Aggregate reports: `../output/markov/aggregated/`

## License

This project is part of the stock-monitor metascripts collection.

## Contributing

Feel free to extend this implementation with:
- Additional market variables
- Different machine learning models
- Enhanced visualization styles
- Support for additional option chains
- Custom state machine transitions

---

**Note**: This is a research/educational tool. Always verify results and use appropriate risk management when making trading decisions.
