# Markov Blanket Analysis for Option Pricing

A Python implementation of Markov blanket discovery for Bayesian networks applied to option pricing prediction. This tool identifies the minimal set of variables that render the option premium conditionally independent of all other variables in a probabilistic graphical model.

## Overview

This project implements a Bayesian network approach to option pricing that goes beyond traditional models like Black-Scholes by identifying causal relationships between market variables. The Markov blanket of the "option premium" node provides an optimal feature set for prediction, potentially capturing market inefficiencies, sentiment-driven effects, and feedback loops that traditional models miss.

## What is a Markov Blanket?

In a Bayesian network, the **Markov blanket** of a target node is the minimal set of nodes that renders the target conditionally independent of all other nodes in the network. For a target node $P$ (option premium), the Markov blanket consists of:

- **Parents** (direct causes): Variables that directly influence $P$
- **Children** (direct effects): Variables that $P$ directly influences  
- **Spouses** (co-parents): Other parents of $P$'s children

This set shields $P$ from irrelevant variables and serves as an optimal feature set for prediction, helping avoid overfitting while capturing essential dependencies.

## Features

- ✅ **Markov Blanket Discovery**: Automatically computes the Markov blanket for the option premium node
- ✅ **Bayesian Network Visualization**: Generates network graphs with highlighted Markov blanket nodes
- ✅ **Real Option Data Analysis**: Loads and analyzes actual SPY option chain data
- ✅ **Option Chain Display**: Shows detailed bid/ask spreads for strikes near current price
- ✅ **Data Export**: Saves option chain data to CSV and formatted text files
- ✅ **Beautiful Terminal Output**: Uses Rich library for professional console formatting
- ✅ **Cyberpunk Styling**: Modern visualization aesthetics with mplcyberpunk

## Installation

### Dependencies

```bash
pip install numpy pandas matplotlib mplcyberpunk rich networkx lightgbm scikit-learn scipy
```

**Optional (for SHAP plots):**
```bash
pip install shap
```

### Required Modules

- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `matplotlib` - Plotting
- `mplcyberpunk` - Cyberpunk-style visualizations
- `rich` - Beautiful terminal output
- `networkx` - Graph operations
- `lightgbm` - Gradient boosting regressor (for testB.py)
- `scikit-learn` - Machine learning utilities (for testB.py)
- `scipy` - Scientific computing (for testB.py)
- `shap` (optional) - SHAP plots for model interpretability

## Usage

### Basic Usage (testA.py - Network Analysis)

```bash
cd markov
python testA.py
```

The script will:
1. Display the Markov blanket analysis for option pricing
2. Load the most recent SPY option chain data
3. Show a summary table of strikes near the current price
4. Save detailed option chain data to `../output/` directory
5. Generate a network visualization (`markov_blanket_network.png`)

### Advanced Usage (testB.py - Predictive Model with Kelly Gate & Markov Masks)

**Quick Start:**
```bash
cd markov
python testB.py --ticker SPY
```

**With Sanity Test (verify mask engine):**
```bash
python testB.py --ticker SPY --sanity-test-masks
```

**With EXPRESSIVE Test (verify escalation):**
```bash
python testB.py --ticker SPY --test-expressive
```

**Full Options:**
```bash
python testB.py \
  --ticker SPY \
  --date 2025-12-28 \
  --folds 5 \
  --skip-shap \
  --debug \
  --sanity-test-masks \
  --test-expressive
```

**Command-Line Arguments:**
- `--ticker SPY` - Stock ticker symbol (default: SPY)
- `--date YYYY-MM-DD` - Option chain date (default: most recent)
- `--folds 5` - Number of cross-validation folds (default: 5)
- `--skip-shap` - Skip SHAP plots for faster execution
- `--use-pca` - Use PCA on classical features (experimental)
- `--no-log-target` or `--raw-premium` - Use raw premium instead of log(premium+1)
- `--debug` - Print debug information for Kelly Gate
- `--sanity-test-masks` - Run sanity test: temporarily set MASK_MAX_DTE=7 to verify top expressives cluster in nearest expiry
- `--test-expressive` - Test EXPRESSIVE masks: temporarily reduce thresholds or bypass PIN damping to verify escalation works

**What testB.py does:**
1. Trains ML regressors (LightGBM) on Markov blanket features
2. Decomposes premium into "classical" (Black-Scholes) and "extra" components
3. Analyzes skew attribution (put-call IV differences)
4. Computes Kelly Gate (regime inference, structure family, Kelly sizing)
5. Generates Markov Masks (contract-level agency encoding)
6. Saves models, visualizations, and analysis results to dated output folder

### Data Requirements

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

## File Structure

```
markov/
├── README.md              # This file
├── testA.py               # Network analysis script (Markov blanket discovery)
├── testB.py               # Predictive model script (ML + Kelly Gate + Markov Masks)
├── markov_mask.py         # Markov Masks module (contract-level agency)
├── kelly_gate.py          # Kelly Gate module (regime inference, sizing)
├── data_loader.py         # Data loading utilities
├── test_viz.py            # Visualization test script
├── wiki                   # Theoretical background on Markov blankets
├── outlineA.md            # Implementation outline and methodology
└── markov_blanket_network.png  # Generated network visualization
```

## Output Files

### Terminal Output

The script displays:
- **Markov Blanket Analysis**: Components (parents, children, spouses) and insights
- **Option Chain Summary**: Compact table showing strikes, bid/ask prices, moneyness, and volume
- **File Save Confirmation**: Paths to saved CSV and text files

### Generated Files

**Network Visualization:**
- `markov_blanket_network.png` - Bayesian network graph with highlighted Markov blanket

**Option Chain Data** (saved to `../output/`):
- `{TICKER}_option_chain_{TIMESTAMP}.csv` - Machine-readable option data
- `{TICKER}_option_chain_{TIMESTAMP}.txt` - Human-readable formatted report

### CSV Format

```csv
Strike,Call_Bid,Call_Ask,Call_Volume,Put_Bid,Put_Ask,Put_Volume,Moneyness,Total_Volume,Stock_Price
681.0,3.95,4.08,23567,0.0,0.01,142132,ITM,165699,684.83
685.0,0.03,0.04,930132,0.04,0.05,493488,ATM,1423620,684.83
```

### Text Report Format

The text file includes:
- Header with generation timestamp and stock price
- Detailed option chain table with all strikes
- Summary statistics (total volume, average spreads)

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

These additional variables can help capture market inefficiencies that Black-Scholes misses.

## Example Output

```
>>> Starting Markov Blanket Analysis for Option Pricing

+----------------------------- Analysis Overview ------------------------------+
| Markov Blanket Analysis for Option Pricing                                   |
|                                                                              |
| This analysis identifies the minimal set of variables that render the option |
| premium conditionally independent of all other variables in the Bayesian     |
| network.                                                                     |
+------------------------------------------------------------------------------+

                  Markov Blanket Components for Option_Premium
+------------------------------------------------------------------------------+
| Component                 | Variables                                | Count |
|---------------------------+------------------------------------------+-------|
| Parents (Direct Causes)   | Spot_Price, Volatility, Interest_Rate,   |     5 |
|                           | Time_to_Expiration, Strike_Price         |       |
| Children (Direct Effects) | Trading_Volume                           |     1 |
| Spouses (Co-parents)      | News                                     |     1 |
| Markov Blanket            | Spot_Price, Volatility, Interest_Rate,   |     7 |
|                           | Time_to_Expiration, Strike_Price,        |       |
|                           | Trading_Volume, News                     |       |
+------------------------------------------------------------------------------+

                   SPY Option Chain Summary
┏━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Strike ┃ Call Bid/Ask ┃ Put Bid/Ask ┃ Moneyness ┃    Volume ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━┩
│   $681 │ $3.95/$4.08  │ $0.00/$0.01 │    ITM    │   165,699 │
│   $682 │ $2.00/$3.06  │ $0.00/$0.01 │    ITM    │   341,873 │
│   $683 │ $1.30/$2.07  │ $0.00/$0.01 │    ITM    │   782,567 │
│   $684 │ $0.79/$1.05  │ $0.00/$0.01 │    ATM    │ 1,322,801 │
│   $685 │ $0.03/$0.04  │ $0.04/$0.05 │    ATM    │ 1,423,620 │
│   $686 │ $0.00/$0.01  │ $0.84/$1.05 │    OTM    │   530,491 │
│   $687 │ $0.00/$0.01  │ $1.92/$2.21 │    OTM    │   189,541 │
│   $688 │ $0.00/$0.01  │ $1.63/$3.03 │    OTM    │    53,211 │
│   $689 │ $0.00/$0.01  │ $2.63/$5.34 │    OTM    │    20,909 │
└────────┴──────────────┴─────────────┴───────────┴───────────┘

SUCCESS: Option chain data saved to:
  CSV: ..\output\SPY_option_chain_20251222_221107.csv
  Text: ..\output\SPY_option_chain_20251222_221107.txt
```

## Theoretical Background

### Markov Condition

The concept of a Markov blanket is rooted in the **Markov condition**, which states that in a probabilistic graphical model, each variable is conditionally independent of its non-descendants given its parents. This condition implies the existence of a minimal separating set—the Markov blanket—that shields a variable from the rest of the network.

### Why This Approach?

1. **Data-Driven**: Unlike Black-Scholes which relies on ideal assumptions, this approach learns from actual market data
2. **Feature Selection**: The Markov blanket provides an optimal feature set, avoiding irrelevant variables
3. **Causal Understanding**: Reveals causal relationships between market variables
4. **Market Inefficiencies**: Can capture effects like volatility smiles, fat tails, and sentiment-driven movements

### References

- Pearl, Judea (1988). *Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference*
- Statnikov, Alexander et al. (2013). "Algorithms for discovery of multiple Markov boundaries"
- See `wiki` and `outlineA.md` for more detailed theoretical background

## Customization

### Changing the Network Structure

Edit the `adjacency_matrix` in `MarkovBlanketAnalyzer.__init__()` to modify the Bayesian network structure.

### Adjusting Strike Range

Modify the strike selection logic in `display_option_chain_table()` to show more or fewer strikes around the current price.

### Output Directory

Change the `output_dir` path in `display_option_chain_table()` to save files to a different location.

## License

This project is part of the stock-monitor metascripts collection.

## Contributing

Feel free to extend this implementation with:
- Additional market variables
- Machine learning models using the Markov blanket features
- Different visualization styles
- Support for additional option chains

---

**Note**: This is a research/educational tool. Always verify results and use appropriate risk management when making trading decisions.