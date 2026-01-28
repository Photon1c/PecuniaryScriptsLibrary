# Setup Guide for Markov Blanket Option Pricing

Complete installation and setup guide for the Markov Blanket analysis system.

## Prerequisites

- Python 3.8 or higher (tested with 3.10-3.12)
- pip package manager
- Access to option chain data (see Data Requirements section)

## Installation Steps

### Option 1: Install in Current Environment

```powershell
# Navigate to the markov directory
cd D:\SereneOcean\cursor_workspaces\stock-monitor\metascripts\markov

# Install all dependencies
pip install -r requirements.txt
```

### Option 2: Create a Virtual Environment (Recommended)

```powershell
# Navigate to the markov directory
cd D:\SereneOcean\cursor_workspaces\stock-monitor\metascripts\markov

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# If you get an execution policy error, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Using Conda (if you prefer)

```powershell
# Create conda environment
conda create -n markov-pricing python=3.10
conda activate markov-pricing

# Install dependencies
pip install -r requirements.txt
```

## Verify Installation

After installation, verify that all core packages are installed:

```powershell
python -c "import numpy, pandas, matplotlib, seaborn, lightgbm, sklearn, rich, mplcyberpunk, networkx; print('All core packages imported successfully!')"
```

For SHAP (optional but recommended):

```powershell
python -c "import shap; print('SHAP installed successfully!')"
```

## Running the Scripts

### testA.py - Network Analysis

Basic Markov blanket discovery and network visualization:

```powershell
python testA.py
```

### testB.py - Full Predictive Model

**Single Ticker:**
```powershell
python testB.py --ticker SPY
```

**Universal Mode (Batch Processing):**
```powershell
python testB.py --universal
```

**With Options:**
```powershell
python testB.py --ticker SPY --capital 50000 --debug --skip-shap
```

### testC.py - Markov Engine Integration (v10)

**Single Ticker:**
```powershell
# Basic run
python testC.py --ticker SPY

# With historical state file
python testC.py --ticker SPY --markov-history-file history.json

# Custom Kolmogorov flow horizon
python testC.py --ticker SPY --markov-horizon 10.0
```

**Universal Mode (Builds State History):**
```powershell
python testC.py --universal --markov-horizon 5.0
```

**Note:** Markov analysis requires at least 3 historical state observations. The system will note insufficient history if running for the first time. State history is saved automatically and can be reused in subsequent runs.

**State History Format:**
```json
{
  "states": ["PIN", "PIN", "PRE_TRANSFER", "PIN", "TRANSFER"]
}
```

States should be Kelly Gate regimes: `PIN`, `PRE_TRANSFER`, or `TRANSFER`.

### aggregate_reports.py - Report Aggregation

Generate summary tables from all ticker reports:

```powershell
python aggregate_reports.py
```

## Dependencies Summary

### Required Core Packages

- **numpy<2.0.0,>=1.21.0** - Numerical computing (NumPy 1.x for compatibility)
- **pandas>=1.5.0,<3.0.0** - Data manipulation
- **scipy>=1.9.0,<2.0.0** - Scientific computing
- **scikit-learn>=1.0.0,<2.0.0** - Machine learning utilities
- **lightgbm>=3.3.0,<5.0.0** - Gradient boosting regressor
- **matplotlib>=3.5.0,<4.0.0** - Plotting
- **seaborn>=0.12.0,<1.0.0** - Enhanced visualizations
- **mplcyberpunk>=0.1.0** - Cyberpunk-style visualizations
- **networkx>=2.8.0,<4.0.0** - Graph operations (for testA.py)
- **rich>=12.0.0,<14.0.0** - Beautiful terminal output

### Optional (but Recommended)

- **shap>=0.41.0,<0.50.0** - SHAP plots for model interpretability
  - Note: SHAP 0.50+ requires NumPy 2.0, but we use NumPy 1.x
  - If you get compatibility errors, try: `pip install "shap<0.50"`
  - The script will work without SHAP, but you'll miss interpretability plots

### Project-Specific Modules

These modules must be available in your Python path:

- **data_loader** - Data loading utilities (should be in parent directory or Python path)
- **testA** - MarkovBlanketAnalyzer class (should be in same directory as testB.py)

The following modules are part of this package and should be in the same directory:
- **markov_engine.py** - Markov chain engine (for testC.py)
- **markov_mask.py** - Markov Masks module
- **reflexive_bifurcation.py** - Reflexive sleeve planning
- **state_machine.py** - Market state classification

## Data Requirements

### Stock Data

**Location:** `F:/inputs/stocks/`  
**Format:** CSV files with columns:
- `Date` - Date column
- `Close/Last` or `Close` - Closing price
- `Volume` - Trading volume
- `Open`, `High`, `Low` - Additional price data (optional)

**Example:** `F:/inputs/stocks/SPY.csv`

### Option Chain Data

**Location:** `F:/inputs/options/log/{ticker}/`  
**Format:** CSV files named `{ticker}_quotedata.csv` in date directories

**Example Structure:**
```
F:/inputs/options/log/spy/
  ├── 12_22_2025/
  │   └── spy_quotedata.csv
  ├── 12_23_2025/
  │   └── spy_quotedata.csv
  └── ...
```

**Required Columns in Option CSV:**
- `Strike` - Strike price
- `Bid`, `Ask` - Bid/ask prices
- `IV` - Implied volatility
- `Volume` - Trading volume
- `Expiration Date` - Option expiration date

### Customizing Data Paths

If your data is stored elsewhere, modify the paths in `data_loader.py`:

```python
STOCK_PATH = "your/stock/path/"
OPTION_PATH = "your/option/path/"
```

## Troubleshooting

### NumPy Version Issues

If you encounter NumPy 2.0 compatibility issues:

```powershell
pip install "numpy<2.0.0"
```

### Missing Project-Specific Modules

**Error:** `ModuleNotFoundError: No module named 'data_loader'`

**Solution:** Ensure `data_loader.py` is in the parent directory or add it to your Python path:

```powershell
# Option 1: Run from parent directory
cd ..
python markov/testB.py --ticker SPY

# Option 2: Add to PYTHONPATH
$env:PYTHONPATH = "D:\SereneOcean\cursor_workspaces\stock-monitor\metascripts"
python testB.py --ticker SPY
```

### SHAP Installation Issues

If SHAP fails to install or has compatibility issues:

```powershell
# Try installing compatible version
pip install "shap<0.50"

# Or skip SHAP entirely - script will work without it
# Just use --skip-shap flag when running
```

The script will automatically detect if SHAP is available and skip plots if not installed.

### LightGBM Installation Issues

If LightGBM fails on Windows:

```powershell
# Option 1: Try with conda
conda install -c conda-forge lightgbm

# Option 2: Use pre-built wheel
pip install lightgbm --no-cache-dir

# Option 3: Install Visual C++ Build Tools if compilation fails
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

### Insufficient Data Errors

**Error:** `Insufficient call data for training (need at least 40 samples)`

**Solution:** 
- Ensure you have recent option chain data for the ticker
- Try a different date with more option contracts
- The minimum threshold is set to 40 samples (configurable in `testB.py` line 417)

### Memory Issues (Universal Mode)

If processing many tickers causes memory issues:

```powershell
# Process in smaller batches by editing tickers.json
# Or process specific tickers:
python testB.py --tickers SPY QQQ IWM
```

The script automatically clears state between tickers in universal mode.

### Path Issues

**Error:** `Output directory not found`

**Solution:** The script creates output directories automatically. If you see this error:
- Check that you have write permissions
- Verify the path calculation (script shows the path it's looking for)
- For universal mode, ensure `output/markov/` directory can be created

## Quick Start Examples

### Example 1: Single Ticker Analysis

```powershell
cd metascripts/markov
python testB.py --ticker SPY --skip-shap
```

### Example 2: Universal Mode with Custom Capital

```powershell
python testB.py --universal --capital 50000
```

### Example 3: Markov Engine Integration (testC.py)

```powershell
# First run (will note insufficient history)
python testC.py --ticker SPY

# Subsequent runs use saved state history
python testC.py --ticker SPY

# Or provide historical states manually
python testC.py --ticker SPY --markov-history-file history.json
```

### Example 4: Generate Aggregate Reports

```powershell
# First, run universal mode to generate reports
python testB.py --universal

# Then aggregate the reports
python aggregate_reports.py
```

### Example 5: Debug Mode

```powershell
python testB.py --ticker SPY --debug
```

This will show detailed Kelly Gate calculations and state machine logic.

### Example 6: Markov Engine with Custom Horizon

```powershell
python testC.py --ticker SPY --markov-horizon 10.0 --markov-rate-scale 1.5 --debug
```

This will show Markov engine diagnostics and flow evolution.

## Output Structure

After running, you'll find:

**Single Ticker Mode:**
```
../output/analysis_YYYYMMDD_HHMMSS/
  ├── *.pkl (models)
  ├── *.csv (data)
  ├── *.json (gate results)
  └── *.png (visualizations)
```

**Universal Mode:**
```
../output/markov/
  ├── SPY/
  │   ├── SPY_report.md
  │   ├── SPY_report.csv
  │   └── reflexive_plan.json (if applicable)
  ├── QQQ/
  │   └── ...
  ├── markov_state_history.json (testC.py only)
  └── aggregated/
      ├── all_tickers_combined_*.csv
      ├── summary_*.csv
      └── summary_report_*.md
```

## Next Steps

1. **Run testA.py** to see Markov blanket discovery in action
2. **Run testB.py** with a single ticker to understand the full pipeline
3. **Run testC.py** to see Markov engine integration (requires state history)
4. **Run universal mode** to process all tickers (testC.py builds state history incrementally)
5. **Generate aggregate reports** to see summary statistics
6. **Review the reports** in markdown and CSV formats

For more details, see [README.md](README.md).
