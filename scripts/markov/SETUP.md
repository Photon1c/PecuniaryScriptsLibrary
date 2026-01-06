# Setup Guide for testB.py

## Prerequisites
- Python 3.8 or higher
- pip package manager

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

After installation, verify that all packages are installed:
```powershell
python -c "import numpy, pandas, matplotlib, seaborn, lightgbm, sklearn, rich, mplcyberpunk; print('All core packages imported successfully!')"
```

For SHAP (optional):
```powershell
python -c "import shap; print('SHAP installed successfully!')"
```

## Running the Script

Once dependencies are installed, run the script:
```powershell
python testB.py --ticker SPY
```

## Troubleshooting

### NumPy Version Issues
If you encounter NumPy 2.0 compatibility issues:
```powershell
pip install "numpy<2.0.0"
```

### Missing Project-Specific Modules
The script requires:
- `data_loader` module (should be in parent directory or Python path)
- `testA` module (MarkovBlanketAnalyzer, should be in same directory)

Make sure these modules are accessible in your Python path.

### SHAP Installation Issues
If SHAP fails to install, the script will still work but without SHAP plots:
```powershell
# Try installing SHAP separately
pip install shap
```

### LightGBM Installation Issues
If LightGBM fails on Windows:
```powershell
# Try installing with conda
conda install -c conda-forge lightgbm

# Or use pre-built wheel
pip install lightgbm --no-cache-dir
```

## Dependencies Summary

### Required
- numpy<2.0.0
- pandas>=1.5.0
- scipy>=1.9.0
- scikit-learn>=1.0.0
- lightgbm>=3.3.0
- matplotlib>=3.5.0
- seaborn>=0.12.0
- mplcyberpunk>=0.1.0
- rich>=12.0.0
- networkx>=2.8.0 (required by testA.py)

### Optional (but recommended)
- shap>=0.41.0 (for SHAP plots and interpretability)

### Project-Specific (must be available)
- data_loader (for loading option chain data)
- testA (for MarkovBlanketAnalyzer)
