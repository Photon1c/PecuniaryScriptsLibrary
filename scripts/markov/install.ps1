# PowerShell installation script for testB.py dependencies
# Run this script from the markov directory

Write-Host "Installing dependencies for Markov Blanket Option Pricing Model..." -ForegroundColor Cyan

# Check Python version
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Python not found. Please install Python 3.8+ first." -ForegroundColor Red
    exit 1
}
Write-Host "Found: $pythonVersion" -ForegroundColor Green

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install core dependencies
Write-Host "`nInstalling core dependencies..." -ForegroundColor Yellow
pip install "numpy<2.0.0,>=1.21.0"
pip install "pandas>=1.5.0,<3.0.0"
pip install "scipy>=1.9.0,<2.0.0"
pip install "scikit-learn>=1.0.0,<2.0.0"
pip install "lightgbm>=3.3.0,<5.0.0"
pip install "matplotlib>=3.5.0,<4.0.0"
pip install "seaborn>=0.12.0,<1.0.0"
pip install "mplcyberpunk>=0.1.0"
pip install "rich>=12.0.0,<14.0.0"
pip install "networkx>=2.8.0,<4.0.0"

# Try to install SHAP (optional)
Write-Host "`nAttempting to install SHAP (optional)..." -ForegroundColor Yellow
pip install "shap<0.50.0" 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "SHAP installed successfully!" -ForegroundColor Green
} else {
    Write-Host "SHAP installation failed (this is optional, script will work without it)" -ForegroundColor Yellow
}

# Verify installation
Write-Host "`nVerifying installation..." -ForegroundColor Yellow
python -c "import numpy, pandas, matplotlib, seaborn, lightgbm, sklearn, rich, mplcyberpunk, networkx; print('✓ All core packages imported successfully!')" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ Installation complete! You can now run: python testB.py --ticker SPY" -ForegroundColor Green
} else {
    Write-Host "`n✗ Some packages failed to import. Please check the error messages above." -ForegroundColor Red
    exit 1
}
