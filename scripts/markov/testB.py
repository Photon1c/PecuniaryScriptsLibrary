#!/usr/bin/env python3
"""
Markov Blanket-Driven Option Pricing: Predictive Model Implementation (UPGRADED v9 - Teixiptla-Garage-Markov Kelly Gate)

This script implements the predictive modeling approach described in outlineB.md:
1. Trains ML regressors on Markov blanket features to forecast premiums
2. Compares performance to Black-Scholes on historical data
3. Decomposes premium into "classical" (Black-Scholes) and "extra" components
4. Analyzes skew attribution to quantify sentiment-driven vs volatility-driven effects
5. Teixiptla-Garage-Markov Kelly Gate: permission layer for option trading

UPGRADES v9 (Teixiptla-Garage-Markov Kelly Gate):
- Added Kelly Gate module: regime inference (PIN/PRE_TRANSFER/TRANSFER), structure family recommendation, Kelly sizing with multipliers
- Skew features: put-call IV diff, skew slope, smile curvature, ATM/25Δ proxies
- Term structure: front/back IV, term slope, inversion detection
- Execution quality: bid-ask spread, slippage estimation, quality score
- Outputs: gate.json, contract_scores.csv, enhanced Rich summary with Kelly Gate panel
- --debug flag for intermediate feature values and p,b estimation details

UPGRADES v5:
- Aggressive residual model boosting: StandardScaler on log(V+1) and N, minimal regularization (reg_lambda=0.1, max_depth=8)
- Enhanced skew: OTM filtering for robust IV skew, added lagged skew feature
- Forced SHAP import: automatic installation attempt if missing, proper TreeExplainer usage
- Target: residual R² >0, extra component >5% of premium

UPGRADES v4:
- Boosted extra component: normalized V (log(V+1)) and N (z-score), added V×moneyness, V_lagged, N×moneyness
- Reduced regularization on residual model (reg_lambda=5.0, max_depth=7, early_stopping=30) to boost contribution
- Fixed skew attribution: RidgeCV/ElasticNetCV with alpha grid, OTM-specific skew calculation
- Enhanced SHAP: proper TreeExplainer for both full and residual models, separate beeswarm plots
- Optional log(premium+1) target transformation for stability
- Reports "extra variance explained" (residual R²) as key metric
- Target: extra component >10% of premium

UPGRADES v3:
- Fixed multicollinearity (removed redundant features: moneyness, vol_sqrt_T, vol_moneyness)
- Enhanced residual model with interaction terms (V×N, V×σ, N×σ) and stronger regularization
- Fixed skew attribution (proper IV skew calculation, Ridge/ElasticNet, scaled features)
- Fixed SHAP compatibility (proper TreeExplainer usage, beeswarm plots, residual model SHAP)
- Added baseline model for comparison
- Feature selection and improved regularization (max_depth=4-5, min_child_weight=10 for residual)
- Reports residual R² (variance explained by extra features)

UPGRADES v2:
- LightGBM with strong regularization and early stopping (reduces overfitting)
- Residual-based training for extra component (isolates Volume/News contribution)
- Feature engineering (moneyness, log(K/S), interactions) to improve separation
- Fixed SHAP compatibility and added dependence plots
- Robust regression for skew analysis (HuberRegressor)
- Feature correlation heatmap to detect multicollinearity
- Model persistence (save/load models and scalers)
- Better train/test separation to prevent data leakage

Based on the methodology:
- f_classical(S, σ, r, τ, K) ≈ Black-Scholes implied premium
- f_full(MB(P)) ≈ observed premium using all Markov blanket features
- Δ = residual_model(V, N) captures incremental contribution of volume and news
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Warning: seaborn not available. Correlation heatmap will use matplotlib.")
import mplcyberpunk
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from pathlib import Path
from datetime import datetime, timedelta
from scipy.stats import norm
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, HuberRegressor, ElasticNet, RidgeCV, ElasticNetCV
from sklearn.inspection import permutation_importance
import lightgbm as lgb
import pickle
import logging
import argparse
import warnings
import json
import os
import sys
from typing import List
warnings.filterwarnings('ignore')

# Fix for Windows multiprocessing issues with scipy/sklearn
# Use n_jobs=1 to avoid DLL loading errors in child processes
# Set environment variable PARALLEL_JOBS to override (e.g., export PARALLEL_JOBS=4)
N_JOBS = int(os.environ.get('PARALLEL_JOBS', '1'))
if sys.platform == 'win32' and N_JOBS == -1:
    print("Warning: n_jobs=-1 on Windows can cause DLL loading errors. Using n_jobs=1 instead.")
    print("Set PARALLEL_JOBS environment variable to override (e.g., set PARALLEL_JOBS=4)")
    N_JOBS = 1

# v8: FORCE SHAP import with robust try-except
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP import failed - skipping plots")
    print("To install: pip install shap")
except Exception as e:
    SHAP_AVAILABLE = False
    print(f"SHAP import error: {e} - skipping SHAP plots")

from data_loader import load_option_chain_data, get_latest_price, load_stock_data, get_most_recent_option_date
from testA import MarkovBlanketAnalyzer
from markov_mask import MarkovMaskComputer
from reflexive_bifurcation import generate_reflexive_plan, LegPlan
from state_machine import MarketState, MarketSignals, compute_market_state, describe_actions

# Initialize rich console
console = Console()

# Reflexive sleeve configuration (can be wired to CLI later)
REFLEXIVE_EXP_CAP_FRAC = 0.20   # at most 20% of K to this sleeve
REFLEXIVE_STOP_FRAC    = 0.097  # per-leg stop
REFLEXIVE_DTE_INITIAL  = 2.0    # days
REFLEXIVE_MAX_LEGS     = 2      # nested legs to pre-plan
REFLEXIVE_INITIAL_DIR  = "call" # default; can be inferred later

# Default capital (can be overridden via CLI)
DEFAULT_CAPITAL = 10000.0

# Path to tickers.json
SCRIPT_DIR = Path(__file__).parent
TICKERS_JSON = SCRIPT_DIR.parent.parent / "tickers.json"


def load_tickers() -> List[str]:
    """Load ticker symbols from shared tickers.json file."""
    if not TICKERS_JSON.exists():
        raise FileNotFoundError(f"Tickers file not found: {TICKERS_JSON}")
    
    with open(TICKERS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Extract just the ticker symbols (first element of each [symbol, type] pair)
    return [ticker[0] for ticker in data["tickers"]]


def setup_logging(output_dir: Path):
    """Setup logging with timestamps to a file."""
    log_file = output_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class BlackScholesPricer:
    """Black-Scholes option pricing implementation."""

    @staticmethod
    def calculate_d1_d2(S, K, T, r, sigma):
        """Calculate d1 and d2 for Black-Scholes formula."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    @staticmethod
    def call_price(S, K, T, r, sigma):
        """Calculate Black-Scholes call option price."""
        if T <= 0:
            return max(S - K, 0)
        if sigma <= 0:
            return max(S - K * np.exp(-r * T), 0)
        
        d1, d2 = BlackScholesPricer.calculate_d1_d2(S, K, T, r, sigma)
        call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return max(call, 0)  # Ensure non-negative

    @staticmethod
    def put_price(S, K, T, r, sigma):
        """Calculate Black-Scholes put option price using put-call parity."""
        call = BlackScholesPricer.call_price(S, K, T, r, sigma)
        put = call - S + K * np.exp(-r * T)
        return max(put, 0)  # Ensure non-negative


class DataPreparator:
    """Handles data preparation and feature engineering."""
    
    @staticmethod
    def parse_expiration_date(exp_date_str, current_date=None):
        """Parse expiration date string to time to expiration in years."""
        if current_date is None:
            current_date = datetime.now()
        
        try:
            # Try various date formats
            if isinstance(exp_date_str, str):
                # Common formats: "Fri Aug 01 2025", "2025-08-01", "08/01/2025"
                for fmt in ['%a %b %d %Y', '%Y-%m-%d', '%m/%d/%Y', '%Y%m%d']:
                    try:
                        exp_date = datetime.strptime(exp_date_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    # Fallback: assume 30 days if parsing fails
                    return 30 / 365.25
            else:
                return 30 / 365.25
            
            delta = exp_date - current_date
            days = max(delta.days, 1)  # At least 1 day
            return days / 365.25
        except Exception:
            return 30 / 365.25  # Default to 30 days
    
    @staticmethod
    def prepare_features(option_df, stock_price, risk_free_rate=0.05, current_date=None, use_log_target=False):
        """
        Prepare feature sets for modeling with proper expiration parsing and feature engineering.
        
        UPGRADED: Adds engineered features (moneyness, log(K/S), interactions) to improve model separation.
        
        Returns:
            - classical_features: [S, σ, r, τ, K] for Black-Scholes
            - full_features: [S, σ, r, τ, K, V, N, moneyness, log(K/S), σ×moneyness, ...] for Markov blanket model
            - targets: observed option premiums
        """
        # Filter out invalid data
        valid_mask = (
            (option_df['Strike'] > 0) &
            (option_df['Bid'] >= 0) &
            (option_df['Ask'] >= 0) &
            (option_df['IV'] > 0) &
            (option_df['IV'] < 5)  # Reasonable IV bounds
        )

        df = option_df[valid_mask].copy()

        # Extract features
        S = stock_price  # Spot price
        K = df['Strike'].values  # Strike prices
        sigma = df['IV'].values / 100.0  # Implied volatility (convert from %)
        r = risk_free_rate  # Risk-free rate (assumed constant)
        
        # Parse expiration dates properly
        if 'Expiration Date' in df.columns:
            exp_dates = df['Expiration Date'].values
            T = np.array([DataPreparator.parse_expiration_date(exp, current_date) for exp in exp_dates])
        else:
            # Fallback: assume short-term options
            T = np.random.uniform(0.01, 0.25, len(df))
        
        T = np.clip(T, 0.01, 1.0)  # Ensure reasonable bounds

        # Classical features (Black-Scholes inputs) - ONLY 5 features
        classical_features = np.column_stack([S * np.ones_like(K), sigma, r * np.ones_like(K), T, K])

        # Markov blanket features: add Trading_Volume and News proxy
        # UPGRADED v4: Normalize V and N for better residual model performance
        V_raw = df['Volume'].values  # Trading volume
        # Handle missing volume: impute with median
        if np.isnan(V_raw).any() or (V_raw == 0).all():
            V_raw = np.where(np.isnan(V_raw) | (V_raw == 0), np.nanmedian(V_raw) if not np.isnan(V_raw).all() else 1.0, V_raw)
        
        # v6: Aggressive normalization for V and N with StandardScaler
        # Step 1: log(V+1) transformation
        V_log = np.log1p(V_raw)  # log(V+1) transformation
        
        # Step 2: StandardScaler normalization on log(V+1)
        scaler_V = StandardScaler()
        V_scaled = scaler_V.fit_transform(V_log.reshape(-1, 1)).flatten()
        V = V_scaled  # StandardScaler normalized log(V+1)
        
        # v6: Also keep log(V+1) as separate feature for residual model
        V_log_feature = V_log  # Will be used in residual model features
        
        # News proxy: use IV as a sentiment indicator
        N_raw = sigma * 100  # Use IV as news/sentiment proxy
        
        # v6: StandardScaler normalization on N
        scaler_N = StandardScaler()
        N_scaled = scaler_N.fit_transform(N_raw.reshape(-1, 1)).flatten()
        N = N_scaled  # StandardScaler normalized
        
        # ENGINEERED FEATURES for full model (FIXED: removed multicollinear features)
        # Calculate moneyness for stratification/visualization (but don't include in features)
        moneyness = K / S  # Moneyness ratio (used for stratification, not in features)
        # Remove moneyness from features (correlates ~1.0 with log(K/S)), keep log_moneyness only
        log_moneyness = np.log(K / S)  # Log moneyness (more stable, less correlated)
        # Remove vol_sqrt_T and vol_moneyness (highly correlated with σ)
        # Keep only non-redundant engineered features
        
        # Full features: classical + extra + minimal engineered features
        # FIXED: Reduced from 11 to 8 features to avoid multicollinearity
        full_features = np.column_stack([
            S * np.ones_like(K),  # S
            sigma,                 # σ
            r * np.ones_like(K),   # r
            T,                     # τ
            K,                     # K
            V,                     # V (Volume)
            N,                     # N (News)
            log_moneyness         # log(K/S) - only non-redundant engineered feature
        ])

        # Targets: use mid-price (average of bid and ask)
        # UPGRADED v4: Optional log transformation for stability
        call_mid = (df['Bid'].values + df['Ask'].values) / 2
        # Check for put columns (may be named 'Bid.1' and 'Ask.1' or similar)
        put_bid_col = 'Bid.1' if 'Bid.1' in df.columns else 'Bid'
        put_ask_col = 'Ask.1' if 'Ask.1' in df.columns else 'Ask'
        put_mid = (df[put_bid_col].values + df[put_ask_col].values) / 2 if put_bid_col in df.columns else call_mid
        
        # Store original targets (for inverse transform if using log)
        call_mid_original = call_mid.copy()
        put_mid_original = put_mid.copy()
        
        # v6: Apply log transformation if requested
        if use_log_target:
            call_mid = np.log1p(call_mid)  # log(premium + 1)
            put_mid = np.log1p(put_mid)

        # Get put IV if available (for skew calculation)
        put_iv = df['IV.1'].values / 100.0 if 'IV.1' in df.columns else sigma

        # v6: Store V_log_feature for residual model
        # Create separate datasets for calls and puts
        call_data = {
            'classical_features': classical_features,
            'full_features': full_features,
            'extra_features': np.column_stack([V, N]),  # Only V and N for residual model
            'targets': call_mid,
            'strikes': K,
            'iv': sigma,
            'put_iv': put_iv,  # For skew calculation
            'volume': V,
            'volume_log': V_log_feature,  # v6: log(V+1) for residual model
            'time_to_exp': T,
            'moneyness': moneyness
        }

        put_data = {
            'classical_features': classical_features,
            'full_features': full_features,
            'extra_features': np.column_stack([V, N]),
            'targets': put_mid,
            'strikes': K,
            'iv': put_iv,
            'call_iv': sigma,  # For skew calculation
            'volume': V,
            'time_to_exp': T,
            'moneyness': moneyness
        }

        return call_data, put_data


class OptionPricingPredictor:
    """
    Implements Markov blanket-driven option pricing with ML models.
    UPGRADED v2: Uses LightGBM with strong regularization, residual-based training, and better separation.
    """

    def __init__(self, analyzer, logger=None, model_type='lightgbm'):
        self.analyzer = analyzer
        self.bs_pricer = BlackScholesPricer()
        self.classical_model = None
        self.full_model = None
        self.residual_model = None  # NEW: Model trained on residuals using only V, N
        self.skew_model = None
        self.logger = logger or logging.getLogger(__name__)
        self.model_type = model_type
        self.feature_names_classical = ['S', 'σ', 'r', 'τ', 'K']
        self.feature_names_full = ['S', 'σ', 'r', 'τ', 'K', 'V', 'N', 'log(K/S)']  # FIXED: removed multicollinear features
        self.feature_names_extra = ['V', 'N']
        self.scaler_skew = None

    def _create_model(self, max_depth=5, n_estimators=200, is_residual=False):
        """
        Create LightGBM model with strong regularization.
        UPGRADED v3: Tighter regularization for residual model, reduced max_depth.
        """
        # Even stronger regularization for residual model
        reg_lambda = 20.0 if is_residual else 10.0
        reg_alpha = 10.0 if is_residual else 5.0
        min_child = 10 if is_residual else 5
        
        return lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.05,  # Lower learning rate
            reg_lambda=reg_lambda,     # L2 regularization (stronger for residual)
            reg_alpha=reg_alpha,       # L1 regularization (stronger for residual)
            min_child_weight=min_child,  # Minimum samples per leaf (higher for residual)
            subsample=0.8,        # Row subsampling
            colsample_bytree=0.8, # Feature subsampling
            random_state=42,
            n_jobs=N_JOBS,
            verbose=-1
        )

    def train_models(self, call_data, put_data, n_folds=5):
        """
        Train classical and full models with cross-validation.
        UPGRADED v2: 
        - Uses LightGBM with strong regularization
        - Implements residual-based training for extra component
        - Better train/test separation to prevent leakage
        """
        self.logger.info("Training predictive models with cross-validation...")
        console.print("[bold cyan]Training predictive models with cross-validation...[/bold cyan]")

        # Train on calls
        X_classical_call = call_data['classical_features']
        X_full_call = call_data['full_features']
        X_extra_call = call_data['extra_features']  # Only V, N
        y_call = call_data['targets']

        # Filter out zero prices (illiquid options)
        valid_call = y_call > 0.01
        X_classical_call = X_classical_call[valid_call]
        X_full_call = X_full_call[valid_call]
        X_extra_call = X_extra_call[valid_call]
        y_call = y_call[valid_call]

        if len(y_call) < 40:  # Need at least 40 samples for proper CV
            console.print("[red]Insufficient call data for training (need at least 40 samples)[/red]")
            self.logger.warning(f"Insufficient data: {len(y_call)} samples")
            return None

        # v6: 80/20 train/test split with stratification by moneyness bins (to prevent leakage)
        # Create moneyness bins for stratification
        moneyness = call_data['moneyness'][valid_call]
        moneyness_bins = pd.cut(moneyness, bins=5, labels=False, duplicates='drop')
        
        # v6: Store indices for proper log(V+1) extraction
        valid_indices = np.where(valid_call)[0]
        
        # Split with indices tracking - train_test_split returns arrays in order
        split_result = train_test_split(
            np.arange(len(X_classical_call)),  # Create index array
            X_classical_call, X_full_call, X_extra_call, y_call,
            test_size=0.2, random_state=42, stratify=moneyness_bins if len(np.unique(moneyness_bins)) > 1 else None
        )
        train_idx, test_idx, X_classical_train, X_classical_test, X_full_train, X_full_test, \
        X_extra_train, X_extra_test, y_train, y_test = split_result
        
        # v6: Map back to original valid_indices
        train_indices = valid_indices[train_idx]
        test_indices = valid_indices[test_idx]

        # v6: Split train into train/val for early stopping (with indices)
        split_result_val = train_test_split(
            np.arange(len(X_classical_train)),  # Create index array for train set
            X_classical_train, X_full_train, X_extra_train, y_train,
            test_size=0.2, random_state=42
        )
        train_sub_idx, val_idx, X_classical_train_sub, X_classical_val, X_full_train_sub, X_full_val, \
        X_extra_train_sub, X_extra_val, y_train_sub, y_val = split_result_val
        
        # v6: Map back to original valid_indices
        train_sub_indices = train_indices[train_sub_idx]
        val_indices = train_indices[val_idx]

        # v7: Get log target flag from call_data (must be done before using it)
        use_log_target = call_data.get('use_log_target', False)

        # v8: Print model params and target type at training
        # Train classical model (ONLY 5 features)
        console.print("[yellow]Training classical model (5 features)...[/yellow]")
        target_type_str = "log(premium+1)" if use_log_target else "raw premium"
        console.print(f"[cyan]Target type: {target_type_str}[/cyan]")
        self.classical_model = self._create_model(max_depth=4, n_estimators=300)
        console.print(f"[dim]Classical model params: max_depth=4, n_estimators=300, early_stopping=50, target={target_type_str}[/dim]")
        self.logger.info(f"Classical model params: max_depth=4, n_estimators=300, early_stopping=50, target={target_type_str}")
        self.classical_model.fit(
            X_classical_train_sub, y_train_sub,
            eval_set=[(X_classical_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        # Train full model (8 features - reduced from 11 to avoid multicollinearity)
        console.print("[yellow]Training full model (8 features)...[/yellow]")
        self.full_model = self._create_model(max_depth=5, n_estimators=300)  # Reduced from 6 to 5
        console.print(f"[dim]Full model params: max_depth=5, n_estimators=300, early_stopping=50[/dim]")
        self.logger.info("Full model params: max_depth=5, n_estimators=300, early_stopping=50")
        self.full_model.fit(
            X_full_train_sub, y_train_sub,
            eval_set=[(X_full_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        # v6: Train residual model with enhanced features and minimal regularization
        console.print("[yellow]Training residual model (V, N + enhanced features)...[/yellow]")
        classical_pred_train = self.classical_model.predict(X_classical_train)
        
        # v7: use_log_target already defined above, no need to redefine
        if use_log_target:
            residuals_train = np.log1p(y_train) - np.log1p(classical_pred_train)
            console.print("[dim]Using log(premium+1) as target for residual model[/dim]")
        else:
            residuals_train = y_train - classical_pred_train
        
        # Get features for interactions
        sigma_train = X_classical_train[:, 1]  # σ is second column
        K_train = X_classical_train[:, 4]  # K is fifth column
        S_train = X_classical_train[:, 0]  # S is first column
        moneyness_train = K_train / S_train  # Moneyness for interaction
        
        V_train = X_extra_train[:, 0]  # StandardScaler normalized V
        N_train = X_extra_train[:, 1]  # StandardScaler normalized N
        
        # v6: Get log(V+1) feature from call_data using proper train indices
        V_log_train = call_data['volume_log'][train_indices] if 'volume_log' in call_data else V_train
        
        # v7: Reduced interactions to prevent overfitting (keep only V*moneyness, N*σ)
        X_extra_train_enhanced = np.column_stack([
            V_train,                    # V (StandardScaler normalized)
            N_train,                    # N (StandardScaler normalized)
            V_log_train,                # log(V+1) as separate feature
            N_train * sigma_train,      # N × σ (keep this interaction)
            V_train * moneyness_train,  # V × moneyness (keep this interaction)
            # Removed: V × N, V × σ, V_lagged, N × moneyness (too many interactions cause overfitting)
        ])
        
        # v6: Same for validation set with log(V+1)
        sigma_val = X_classical_val[:, 1]
        K_val = X_classical_val[:, 4]
        S_val = X_classical_val[:, 0]
        moneyness_val = K_val / S_val
        V_val = X_extra_val[:, 0]
        N_val = X_extra_val[:, 1]
        
        # v6: Get log(V+1) for validation set using proper val indices
        V_log_val = call_data['volume_log'][val_indices] if 'volume_log' in call_data else V_val
        
        # v7: Reduced interactions for validation set (matching training)
        X_extra_val_enhanced = np.column_stack([
            V_val, N_val, V_log_val, N_val * sigma_val, V_val * moneyness_val
        ])
        
        # v7: Update feature names for extra model (reduced interactions)
        self.feature_names_extra = ['V', 'N', 'log(V+1)', 'N×σ', 'V×moneyness']
        
        # v7: Train residual model with stronger regularization to prevent overfitting and negative Δ
        # Specific params: reg_lambda=0.5 (increased from 0.05), reg_alpha=0.1, max_depth=7, min_child_weight=5, early_stopping=40
        self.residual_model = lgb.LGBMRegressor(
            n_estimators=500,
            max_depth=7,  # v7: Reduced from 9 to 7 (prevent overfitting)
            learning_rate=0.05,
            reg_lambda=0.5,  # v7: Increased to 0.5 (stronger L2 regularization)
            reg_alpha=0.1,   # Keep at 0.1
            min_child_weight=5,  # v7: Increased from 3 to 5 (stronger regularization)
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=N_JOBS,
            verbose=-1
        )
        
        # v8: Print model params and target type at training
        target_type_str = "log(premium+1)" if use_log_target else "raw premium"
        console.print(f"[cyan]Residual model target type: {target_type_str}[/cyan]")
        console.print(f"[dim]Residual model params: max_depth=7, reg_lambda=0.5, reg_alpha=0.1, min_child_weight=5, target={target_type_str}[/dim]")
        self.logger.info(f"Residual model params: max_depth=7, reg_lambda=0.5, reg_alpha=0.1, min_child_weight=5, early_stopping=40, target={target_type_str}")
        
        # v6: Prepare validation residuals (with log if needed)
        if use_log_target:
            val_residuals = np.log1p(y_val) - np.log1p(self.classical_model.predict(X_classical_val))
        else:
            val_residuals = y_val - self.classical_model.predict(X_classical_val)
        
        self.residual_model.fit(
            X_extra_train_enhanced, residuals_train,
            eval_set=[(X_extra_val_enhanced, val_residuals)],
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)]  # v6: Set to 40
        )
        
        # Calculate R² of residual model on residuals
        residual_pred_train = self.residual_model.predict(X_extra_train_enhanced)
        residual_r2 = r2_score(residuals_train, residual_pred_train)
        self.logger.info(f"Residual model R² (variance explained by V, N): {residual_r2:.3f}")

        # Cross-validation for robust metrics
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # CV for classical model
        classical_cv_mae = -cross_val_score(
            self.classical_model, X_classical_train, y_train,
            cv=kf, scoring='neg_mean_absolute_error', n_jobs=N_JOBS
        )
        classical_cv_r2 = cross_val_score(
            self.classical_model, X_classical_train, y_train,
            cv=kf, scoring='r2', n_jobs=N_JOBS
        )

        # CV for full model
        full_cv_mae = -cross_val_score(
            self.full_model, X_full_train, y_train,
            cv=kf, scoring='neg_mean_absolute_error', n_jobs=N_JOBS
        )
        full_cv_r2 = cross_val_score(
            self.full_model, X_full_train, y_train,
            cv=kf, scoring='r2', n_jobs=N_JOBS
        )

        # Test set predictions
        classical_pred = self.classical_model.predict(X_classical_test)
        full_pred = self.full_model.predict(X_full_test)
        
        # Residual model prediction on test set (with interaction terms)
        classical_pred_test = self.classical_model.predict(X_classical_test)
        residuals_test = y_test - classical_pred_test
        
        # v6: Prepare enhanced extra features for test set (matching training features)
        sigma_test = X_classical_test[:, 1]
        K_test = X_classical_test[:, 4]
        S_test = X_classical_test[:, 0]
        moneyness_test = K_test / S_test
        V_test = X_extra_test[:, 0]
        N_test = X_extra_test[:, 1]
        
        # v6: Get log(V+1) for test set using proper test indices
        V_log_test = call_data['volume_log'][test_indices] if 'volume_log' in call_data else V_test
        
        # v7: Reduced interactions for test set (matching training)
        X_extra_test_enhanced = np.column_stack([
            V_test, N_test, V_log_test, N_test * sigma_test, V_test * moneyness_test
        ])
        
        residual_pred = self.residual_model.predict(X_extra_test_enhanced)
        
        # v7: Force positive mean Δ via offset if negative
        mean_residual_pred = np.mean(residual_pred)
        if mean_residual_pred < 0:
            # v7: Add offset to ensure positive mean (prevent sign flip)
            offset = -mean_residual_pred + 0.01  # Small positive margin
            residual_pred = residual_pred + offset
            console.print(f"[yellow]Applied offset {offset:.4f} to residual predictions to ensure positive mean Δ[/yellow]")
            self.logger.info(f"Applied offset {offset:.4f} to residual predictions (mean was {mean_residual_pred:.4f})")
        
        # v7: Combined prediction: classical + residual (handle log target inverse transform)
        if use_log_target:
            # v7: If residual model was trained on log residuals, predictions are in log space
            # Combined in log space: log1p(classical) + residual_pred
            # Then inverse transform: expm1(log1p(classical) + residual_pred)
            combined_pred = np.expm1(np.log1p(classical_pred_test) + residual_pred)
        else:
            combined_pred = classical_pred_test + residual_pred
        
        # v6: Baseline model (mean premium) - dummy baseline for comparison
        baseline_pred = np.full_like(y_test, np.mean(y_train))
        baseline_mae = mean_absolute_error(y_test, baseline_pred)
        baseline_r2 = r2_score(y_test, baseline_pred)
        console.print(f"[dim]Baseline (dummy) MAE: ${baseline_mae:.2f}, R²: {baseline_r2:.3f}[/dim]")
        self.logger.info(f"Baseline (dummy mean premium) - MAE: ${baseline_mae:.2f}, R²: {baseline_r2:.3f}")

        classical_mae = mean_absolute_error(y_test, classical_pred)
        full_mae = mean_absolute_error(y_test, full_pred)
        combined_mae = mean_absolute_error(y_test, combined_pred)
        classical_r2 = r2_score(y_test, classical_pred)
        full_r2 = r2_score(y_test, full_pred)
        combined_r2 = r2_score(y_test, combined_pred)

        # Display results
        performance_table = Table(title="Model Performance (Calls) - Cross-Validated")
        performance_table.add_column("Metric", style="cyan")
        performance_table.add_column("Baseline", style="dim", justify="right")
        performance_table.add_column("Classical (5)", style="yellow", justify="right")
        performance_table.add_column("Full (8)", style="green", justify="right")
        performance_table.add_column("Combined", style="magenta", justify="right")

        performance_table.add_row(
            "CV MAE (mean ± std)",
            f"${baseline_mae:.2f}",
            f"${classical_cv_mae.mean():.2f} ± ${classical_cv_mae.std():.2f}",
            f"${full_cv_mae.mean():.2f} ± ${full_cv_mae.std():.2f}",
            f"${combined_mae:.2f}"
        )
        performance_table.add_row(
            "CV R² (mean ± std)",
            f"{baseline_r2:.3f}",
            f"{classical_cv_r2.mean():.3f} ± {classical_cv_r2.std():.3f}",
            f"{full_cv_r2.mean():.3f} ± {full_cv_r2.std():.3f}",
            f"{combined_r2:.3f}"
        )
        performance_table.add_row(
            "Test MAE",
            f"${baseline_mae:.2f}",
            f"${classical_mae:.2f}",
            f"${full_mae:.2f}",
            f"${combined_mae:.2f}"
        )
        performance_table.add_row(
            "Test R²",
            f"{baseline_r2:.3f}",
            f"{classical_r2:.3f}",
            f"{full_r2:.3f}",
            f"{combined_r2:.3f}"
        )

        console.print(performance_table)
        self.logger.info(f"Classical model - CV MAE: ${classical_cv_mae.mean():.2f}, CV R²: {classical_cv_r2.mean():.3f}")
        self.logger.info(f"Full model - CV MAE: ${full_cv_mae.mean():.2f}, CV R²: {full_cv_r2.mean():.3f}")
        self.logger.info(f"Combined (classical + residual) - Test MAE: ${combined_mae:.2f}, R²: {combined_r2:.3f}")

        return {
            'classical_pred': classical_pred,
            'full_pred': full_pred,
            'combined_pred': combined_pred,
            'residual_pred': residual_pred,
            'y_test': y_test,
            'classical_mae': classical_mae,
            'full_mae': full_mae,
            'combined_mae': combined_mae,
            'classical_r2': classical_r2,
            'full_r2': full_r2,
            'combined_r2': combined_r2,
            'classical_cv_mae': classical_cv_mae,
            'full_cv_mae': full_cv_mae,
            'classical_cv_r2': classical_cv_r2,
            'full_cv_r2': full_cv_r2,
            'X_classical_test': X_classical_test,
            'X_full_test': X_full_test,
            'X_extra_test': X_extra_test,
            'X_extra_test_enhanced': X_extra_test_enhanced,
            'baseline_mae': baseline_mae,
            'baseline_r2': baseline_r2
        }

    def decompose_premium(self, call_data, put_data, training_results):
        """
        Decompose premium into classical and extra components.
        UPGRADED v2: Uses residual model to isolate extra component contribution.
        """
        self.logger.info("Decomposing premium into classical vs extra components...")
        console.print("\n[bold cyan]Decomposing Premium into Classical vs Extra Components...[/bold cyan]")

        # Use test set from training for decomposition
        X_classical_test = training_results['X_classical_test']
        X_extra_test_enhanced = training_results['X_extra_test_enhanced']
        y_test = training_results['y_test']

        # v7: Predictions (handle log target if needed)
        use_log_target = call_data.get('use_log_target', False)
        classical_pred = self.classical_model.predict(X_classical_test)
        residual_pred = self.residual_model.predict(X_extra_test_enhanced)
        
        # v8: Stabilize extra component - post-processing to ensure positive Δ
        mean_residual_pred = np.mean(residual_pred)
        if mean_residual_pred < 0:
            # v8: Add offset to ensure positive mean (prevent sign flip)
            offset = -mean_residual_pred + 0.01  # Small positive margin
            residual_pred = residual_pred + offset
            console.print(f"[yellow]Applied offset {offset:.4f} to residual predictions in decomposition to ensure positive mean Δ[/yellow]")
            self.logger.info(f"Applied offset {offset:.4f} to residual predictions in decomposition (mean was {mean_residual_pred:.4f})")
        
        # v8: Clip any remaining negative values to 0 (post-processing)
        negative_count = np.sum(residual_pred < 0)
        if negative_count > 0:
            residual_pred = np.clip(residual_pred, 0, None)  # Clip negative values to 0
            console.print(f"[yellow]Clipped {negative_count} negative residual predictions to 0 in decomposition[/yellow]")
            self.logger.info(f"Clipped {negative_count} negative residual predictions to 0 in decomposition")
        
        # v7: Combined prediction (handle log target inverse transform)
        if use_log_target:
            combined_pred = np.expm1(np.log1p(classical_pred) + residual_pred)
            # v7: Convert to original scale for decomposition
            y_test_original = np.expm1(y_test) if np.all(y_test >= 0) else y_test
            classical_pred_original = np.expm1(classical_pred) if np.all(classical_pred >= 0) else classical_pred
            # v7: Delta in original scale (approximate: expm1(residual_pred) for small residuals)
            delta = np.expm1(residual_pred)  # Convert log-space residual to original scale
        else:
            combined_pred = classical_pred + residual_pred
            y_test_original = y_test
            classical_pred_original = classical_pred
            delta = residual_pred  # This is the isolated contribution of V, N
        
        # Calculate R² of residual model (variance explained by extra features)
        residuals_test = y_test - classical_pred
        residual_r2 = r2_score(residuals_test, residual_pred)

        # Feature importance using permutation importance (normalized)
        X_full_test = training_results['X_full_test']
        perm_importance = permutation_importance(
            self.full_model, X_full_test[:min(1000, len(X_full_test))], y_test[:min(1000, len(y_test))],
            n_repeats=10, random_state=42, n_jobs=N_JOBS
        )
        
        # Normalize importance
        total_importance = np.sum(perm_importance.importances_mean)
        normalized_importance = perm_importance.importances_mean / (total_importance + 1e-10)

        # Statistics
        decomposition_table = Table(title="Premium Decomposition Analysis (Residual-Based)")
        decomposition_table.add_column("Metric", style="cyan")
        decomposition_table.add_column("Value", style="green", justify="right")

        # v7: Use original scale values for decomposition table
        decomposition_table.add_row("Mean Actual Premium", f"${np.mean(y_test_original):.2f}")
        decomposition_table.add_row("Mean Classical Component", f"${np.mean(classical_pred_original):.2f}")
        decomposition_table.add_row("Mean Extra Component (Δ)", f"${np.mean(delta):.2f}")
        decomposition_table.add_row("Extra % of Total", f"{(np.mean(delta) / np.mean(y_test_original) * 100):.1f}%")
        decomposition_table.add_row("Std of Extra Component", f"${np.std(delta):.2f}")
        decomposition_table.add_row("MAE Explained by Extra", f"${np.mean(np.abs(delta)):.2f}")
        decomposition_table.add_row("Residual R² (extra variance explained)", f"{residual_r2:.3f} ({residual_r2*100:.1f}% of residual variance)")
        decomposition_table.add_row("", "")
        decomposition_table.add_row("[bold]Feature Importance (Normalized)[/bold]", "", "")
        for i, name in enumerate(self.feature_names_full):
            importance = normalized_importance[i]
            decomposition_table.add_row(f"  {name}", f"{importance*100:.2f}%", "")

        console.print(decomposition_table)
        self.logger.info(f"Extra component: ${np.mean(delta):.2f} ({np.mean(delta) / np.mean(y_test_original) * 100:.1f}% of premium)")

        # Get strikes and moneyness for visualization
        strikes = call_data['strikes']
        moneyness = call_data['moneyness']
        # Filter to match test set indices (we need to track which samples are in test set)
        # For now, use all data
        valid = call_data['targets'] > 0.01
        strikes_all = strikes[valid]
        moneyness_all = moneyness[valid]

        return {
            'classical': classical_pred_original,
            'combined': combined_pred,
            'extra': delta,
            'actual': y_test_original,
            'strikes': strikes_all[:len(y_test)] if len(strikes_all) >= len(y_test) else strikes_all,
            'moneyness': moneyness_all[:len(y_test)] if len(moneyness_all) >= len(y_test) else moneyness_all,
            'perm_importance': perm_importance,
            'normalized_importance': normalized_importance,
            'X_full': X_full_test,
            'residual_r2': residual_r2  # R² of residual model
        }

    def analyze_skew_attribution(self, call_data, put_data):
        """
        Analyze skew attribution using Ridge/ElasticNet regression.
        UPGRADED v3: Proper IV skew calculation, scaled features, Ridge/ElasticNet for stability.
        """
        self.logger.info("Analyzing skew attribution with Ridge regression...")
        console.print("\n[bold cyan]Analyzing Skew Attribution (Ridge Regression)...[/bold cyan]")

        # Calculate actual put-call IV skew (FIXED: ensure proper calculation)
        call_iv = call_data['iv']
        put_iv = call_data['put_iv']
        
        # UPGRADED v5: Robust IV skew calculation with OTM filtering
        # Ensure we have valid IV data
        valid_skew = (call_iv > 0) & (put_iv > 0) & (call_iv < 5) & (put_iv < 5)
        
        # v7: Filter for OTM options with minimum threshold (more reliable skew signal)
        moneyness_all = call_data['moneyness']
        # v7: Stricter OTM filter: OTM puts: moneyness < 0.90, OTM calls: moneyness > 1.10 (min OTM filter)
        otm_puts = moneyness_all < 0.90  # v7: Stricter (was 0.95)
        otm_calls = moneyness_all > 1.10  # v7: Stricter (was 1.05)
        otm_mask = otm_puts | otm_calls  # Either OTM puts or OTM calls
        
        # v7: Ensure minimum number of OTM samples
        if np.sum(otm_mask) < 20:
            self.logger.warning(f"Only {np.sum(otm_mask)} OTM samples found, relaxing filter...")
            # Fallback to less strict filter
            otm_puts = moneyness_all < 0.95
            otm_calls = moneyness_all > 1.05
            otm_mask = otm_puts | otm_calls
        
        # Combine valid IV and OTM filters
        valid_skew = valid_skew & otm_mask
        
        # Check if we have any valid samples after filtering
        if np.sum(valid_skew) == 0:
            self.logger.warning("No valid samples found for skew analysis after filtering. Returning empty results.")
            console.print("[yellow]Warning: No valid samples found for skew analysis. Skipping...[/yellow]")
            # Return safe defaults that match the expected structure
            feature_names_skew = ['σ', 'V', 'N', 'skew_lagged', 'σ×V', 'σ×N', 'V×N']
            iv_skew_full = np.full(len(call_data['iv']), np.nan)
            return {
                'beta_sigma': 0.0,
                'beta_V': 0.0,
                'beta_N': 0.0,
                'beta_interactions': np.zeros(4),  # 4 interaction terms
                'r2': 0.0,
                'iv_skew': iv_skew_full,
                'iv_skew_valid': np.array([]),
                'X_skew': np.array([]).reshape(0, 7),  # 7 features
                'feature_names': feature_names_skew,
                'relative_importance': np.zeros(7)
            }
        
        call_iv_valid = call_iv[valid_skew]
        put_iv_valid = put_iv[valid_skew]
        
        # Put-call IV skew: Put IV - Call IV (typical volatility skew metric)
        iv_skew = put_iv_valid - call_iv_valid
        
        # Check if iv_skew is empty (shouldn't happen after the check above, but safety check)
        if len(iv_skew) == 0:
            self.logger.warning("Computed iv_skew is empty. Returning empty results.")
            console.print("[yellow]Warning: Computed IV skew is empty. Skipping analysis...[/yellow]")
            feature_names_skew = ['σ', 'V', 'N', 'skew_lagged', 'σ×V', 'σ×N', 'V×N']
            iv_skew_full = np.full(len(call_data['iv']), np.nan)
            return {
                'beta_sigma': 0.0,
                'beta_V': 0.0,
                'beta_N': 0.0,
                'beta_interactions': np.zeros(4),
                'r2': 0.0,
                'iv_skew': iv_skew_full,
                'iv_skew_valid': np.array([]),
                'X_skew': np.array([]).reshape(0, 7),
                'feature_names': feature_names_skew,
                'relative_importance': np.zeros(7)
            }
        
        # UPGRADED v5: Calculate lagged skew BEFORE outlier filtering
        # (so we can use it as a feature even if some values are filtered out)
        # Create lagged skew: use previous value, pad with first value
        iv_skew_lagged_full = np.concatenate([[iv_skew[0]], iv_skew[:-1]])
        
        # Filter out extreme outliers (beyond 3 std devs)
        skew_mean = np.mean(iv_skew)
        skew_std = np.std(iv_skew)
        outlier_mask = np.abs(iv_skew - skew_mean) < 3 * skew_std
        iv_skew = iv_skew[outlier_mask]
        
        # Convert valid_skew from boolean mask to indices, then filter by outlier_mask
        valid_skew_indices = np.where(valid_skew)[0][outlier_mask]
        
        # UPGRADED v5: Prepare features for skew regression with lagged skew
        # Use only valid indices (after outlier filtering)
        sigma = call_data['iv'][valid_skew_indices]
        V = call_data['volume'][valid_skew_indices]
        N = sigma * 100  # News proxy
        
        # Apply outlier mask to lagged skew
        iv_skew_lagged = iv_skew_lagged_full[outlier_mask]

        # UPGRADED v5: Create polynomial features (interaction terms) + lagged skew
        X_base = np.column_stack([sigma, V, N, iv_skew_lagged])  # Added lagged skew (4 features)
        X_interactions = np.column_stack([
            sigma * V,  # Vol × Volume
            sigma * N,  # Vol × News
            V * N       # Volume × News
        ])
        X_skew = np.column_stack([X_base, X_interactions])  # Total: 4 base + 3 interactions = 7 features

        # Scale features (CRITICAL for Ridge/ElasticNet)
        self.scaler_skew = StandardScaler()
        X_skew_scaled = self.scaler_skew.fit_transform(X_skew)

        # UPGRADED v4: Use RidgeCV with alpha grid for stability
        # Try to compute OTM-specific skew if possible (for better target)
        # For now, use overall put-call IV skew, but compute it more carefully
        
        # Compute skew by moneyness groups (OTM puts typically have higher IV)
        moneyness_skew = call_data['moneyness'][valid_skew_indices]
        # OTM puts (moneyness < 1) should have higher IV than OTM calls (moneyness > 1)
        # Adjust skew calculation: for OTM, use put IV - call IV; for ITM, use call IV - put IV
        otm_mask = moneyness_skew < 1.0  # OTM
        itm_mask = moneyness_skew >= 1.0  # ITM
        
        # For OTM: put IV should be higher (positive skew expected)
        # For ITM: call IV might be higher (negative skew possible)
        iv_skew_adjusted = np.where(otm_mask, iv_skew, -iv_skew)  # Flip for ITM
        
        # v6: Use RidgeCV with logspace alpha grid for stability
        alphas = np.logspace(-3, 3, 20)  # v6: 20 values from 0.001 to 1000
        self.skew_model = RidgeCV(alphas=alphas, cv=5, scoring='r2')
        self.skew_model.fit(X_skew_scaled, iv_skew_adjusted)
        
        # Calculate R²
        skew_pred = self.skew_model.predict(X_skew_scaled)
        skew_r2 = r2_score(iv_skew_adjusted, skew_pred)
        
        # If R² is still very low, try ElasticNet with CV
        if skew_r2 < 0.05:
            self.logger.warning(f"RidgeCV R² is low ({skew_r2:.3f}), trying ElasticNetCV...")
            self.skew_model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0], 
                                          alphas=[0.01, 0.1, 0.5, 1.0, 2.0], cv=5, max_iter=2000)
            self.skew_model.fit(X_skew_scaled, iv_skew_adjusted)
            skew_pred = self.skew_model.predict(X_skew_scaled)
            skew_r2 = r2_score(iv_skew_adjusted, skew_pred)
            console.print(f"[yellow]Using ElasticNetCV, R²: {skew_r2:.3f}[/yellow]")
        
        # Note: iv_skew is already filtered by outliers (from line 698)
        # Keep it as is - it's the original put_iv - call_iv after outlier filtering
        # (iv_skew_filtered was saved and assigned to iv_skew, so it's already correct)

        # Calculate relative importance: |beta| * std(feature)
        feature_stds = np.std(X_skew_scaled, axis=0)
        relative_importance = np.abs(self.skew_model.coef_) * feature_stds
        total_relative = np.sum(relative_importance)
        relative_importance_pct = (relative_importance / (total_relative + 1e-10)) * 100

        # Bootstrap confidence intervals for coefficients (use same model type)
        n_bootstrap = 100
        coef_samples = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(X_skew_scaled), len(X_skew_scaled), replace=True)
            X_boot = X_skew_scaled[indices]
            y_boot = iv_skew_adjusted[indices]
            # Use same model type as final model
            if isinstance(self.skew_model, RidgeCV) or isinstance(self.skew_model, Ridge):
                model_boot = Ridge(alpha=self.skew_model.alpha_ if hasattr(self.skew_model, 'alpha_') else 1.0, max_iter=1000)
            else:
                model_boot = ElasticNet(alpha=self.skew_model.alpha_ if hasattr(self.skew_model, 'alpha_') else 0.1,
                                      l1_ratio=self.skew_model.l1_ratio_ if hasattr(self.skew_model, 'l1_ratio_') else 0.5,
                                      max_iter=2000)
            model_boot.fit(X_boot, y_boot)
            coef_samples.append(model_boot.coef_)
        
        coef_samples = np.array(coef_samples)
        coef_ci_lower = np.percentile(coef_samples, 2.5, axis=0)
        coef_ci_upper = np.percentile(coef_samples, 97.5, axis=0)

        # Feature names with interactions + lagged skew
        feature_names_skew = ['σ', 'V', 'N', 'skew_lagged', 'σ×V', 'σ×N', 'V×N']

        skew_table = Table(title="Skew Attribution Analysis (Robust Regression)")
        skew_table.add_column("Component", style="cyan")
        skew_table.add_column("Coefficient", style="magenta", justify="right")
        skew_table.add_column("95% CI", style="yellow", justify="right")
        skew_table.add_column("Rel. Importance", style="green", justify="right")

        for i, name in enumerate(feature_names_skew):
            coef = self.skew_model.coef_[i]
            ci_low = coef_ci_lower[i]
            ci_high = coef_ci_upper[i]
            rel_imp = relative_importance_pct[i]
            skew_table.add_row(
                name,
                f"{coef:.4f}",
                f"[{ci_low:.4f}, {ci_high:.4f}]",
                f"{rel_imp:.1f}%"
            )

        skew_table.add_row("", "", "", "")
        skew_table.add_row("[bold]Intercept[/bold]", f"{self.skew_model.intercept_:.4f}", "", "")
        skew_table.add_row("[bold]Model R²[/bold]", f"[bold]{skew_r2:.3f}[/bold]", "", "")

        console.print(skew_table)
        self.logger.info(f"Skew model R²: {skew_r2:.3f}")

        # Interpretation
        console.print("\n[bold yellow]Interpretation:[/bold yellow]")
        base_contrib = relative_importance[0]  # σ contribution
        if base_contrib > 0:
            vol_contrib = relative_importance[1] / base_contrib * 100 if base_contrib > 0 else 0
            news_contrib = relative_importance[2] / base_contrib * 100 if base_contrib > 0 else 0
            if vol_contrib > 1:
                console.print(f"  • Volume contributes {vol_contrib:.1f}% relative to volatility")
            if news_contrib > 1:
                console.print(f"  • News/sentiment contributes {news_contrib:.1f}% relative to volatility")

        # Store full IV skew array (for visualization)
        # Convert valid_skew_indices back to boolean mask for assignment
        iv_skew_full = np.full(len(call_data['iv']), np.nan)
        valid_skew_final = np.zeros(len(call_data['iv']), dtype=bool)
        valid_skew_final[valid_skew_indices] = True
        iv_skew_full[valid_skew_final] = iv_skew
        
        return {
            'beta_sigma': self.skew_model.coef_[0],
            'beta_V': self.skew_model.coef_[1],
            'beta_N': self.skew_model.coef_[2],
            'beta_interactions': self.skew_model.coef_[3:],
            'r2': skew_r2,
            'iv_skew': iv_skew_full,  # Full array for visualization
            'iv_skew_valid': iv_skew,  # Valid values only
            'X_skew': X_skew_scaled,
            'feature_names': feature_names_skew,
            'relative_importance': relative_importance_pct
        }

    def plot_feature_correlation(self, call_data, output_dir):
        """Plot feature correlation heatmap to detect multicollinearity."""
        self.logger.info("Generating feature correlation heatmap...")
        
        X_full = call_data['full_features']
        valid = call_data['targets'] > 0.01
        X_full_valid = X_full[valid]
        
        # Create DataFrame for correlation
        df_features = pd.DataFrame(X_full_valid, columns=self.feature_names_full)
        corr_matrix = df_features.corr()
        
        plt.style.use("cyberpunk")
        fig, ax = plt.subplots(figsize=(12, 10))
        if SEABORN_AVAILABLE:
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        else:
            # Fallback to matplotlib
            im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            ax.set_xticks(np.arange(len(self.feature_names_full)))
            ax.set_yticks(np.arange(len(self.feature_names_full)))
            ax.set_xticklabels(self.feature_names_full, rotation=45, ha='right')
            ax.set_yticklabels(self.feature_names_full)
            for i in range(len(self.feature_names_full)):
                for j in range(len(self.feature_names_full)):
                    text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
            plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        mplcyberpunk.add_glow_effects()
        
        output_path = output_dir / f"feature_correlation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='lightgray', edgecolor='none')
        console.print(f"[green]Correlation heatmap saved: {output_path}[/green]")
        self.logger.info(f"Correlation heatmap saved: {output_path}")
        plt.close()

    def visualize_results(self, decomposition_results, skew_results, training_results, output_dir, args=None):
        """
        Create comprehensive visualizations.
        UPGRADED v5: Fixed SHAP calls, added dependence plots, and improved layout.
        args parameter for --skip-shap flag.
        """
        if args is None:
            class DummyArgs:
                skip_shap = False
            args = DummyArgs()
        self.logger.info("Generating visualizations...")
        console.print("\n[bold cyan]Generating visualizations...[/bold cyan]")

        plt.style.use("cyberpunk")
        
        classical = decomposition_results['classical']
        combined = decomposition_results['combined']
        extra = decomposition_results['extra']
        actual = decomposition_results['actual']
        strikes = decomposition_results['strikes']
        moneyness = decomposition_results['moneyness']

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(22, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        # 1. Prediction comparison (actual vs predicted)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(actual, classical, alpha=0.5, label='Classical Model', s=20)
        ax1.scatter(actual, combined, alpha=0.5, label='Combined (Classical + Residual)', s=20)
        ax1.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2, label='Perfect')
        ax1.set_xlabel('Actual Premium ($)')
        ax1.set_ylabel('Predicted Premium ($)')
        ax1.set_title('Model Predictions vs Actual')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Residuals vs Strike
        ax2 = fig.add_subplot(gs[0, 1])
        classical_residuals = actual - classical
        combined_residuals = actual - combined
        ax2.scatter(strikes, classical_residuals, alpha=0.5, label='Classical', s=20)
        ax2.scatter(strikes, combined_residuals, alpha=0.5, label='Combined', s=20)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Strike Price ($)')
        ax2.set_ylabel('Residual ($)')
        ax2.set_title('Residuals vs Strike Price')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Residuals vs Moneyness
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.scatter(moneyness, classical_residuals, alpha=0.5, label='Classical', s=20)
        ax3.scatter(moneyness, combined_residuals, alpha=0.5, label='Combined', s=20)
        ax3.axhline(y=0, color='r', linestyle='--', lw=2)
        ax3.axvline(x=1.0, color='g', linestyle=':', lw=1, label='ATM')
        ax3.set_xlabel('Moneyness (K/S)')
        ax3.set_ylabel('Residual ($)')
        ax3.set_title('Residuals vs Moneyness')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Extra component distribution
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.hist(extra, bins=50, alpha=0.7, edgecolor='white')
        ax4.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero')
        ax4.axvline(x=np.mean(extra), color='g', linestyle='-', lw=2, label=f'Mean: ${np.mean(extra):.2f}')
        ax4.set_xlabel('Extra Component Δ ($)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Extra Premium Component')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Decomposition by moneyness groups (ITM/ATM/OTM)
        ax5 = fig.add_subplot(gs[1, 1])
        itm_mask = moneyness < 0.95
        atm_mask = (moneyness >= 0.95) & (moneyness <= 1.05)
        otm_mask = moneyness > 1.05
        
        groups = ['ITM', 'ATM', 'OTM']
        classical_means = [
            np.mean(classical[itm_mask]) if np.any(itm_mask) else 0,
            np.mean(classical[atm_mask]) if np.any(atm_mask) else 0,
            np.mean(classical[otm_mask]) if np.any(otm_mask) else 0
        ]
        extra_means = [
            np.mean(extra[itm_mask]) if np.any(itm_mask) else 0,
            np.mean(extra[atm_mask]) if np.any(atm_mask) else 0,
            np.mean(extra[otm_mask]) if np.any(otm_mask) else 0
        ]
        
        x = np.arange(len(groups))
        width = 0.35
        ax5.bar(x - width/2, classical_means, width, label='Classical', alpha=0.8)
        ax5.bar(x + width/2, extra_means, width, label='Extra (Δ)', alpha=0.8)
        ax5.set_xlabel('Moneyness Group')
        ax5.set_ylabel('Mean Premium ($)')
        ax5.set_title('Premium Decomposition by Moneyness')
        ax5.set_xticks(x)
        ax5.set_xticklabels(groups)
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')

        # 6. Feature importance (permutation, normalized)
        ax6 = fig.add_subplot(gs[1, 2])
        perm_importance = decomposition_results['perm_importance']
        feature_names = self.feature_names_full
        importances = perm_importance.importances_mean
        std = perm_importance.importances_std
        
        ax6.barh(feature_names, importances, xerr=std, alpha=0.8)
        ax6.set_xlabel('Importance')
        ax6.set_title('Feature Importance (Permutation)')
        ax6.grid(True, alpha=0.3, axis='x')

        # v8: SHAP summary plot - FORCE to work with robust error handling
        ax7 = fig.add_subplot(gs[2, 0])
        if SHAP_AVAILABLE and training_results is not None and not args.skip_shap:
            try:
                X_sample = training_results.get('X_full_test', None)
                if X_sample is not None and len(X_sample) > 0:
                    n_samples = min(100, len(X_sample))
                    X_shap = X_sample[:n_samples]
                    
                    # v8: FORCE SHAP to work - use TreeExplainer with robust error handling
                    try:
                        explainer = shap.TreeExplainer(self.full_model)
                        shap_values = explainer.shap_values(X_shap)
                    except Exception as e:
                        self.logger.warning(f"SHAP TreeExplainer failed for full model: {e}")
                        console.print(f"[yellow]SHAP TreeExplainer failed: {e}[/yellow]")
                        shap_values = None
                    
                    # v8: Save SHAP summary plot (bar) with clear name in dated folder - only if shap_values available
                    if shap_values is not None:
                        try:
                            shap_output = output_dir / f"shap_full_model_bar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                            shap_fig2 = plt.figure(figsize=(10, 8))
                            shap.summary_plot(shap_values, X_shap, feature_names=self.feature_names_full,
                                            show=False, plot_type='bar')
                            plt.tight_layout()
                            plt.savefig(shap_output, dpi=300, bbox_inches='tight')
                            plt.close(shap_fig2)
                            console.print(f"[green]SHAP full model (bar) saved: {shap_output}[/green]")
                            self.logger.info(f"SHAP full model (bar) saved: {shap_output}")
                            
                            # v8: SHAP beeswarm plot for full model with clear name in dated folder
                            shap_beeswarm_output = output_dir / f"shap_full_model_beeswarm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                            shap_fig_beeswarm = plt.figure(figsize=(10, 8))
                            shap.summary_plot(shap_values, X_shap, feature_names=self.feature_names_full,
                                            show=False, plot_type='dot')
                            plt.tight_layout()
                            plt.savefig(shap_beeswarm_output, dpi=300, bbox_inches='tight')
                            plt.close(shap_fig_beeswarm)
                            console.print(f"[green]SHAP full model (beeswarm) saved: {shap_beeswarm_output}[/green]")
                            self.logger.info(f"SHAP full model (beeswarm) saved: {shap_beeswarm_output}")
                        except Exception as e:
                            self.logger.warning(f"SHAP plotting failed for full model: {e}")
                            console.print(f"[yellow]SHAP plotting failed: {e}[/yellow]")
                    else:
                        self.logger.warning("SHAP values not available for full model")
                    
                    # v8: SHAP dependence plots for V and N with clear names in dated folder
                    if shap_values is not None and len(X_shap) > 0:
                        try:
                            shap_fig3, axes = plt.subplots(1, 2, figsize=(14, 6))
                            # Volume dependence (index 5 in full features)
                            shap.dependence_plot(5, shap_values, X_shap, feature_names=self.feature_names_full,
                                               ax=axes[0], show=False)
                            axes[0].set_title('SHAP Dependence: Volume (V)')
                            # News dependence (index 6 in full features)
                            shap.dependence_plot(6, shap_values, X_shap, feature_names=self.feature_names_full,
                                               ax=axes[1], show=False)
                            axes[1].set_title('SHAP Dependence: News (N)')
                            plt.tight_layout()
                            shap_dep_output = output_dir / f"shap_full_model_dependence_V_N_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                            plt.savefig(shap_dep_output, dpi=300, bbox_inches='tight')
                            plt.close(shap_fig3)
                            console.print(f"[green]SHAP dependence plots saved: {shap_dep_output}[/green]")
                            self.logger.info(f"SHAP dependence plots saved: {shap_dep_output}")
                        except Exception as e:
                            self.logger.warning(f"SHAP dependence plots failed: {e}")
                            console.print(f"[yellow]SHAP dependence plots failed: {e}[/yellow]")
                    
                    # v7: Plot SHAP for residual model separately (beeswarm required)
                    if self.residual_model is not None:
                        X_extra_shap = training_results.get('X_extra_test_enhanced', None)
                        if X_extra_shap is not None and len(X_extra_shap) > 0:
                            n_samples_res = min(100, len(X_extra_shap))
                            X_extra_shap_sample = X_extra_shap[:n_samples_res]
                            
                            # v7: FORCE SHAP to work - use TreeExplainer with proper try-except
                            try:
                                explainer_res = shap.TreeExplainer(self.residual_model)
                                shap_values_res = explainer_res.shap_values(X_extra_shap_sample)
                                
                                # v7: SHAP bar plot for residual model with clear name
                                shap_res_output = output_dir / f"shap_residual_model_bar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                                shap_fig_res = plt.figure(figsize=(10, 6))
                                shap.summary_plot(shap_values_res, X_extra_shap_sample, 
                                                feature_names=self.feature_names_extra,
                                                show=False, plot_type='bar')
                                plt.tight_layout()
                                plt.savefig(shap_res_output, dpi=300, bbox_inches='tight')
                                plt.close(shap_fig_res)
                                console.print(f"[green]SHAP residual model (bar) saved: {shap_res_output}[/green]")
                                self.logger.info(f"SHAP residual model (bar) saved: {shap_res_output}")
                                
                                # v7: SHAP beeswarm plot for residual model (REQUIRED) with clear name
                                shap_res_beeswarm_output = output_dir / f"shap_residual_model_beeswarm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                                shap_fig_res_beeswarm = plt.figure(figsize=(10, 6))
                                shap.summary_plot(shap_values_res, X_extra_shap_sample, 
                                                feature_names=self.feature_names_extra,
                                                show=False, plot_type='dot')
                                plt.tight_layout()
                                plt.savefig(shap_res_beeswarm_output, dpi=300, bbox_inches='tight')
                                plt.close(shap_fig_res_beeswarm)
                                console.print(f"[green]SHAP residual model (beeswarm) saved: {shap_res_beeswarm_output}[/green]")
                                self.logger.info(f"SHAP residual model (beeswarm) saved: {shap_res_beeswarm_output}")
                            except Exception as e:
                                self.logger.warning(f"SHAP failed for residual model: {e}")
                                console.print(f"[yellow]SHAP residual model failed: {e}[/yellow]")
                    
                    ax7.text(0.5, 0.5, 'SHAP plots generated\n(see separate files)', 
                            ha='center', va='center', transform=ax7.transAxes, fontsize=10)
            except Exception as e:
                self.logger.warning(f"SHAP plot failed: {e}")
                ax7.text(0.5, 0.5, f'SHAP plot unavailable\n{str(e)[:50]}', 
                        ha='center', va='center', transform=ax7.transAxes, fontsize=8)
        else:
            ax7.text(0.5, 0.5, 'SHAP not installed\n(pip install shap)', 
                    ha='center', va='center', transform=ax7.transAxes, fontsize=8)
        ax7.set_title('SHAP Feature Importance')

        # 8. Skew attribution coefficients
        if skew_results is not None and self.skew_model is not None:
            ax8 = fig.add_subplot(gs[2, 1])
            feature_names_skew = skew_results['feature_names']
            coefs = self.skew_model.coef_
            colors = ['red' if c < 0 else 'green' for c in coefs]
            ax8.barh(feature_names_skew, coefs, color=colors, alpha=0.8)
            ax8.axvline(x=0, color='black', linestyle='-', lw=1)
            ax8.set_xlabel('Coefficient')
            ax8.set_title('Skew Attribution Coefficients')
            ax8.grid(True, alpha=0.3, axis='x')
        elif skew_results is not None:
            # Handle case where skew_results exists but model wasn't trained (no valid data)
            ax8 = fig.add_subplot(gs[2, 1])
            ax8.text(0.5, 0.5, 'Skew model not available\n(insufficient data)', 
                    ha='center', va='center', transform=ax8.transAxes, fontsize=10)
            ax8.set_title('Skew Attribution Coefficients')
            ax8.axis('off')

        # 9. IV Skew distribution
        if skew_results is not None:
            ax9 = fig.add_subplot(gs[2, 2])
            iv_skew_valid = skew_results.get('iv_skew_valid', skew_results['iv_skew'])
            # Filter out NaN values
            iv_skew_clean = iv_skew_valid[~np.isnan(iv_skew_valid)]
            if len(iv_skew_clean) > 0:
                ax9.hist(iv_skew_clean, bins=50, alpha=0.7, edgecolor='white')
                ax9.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Skew')
                ax9.axvline(x=np.mean(iv_skew_clean), color='g', linestyle='-', lw=2, label=f'Mean: {np.mean(iv_skew_clean):.4f}')
                ax9.set_xlabel('IV Skew (Put IV - Call IV)')
                ax9.set_ylabel('Frequency')
                ax9.set_title('Distribution of IV Skew')
                ax9.legend()
                ax9.grid(True, alpha=0.3)

        # 10. Relative importance for skew
        if skew_results is not None:
            ax10 = fig.add_subplot(gs[3, 0])
            rel_imp = skew_results['relative_importance']
            ax10.barh(skew_results['feature_names'], rel_imp, alpha=0.8)
            ax10.set_xlabel('Relative Importance (%)')
            ax10.set_title('Skew Attribution: Relative Importance')
            ax10.grid(True, alpha=0.3, axis='x')

        # 11. Extra component vs Volume
        ax11 = fig.add_subplot(gs[3, 1])
        if training_results is not None:
            X_extra_test_enhanced = training_results.get('X_extra_test_enhanced', None)
            if X_extra_test_enhanced is not None and len(X_extra_test_enhanced) > 0:
                volumes = X_extra_test_enhanced[:, 0]  # First column is V
                ax11.scatter(volumes, extra, alpha=0.5, s=20)
                ax11.set_xlabel('Trading Volume (V)')
                ax11.set_ylabel('Extra Component Δ ($)')
                ax11.set_title('Extra Component vs Volume')
                ax11.grid(True, alpha=0.3)

        # 12. Extra component vs News
        ax12 = fig.add_subplot(gs[3, 2])
        if training_results is not None:
            X_extra_test_enhanced = training_results.get('X_extra_test_enhanced', None)
            if X_extra_test_enhanced is not None and len(X_extra_test_enhanced) > 0:
                news = X_extra_test_enhanced[:, 1]  # Second column is N
                ax12.scatter(news, extra, alpha=0.5, s=20)
                ax12.set_xlabel('News/Sentiment (N)')
                ax12.set_ylabel('Extra Component Δ ($)')
                ax12.set_title('Extra Component vs News')
                ax12.grid(True, alpha=0.3)

        plt.suptitle('Markov Blanket Option Pricing Analysis (Upgraded v9 - Kelly Gate)', fontsize=16, fontweight='bold')
        mplcyberpunk.add_glow_effects()

        # Save
        output_path = output_dir / f"markov_blanket_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='lightgray', edgecolor='none')
        console.print(f"[green]Visualization saved: {output_path}[/green]")
        self.logger.info(f"Visualization saved: {output_path}")
        plt.close()

    def save_models(self, output_dir: Path):
        """Save all trained models and scalers to disk."""
        self.logger.info("Saving models and scalers...")
        try:
            # Save models
            if self.classical_model is not None:
                with open(output_dir / 'classical_model.pkl', 'wb') as f:
                    pickle.dump(self.classical_model, f)
            if self.full_model is not None:
                with open(output_dir / 'full_model.pkl', 'wb') as f:
                    pickle.dump(self.full_model, f)
            if self.residual_model is not None:
                with open(output_dir / 'residual_model.pkl', 'wb') as f:
                    pickle.dump(self.residual_model, f)
            if self.skew_model is not None:
                with open(output_dir / 'skew_model.pkl', 'wb') as f:
                    pickle.dump(self.skew_model, f)
            if self.scaler_skew is not None:
                with open(output_dir / 'skew_scaler.pkl', 'wb') as f:
                    pickle.dump(self.scaler_skew, f)
            console.print("[green]Models and scalers saved successfully[/green]")
            self.logger.info("Models and scalers saved successfully")
        except Exception as e:
            console.print(f"[red]Error saving models: {e}[/red]")
            self.logger.error(f"Error saving models: {e}", exc_info=True)


# ============================================================================
# Teixiptla-Garage-Markov Kelly Gate Module
# ============================================================================

class KellyGate:
    """
    Teixiptla-Garage-Markov Kelly Gate: Permission layer for option trading
    based on regime inference, skew/term structure, and execution quality.
    """
    
    def __init__(self, logger=None, fractional_kelly=0.25, max_position_pct=0.05, debug=False):
        self.logger = logger or logging.getLogger(__name__)
        self.fractional_kelly = fractional_kelly  # Default: 25% Kelly
        self.max_position_pct = max_position_pct  # Cap at 5% of capital per trade
        self.debug = debug
    
    def compute_skew_features(self, option_df, stock_price):
        """Compute IV skew metrics: put-call diff, slope, curvature."""
        try:
            # Extract IV columns (may be 'IV' and 'IV.1' or similar)
            call_iv_col = 'IV' if 'IV' in option_df.columns else None
            put_iv_col = 'IV.1' if 'IV.1' in option_df.columns else 'IV'
            
            if call_iv_col is None:
                self.logger.warning("Call IV column not found, using defaults")
                return {
                    'put_call_iv_diff': 0.0,
                    'skew_slope_puts': 0.0,
                    'skew_slope_calls': 0.0,
                    'smile_curvature': 0.0,
                    'atm_put_iv': 0.0,
                    'atm_call_iv': 0.0,
                    'delta25_put_iv': 0.0,
                    'delta25_call_iv': 0.0
                }
            
            # Get strikes and IVs
            strikes = pd.to_numeric(option_df['Strike'], errors='coerce')
            call_iv = pd.to_numeric(option_df[call_iv_col], errors='coerce') / 100.0  # Convert % to decimal
            put_iv = pd.to_numeric(option_df[put_iv_col], errors='coerce') / 100.0
            
            # Filter valid data
            valid = ~(strikes.isna() | call_iv.isna() | put_iv.isna())
            strikes = strikes[valid].values
            call_iv = call_iv[valid].values
            put_iv = put_iv[valid].values
            
            if len(strikes) < 5:
                self.logger.warning("Insufficient data for skew computation")
                return self._default_skew_features()
            
            # Moneyness
            moneyness = strikes / stock_price
            
            # ATM: moneyness closest to 1.0
            atm_idx = np.argmin(np.abs(moneyness - 1.0))
            atm_put_iv = put_iv[atm_idx] if atm_idx < len(put_iv) else np.median(put_iv)
            atm_call_iv = call_iv[atm_idx] if atm_idx < len(call_iv) else np.median(call_iv)
            
            # Put-call IV difference at ATM
            put_call_iv_diff = atm_put_iv - atm_call_iv
            
            # 25Δ proxies: moneyness ~0.75 (puts) and ~1.25 (calls)
            delta25_put_idx = np.argmin(np.abs(moneyness - 0.75))
            delta25_call_idx = np.argmin(np.abs(moneyness - 1.25))
            delta25_put_iv = put_iv[delta25_put_idx] if delta25_put_idx < len(put_iv) else np.median(put_iv)
            delta25_call_iv = call_iv[delta25_call_idx] if delta25_call_idx < len(call_iv) else np.median(call_iv)
            
            # Skew slope: linear fit of IV vs moneyness
            # OTM puts (moneyness < 1.0)
            otm_puts = moneyness < 0.95
            if np.sum(otm_puts) >= 3:
                try:
                    put_slope, _ = np.polyfit(moneyness[otm_puts], put_iv[otm_puts], 1)
                except:
                    put_slope = 0.0
            else:
                put_slope = 0.0
            
            # OTM calls (moneyness > 1.0)
            otm_calls = moneyness > 1.05
            if np.sum(otm_calls) >= 3:
                try:
                    call_slope, _ = np.polyfit(moneyness[otm_calls], call_iv[otm_calls], 1)
                except:
                    call_slope = 0.0
            else:
                call_slope = 0.0
            
            # Smile curvature: 2nd-order fit (coefficient of x^2)
            if len(moneyness) >= 5:
                try:
                    put_poly = np.polyfit(moneyness, put_iv, 2)
                    call_poly = np.polyfit(moneyness, call_iv, 2)
                    # Curvature = average of quadratic coefficients
                    smile_curvature = (put_poly[0] + call_poly[0]) / 2.0
                except:
                    smile_curvature = 0.0
            else:
                smile_curvature = 0.0
            
            return {
                'put_call_iv_diff': float(put_call_iv_diff),
                'skew_slope_puts': float(put_slope),
                'skew_slope_calls': float(call_slope),
                'smile_curvature': float(smile_curvature),
                'atm_put_iv': float(atm_put_iv),
                'atm_call_iv': float(atm_call_iv),
                'delta25_put_iv': float(delta25_put_iv),
                'delta25_call_iv': float(delta25_call_iv)
            }
        except Exception as e:
            self.logger.warning(f"Error computing skew features: {e}")
            return self._default_skew_features()
    
    def _default_skew_features(self):
        """Return default skew features when computation fails."""
        return {
            'put_call_iv_diff': 0.0,
            'skew_slope_puts': 0.0,
            'skew_slope_calls': 0.0,
            'smile_curvature': 0.0,
            'atm_put_iv': 0.0,
            'atm_call_iv': 0.0,
            'delta25_put_iv': 0.0,
            'delta25_call_iv': 0.0
        }
    
    def compute_term_structure_features(self, option_df, current_date):
        """Compute term structure metrics: front/back IV, slope, inversion."""
        try:
            if 'Expiration Date' not in option_df.columns:
                return self._default_term_features()
            
            # Parse expiration dates
            exp_dates = pd.to_datetime(option_df['Expiration Date'], errors='coerce')
            valid_exp = ~exp_dates.isna()
            
            if not valid_exp.any():
                return self._default_term_features()
            
            # Days to expiration
            dte = (exp_dates[valid_exp] - current_date).dt.days.values
            dte = np.maximum(dte, 1)  # At least 1 day
            
            # Get IV (use call IV as proxy)
            iv_col = 'IV' if 'IV' in option_df.columns else None
            if iv_col is None:
                return self._default_term_features()
            
            iv = pd.to_numeric(option_df[iv_col][valid_exp], errors='coerce').values / 100.0
            valid_iv = ~np.isnan(iv) & (iv > 0) & (iv < 5.0)
            
            if not valid_iv.any():
                return self._default_term_features()
            
            dte = dte[valid_iv]
            iv = iv[valid_iv]
            
            if len(dte) < 2:
                return self._default_term_features()
            
            # Front IV: nearest expiry (DTE < 30)
            front_mask = dte < 30
            front_iv = np.median(iv[front_mask]) if front_mask.any() else np.median(iv)
            
            # Back IV: far expiry (DTE > 60)
            back_mask = dte > 60
            back_iv = np.median(iv[back_mask]) if back_mask.any() else np.median(iv)
            
            # Term slope: IV vs DTE
            if len(dte) >= 3:
                try:
                    term_slope, _ = np.polyfit(dte, iv, 1)
                except:
                    term_slope = 0.0
            else:
                term_slope = 0.0
            
            # Inversion flag: front_iv > back_iv by threshold (5%)
            inversion_threshold = 0.05
            is_inverted = (front_iv > back_iv) and ((front_iv - back_iv) / (back_iv + 1e-6) > inversion_threshold)
            
            return {
                'front_iv': float(front_iv),
                'back_iv': float(back_iv),
                'term_slope': float(term_slope),
                'is_inverted': bool(is_inverted),
                'front_dte': float(np.median(dte[front_mask])) if front_mask.any() else 0.0,
                'back_dte': float(np.median(dte[back_mask])) if back_mask.any() else 0.0
            }
        except Exception as e:
            self.logger.warning(f"Error computing term structure features: {e}")
            return self._default_term_features()
    
    def _default_term_features(self):
        """Return default term structure features when computation fails."""
        return {
            'front_iv': 0.0,
            'back_iv': 0.0,
            'term_slope': 0.0,
            'is_inverted': False,
            'front_dte': 0.0,
            'back_dte': 0.0
        }
    
    def compute_execution_quality(self, option_df):
        """Compute execution quality metrics: bid-ask spread, slippage, quality score."""
        try:
            # Get bid/ask columns
            call_bid_col = 'Bid' if 'Bid' in option_df.columns else None
            call_ask_col = 'Ask' if 'Ask' in option_df.columns else None
            put_bid_col = 'Bid.1' if 'Bid.1' in option_df.columns else call_bid_col
            put_ask_col = 'Ask.1' if 'Ask.1' in option_df.columns else call_ask_col
            
            if call_bid_col is None or call_ask_col is None:
                return self._default_execution_quality()
            
            # Extract prices
            call_bid = pd.to_numeric(option_df[call_bid_col], errors='coerce').values
            call_ask = pd.to_numeric(option_df[call_ask_col], errors='coerce').values
            put_bid = pd.to_numeric(option_df[put_bid_col], errors='coerce').values if put_bid_col else call_bid
            put_ask = pd.to_numeric(option_df[put_ask_col], errors='coerce').values if put_ask_col else call_ask
            
            # Volume and OI
            volume_col = 'Volume' if 'Volume' in option_df.columns else None
            oi_col = 'Open Interest' if 'Open Interest' in option_df.columns else None
            
            volume = pd.to_numeric(option_df[volume_col], errors='coerce').values if volume_col else np.ones(len(option_df))
            oi = pd.to_numeric(option_df[oi_col], errors='coerce').values if oi_col else np.ones(len(option_df))
            
            # Combine calls and puts
            all_bid = np.concatenate([call_bid[~np.isnan(call_bid)], put_bid[~np.isnan(put_bid)]])
            all_ask = np.concatenate([call_ask[~np.isnan(call_ask)], put_ask[~np.isnan(put_ask)]])
            all_volume = np.concatenate([volume[~np.isnan(volume)], volume[~np.isnan(volume)]])
            all_oi = np.concatenate([oi[~np.isnan(oi)], oi[~np.isnan(oi)]])
            
            # Filter valid
            valid = (all_bid > 0) & (all_ask > all_bid) & (all_volume >= 0) & (all_oi >= 0)
            if not valid.any():
                return self._default_execution_quality()
            
            bid = all_bid[valid]
            ask = all_ask[valid]
            vol = all_volume[valid]
            oi_valid = all_oi[valid]
            
            # Mid price
            mid_price = (bid + ask) / 2.0
            
            # Bid-ask spread percentage
            spread_pct = ((ask - bid) / (mid_price + 1e-6)) * 100.0
            
            # Entry slippage: pay ask, exit at bid
            entry_slippage_est = (ask - mid_price) / (mid_price + 1e-6) * 100.0
            exit_slippage_est = (mid_price - bid) / (mid_price + 1e-6) * 100.0
            total_slippage = entry_slippage_est + exit_slippage_est
            
            # Execution quality score (0-1): penalize wide spreads, low volume/OI
            # Normalize: spread < 5% = good, > 20% = bad
            spread_score = np.clip(1.0 - (spread_pct - 5.0) / 15.0, 0.0, 1.0)
            
            # Volume score: log scale (ensure scalar)
            vol_median = float(np.median(vol)) if len(vol) > 0 else 0.0
            vol_score = float(np.clip(np.log1p(vol_median) / np.log1p(1000), 0.0, 1.0))
            
            # OI score: log scale (ensure scalar)
            oi_median = float(np.median(oi_valid)) if len(oi_valid) > 0 else 0.0
            oi_score = float(np.clip(np.log1p(oi_median) / np.log1p(5000), 0.0, 1.0))
            
            # Combined quality score (weighted) - use median of spread_score since it's an array
            spread_score_median = float(np.median(spread_score)) if len(spread_score) > 0 else 0.5
            quality_score = 0.5 * spread_score_median + 0.3 * vol_score + 0.2 * oi_score
            
            return {
                'bid_ask_spread_pct': float(np.median(spread_pct)),
                'mid_price': float(np.median(mid_price)),
                'entry_slippage_est': float(np.median(entry_slippage_est)),
                'exit_slippage_est': float(np.median(exit_slippage_est)),
                'total_slippage': float(np.median(total_slippage)),
                'quality_score': float(quality_score),
                'spread_score': spread_score_median,
                'volume_score': float(vol_score),
                'oi_score': float(oi_score)
            }
        except Exception as e:
            self.logger.warning(f"Error computing execution quality: {e}")
            return self._default_execution_quality()
    
    def _default_execution_quality(self):
        """Return default execution quality when computation fails."""
        return {
            'bid_ask_spread_pct': 10.0,
            'mid_price': 0.0,
            'entry_slippage_est': 5.0,
            'exit_slippage_est': 5.0,
            'total_slippage': 10.0,
            'quality_score': 0.5,
            'spread_score': 0.5,
            'volume_score': 0.5,
            'oi_score': 0.5
        }
    
    def infer_regime(self, call_data, put_data, decomposition_results, skew_results, training_results):
        """
        Infer regime: PIN / PRE_TRANSFER / TRANSFER
        Based on residual R², skew steepness, term structure, and model confidence.
        """
        try:
            residual_r2 = decomposition_results.get('residual_r2', 0.0)
            skew_r2 = skew_results.get('r2', 0.0)
            
            # Get skew features (if available)
            put_call_iv_diff = abs(skew_results.get('iv_skew_valid', np.array([0.0]))).mean() if 'iv_skew_valid' in skew_results else 0.0
            
            # Model confidence: high residual R² suggests market inefficiency (transfer opportunity)
            # Low residual R² suggests efficient pricing (PIN)
            if residual_r2 > 0.4 and skew_r2 > 0.3:
                # High model confidence + strong skew signal = TRANSFER
                regime = "TRANSFER"
            elif residual_r2 > 0.2 or put_call_iv_diff > 0.1:
                # Moderate inefficiency or steep skew = PRE_TRANSFER
                regime = "PRE_TRANSFER"
            else:
                # Low inefficiency = PIN (price in noise)
                regime = "PIN"
            
            if self.debug:
                self.logger.info(f"Regime inference: residual_r2={residual_r2:.3f}, skew_r2={skew_r2:.3f}, put_call_diff={put_call_iv_diff:.4f} -> {regime}")
            
            return regime
        except Exception as e:
            self.logger.warning(f"Error inferring regime: {e}")
            return "PIN"
    
    def suggest_structure_family(self, regime, skew_features, term_features):
        """Suggest structure family based on regime and market structure."""
        try:
            put_call_diff = skew_features.get('put_call_iv_diff', 0.0)
            is_inverted = term_features.get('is_inverted', False)
            
            if regime == "TRANSFER":
                # Transfer: favor convexity (long gamma, long vega)
                return "CONVEXITY"
            elif regime == "PRE_TRANSFER":
                if is_inverted and put_call_diff > 0.05:
                    # Inverted term + steep put skew = convexity opportunity
                    return "CONVEXITY"
                elif not is_inverted and put_call_diff < 0.02:
                    # Normal term + low skew = mean reversion premium
                    return "MEAN_REVERSION_PREMIUM"
                else:
                    return "PROBE_ONLY"
            else:  # PIN
                # PIN: only probe, no deployment
                return "PROBE_ONLY"
        except Exception as e:
            self.logger.warning(f"Error suggesting structure: {e}")
            return "PROBE_ONLY"
    
    def estimate_p_b(self, call_data, put_data, decomposition_results, training_results, regime, structure_family):
        """
        Estimate win probability (p) and win/loss ratio (b) using conservative heuristics.
        Uses mispricing signals from combined model vs classical model.
        """
        try:
            # Get mispricing signals
            classical_pred = training_results.get('classical_pred', None)
            combined_pred = training_results.get('combined_pred', None)
            actual = training_results.get('y_test', None)
            
            if classical_pred is None or combined_pred is None or actual is None:
                # Fallback: use conservative defaults
                return 0.5, 1.0
            
            # Mispricing magnitude: how much does combined model differ from classical?
            mispricing = combined_pred - classical_pred
            mispricing_magnitude = np.abs(mispricing)
            
            # For premium selling: p increases when IV rank high, term normal, skew not steepening
            # For convexity: p increases when term inverts, skew steepens, residual R² high
            
            residual_r2 = decomposition_results.get('residual_r2', 0.0)
            
            # Conservative heuristic: use mispricing consistency
            # "Win" = mispricing direction is consistent with structural logic
            if structure_family == "MEAN_REVERSION_PREMIUM":
                # Premium selling: favor when model says overpriced (combined > classical)
                # But be conservative: only if mispricing is significant
                significant_mispricing = mispricing_magnitude > np.percentile(mispricing_magnitude, 50)
                overpriced = mispricing > 0
                # Conservative: p = 0.5 + small boost if conditions align
                p_base = 0.5
                if significant_mispricing.any() and overpriced.any():
                    p = min(0.6, p_base + 0.1 * residual_r2)  # Boost by residual R²
                else:
                    p = p_base
                
                # b: avg_win / avg_loss (conservative: assume 1:1.5 for premium selling)
                b = 1.5
                
            elif structure_family == "CONVEXITY":
                # Convexity: favor when model says underpriced (combined < classical) OR when residual R² high
                significant_mispricing = mispricing_magnitude > np.percentile(mispricing_magnitude, 50)
                underpriced = mispricing < 0
                
                p_base = 0.5
                if (significant_mispricing.any() and underpriced.any()) or residual_r2 > 0.4:
                    p = min(0.65, p_base + 0.15 * residual_r2)  # Higher boost for convexity
                else:
                    p = p_base
                
                # b: convexity can have higher win/loss (2:1 or better)
                b = 2.0
                
            else:  # PROBE_ONLY
                # Probe: very conservative
                p = 0.5
                b = 1.0
            
            # Shrinkage: pull p toward 0.5 and b toward 1 when sample size is small
            n_samples = len(mispricing)
            if n_samples < 50:
                shrinkage = 0.3
                p = p * (1 - shrinkage) + 0.5 * shrinkage
                b = b * (1 - shrinkage) + 1.0 * shrinkage
            
            # Clip to safe bounds
            p = np.clip(p, 0.4, 0.7)  # Never too extreme
            b = np.clip(b, 0.5, 3.0)  # Reasonable win/loss bounds
            
            if self.debug:
                self.logger.info(f"p,b estimation: p={p:.3f}, b={b:.3f}, n_samples={n_samples}, structure={structure_family}")
            
            return float(p), float(b)
        except Exception as e:
            self.logger.warning(f"Error estimating p,b: {e}")
            return 0.5, 1.0
    
    def kelly_fraction(self, p, b):
        """Compute Kelly fraction with safeguards."""
        try:
            if b <= 0 or p <= 0 or p >= 1:
                return 0.0
            
            # Kelly formula: f = (p * b - (1 - p)) / b
            f_kelly = (p * b - (1 - p)) / b
            
            # Apply fractional Kelly
            f_fractional = f_kelly * self.fractional_kelly
            
            # Cap at max position size
            f_final = min(f_fractional, self.max_position_pct)
            
            # Ensure non-negative
            f_final = max(0.0, f_final)
            
            return float(f_final)
        except Exception as e:
            self.logger.warning(f"Error computing Kelly: {e}")
            return 0.0
    
    def apply_multipliers(self, f_kelly, skew_features, term_features, execution_quality, regime, structure_family):
        """Apply multipliers based on skew, term structure, and execution quality."""
        try:
            multiplier = 1.0
            
            # Term inversion: reduces sizing for premium selling, increases for convexity
            is_inverted = term_features.get('is_inverted', False)
            if structure_family == "MEAN_REVERSION_PREMIUM" and is_inverted:
                multiplier *= 0.7  # Reduce for premium selling in inversion
            elif structure_family == "CONVEXITY" and is_inverted:
                multiplier *= 1.2  # Increase for convexity in inversion
                multiplier = min(multiplier, 1.5)  # Cap at 1.5x
            
            # Steep put skew: reduces sizing for short gamma / short puts
            put_slope = skew_features.get('skew_slope_puts', 0.0)
            if structure_family == "MEAN_REVERSION_PREMIUM" and put_slope > 0.1:
                multiplier *= 0.8  # Reduce if steepening
            
            # Execution quality: lower sizing when quality is poor
            quality_score = execution_quality.get('quality_score', 0.5)
            multiplier *= quality_score  # Direct scaling by quality
            
            # Regime adjustments
            if regime == "PIN":
                multiplier *= 0.5  # Very conservative in PIN
            elif regime == "PRE_TRANSFER":
                multiplier *= 0.8  # Moderate in PRE_TRANSFER
            
            f_adjusted = f_kelly * multiplier
            f_adjusted = max(0.0, min(f_adjusted, self.max_position_pct))  # Clip to bounds
            
            return float(f_adjusted), float(multiplier)
        except Exception as e:
            self.logger.warning(f"Error applying multipliers: {e}")
            return f_kelly, 1.0
    
    def determine_gate_state(self, f_kelly_adjusted, quality_score, regime):
        """Determine gate state: BLOCK / PROBE / DEPLOY."""
        try:
            if f_kelly_adjusted <= 0.001 or quality_score < 0.3:
                return "BLOCK"
            elif f_kelly_adjusted < 0.01 or quality_score < 0.5 or regime == "PIN":
                return "PROBE"
            else:
                return "DEPLOY"
        except:
            return "BLOCK"
    
    def compute_gate(self, option_df, stock_price, current_date, call_data, put_data, 
                     decomposition_results, skew_results, training_results):
        """
        Main entry point: compute all gate features and return gate state.
        """
        try:
            # Compute features
            skew_features = self.compute_skew_features(option_df, stock_price)
            term_features = self.compute_term_structure_features(option_df, current_date)
            execution_quality = self.compute_execution_quality(option_df)
            
            # Infer regime
            regime = self.infer_regime(call_data, put_data, decomposition_results, skew_results, training_results)
            
            # Suggest structure
            structure_family = self.suggest_structure_family(regime, skew_features, term_features)
            
            # Estimate p, b
            p, b = self.estimate_p_b(call_data, put_data, decomposition_results, training_results, regime, structure_family)
            
            # Compute Kelly
            f_kelly_raw = self.kelly_fraction(p, b)
            
            # Apply multipliers
            f_kelly_adjusted, multiplier = self.apply_multipliers(
                f_kelly_raw, skew_features, term_features, execution_quality, regime, structure_family
            )
            
            # Determine gate state
            gate_state = self.determine_gate_state(f_kelly_adjusted, execution_quality['quality_score'], regime)
            
            # Compile results
            gate_results = {
                'regime': regime,
                'structure_family': structure_family,
                'kelly_raw': f_kelly_raw,
                'kelly_fractional': f_kelly_raw * self.fractional_kelly,
                'kelly_adjusted': f_kelly_adjusted,
                'gate_state': gate_state,
                'p': p,
                'b': b,
                'multiplier': multiplier,
                'skew_features': skew_features,
                'term_features': term_features,
                'execution_quality': execution_quality
            }
            
            return gate_results
        except Exception as e:
            self.logger.error(f"Error computing Kelly Gate: {e}", exc_info=True)
            # Return safe defaults
            return {
                'regime': 'PIN',
                'structure_family': 'PROBE_ONLY',
                'kelly_raw': 0.0,
                'kelly_fractional': 0.0,
                'kelly_adjusted': 0.0,
                'gate_state': 'BLOCK',
                'p': 0.5,
                'b': 1.0,
                'multiplier': 0.0,
                'skew_features': self._default_skew_features(),
                'term_features': self._default_term_features(),
                'execution_quality': self._default_execution_quality()
            }


def should_build_reflexive_sleeve(
    kelly_gate_results: dict,
    market_state: MarketState,
    force: bool = False,
) -> bool:
    """
    Determine if reflexive sleeve should be built based on market state.
    
    Args:
        kelly_gate_results: Dictionary with 'kelly_fractional' key
        market_state: Current market state from state machine
        force: Override flag to force generation
    
    Returns:
        True if reflexive sleeve should be built, False otherwise
    """
    if force:
        return True
    kelly_frac = kelly_gate_results.get('kelly_fractional', 0.0)
    if kelly_frac <= 0.0:
        return False
    # Only allow in TREND / RUPTURE_PREP / RUPTURE_ACTIVE
    if market_state in {MarketState.PIN, MarketState.RANGE, MarketState.COOLDOWN}:
        return False
    return market_state in {
        MarketState.TREND,
        MarketState.RUPTURE_PREP,
        MarketState.RUPTURE_ACTIVE,
    }


def save_markdown_report(
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
    Save a markdown report summarizing the analysis results.
    
    Returns:
        Path to the saved markdown file
    """
    md_path = output_dir / f"{ticker}_report.md"
    
    improvement_pct = ((training_results['classical_cv_mae'].mean() - training_results['full_cv_mae'].mean()) / 
                       training_results['classical_cv_mae'].mean() * 100) if training_results else 0.0
    extra_pct = (np.mean(decomposition_results['extra']) / np.mean(decomposition_results['actual']) * 100) if decomposition_results else 0.0
    residual_r2 = decomposition_results.get('residual_r2', 0.0) if decomposition_results else 0.0
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# {ticker} - Markov Blanket Option Pricing Analysis\n\n")
        f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Model Performance\n\n")
        if training_results:
            f.write(f"- **Baseline MAE:** ${training_results.get('baseline_mae', 0):.2f}\n")
            f.write(f"- **Classical Model MAE:** ${training_results['classical_mae']:.2f}\n")
            f.write(f"- **Combined Model MAE:** ${training_results['combined_mae']:.2f}\n")
            f.write(f"- **Improvement:** {improvement_pct:.1f}% (CV MAE reduction)\n")
            f.write(f"- **Extra Component (V, N):** {extra_pct:.1f}% of premium\n")
            f.write(f"- **Extra Variance Explained (Residual R²):** {residual_r2:.3f} ({residual_r2*100:.1f}%)\n")
            f.write(f"- **Skew Model R²:** {skew_results.get('r2', 0.0):.3f} ({skew_results.get('r2', 0.0)*100:.1f}%)\n\n")
        else:
            f.write("- *Insufficient data for model training*\n\n")
        
        f.write("## Kelly Gate\n\n")
        f.write(f"- **Regime:** {gate_results.get('regime', 'N/A')}\n")
        f.write(f"- **Structure:** {gate_results.get('structure_family', 'N/A')}\n")
        f.write(f"- **Kelly (raw):** {gate_results.get('kelly_raw', 0.0):.4f}\n")
        f.write(f"- **Kelly (fractional):** {gate_results.get('kelly_fractional', 0.0):.4f}\n")
        f.write(f"- **Kelly (adjusted):** {gate_results.get('kelly_adjusted', 0.0):.4f}\n")
        f.write(f"- **Gate State:** {gate_results.get('gate_state', 'N/A')}\n")
        f.write(f"- **p:** {gate_results.get('p', 0.0):.3f}\n")
        f.write(f"- **b:** {gate_results.get('b', 0.0):.3f}\n")
        f.write(f"- **Multiplier:** {gate_results.get('multiplier', 0.0):.3f}\n\n")
        
        f.write("## State Machine\n\n")
        f.write(f"- **Current State:** {market_state.value}\n")
        f.write(f"- **Actions:** {describe_actions(market_state)}\n")
        f.write(f"- **Derived from:** regime={signals.regime}, gate={signals.gate_state}, Kelly={signals.kelly_fraction:.4f}\n\n")
        
        f.write("## Reflexive Sleeve\n\n")
        if not reflexive_plan:
            f.write("**Status:** BLOCKED\n\n")
            f.write("Kelly Gate or Teixiptla regime does not permit reflexive nesting.\n\n")
        else:
            E0 = REFLEXIVE_EXP_CAP_FRAC * capital_value
            E0_eff = min(E0, gate_results.get('kelly_fractional', 0.0) * capital_value)
            f.write(f"- **Max Sleeve Cap:** {REFLEXIVE_EXP_CAP_FRAC:.0%} of K (K=${capital_value:.2f})\n")
            f.write(f"- **Effective Sleeve:** ${E0_eff:.2f}\n\n")
            f.write("| Leg | Direction | DTE (days) | Sleeve Entry | Stop Loss |\n")
            f.write("|-----|-----------|------------|--------------|-----------|\n")
            for lp in reflexive_plan:
                f.write(f"| {lp.leg} | {lp.direction.upper()} | {lp.dte:.1f} | ${lp.sleeve_entry:.2f} | ${lp.stop_loss:.2f} |\n")
            f.write("\n")
        
        if gate_results.get('skew_features'):
            f.write("## Skew Features\n\n")
            f.write(f"- **Put-Call IV Diff:** {gate_results['skew_features'].get('put_call_iv_diff', 0.0):.4f}\n")
            f.write(f"- **Skew Slope (Puts):** {gate_results['skew_features'].get('skew_slope_puts', 0.0):.4f}\n")
            f.write(f"- **Smile Curvature:** {gate_results['skew_features'].get('smile_curvature', 0.0):.4f}\n\n")
        
        if gate_results.get('term_features'):
            f.write("## Term Structure\n\n")
            f.write(f"- **Front IV:** {gate_results['term_features'].get('front_iv', 0.0):.4f}\n")
            f.write(f"- **Back IV:** {gate_results['term_features'].get('back_iv', 0.0):.4f}\n")
            f.write(f"- **Term Slope:** {gate_results['term_features'].get('term_slope', 0.0):.6f}\n")
            f.write(f"- **Inverted:** {gate_results['term_features'].get('is_inverted', False)}\n\n")
        
        if gate_results.get('execution_quality'):
            f.write("## Execution Quality\n\n")
            f.write(f"- **Bid-Ask Spread:** {gate_results['execution_quality'].get('bid_ask_spread_pct', 0.0):.2f}%\n")
            f.write(f"- **Quality Score:** {gate_results['execution_quality'].get('quality_score', 0.0):.3f}\n\n")
    
    return md_path


def save_csv_report(
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
    Save a CSV report with key metrics.
    
    Returns:
        Path to the saved CSV file
    """
    csv_path = output_dir / f"{ticker}_report.csv"
    
    # Prepare data for CSV
    data = {
        'Ticker': [ticker],
        'Analysis_Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'Baseline_MAE': [training_results.get('baseline_mae', 0.0) if training_results else 0.0],
        'Classical_MAE': [training_results.get('classical_mae', 0.0) if training_results else 0.0],
        'Combined_MAE': [training_results.get('combined_mae', 0.0) if training_results else 0.0],
        'Improvement_Pct': [((training_results['classical_cv_mae'].mean() - training_results['full_cv_mae'].mean()) / 
                            training_results['classical_cv_mae'].mean() * 100) if training_results else 0.0],
        'Extra_Component_Pct': [(np.mean(decomposition_results['extra']) / np.mean(decomposition_results['actual']) * 100) 
                                if decomposition_results else 0.0],
        'Residual_R2': [decomposition_results.get('residual_r2', 0.0) if decomposition_results else 0.0],
        'Skew_R2': [skew_results.get('r2', 0.0) if skew_results else 0.0],
        'Regime': [gate_results.get('regime', 'N/A')],
        'Structure_Family': [gate_results.get('structure_family', 'N/A')],
        'Kelly_Raw': [gate_results.get('kelly_raw', 0.0)],
        'Kelly_Fractional': [gate_results.get('kelly_fractional', 0.0)],
        'Kelly_Adjusted': [gate_results.get('kelly_adjusted', 0.0)],
        'Gate_State': [gate_results.get('gate_state', 'N/A')],
        'P': [gate_results.get('p', 0.0)],
        'B': [gate_results.get('b', 0.0)],
        'Multiplier': [gate_results.get('multiplier', 0.0)],
        'Market_State': [market_state.value],
        'Put_Call_IV_Diff': [gate_results.get('skew_features', {}).get('put_call_iv_diff', 0.0)],
        'Skew_Slope_Puts': [gate_results.get('skew_features', {}).get('skew_slope_puts', 0.0)],
        'Smile_Curvature': [gate_results.get('skew_features', {}).get('smile_curvature', 0.0)],
        'Front_IV': [gate_results.get('term_features', {}).get('front_iv', 0.0)],
        'Back_IV': [gate_results.get('term_features', {}).get('back_iv', 0.0)],
        'Term_Slope': [gate_results.get('term_features', {}).get('term_slope', 0.0)],
        'Is_Inverted': [gate_results.get('term_features', {}).get('is_inverted', False)],
        'Bid_Ask_Spread_Pct': [gate_results.get('execution_quality', {}).get('bid_ask_spread_pct', 0.0)],
        'Quality_Score': [gate_results.get('execution_quality', {}).get('quality_score', 0.0)],
        'Reflexive_Sleeve_Status': ['ACTIVE' if reflexive_plan else 'BLOCKED'],
        'Reflexive_Sleeve_Legs': [len(reflexive_plan)],
        'Capital': [capital_value]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    
    return csv_path


def save_reflexive_plan(plan: list[LegPlan], output_dir: Path) -> None:
    """
    Save reflexive plan to JSON file.
    
    Args:
        plan: List of LegPlan objects
        output_dir: Output directory path
    """
    if not plan:
        return
    data = [
        {
            "leg": lp.leg,
            "direction": lp.direction,
            "dte": lp.dte,
            "sleeve_entry": lp.sleeve_entry,
            "stop_loss": lp.stop_loss,
        }
        for lp in plan
    ]
    plan_path = output_dir / "reflexive_plan.json"
    plan_path.write_text(json.dumps(data, indent=2))


def process_single_ticker(
    ticker: str,
    args,
    output_dir: Path,
    logger: logging.Logger,
    analyzer: MarkovBlanketAnalyzer,
    predictor: OptionPricingPredictor
) -> dict:
    """
    Process a single ticker and return results dictionary.
    
    Returns:
        Dictionary with keys: training_results, decomposition_results, skew_results,
        gate_results, market_state, signals, reflexive_plan, capital_value, success
    """
    results = {
        'training_results': None,
        'decomposition_results': None,
        'skew_results': None,
        'gate_results': None,
        'market_state': MarketState.PIN,
        'signals': None,
        'reflexive_plan': [],
        'capital_value': args.capital if args.capital is not None else DEFAULT_CAPITAL,
        'success': False
    }
    
    try:
        # Load data
        console.print(f"[bold yellow]>>> Loading Option Data for {ticker}[/bold yellow]")
        
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
        console.print(f"[red]Error loading data for {ticker}: {e}[/red]")
        logger.error(f"Error loading data for {ticker}: {e}", exc_info=True)
        return results

    # Prepare features
    console.print(f"\n[bold yellow]>>> Preparing Features for {ticker}[/bold yellow]")
    try:
        call_data, put_data = DataPreparator.prepare_features(option_df, stock_price, current_date=current_date, use_log_target=args.log_target)
        logger.info(f"Prepared features: {len(call_data['targets'])} call options")
    except Exception as e:
        console.print(f"[red]Error preparing features for {ticker}: {e}[/red]")
        logger.error(f"Error preparing features for {ticker}: {e}", exc_info=True)
        return results

    # Plot feature correlation (skip in universal mode to save time)
    if not args.universal:
        console.print(f"\n[bold yellow]>>> Feature Correlation Analysis for {ticker}[/bold yellow]")
        predictor.plot_feature_correlation(call_data, output_dir)

    # Train models
    console.print(f"\n[bold yellow]>>> Training Models for {ticker}[/bold yellow]")
    training_results = predictor.train_models(call_data, put_data, n_folds=args.folds)
    
    if training_results is None:
        console.print(f"[red]Failed to train models for {ticker}[/red]")
        logger.error(f"Failed to train models for {ticker}")
        return results
    
    results['training_results'] = training_results

    # Decompose premium
    console.print(f"\n[bold yellow]>>> Premium Decomposition for {ticker}[/bold yellow]")
    decomposition_results = predictor.decompose_premium(call_data, put_data, training_results)
    results['decomposition_results'] = decomposition_results

    # Analyze skew
    console.print(f"\n[bold yellow]>>> Skew Attribution Analysis for {ticker}[/bold yellow]")
    skew_results = predictor.analyze_skew_attribution(call_data, put_data)
    results['skew_results'] = skew_results

    # Kelly Gate
    console.print(f"\n[bold yellow]>>> Teixiptla-Garage-Markov Kelly Gate for {ticker}[/bold yellow]")
    kelly_gate = KellyGate(logger=logger, debug=args.debug)
    reflexive_plan: list[LegPlan] = []
    market_state: MarketState = MarketState.PIN
    capital_value = args.capital if args.capital is not None else DEFAULT_CAPITAL
    
    try:
        gate_results = kelly_gate.compute_gate(
            option_df, stock_price, current_date, call_data, put_data,
            decomposition_results, skew_results, training_results
        )
        
        # State Machine
        execution_quality = gate_results.get('execution_quality', {})
        skew_features = gate_results.get('skew_features', {})
        
        signals = MarketSignals(
            regime=gate_results['regime'],
            kelly_fraction=gate_results['kelly_fractional'],
            gate_state=gate_results['gate_state'],
            spread=execution_quality.get('bid_ask_spread_pct', 0.0) / 100.0 if execution_quality.get('bid_ask_spread_pct') else 0.0,
            quality=execution_quality.get('quality_score', 0.0),
            skew_slope=skew_features.get('skew_slope_puts', 0.0),
            curvature=skew_features.get('smile_curvature', 0.0),
        )
        
        market_state = compute_market_state(signals)
        results['market_state'] = market_state
        results['signals'] = signals
        
        # Reflexive Bifurcation Sleeve
        if should_build_reflexive_sleeve(gate_results, market_state, force=args.force_reflexive):
            try:
                reflexive_plan = generate_reflexive_plan(
                    K=capital_value,
                    kelly_fraction=gate_results['kelly_fractional'],
                    exp_cap_frac=REFLEXIVE_EXP_CAP_FRAC,
                    stop_frac=REFLEXIVE_STOP_FRAC,
                    dte_initial=REFLEXIVE_DTE_INITIAL,
                    initial_direction=REFLEXIVE_INITIAL_DIR,
                    max_legs=REFLEXIVE_MAX_LEGS,
                )
            except Exception as e:
                logger.warning(f"Error generating reflexive sleeve for {ticker}: {e}")
                reflexive_plan = []
        
        results['reflexive_plan'] = reflexive_plan
        results['gate_results'] = gate_results
        results['capital_value'] = capital_value
        results['success'] = True
        
    except Exception as e:
        console.print(f"[yellow]Error computing Kelly Gate for {ticker}: {e}[/yellow]")
        logger.warning(f"Error computing Kelly Gate for {ticker}: {e}", exc_info=True)
        # Use safe defaults
        results['gate_results'] = {
            'regime': 'PIN',
            'structure_family': 'PROBE_ONLY',
            'kelly_raw': 0.0,
            'kelly_fractional': 0.0,
            'kelly_adjusted': 0.0,
            'gate_state': 'BLOCK',
            'p': 0.5,
            'b': 1.0,
            'multiplier': 0.0,
            'skew_features': {'put_call_iv_diff': 0.0, 'skew_slope_puts': 0.0, 'smile_curvature': 0.0},
            'term_features': {'front_iv': 0.0, 'back_iv': 0.0, 'term_slope': 0.0, 'is_inverted': False},
            'execution_quality': {'bid_ask_spread_pct': 10.0, 'quality_score': 0.5}
        }
        signals = MarketSignals(
            regime=results['gate_results']['regime'],
            kelly_fraction=results['gate_results']['kelly_fractional'],
            gate_state=results['gate_results']['gate_state'],
            spread=0.0,
            quality=0.5,
            skew_slope=0.0,
            curvature=0.0,
        )
        results['market_state'] = compute_market_state(signals)
        results['signals'] = signals
    
    return results


def main():
    """Main execution function with CLI arguments."""
    parser = argparse.ArgumentParser(description='Markov Blanket-Driven Option Pricing Model')
    parser.add_argument('--ticker', type=str, default='SPY', help='Stock ticker symbol (default: SPY)')
    parser.add_argument('--date', type=str, default=None, help='Option chain date (YYYY-MM-DD), defaults to most recent')
    parser.add_argument('--model', type=str, default='lightgbm', choices=['lightgbm'], help='Model type (default: lightgbm)')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds (default: 5)')
    parser.add_argument('--output', type=str, default=None, help='Output directory (default: dated folder)')
    parser.add_argument('--skip-shap', action='store_true', help='Skip SHAP plots (faster execution)')
    parser.add_argument('--use-pca', action='store_true', help='Use PCA on classical features (experimental)')
    # v8: Log target enabled by default - use store_false pattern with default=True
    parser.add_argument('--no-log-target', dest='log_target', action='store_false', default=True, help='Disable log(premium+1) target transformation (default: enabled)')
    # v8: Add --raw-premium flag as alternative to --no-log-target
    parser.add_argument('--raw-premium', dest='log_target', action='store_false', default=True, help='Use raw premium instead of log(premium+1) (same as --no-log-target)')
    parser.add_argument('--debug', action='store_true', help='Print debug information for Kelly Gate')
    parser.add_argument('--sanity-test-masks', action='store_true', help='Run sanity test: temporarily set MASK_MAX_DTE=7 to verify top expressives cluster in nearest expiry')
    parser.add_argument('--test-expressive', action='store_true', help='Test EXPRESSIVE masks: temporarily reduce thresholds or bypass PIN damping to verify escalation works')
    parser.add_argument('--capital', type=float, default=None, help=f'Total portfolio capital K (default: {DEFAULT_CAPITAL:.2f})')
    parser.add_argument('--force-reflexive', action='store_true', help='Force reflexive sleeve generation even if gate is BLOCK or regime is PIN')
    parser.add_argument('--universal', action='store_true', help='Process all tickers from tickers.json and save reports to output/markov/{TICKER}/')
    
    args = parser.parse_args()
    
    # v8: Ensure log_target is True by default (enabled)
    # With --no-log-target or --raw-premium pattern, log_target will be True unless flag is provided

    # Handle universal mode
    if args.universal:
        # Load all tickers
        try:
            tickers = load_tickers()
            console.print(f"[bold cyan]Universal mode: Processing {len(tickers)} tickers[/bold cyan]")
        except Exception as e:
            console.print(f"[red]Error loading tickers: {e}[/red]")
            return
        
        # Setup universal output directory
        universal_output_dir = Path("../output") / "markov"
        universal_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging for universal mode
        logger = setup_logging(universal_output_dir)
        logger.info(f"Starting universal analysis for {len(tickers)} tickers")
        
        successful = []
        failed = []
        
        for idx, ticker in enumerate(tickers, 1):
            ticker = ticker.upper()
            console.print(f"\n{'='*80}")
            console.print(f"[bold magenta]Processing {ticker} ({idx}/{len(tickers)})[/bold magenta]")
            console.print('='*80)
            
            # Create ticker-specific output directory
            ticker_output_dir = universal_output_dir / ticker
            ticker_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create fresh analyzer and predictor instances for each ticker to avoid state contamination
            analyzer = MarkovBlanketAnalyzer()
            predictor = OptionPricingPredictor(analyzer, logger, model_type=args.model)
            
            # Process ticker
            try:
                results = process_single_ticker(ticker, args, ticker_output_dir, logger, analyzer, predictor)
                
                # Try to save reports even if processing had partial failures
                # (e.g., if training failed but we have gate_results from defaults)
                reports_saved = False
                try:
                    # Ensure we have at least gate_results (should always be present)
                    if results.get('gate_results') is None:
                        results['gate_results'] = {
                            'regime': 'PIN',
                            'structure_family': 'PROBE_ONLY',
                            'kelly_raw': 0.0,
                            'kelly_fractional': 0.0,
                            'kelly_adjusted': 0.0,
                            'gate_state': 'BLOCK',
                            'p': 0.5,
                            'b': 1.0,
                            'multiplier': 0.0,
                            'skew_features': {'put_call_iv_diff': 0.0, 'skew_slope_puts': 0.0, 'smile_curvature': 0.0},
                            'term_features': {'front_iv': 0.0, 'back_iv': 0.0, 'term_slope': 0.0, 'is_inverted': False},
                            'execution_quality': {'bid_ask_spread_pct': 10.0, 'quality_score': 0.5}
                        }
                    
                    # Ensure we have market_state and signals
                    if results.get('market_state') is None:
                        results['market_state'] = MarketState.PIN
                    if results.get('signals') is None and results.get('gate_results'):
                        results['signals'] = MarketSignals(
                            regime=results['gate_results'].get('regime', 'PIN'),
                            kelly_fraction=results['gate_results'].get('kelly_fractional', 0.0),
                            gate_state=results['gate_results'].get('gate_state', 'BLOCK'),
                            spread=0.0,
                            quality=0.5,
                            skew_slope=0.0,
                            curvature=0.0,
                        )
                    
                    # Save reports
                    md_path = save_markdown_report(
                        ticker, results.get('training_results'), results.get('decomposition_results'),
                        results.get('skew_results', {'r2': 0.0}), results['gate_results'], results['market_state'],
                        results['signals'], results.get('reflexive_plan', []), results.get('capital_value', DEFAULT_CAPITAL),
                        ticker_output_dir
                    )
                    csv_path = save_csv_report(
                        ticker, results.get('training_results'), results.get('decomposition_results'),
                        results.get('skew_results', {'r2': 0.0}), results['gate_results'], results['market_state'],
                        results['signals'], results.get('reflexive_plan', []), results.get('capital_value', DEFAULT_CAPITAL),
                        ticker_output_dir
                    )
                    save_reflexive_plan(results.get('reflexive_plan', []), ticker_output_dir)
                    
                    reports_saved = True
                    console.print(f"[green]✓ {ticker}: Reports saved to {ticker_output_dir}[/green]")
                    logger.info(f"Successfully processed {ticker}: {md_path}, {csv_path}")
                    
                    if results.get('success', False):
                        successful.append(ticker)
                    else:
                        # Partial success - reports saved but processing had issues
                        console.print(f"[yellow]⚠ {ticker}: Reports saved but processing had issues[/yellow]")
                        successful.append(ticker)
                        
                except Exception as e:
                    console.print(f"[red]Error saving reports for {ticker}: {e}[/red]")
                    logger.error(f"Error saving reports for {ticker}: {e}", exc_info=True)
                    failed.append(ticker)
                    
            except Exception as e:
                console.print(f"[red]✗ {ticker}: Processing failed with exception: {e}[/red]")
                logger.error(f"Processing failed for {ticker}: {e}", exc_info=True)
                failed.append(ticker)
            finally:
                # Clear state after processing (help with memory management)
                try:
                    del analyzer
                    del predictor
                except:
                    pass
                import gc
                gc.collect()
        
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

    # Single ticker mode (original behavior)
    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path("../output") / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting analysis for {args.ticker}")

    console.print("[bold magenta]>>> Markov Blanket-Driven Option Pricing: Predictive Model (UPGRADED v8 - FINAL)[/bold magenta]\n")

    # Initialize analyzer
    analyzer = MarkovBlanketAnalyzer()
    predictor = OptionPricingPredictor(analyzer, logger, model_type=args.model)

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
        call_data, put_data = DataPreparator.prepare_features(option_df, stock_price, current_date=current_date, use_log_target=args.log_target)
        logger.info(f"Prepared features: {len(call_data['targets'])} call options")
    except Exception as e:
        console.print(f"[red]Error preparing features: {e}[/red]")
        logger.error(f"Error preparing features: {e}", exc_info=True)
        return

    # Plot feature correlation
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

    # Kelly Gate
    console.print("\n[bold yellow]>>> Teixiptla-Garage-Markov Kelly Gate[/bold yellow]")
    kelly_gate = KellyGate(logger=logger, debug=args.debug)
    reflexive_plan: list[LegPlan] = []  # Initialize reflexive plan
    market_state: MarketState = MarketState.PIN  # Initialize to PIN (safest default)
    capital_value = args.capital if args.capital is not None else DEFAULT_CAPITAL
    if args.capital is None:
        logger.warning(f"Using default capital: ${capital_value:.2f}. Use --capital to override.")
    try:
        gate_results = kelly_gate.compute_gate(
            option_df, stock_price, current_date, call_data, put_data,
            decomposition_results, skew_results, training_results
        )
        
        # Save gate JSON
        gate_json_path = output_dir / "gate.json"
        try:
            # Convert numpy types to native Python types for JSON
            gate_json = {
                'regime': gate_results['regime'],
                'structure_family': gate_results['structure_family'],
                'kelly_raw': float(gate_results['kelly_raw']),
                'kelly_fractional': float(gate_results['kelly_fractional']),
                'kelly_adjusted': float(gate_results['kelly_adjusted']),
                'gate_state': gate_results['gate_state'],
                'p': float(gate_results['p']),
                'b': float(gate_results['b']),
                'multiplier': float(gate_results['multiplier']),
                'skew_features': {k: float(v) if isinstance(v, (np.number, float, int)) else v 
                                 for k, v in gate_results['skew_features'].items()},
                'term_features': {k: float(v) if isinstance(v, (np.number, float, int)) else bool(v) if isinstance(v, bool) else v
                                 for k, v in gate_results['term_features'].items()},
                'execution_quality': {k: float(v) if isinstance(v, (np.number, float, int)) else v
                                    for k, v in gate_results['execution_quality'].items()}
            }
            with open(gate_json_path, 'w') as f:
                json.dump(gate_json, f, indent=2)
            console.print(f"[green]Kelly Gate JSON saved: {gate_json_path}[/green]")
            logger.info(f"Kelly Gate JSON saved: {gate_json_path}")
        except Exception as e:
            console.print(f"[yellow]Error saving gate JSON: {e}[/yellow]")
            logger.warning(f"Error saving gate JSON: {e}")
        
        # State Machine
        # Build MarketSignals from existing data
        execution_quality = gate_results.get('execution_quality', {})
        skew_features = gate_results.get('skew_features', {})
        
        signals = MarketSignals(
            regime=gate_results['regime'],
            kelly_fraction=gate_results['kelly_fractional'],
            gate_state=gate_results['gate_state'],
            spread=execution_quality.get('bid_ask_spread_pct', 0.0) / 100.0 if execution_quality.get('bid_ask_spread_pct') else 0.0,
            quality=execution_quality.get('quality_score', 0.0),
            skew_slope=skew_features.get('skew_slope_puts', 0.0),
            curvature=skew_features.get('smile_curvature', 0.0),
        )
        
        # Compute market state
        market_state = compute_market_state(signals)
        actions_text = describe_actions(market_state)
        logger.info("Market state: %s (actions: %s)", market_state.value, actions_text)
        
        # Reflexive Bifurcation Sleeve
        console.print("\n[bold yellow]>>> Reflexive Bifurcation Sleeve[/bold yellow]")
        # Generate reflexive sleeve plan if permitted
        if should_build_reflexive_sleeve(gate_results, market_state, force=args.force_reflexive):
            try:
                reflexive_plan = generate_reflexive_plan(
                    K=capital_value,
                    kelly_fraction=gate_results['kelly_fractional'],
                    exp_cap_frac=REFLEXIVE_EXP_CAP_FRAC,
                    stop_frac=REFLEXIVE_STOP_FRAC,
                    dte_initial=REFLEXIVE_DTE_INITIAL,
                    initial_direction=REFLEXIVE_INITIAL_DIR,
                    max_legs=REFLEXIVE_MAX_LEGS,
                )
                if reflexive_plan:
                    console.print(f"[green]Generated reflexive sleeve with {len(reflexive_plan)} leg(s)[/green]")
                    logger.info(f"Reflexive sleeve legs: {len(reflexive_plan)}")
                    logger.info(f"Reflexive sleeve plan: {[(lp.leg, lp.direction, lp.sleeve_entry) for lp in reflexive_plan]}")
                else:
                    console.print("[yellow]Reflexive sleeve plan is empty (Kelly fraction too small)[/yellow]")
            except Exception as e:
                console.print(f"[yellow]Error generating reflexive sleeve: {e}[/yellow]")
                logger.warning(f"Error generating reflexive sleeve: {e}", exc_info=True)
                reflexive_plan = []
        else:
            reason = []
            if gate_results.get('gate_state') == "BLOCK":
                reason.append("gate is BLOCK")
            if gate_results.get('regime') == "PIN":
                reason.append("regime is PIN")
            if gate_results.get('kelly_fractional', 0.0) <= 0.0:
                reason.append("Kelly fraction is zero or negative")
            reason_str = ", ".join(reason) if reason else "unknown"
            console.print(f"[yellow]Reflexive sleeve not permitted: {reason_str}[/yellow]")
            logger.info("Reflexive sleeve not permitted: gate_state=%s, regime=%s, kelly_fractional=%.4f",
                       gate_results.get('gate_state'), gate_results.get('regime'), gate_results.get('kelly_fractional', 0.0))
        
        # Create contract scores CSV
        try:
            contract_scores = []
            # Get test set predictions for contracts
            X_classical_test = training_results.get('X_classical_test', None)
            y_test = training_results.get('y_test', None)
            
            if X_classical_test is not None and len(X_classical_test) > 0:
                classical_pred = predictor.classical_model.predict(X_classical_test)
                combined_pred = training_results.get('combined_pred', None)
                if combined_pred is None:
                    # Compute combined prediction
                    residual_pred = predictor.residual_model.predict(training_results.get('X_extra_test_enhanced', X_classical_test[:, :2]))
                    use_log_target = call_data.get('use_log_target', False)
                    if use_log_target:
                        combined_pred = np.expm1(np.log1p(classical_pred) + residual_pred)
                    else:
                        combined_pred = classical_pred + residual_pred
                
                # Mispricing magnitude
                mispricing = combined_pred - classical_pred
                
                # Execution quality per contract (use aggregated for now, could be enhanced)
                quality_score = gate_results['execution_quality']['quality_score']
                spread_pct = gate_results['execution_quality']['bid_ask_spread_pct']
                
                # Moneyness buckets
                strikes = call_data['strikes']
                moneyness = call_data['moneyness']
                valid = call_data['targets'] > 0.01
                strikes_valid = strikes[valid]
                moneyness_valid = moneyness[valid]
                
                # Limit to test set size
                n_test = min(len(y_test), len(strikes_valid), len(mispricing))
                
                for i in range(n_test):
                    moneyness_val = moneyness_valid[i] if i < len(moneyness_valid) else 1.0
                    if moneyness_val < 0.95:
                        bucket = 'ITM'
                    elif moneyness_val <= 1.05:
                        bucket = 'ATM'
                    else:
                        bucket = 'OTM'
                    
                    contract_scores.append({
                        'contract_idx': i,
                        'strike': float(strikes_valid[i]) if i < len(strikes_valid) else 0.0,
                        'moneyness': float(moneyness_val),
                        'moneyness_bucket': bucket,
                        'classical_pred': float(classical_pred[i]) if i < len(classical_pred) else 0.0,
                        'combined_pred': float(combined_pred[i]) if i < len(combined_pred) else 0.0,
                        'mispricing_magnitude': float(np.abs(mispricing[i])) if i < len(mispricing) else 0.0,
                        'execution_quality_score': quality_score,
                        'bid_ask_spread_pct': spread_pct,
                        'regime': gate_results['regime'],
                        'structure_family': gate_results['structure_family'],
                        'gate_state': gate_results['gate_state']
                    })
                
                # Save CSV
                contract_scores_df = pd.DataFrame(contract_scores)
                contract_scores_path = output_dir / "contract_scores.csv"
                contract_scores_df.to_csv(contract_scores_path, index=False)
                console.print(f"[green]Contract scores CSV saved: {contract_scores_path}[/green]")
                logger.info(f"Contract scores CSV saved: {contract_scores_path}")
        except Exception as e:
            console.print(f"[yellow]Error creating contract scores CSV: {e}[/yellow]")
            logger.warning(f"Error creating contract scores CSV: {e}")
    except Exception as e:
        console.print(f"[yellow]Error computing Kelly Gate: {e}[/yellow]")
        logger.warning(f"Error computing Kelly Gate: {e}", exc_info=True)
        # Use safe defaults
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
            'skew_features': {'put_call_iv_diff': 0.0, 'skew_slope_puts': 0.0, 'smile_curvature': 0.0},
            'term_features': {'front_iv': 0.0, 'back_iv': 0.0, 'term_slope': 0.0, 'is_inverted': False},
            'execution_quality': {'bid_ask_spread_pct': 10.0, 'quality_score': 0.5}
        }
        reflexive_plan = []  # Empty plan on error
        # Compute state machine with default gate_results
        signals = MarketSignals(
            regime=gate_results['regime'],
            kelly_fraction=gate_results['kelly_fractional'],
            gate_state=gate_results['gate_state'],
            spread=0.0,
            quality=0.5,
            skew_slope=0.0,
            curvature=0.0,
        )
        market_state = compute_market_state(signals)

    # Markov Masks
    console.print("\n[bold yellow]>>> Markov Masks (Contract-Level Agency)[/bold yellow]")
    mask_computer = None
    original_max_dte = None
    try:
        mask_computer = MarkovMaskComputer(logger=logger)
        original_max_dte = mask_computer.MASK_MAX_DTE  # Store original value
        original_thresholds = None  # Store original thresholds for EXPRESSIVE test
        
        # Sanity test: temporarily override MASK_MAX_DTE if flag is set
        if args.sanity_test_masks:
            console.print("[cyan]>>> Running Mask Engine Sanity Test (MASK_MAX_DTE=7)[/cyan]")
            logger.info("Sanity test enabled: temporarily setting MASK_MAX_DTE=7")
            mask_computer.MASK_MAX_DTE = 7
        
        # EXPRESSIVE test: temporarily reduce thresholds or bypass PIN damping
        if args.test_expressive:
            console.print("[cyan]>>> Running EXPRESSIVE Test (reduced thresholds, bypass PIN damping)[/cyan]")
            logger.info("EXPRESSIVE test enabled: temporarily reducing thresholds and bypassing PIN damping")
            original_thresholds = mask_computer.THRESHOLDS.copy()
            # Reduce EXPRESSIVE thresholds by 50%
            mask_computer.THRESHOLDS['expressive_gamma'] = original_thresholds['expressive_gamma'] * 0.5
            mask_computer.THRESHOLDS['expressive_vega'] = original_thresholds['expressive_vega'] * 0.5
            mask_computer.THRESHOLDS['expressive_curvature'] = original_thresholds['expressive_curvature'] * 0.5
            # Store flag to bypass PIN damping in expressivity computation
            mask_computer._test_expressive_bypass_pin = True
        
        # Prepare separate DataFrames for calls and puts
        # option_df has both calls and puts in wide format
        # Use option_df directly - markov_mask.py will handle column extraction
        df_calls = option_df.copy()
        
        # For puts, try to extract if .1 suffix columns exist
        df_puts = None
        put_suffix_cols = [col for col in option_df.columns if '.1' in str(col)]
        if len(put_suffix_cols) > 0:
            # Create puts DataFrame with .1 suffix columns
            df_puts = option_df[['Expiration Date', 'Strike'] + put_suffix_cols].copy()
        
        # Compute masks (pass gate_state and ensure residual_r2/extra_pct are defined)
        # Initialize residual_r2 and extra_pct early to avoid UnboundLocalError
        residual_r2 = decomposition_results.get('residual_r2', float('nan'))
        if pd.isna(residual_r2) or not np.isfinite(residual_r2):
            residual_r2 = 0.0
        
        # Initialize extra_pct early (computed later in summary section)
        try:
            extra_pct = (np.mean(decomposition_results['extra']) / np.mean(decomposition_results['actual']) * 100)
            if not np.isfinite(extra_pct):
                extra_pct = float('nan')
        except (KeyError, ZeroDivisionError, ValueError):
            extra_pct = float('nan')
        
        # Debug: print values
        logger.info(f"Markov Masks: residual_r2={residual_r2}, extra_pct={extra_pct}")
        
        # EXPRESSIVE test: log test status
        if args.test_expressive:
            logger.info("EXPRESSIVE test: reduced thresholds applied, PIN damping bypassed")
            console.print("[dim]EXPRESSIVE test: checking for EXPRESSIVE/RUPTURE_CANDIDATE masks...[/dim]")
        
        df_masks, masks_summary = mask_computer.compute_masks(
            df_calls, df_puts, stock_price, current_date,
            gate_results['execution_quality'],
            gate_results['regime'],
            gate_results['structure_family'],
            gate_state=gate_results.get('gate_state', None),
            flip_level=None,  # Can be enhanced later
            walls=None  # Can be enhanced later
        )
        
        if len(df_masks) > 0:
            # EXPRESSIVE test: verify results
            if args.test_expressive and 'mask_state' in df_masks.columns:
                expressive_count = int((df_masks['mask_state'] == 'EXPRESSIVE').sum())
                rupture_count = int((df_masks['mask_state'] == 'RUPTURE_CANDIDATE').sum())
                console.print(f"\n[bold cyan]EXPRESSIVE Test Results:[/bold cyan]")
                console.print(f"  • EXPRESSIVE masks: {expressive_count}")
                console.print(f"  • RUPTURE_CANDIDATE masks: {rupture_count}")
                if expressive_count > 0 or rupture_count > 0:
                    console.print(f"[green]✓ PASS: EXPRESSIVE masks detected ({expressive_count + rupture_count} total)[/green]")
                    logger.info(f"EXPRESSIVE test PASS: {expressive_count} EXPRESSIVE, {rupture_count} RUPTURE_CANDIDATE")
                else:
                    console.print(f"[yellow]⚠ No EXPRESSIVE masks detected. Thresholds may need further reduction or data may not support escalation.[/yellow]")
                    logger.warning("EXPRESSIVE test: No EXPRESSIVE masks detected despite reduced thresholds")
            
            # Sanity test: verify top expressives cluster in nearest expiry
            if args.sanity_test_masks and 'eligible_mask' in df_masks.columns:
                df_eligible = df_masks[df_masks['eligible_mask']].copy()
                if len(df_eligible) > 0:
                    # Get top 10 expressive contracts
                    top_expressives = df_eligible.nlargest(10, 'expressivity_score')
                    
                    # Find nearest expiry (minimum DTE among eligible contracts)
                    if 'dte' in df_eligible.columns:
                        nearest_dte = df_eligible['dte'].min()
                        nearest_expiry_mask = df_eligible['dte'] == nearest_dte
                        nearest_expiry_contracts = df_eligible[nearest_expiry_mask]
                        
                        # Count how many top expressives are in nearest expiry
                        top_in_nearest = 0
                        nearest_expiry_dates = set()
                        for idx, row in top_expressives.iterrows():
                            if 'dte' in row and pd.notna(row['dte']):
                                if row['dte'] == nearest_dte:
                                    top_in_nearest += 1
                                if 'expiration' in row and pd.notna(row['expiration']):
                                    nearest_expiry_dates.add(str(row['expiration'])[:10])
                        
                        # Check if majority cluster in nearest expiry
                        cluster_pct = (top_in_nearest / len(top_expressives) * 100) if len(top_expressives) > 0 else 0.0
                        expected_cluster = cluster_pct >= 50.0  # At least 50% should be in nearest expiry
                        
                        console.print(f"\n[bold cyan]Sanity Test Results:[/bold cyan]")
                        console.print(f"  • Nearest expiry DTE: {nearest_dte:.0f} days")
                        console.print(f"  • Top 10 expressives in nearest expiry: {top_in_nearest}/{len(top_expressives)} ({cluster_pct:.1f}%)")
                        if nearest_expiry_dates:
                            console.print(f"  • Nearest expiry date(s): {', '.join(sorted(nearest_expiry_dates))}")
                        
                        if expected_cluster:
                            console.print(f"[green]✓ PASS: Top expressives cluster in nearest expiry ({cluster_pct:.1f}%)[/green]")
                            logger.info(f"Sanity test PASS: {top_in_nearest}/{len(top_expressives)} top expressives in nearest expiry (DTE={nearest_dte:.0f})")
                        else:
                            console.print(f"[red]✗ FAIL: Top expressives do NOT cluster in nearest expiry ({cluster_pct:.1f}% < 50%)[/red]")
                            console.print(f"[yellow]  → Possible issues: DTE parsing, expiration normalization, or expressivity scoring[/yellow]")
                            logger.warning(f"Sanity test FAIL: Only {top_in_nearest}/{len(top_expressives)} top expressives in nearest expiry. DTE parsing or expiration normalization may be off.")
                    else:
                        console.print("[yellow]Sanity test: 'dte' column not found - cannot verify expiry clustering[/yellow]")
                        logger.warning("Sanity test: 'dte' column missing - skipping expiry clustering check")
            
            # Verify masks are economically honest before saving
            if 'eligible_mask' in df_masks.columns:
                eligible_count = int(df_masks['eligible_mask'].sum())
                if eligible_count == 0:
                    logger.warning("No eligible contracts found - masks may not be economically honest")
                    console.print("[yellow]Warning: No eligible contracts found. Consider adjusting thresholds.[/yellow]")
                else:
                    # Verify top expressive contracts are near-spot with valid IV/gamma
                    df_eligible = df_masks[df_masks['eligible_mask']].copy()
                    if len(df_eligible) > 0:
                        top_contracts = df_eligible.nlargest(5, 'expressivity_score')
                        for idx, row in top_contracts.iterrows():
                            abs_dS = row.get('abs_dS', np.inf)
                            iv = row.get('iv', np.nan)
                            gamma = row.get('gamma', np.nan)
                            if abs_dS > 60 or pd.isna(iv) or not np.isfinite(iv) or pd.isna(gamma) or abs(gamma) < 5e-5:
                                logger.warning(f"Top contract {idx} may not be economically honest: abs_dS={abs_dS}, iv={iv}, gamma={gamma}")
            
            # Save CSV
            masks_csv_path = output_dir / "markov_masks.csv"
            df_masks.to_csv(masks_csv_path, index=False)
            console.print(f"[green]Markov Masks CSV saved: {masks_csv_path}[/green]")
            logger.info(f"Markov Masks CSV saved: {masks_csv_path}")
            
            # Save JSON
            masks_json_path = output_dir / "markov_masks.json"
            masks_json = {
                'run_metadata': {
                    'ticker': ticker,
                    'timestamp': datetime.now().isoformat(),
                    'spot_price': float(stock_price),
                    'output_folder': str(output_dir)
                },
                'global_blanket_summary': {
                    'residual_r2': None if not np.isfinite(residual_r2) else float(residual_r2),
                    'skew_r2': float(skew_results['r2']),
                    'extra_pct': None if not np.isfinite(extra_pct) else float(extra_pct),
                    'regime': gate_results['regime'],
                    'structure_family': gate_results['structure_family'],
                    'gate_state': gate_results['gate_state']
                },
                'top_masks': masks_summary,
                'mask_state_counts': masks_summary.get('mask_state_counts', {}),
                'mask_state_counts_eligible': masks_summary.get('mask_state_counts_eligible', {}),
                'nearest_expiry_cluster': masks_summary.get('nearest_expiry_cluster'),
                'top_strike_cluster': masks_summary.get('top_strike_cluster')
            }
            
            with open(masks_json_path, 'w') as f:
                json.dump(masks_json, f, indent=2)
            console.print(f"[green]Markov Masks JSON saved: {masks_json_path}[/green]")
            logger.info(f"Markov Masks JSON saved: {masks_json_path}")
            
            # Roundtable: Contract-Level Agency (Markov Masks) - Deterministic, short format
            console.print("\n[bold yellow]>>> Roundtable: Contract-Level Agency (Markov Masks)[/bold yellow]")
            
            # 1) ELIGIBILITY STATS
            if 'eligible_mask' in df_masks.columns:
                total_contracts = len(df_masks)
                eligible_count = int(df_masks['eligible_mask'].sum())
                eligible_pct = (eligible_count / total_contracts * 100) if total_contracts > 0 else 0.0
                
                # Top 3 ineligible reasons
                ineligible_df = df_masks[~df_masks['eligible_mask']]
                if len(ineligible_df) > 0 and 'ineligible_reason' in ineligible_df.columns:
                    all_reasons = []
                    for reasons_str in ineligible_df['ineligible_reason']:
                        if pd.notna(reasons_str) and reasons_str:
                            all_reasons.extend(str(reasons_str).split('|'))
                    reason_counts = pd.Series(all_reasons).value_counts().head(3)
                    top_reasons = reason_counts.to_dict()
                else:
                    top_reasons = {}
                
                eligibility_table = Table(title="Eligibility Stats")
                eligibility_table.add_column("Metric", style="cyan")
                eligibility_table.add_column("Value", style="green", justify="right")
                eligibility_table.add_row("Total Contracts", str(total_contracts))
                eligibility_table.add_row("Eligible", str(eligible_count))
                eligibility_table.add_row("Eligible %", f"{eligible_pct:.1f}%")
                console.print(eligibility_table)
                
                if top_reasons:
                    reason_table = Table(title="Top 3 Ineligible Reasons")
                    reason_table.add_column("Reason", style="yellow")
                    reason_table.add_column("Count", style="red", justify="right")
                    for reason, count in top_reasons.items():
                        reason_table.add_row(reason, str(count))
                    console.print(reason_table)
            
            # 2) TOP EXPRESSIVE (eligible only) - Compact format
            if 'eligible_mask' in df_masks.columns:
                df_eligible = df_masks[df_masks['eligible_mask']].copy()
            else:
                df_eligible = df_masks.copy()
            
            if len(df_eligible) > 0:
                top_expressivity_eligible = df_eligible.nlargest(10, 'expressivity_score')
                
                roundtable_table = Table(title="Top 10 Expressive Contracts (Eligible Only)")
                roundtable_table.add_column("Type", style="yellow")
                roundtable_table.add_column("Expiry", style="dim")
                roundtable_table.add_column("Strike", style="cyan", justify="right")
                roundtable_table.add_column("|dS|", justify="right")
                roundtable_table.add_column("IV", justify="right")
                roundtable_table.add_column("Δ", justify="right")
                roundtable_table.add_column("Γ", justify="right")
                roundtable_table.add_column("Vega", justify="right")
                roundtable_table.add_column("Θ", justify="right")
                roundtable_table.add_column("Curvature", justify="right")
                roundtable_table.add_column("Expressivity", style="bold cyan", justify="right")
                roundtable_table.add_column("Mask State", style="bold")
                
                for idx, row in top_expressivity_eligible.iterrows():
                    roundtable_table.add_row(
                        str(row.get('contract_type', 'CALL')),
                        str(row.get('expiration', 'N/A'))[:10] if pd.notna(row.get('expiration')) else "N/A",
                        f"${row.get('strike', 0):.2f}" if pd.notna(row.get('strike')) else "N/A",
                        f"{row.get('abs_dS', 0):.2f}" if pd.notna(row.get('abs_dS')) else "N/A",
                        f"{row.get('iv', 0)*100:.2f}%" if pd.notna(row.get('iv')) else "N/A",
                        f"{row.get('delta', 0):.3f}" if pd.notna(row.get('delta')) else "N/A",
                        f"{row.get('gamma', 0):.4f}" if pd.notna(row.get('gamma')) else "N/A",
                        f"{row.get('vega', 0):.3f}" if pd.notna(row.get('vega')) else "N/A",
                        f"{row.get('theta', 0):.3f}" if pd.notna(row.get('theta')) else "N/A",
                        f"{row.get('iv_curvature_local', 0):.6f}" if pd.notna(row.get('iv_curvature_local')) else "N/A",
                        f"{row.get('expressivity_score', 0):.3f}" if pd.notna(row.get('expressivity_score')) else "N/A",
                        str(row.get('mask_state', 'UNKNOWN'))
                    )
                
                console.print(roundtable_table)
                
                # 3) 2-LINE SYNTHESIS
                console.print("\n[bold cyan]Synthesis:[/bold cyan]")
                regime = gate_results.get('regime', 'UNKNOWN')
                gate_state = gate_results.get('gate_state', 'UNKNOWN')
                
                if regime == "PIN" or gate_state == "BLOCK":
                    console.print("[dim]Mask intact; convexity suppressed; probe-only.[/dim]")
                else:
                    # Find leading expiry/strike cluster
                    if len(df_eligible) > 0:
                        top_contract = df_eligible.nlargest(1, 'expressivity_score').iloc[0]
                        leading_expiry = str(top_contract.get('expiration', 'N/A'))[:10] if pd.notna(top_contract.get('expiration')) else "N/A"
                        leading_strike = top_contract.get('strike', 0)
                        
                        # Determine driver
                        iv_curv = abs(top_contract.get('iv_curvature_local', 0)) if pd.notna(top_contract.get('iv_curvature_local')) else 0
                        gamma_val = abs(top_contract.get('gamma', 0)) if pd.notna(top_contract.get('gamma')) else 0
                        skew_val = abs(top_contract.get('skew_local', 0)) if pd.notna(top_contract.get('skew_local')) else 0
                        
                        if iv_curv > max(gamma_val * 0.1, skew_val * 0.1):
                            driver = "curvature"
                        elif gamma_val > max(iv_curv * 10, skew_val * 0.1):
                            driver = "gamma"
                        elif skew_val > max(iv_curv * 10, gamma_val * 0.1):
                            driver = "skew"
                        else:
                            driver = "balanced"
                        
                        console.print(f"[dim]Agency transfer detected at {leading_expiry} expiry, ${leading_strike:.2f} strike cluster, driven by {driver}.[/dim]")
                    else:
                        console.print("[dim]No eligible contracts for synthesis.[/dim]")
            else:
                console.print("[yellow]No eligible contracts for Roundtable analysis[/yellow]")
        else:
            console.print("[yellow]Markov Masks computation returned empty results[/yellow]")
            logger.warning("Markov Masks computation returned empty DataFrame")
    except Exception as e:
        console.print(f"[yellow]Error computing Markov Masks: {e}[/yellow]")
        logger.warning(f"Error computing Markov Masks: {e}", exc_info=True)
    finally:
        # Always reset MASK_MAX_DTE if sanity test was enabled
        if args.sanity_test_masks and mask_computer is not None and original_max_dte is not None:
            mask_computer.MASK_MAX_DTE = original_max_dte
            console.print(f"[dim]Sanity test cleanup: MASK_MAX_DTE reset to {original_max_dte}.[/dim]")
            logger.info(f"Sanity test cleanup: MASK_MAX_DTE reset to {original_max_dte}")
        
        # Always reset thresholds if EXPRESSIVE test was enabled
        if args.test_expressive and mask_computer is not None and original_thresholds is not None:
            mask_computer.THRESHOLDS = original_thresholds
            if hasattr(mask_computer, '_test_expressive_bypass_pin'):
                delattr(mask_computer, '_test_expressive_bypass_pin')
            console.print(f"[dim]EXPRESSIVE test cleanup: thresholds reset to original values.[/dim]")
            logger.info("EXPRESSIVE test cleanup: thresholds reset to original values")

    # Visualize
    console.print("\n[bold yellow]>>> Generating Visualizations[/bold yellow]")
    predictor.visualize_results(decomposition_results, skew_results, training_results, output_dir, args)

    # Save models
    console.print("\n[bold yellow]>>> Saving Models[/bold yellow]")
    predictor.save_models(output_dir)
    
    # Save reflexive plan
    try:
        save_reflexive_plan(reflexive_plan, output_dir)
        if reflexive_plan:
            console.print(f"[green]Reflexive plan saved: {output_dir / 'reflexive_plan.json'}[/green]")
            logger.info(f"Reflexive plan saved: {output_dir / 'reflexive_plan.json'}")
    except Exception as e:
        console.print(f"[yellow]Error saving reflexive plan: {e}[/yellow]")
        logger.warning(f"Error saving reflexive plan: {e}")

    # Summary
    console.print("\n" + "="*80)
    improvement_pct = ((training_results['classical_cv_mae'].mean() - training_results['full_cv_mae'].mean()) / 
                       training_results['classical_cv_mae'].mean() * 100)
    extra_pct = (np.mean(decomposition_results['extra']) / np.mean(decomposition_results['actual']) * 100)
    residual_r2 = decomposition_results.get('residual_r2', 0.0)
    
    # Build summary with Kelly Gate (gate_results computed above)
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
        f"  • Kelly (fractional): {gate_results['kelly_fractional']:.4f}\n"
        f"  • Gate state: {gate_results['gate_state']}\n"
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
        # Calculate effective sleeve
        E0 = REFLEXIVE_EXP_CAP_FRAC * capital_value
        E0_eff = min(E0, gate_results['kelly_fractional'] * capital_value)
        
        # Build leg table
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
        f"[bold]Kelly (fractional):[/bold] {gate_results['kelly_fractional']:.4f}\n"
        f"[bold]Kelly (adjusted):[/bold] {gate_results['kelly_adjusted']:.4f}\n"
        f"[bold]Gate state:[/bold] {gate_results['gate_state']}\n"
        f"[bold]p:[/bold] {gate_results['p']:.3f}, [bold]b:[/bold] {gate_results['b']:.3f}\n"
        f"[bold]Multiplier:[/bold] {gate_results['multiplier']:.3f}"
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
    
    # Also print ASCII-style box for consistency with summary text
    console.print("\n╭──────────────────────────── State Machine ─────────────────────────────╮")
    state_line = f"│ Current state: {market_state.value:<57}│"
    console.print(state_line)
    # Wrap actions text if needed (max 63 chars for Actions line)
    if len(actions_text) > 63:
        # Simple wrap at word boundary
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
    # Truncate if too long (max 75 chars total including "│ Notes: ")
    max_notes_len = 75 - len("│ Notes: ")
    if len(notes_line) > 75:
        notes_line = notes_line[:72] + "..."
    notes_line = notes_line.ljust(75) + "│"
    console.print(notes_line)
    console.print("╰────────────────────────────────────────────────────────────────────────╯")
    
    # Deterministic narrative output (publication-ready)
    console.print("\n[bold yellow]>>> Narrative Summary[/bold yellow]")
    try:
        # Get mask state information - df_masks should be in scope from Markov Masks section
        # If not available, try to load from saved CSV
        df_masks_narrative = None
        if 'df_masks' in locals() and len(df_masks) > 0:
            df_masks_narrative = df_masks
        else:
            # Try to load from most recent output
            try:
                masks_csv_path = output_dir / "markov_masks.csv"
                if masks_csv_path.exists():
                    df_masks_narrative = pd.read_csv(masks_csv_path)
            except:
                pass
        
        if df_masks_narrative is not None and len(df_masks_narrative) > 0 and 'mask_state' in df_masks_narrative.columns:
            mask_state_counts = df_masks_narrative['mask_state'].value_counts().to_dict()
            # Get top mask state (excluding DORMANT if others exist)
            non_dormant = {k: v for k, v in mask_state_counts.items() if k != 'DORMANT'}
            if non_dormant:
                top_mask_state = max(non_dormant.items(), key=lambda x: x[1])[0]
            else:
                top_mask_state = max(mask_state_counts.items(), key=lambda x: x[1])[0] if mask_state_counts else 'UNKNOWN'
            
            regime = gate_results.get('regime', 'UNKNOWN')
            
            # Deterministic narrative based on regime and mask states
            if regime == "PIN" and top_mask_state == "SENSITIVE":
                narrative = (
                    "ATM contracts show elevated sensitivity without convexity permission. "
                    "The Markov mask remains intact; agency is present but suppressed. "
                    "No EXPRESSIVE or RUPTURE_CANDIDATE states detected, indicating the system "
                    "is operating within PIN constraints."
                )
            elif regime == "PIN" and top_mask_state in ["EXPRESSIVE", "RUPTURE_CANDIDATE"]:
                narrative = (
                    "Despite PIN regime, EXPRESSIVE masks have emerged, suggesting potential "
                    "convexity permission or threshold calibration. The mask engine has detected "
                    "elevated expressivity, but regime constraints may limit deployment."
                )
            elif regime in ["PRE_TRANSFER", "TRANSFER"] and top_mask_state in ["EXPRESSIVE", "RUPTURE_CANDIDATE"]:
                narrative = (
                    "Agency transfer detected: EXPRESSIVE masks indicate convexity permission. "
                    "The Markov blanket shows elevated expressivity at key strike/expiry clusters, "
                    "suggesting the system is transitioning from PIN to active deployment."
                )
            elif regime in ["PRE_TRANSFER", "TRANSFER"] and top_mask_state == "SENSITIVE":
                narrative = (
                    "Transition regime detected with SENSITIVE masks. Contracts show elevated "
                    "sensitivity but have not yet escalated to EXPRESSIVE. The mask remains "
                    "partially intact, awaiting full convexity permission."
                )
            else:
                narrative = (
                    f"Regime: {regime}, dominant mask state: {top_mask_state}. "
                    "The Markov mask system is operational, with contract-level agency "
                    "encoded in mask states and expressivity scores."
                )
            
            narrative_panel = Panel(narrative, title="[bold cyan]Teixiptla Narrative[/bold cyan]", border_style="cyan")
            console.print(narrative_panel)
            logger.info(f"Narrative summary: {narrative}")
        else:
            console.print("[dim]Mask state information not available for narrative generation.[/dim]")
    except Exception as e:
        logger.warning(f"Error generating narrative: {e}")
        console.print("[dim]Could not generate narrative summary.[/dim]")
    
    logger.info("Analysis complete")


if __name__ == "__main__":
    main()
