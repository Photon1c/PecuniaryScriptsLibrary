"""
Markov Masks: Contract-Level Agency Detection

Computes per-contract "masks" that encode structural position, volatility geometry,
and sensitivity profiles to detect local agency transfer and expressivity.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from datetime import datetime
import logging


class MarkovMaskComputer:
    """Computes Markov masks for option contracts."""
    
    # Eligibility filter thresholds (tunable)
    MASK_MAX_ABS_DS = 60.0  # Maximum absolute distance from spot (points) - increased for wide strike ladders
    MASK_MIN_ABS_GAMMA = 5e-5  # Minimum absolute gamma for eligibility - reduced slightly
    MASK_MAX_DTE = 45  # Maximum days to expiration (optional filter)
    
    # Expressivity score weights (tunable)
    EXPRESSIVITY_WEIGHTS = {
        'proximity_to_spot': 0.20,
        'gamma_strength': 0.25,
        'vega_strength': 0.20,
        'curvature_signal': 0.15,
        'instability_gate': 0.10,
        'execution_quality': 0.10
    }
    
    # Mask state thresholds
    THRESHOLDS = {
        'dormant_gamma': 0.01,
        'dormant_vega': 0.05,
        'dormant_distance': 0.10,  # 10% from spot
        'sensitive_distance': 0.05,  # 5% from spot
        'sensitive_gamma': 0.05,
        'expressive_gamma': 0.10,
        'expressive_vega': 0.20,
        'expressive_curvature': 0.01,
        'toxic_spread_pct': 15.0,  # >15% spread = toxic
        'toxic_quality': 0.3  # quality < 0.3 = toxic
    }
    
    # IV regression window (strikes)
    IV_REGRESSION_WINDOW = 5
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
    
    def compute_eligibility_filter(self, df, spot_price, current_date, structural, volatility, sensitivity):
        """
        Compute eligibility mask: filters to economically-relevant, curvature-capable contracts.
        Returns eligible_mask (pandas Series) and ineligible_reason (pandas Series).
        """
        try:
            n = len(df)
            
            # Coerce numeric columns first
            numeric_cols = ['iv', 'delta', 'gamma', 'vega', 'theta', 'premium_mid', 'strike', 'dS', 'abs_dS']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Add structural and sensitivity data to df for easier access
            df['abs_dS'] = pd.Series(structural['abs_dS'])
            df['iv'] = pd.Series(volatility['iv_local'])
            df['gamma'] = pd.Series(sensitivity['gamma'])
            df['premium_mid'] = (pd.to_numeric(df.get('Bid', 0), errors='coerce') + 
                                 pd.to_numeric(df.get('Ask', 0), errors='coerce')) / 2.0
            
            # Build eligibility boolean Series using & operators
            cfg = {
                'MASK_MAX_ABS_DS': self.MASK_MAX_ABS_DS,
                'MASK_MIN_ABS_GAMMA': self.MASK_MIN_ABS_GAMMA,
                'MASK_MAX_DTE': self.MASK_MAX_DTE
            }
            
            eligible = (
                df['abs_dS'].le(cfg['MASK_MAX_ABS_DS']) &
                df['premium_mid'].gt(0) &
                df['iv'].notna() &
                np.isfinite(df['iv']) &
                df['gamma'].notna() &
                df['gamma'].abs().ge(cfg['MASK_MIN_ABS_GAMMA'])
            )
            
            # Filter by DTE if available
            if 'Expiration Date' in df.columns:
                try:
                    exp_dates = pd.to_datetime(df['Expiration Date'], errors='coerce')
                    dte = (exp_dates - current_date).dt.days
                    df['dte'] = dte
                    eligible &= df['dte'].le(cfg['MASK_MAX_DTE'])
                except:
                    pass  # Skip DTE filter if date parsing fails
            
            # Add eligible_mask to df
            df['eligible_mask'] = eligible
            
            # Build ineligible_reason Series
            df['ineligible_reason'] = ''
            df.loc[df['abs_dS'].gt(cfg['MASK_MAX_ABS_DS']), 'ineligible_reason'] += '|abs_dS'
            df.loc[df['premium_mid'].le(0) | df['premium_mid'].isna(), 'ineligible_reason'] += '|premium'
            df.loc[df['iv'].isna() | ~np.isfinite(df['iv']), 'ineligible_reason'] += '|iv'
            df.loc[df['gamma'].isna() | (df['gamma'].abs() < cfg['MASK_MIN_ABS_GAMMA']), 'ineligible_reason'] += '|gamma'
            if 'dte' in df.columns:
                df.loc[df['dte'].isna() | (df['dte'] > cfg['MASK_MAX_DTE']), 'ineligible_reason'] += '|dte'
            df['ineligible_reason'] = df['ineligible_reason'].str.strip('|')
            
            return df['eligible_mask'], df['ineligible_reason']
        except Exception as e:
            self.logger.warning(f"Error computing eligibility filter: {e}")
            # Default: all eligible
            n = len(df)
            return pd.Series([True] * n, index=df.index), pd.Series([''] * n, index=df.index)
    
    def compute_structural_position(self, df, spot_price, flip_level=None, walls=None):
        """Compute structural position metrics: dS, abs_dS, log_moneyness, dFlip, wall_distance."""
        try:
            strikes = pd.to_numeric(df['Strike'], errors='coerce').values
            
            # Distance to spot
            dS = spot_price - strikes
            abs_dS = np.abs(dS)
            log_moneyness = np.log(strikes / spot_price)
            
            # Distance to gamma flip (if provided)
            if flip_level is not None:
                dFlip = spot_price - flip_level
                abs_dFlip = np.abs(dFlip)
            else:
                dFlip = np.full(len(df), np.nan)
                abs_dFlip = np.full(len(df), np.nan)
            
            # Wall distance (if provided)
            if walls is not None and isinstance(walls, dict):
                # Assume walls is dict with 'call_walls' and 'put_walls' arrays
                call_walls = walls.get('call_walls', [])
                put_walls = walls.get('put_walls', [])
                all_walls = np.concatenate([call_walls, put_walls]) if len(call_walls) > 0 or len(put_walls) > 0 else np.array([spot_price])
                
                # Distance to nearest wall
                wall_distances = []
                for strike in strikes:
                    if len(all_walls) > 0:
                        distances = np.abs(all_walls - strike)
                        wall_distances.append(np.min(distances))
                    else:
                        wall_distances.append(np.nan)
                wall_distance = np.array(wall_distances)
            else:
                wall_distance = np.full(len(df), np.nan)
            
            # Moneyness bucket
            moneyness = strikes / spot_price
            moneyness_bucket = pd.cut(
                moneyness,
                bins=[0, 0.95, 1.05, np.inf],
                labels=['ITM', 'ATM', 'OTM'],
                include_lowest=True
            ).astype(str)
            
            return {
                'dS': dS,
                'abs_dS': abs_dS,
                'log_moneyness': log_moneyness,
                'dFlip': dFlip,
                'abs_dFlip': abs_dFlip,
                'wall_distance': wall_distance,
                'moneyness_bucket': moneyness_bucket
            }
        except Exception as e:
            self.logger.warning(f"Error computing structural position: {e}")
            return {
                'dS': np.full(len(df), np.nan),
                'abs_dS': np.full(len(df), np.nan),
                'log_moneyness': np.full(len(df), np.nan),
                'dFlip': np.full(len(df), np.nan),
                'abs_dFlip': np.full(len(df), np.nan),
                'wall_distance': np.full(len(df), np.nan),
                'moneyness_bucket': np.full(len(df), 'UNKNOWN', dtype=object)
            }
    
    def compute_volatility_geometry(self, df, spot_price, expiry_col='Expiration Date', eligible_mask=None):
        """Compute volatility geometry: IV level, slope, curvature, term structure, skew."""
        try:
            # Coerce numeric columns
            if 'Strike' in df.columns:
                df['Strike'] = pd.to_numeric(df['Strike'], errors='coerce')
            strikes = df['Strike'].values if 'Strike' in df.columns else np.full(len(df), np.nan)
            
            # Get IV column
            iv_col = 'IV' if 'IV' in df.columns else None
            if iv_col is None:
                return self._default_volatility_geometry(len(df))
            
            # Coerce IV
            df[iv_col] = pd.to_numeric(df[iv_col], errors='coerce')
            iv = df[iv_col].values / 100.0  # Convert % to decimal
            
            # Filter to eligible contracts for geometry computation
            # eligible_indices will be used for term structure computation
            if eligible_mask is not None:
                if isinstance(eligible_mask, pd.Series):
                    eligible_indices_list = df.index[eligible_mask].tolist()
                else:
                    eligible_indices_list = df.index[eligible_mask].tolist()
                if len(eligible_indices_list) == 0:
                    # No eligible contracts - return defaults
                    return self._default_volatility_geometry(len(df))
            else:
                eligible_indices_list = df.index.tolist()
            
            # Get expiration dates for term structure
            if expiry_col in df.columns:
                exp_dates = pd.to_datetime(df[expiry_col], errors='coerce')
                unique_expiries = exp_dates.dropna().unique()
            else:
                unique_expiries = []
            
            # Initialize arrays
            iv_slope_local = np.full(len(df), np.nan)
            iv_curvature_local = np.full(len(df), np.nan)
            term_slope = np.full(len(df), np.nan)
            skew_local = np.full(len(df), np.nan)
            
            # Get expiration dates for term structure
            if expiry_col in df.columns:
                exp_dates = pd.to_datetime(df[expiry_col], errors='coerce')
                unique_expiries = exp_dates.dropna().unique()
            else:
                unique_expiries = []
            
            # Compute per-expiry IV slope and curvature (only on eligible contracts)
            if len(unique_expiries) > 0:
                for exp_date in unique_expiries:
                    # Filter by expiry
                    exp_mask = (exp_dates == exp_date)
                    if eligible_mask is not None:
                        # Use .any() check, not scalar if
                        if isinstance(eligible_mask, pd.Series):
                            exp_mask = exp_mask & eligible_mask
                        else:
                            exp_mask = exp_mask & pd.Series(eligible_mask, index=df.index)
                    
                    # Additional validity checks using pandas Series
                    iv_valid = pd.Series(iv, index=df.index).notna() & np.isfinite(pd.Series(iv, index=df.index))
                    strikes_series = pd.Series(strikes, index=df.index)
                    strikes_valid = strikes_series.notna() & np.isfinite(strikes_series)
                    exp_mask = exp_mask & iv_valid & strikes_valid
                    
                    # Check if enough points (use .sum() not len)
                    if exp_mask.sum() < 3:
                        continue
                    
                    # Get data for this expiry
                    exp_indices = df.index[exp_mask]
                    exp_strikes = strikes_series.loc[exp_indices].values
                    exp_iv = pd.Series(iv, index=df.index).loc[exp_indices].values
                    
                    # Drop NaNs explicitly
                    x = exp_strikes
                    y = exp_iv
                    mask = np.isfinite(x) & np.isfinite(y)
                    x, y = x[mask], y[mask]
                    exp_indices_clean = exp_indices[mask]
                    
                    if len(x) < 3:
                        continue
                    
                    # Sort by strike
                    sort_idx = np.argsort(x)
                    x_sorted = x[sort_idx]
                    y_sorted = y[sort_idx]
                    indices_sorted = exp_indices_clean[sort_idx]
                    
                    # Windowed regression for each contract
                    window = self.IV_REGRESSION_WINDOW
                    for pos, orig_idx in enumerate(indices_sorted):
                        # Get window around this strike
                        start = max(0, pos - window // 2)
                        end = min(len(x_sorted), pos + window // 2 + 1)
                        
                        if end - start < 3:
                            continue
                        
                        window_x = x_sorted[start:end]
                        window_y = y_sorted[start:end]
                        
                        # Final NaN check
                        mask_window = np.isfinite(window_x) & np.isfinite(window_y)
                        if mask_window.sum() < 3:
                            continue
                        
                        x_clean = window_x[mask_window]
                        y_clean = window_y[mask_window]
                        
                        try:
                            # Linear fit for slope
                            slope, intercept = np.polyfit(x_clean, y_clean, 1)
                            # Find position in original df
                            orig_pos = df.index.get_loc(orig_idx)
                            iv_slope_local[orig_pos] = slope
                            
                            # Quadratic fit for curvature
                            if len(x_clean) >= 3:
                                poly = np.polyfit(x_clean, y_clean, 2)
                                iv_curvature_local[orig_pos] = poly[0]  # Quadratic coefficient
                        except:
                            pass
            
            # Term structure: compare IV between nearest expiries (only eligible)
            if len(unique_expiries) >= 2:
                sorted_expiries = sorted(unique_expiries)
                
                for orig_idx in eligible_indices_list:
                    i = df.index.get_loc(orig_idx)
                    if pd.isna(exp_dates.iloc[i]):
                        continue
                    
                    # Find nearest expiries
                    exp_date = exp_dates.iloc[i]
                    strike = strikes[i]
                    if pd.isna(strike):
                        continue
                    
                    # Get IVs for same/similar strike across expiries (eligible only)
                    ivs_by_expiry = []
                    for exp in sorted_expiries:
                        exp_mask = (exp_dates == exp) & (np.abs(strikes - strike) < strike * 0.05)  # ±5% strike window
                        if eligible_mask is not None:
                            if isinstance(eligible_mask, pd.Series):
                                exp_mask = exp_mask & eligible_mask
                            else:
                                eligible_series = pd.Series(eligible_mask, index=df.index)
                                exp_mask = exp_mask & eligible_series
                        if exp_mask.any():
                            exp_ivs = iv[exp_mask]
                            exp_ivs_clean = exp_ivs[pd.Series(exp_ivs).notna() & np.isfinite(exp_ivs)]
                            if len(exp_ivs_clean) > 0:
                                ivs_by_expiry.append((exp, float(np.median(exp_ivs_clean))))
                    
                    if len(ivs_by_expiry) >= 2:
                        # Compute term slope (IV vs days to expiry)
                        days_to_exp = [(exp - exp_date).days for exp, _ in ivs_by_expiry]
                        iv_values = [iv_val for _, iv_val in ivs_by_expiry]
                        try:
                            term_slope_val, _ = np.polyfit(days_to_exp, iv_values, 1)
                            term_slope[i] = term_slope_val
                        except:
                            pass
            
            # Local skew: put IV - call IV at same/similar strike
            # This requires both calls and puts - will be computed in main function
            # For now, set to NaN (will be filled if put data available)
            
            return {
                'iv_local': iv,
                'iv_slope_local': iv_slope_local,
                'iv_curvature_local': iv_curvature_local,
                'term_slope': term_slope,
                'skew_local': skew_local
            }
        except Exception as e:
            self.logger.warning(f"Error computing volatility geometry: {e}")
            return self._default_volatility_geometry(len(df))
    
    def _default_volatility_geometry(self, n):
        """Return default volatility geometry when computation fails."""
        return {
            'iv_local': np.full(n, np.nan),
            'iv_slope_local': np.full(n, np.nan),
            'iv_curvature_local': np.full(n, np.nan),
            'term_slope': np.full(n, np.nan),
            'skew_local': np.full(n, np.nan)
        }
    
    def compute_sensitivity_profile(self, df):
        """Extract sensitivity profile: delta, gamma, vega, theta, rho."""
        try:
            # Try to get existing Greeks from dataframe
            delta = pd.to_numeric(df.get('Delta', df.get('delta', None)), errors='coerce').values if 'Delta' in df.columns or 'delta' in df.columns else np.full(len(df), np.nan)
            gamma = pd.to_numeric(df.get('Gamma', df.get('gamma', None)), errors='coerce').values if 'Gamma' in df.columns or 'gamma' in df.columns else np.full(len(df), np.nan)
            vega = pd.to_numeric(df.get('Vega', df.get('vega', None)), errors='coerce').values if 'Vega' in df.columns or 'vega' in df.columns else np.full(len(df), np.nan)
            theta = pd.to_numeric(df.get('Theta', df.get('theta', None)), errors='coerce').values if 'Theta' in df.columns or 'theta' in df.columns else np.full(len(df), np.nan)
            rho = pd.to_numeric(df.get('Rho', df.get('rho', None)), errors='coerce').values if 'Rho' in df.columns or 'rho' in df.columns else np.full(len(df), np.nan)
            
            return {
                'delta': delta,
                'gamma': gamma,
                'vega': vega,
                'theta': theta,
                'rho': rho
            }
        except Exception as e:
            self.logger.warning(f"Error computing sensitivity profile: {e}")
            return {
                'delta': np.full(len(df), np.nan),
                'gamma': np.full(len(df), np.nan),
                'vega': np.full(len(df), np.nan),
                'theta': np.full(len(df), np.nan),
                'rho': np.full(len(df), np.nan)
            }
    
    def compute_local_skew(self, df_calls, df_puts, spot_price):
        """Compute local skew: put IV - call IV at same/similar strikes."""
        try:
            if df_calls is None or df_puts is None or len(df_calls) == 0 or len(df_puts) == 0:
                return np.full(len(df_calls), np.nan)
            
            call_strikes = pd.to_numeric(df_calls['Strike'], errors='coerce').values
            put_strikes = pd.to_numeric(df_puts['Strike'], errors='coerce').values
            
            call_iv_col = 'IV' if 'IV' in df_calls.columns else None
            put_iv_col = 'IV.1' if 'IV.1' in df_puts.columns else ('IV' if 'IV' in df_puts.columns else None)
            
            if call_iv_col is None or put_iv_col is None:
                return np.full(len(df_calls), np.nan)
            
            call_iv = pd.to_numeric(df_calls[call_iv_col], errors='coerce').values / 100.0
            put_iv = pd.to_numeric(df_puts[put_iv_col], errors='coerce').values / 100.0
            
            # Match strikes (within 1% tolerance)
            skew_local = np.full(len(df_calls), np.nan)
            for i, call_strike in enumerate(call_strikes):
                if np.isnan(call_strike):
                    continue
                
                # Find matching put strike
                put_distances = np.abs(put_strikes - call_strike)
                tolerance = call_strike * 0.01  # 1% tolerance
                matches = put_distances < tolerance
                
                if matches.any():
                    match_idx = np.argmin(put_distances[matches])
                    put_iv_val = put_iv[matches][match_idx]
                    call_iv_val = call_iv[i]
                    
                    if not (np.isnan(put_iv_val) or np.isnan(call_iv_val)):
                        skew_local[i] = put_iv_val - call_iv_val
            
            return skew_local
        except Exception as e:
            self.logger.warning(f"Error computing local skew: {e}")
            return np.full(len(df_calls), np.nan)
    
    def compute_expressivity_score(self, structural, volatility, sensitivity, execution_quality, regime, structure_family, gate_state=None, eligible_mask=None):
        """Compute expressivity score [0, 1] using weighted components with robust normalization."""
        try:
            n = len(structural['abs_dS'])
            if n == 0:
                return np.array([])
            scores = np.zeros(n)
            
            # Filter to eligible for normalization
            if eligible_mask is not None:
                eligible_indices = np.where(eligible_mask)[0]
            else:
                eligible_indices = np.arange(n)
            
            if len(eligible_indices) == 0:
                # No eligible contracts - return low scores
                return np.full(n, 0.1)
            
            # Proximity to spot (higher when close) - use percentile scaling
            abs_dS = structural['abs_dS']
            abs_dS_series = pd.Series(abs_dS)
            abs_dS_eligible = abs_dS_series.iloc[eligible_indices]
            if len(abs_dS_eligible) > 0 and abs_dS_eligible.notna().any():
                # Use percentile: lower abs_dS = higher proximity
                proximity = 1.0 - abs_dS_series.rank(pct=True, method='min', na_option='keep')
                proximity = proximity.fillna(0.0).values
            else:
                proximity = np.full(n, 0.5)
            proximity = np.nan_to_num(proximity, nan=0.0)
            
            # Gamma strength (percentile scaling)
            gamma = sensitivity['gamma']
            abs_gamma = np.abs(gamma)
            abs_gamma_series = pd.Series(abs_gamma)
            abs_gamma_eligible = abs_gamma_series.iloc[eligible_indices]
            if len(abs_gamma_eligible) > 0 and abs_gamma_eligible.notna().any():
                gamma_strength = abs_gamma_series.rank(pct=True, method='min', na_option='keep')
                gamma_strength = gamma_strength.fillna(0.0).values
            else:
                gamma_strength = np.full(n, 0.0)
            gamma_strength = np.nan_to_num(gamma_strength, nan=0.0)
            
            # Vega strength (percentile scaling)
            vega = sensitivity['vega']
            vega_series = pd.Series(vega)
            vega_eligible = vega_series.iloc[eligible_indices]
            if len(vega_eligible) > 0 and vega_eligible.notna().any():
                vega_strength = vega_series.rank(pct=True, method='min', na_option='keep')
                vega_strength = vega_strength.fillna(0.0).values
            else:
                vega_strength = np.full(n, 0.0)
            vega_strength = np.nan_to_num(vega_strength, nan=0.0)
            
            # Curvature signal (percentile scaling)
            curvature = volatility['iv_curvature_local']
            abs_curvature = np.abs(curvature)
            abs_curvature_series = pd.Series(abs_curvature)
            abs_curvature_eligible = abs_curvature_series.iloc[eligible_indices]
            if len(abs_curvature_eligible) > 0 and abs_curvature_eligible.notna().any():
                curvature_signal = abs_curvature_series.rank(pct=True, method='min', na_option='keep')
                curvature_signal = curvature_signal.fillna(0.0).values
            else:
                curvature_signal = np.full(n, 0.0)
            curvature_signal = np.nan_to_num(curvature_signal, nan=0.0)
            
            # Instability gate (higher when regime not PIN and/or convexity)
            # Check if EXPRESSIVE test bypasses PIN damping
            bypass_pin = getattr(self, '_test_expressive_bypass_pin', False)
            if bypass_pin:
                # Bypass PIN damping for test
                if regime == "PRE_TRANSFER":
                    instability = 0.6
                elif regime == "TRANSFER":
                    instability = 1.0
                else:
                    instability = 0.8  # Higher baseline for test
            elif regime == "PIN" or (gate_state is not None and gate_state == "BLOCK"):
                instability = 0.25  # Damping for PIN/BLOCK
            elif regime == "PRE_TRANSFER":
                instability = 0.6
            elif regime == "TRANSFER":
                instability = 1.0
            else:
                instability = 0.5
            
            if structure_family == "CONVEXITY":
                instability = min(1.0, instability * 1.2)  # Boost for convexity
            
            instability_gate = np.full(n, instability)
            
            # Execution quality (reuse from execution_quality dict)
            exec_quality = execution_quality.get('quality_score', 0.5)
            spread_pct = execution_quality.get('bid_ask_spread_pct', 10.0)
            
            # Cap score if execution quality is poor
            if exec_quality < 0.4 or spread_pct > 8.0:
                exec_quality_capped = 0.15
            else:
                exec_quality_capped = exec_quality
            
            execution_quality_score = np.full(n, exec_quality_capped)
            
            # Weighted sum
            weights = self.EXPRESSIVITY_WEIGHTS
            scores = (
                weights['proximity_to_spot'] * proximity +
                weights['gamma_strength'] * gamma_strength +
                weights['vega_strength'] * vega_strength +
                weights['curvature_signal'] * curvature_signal +
                weights['instability_gate'] * instability_gate +
                weights['execution_quality'] * execution_quality_score
            )
            
            # Clip to [0, 1]
            scores = np.clip(scores, 0.0, 1.0)
            
            return scores
        except Exception as e:
            self.logger.warning(f"Error computing expressivity score: {e}")
            n = len(structural.get('abs_dS', []))
            return np.full(n, 0.5) if n > 0 else np.array([])
    
    def assign_mask_state(self, structural, volatility, sensitivity, execution_quality, expressivity):
        """Assign mask_state: DORMANT, SENSITIVE, EXPRESSIVE, RUPTURE_CANDIDATE, TOXIC."""
        try:
            n = len(expressivity)
            mask_states = np.full(n, 'DORMANT', dtype=object)
            
            gamma = sensitivity['gamma']
            vega = sensitivity['vega']
            abs_gamma = np.abs(gamma)
            abs_dS = structural['abs_dS']
            curvature = np.abs(volatility['iv_curvature_local'])
            spread_pct = execution_quality.get('bid_ask_spread_pct', 10.0)
            quality_score = execution_quality.get('quality_score', 0.5)
            
            # Handle NaN values
            abs_gamma = np.nan_to_num(abs_gamma, nan=0.0)
            vega = np.nan_to_num(vega, nan=0.0)
            abs_dS = np.nan_to_num(abs_dS, nan=1.0)
            curvature = np.nan_to_num(curvature, nan=0.0)
            
            th = self.THRESHOLDS
            
            for i in range(n):
                # TOXIC: poor execution quality
                if spread_pct > th['toxic_spread_pct'] or quality_score < th['toxic_quality']:
                    mask_states[i] = 'TOXIC'
                    continue
                
                # DORMANT: low gamma/vega and far from spot
                if (abs_gamma[i] < th['dormant_gamma'] and 
                    vega[i] < th['dormant_vega'] and 
                    abs_dS[i] > th['dormant_distance'] * abs_dS.max() if abs_dS.max() > 0 else abs_dS[i] > 0.1):
                    mask_states[i] = 'DORMANT'
                    continue
                
                # SENSITIVE: near spot, moderate gamma/vega, low curvature
                if (abs_dS[i] < th['sensitive_distance'] * abs_dS.max() if abs_dS.max() > 0 else abs_dS[i] < 0.05 and
                    abs_gamma[i] >= th['sensitive_gamma'] and 
                    abs_gamma[i] < th['expressive_gamma'] and
                    curvature[i] < th['expressive_curvature']):
                    mask_states[i] = 'SENSITIVE'
                    continue
                
                # EXPRESSIVE: high gamma/vega and strong curvature
                if (abs_gamma[i] >= th['expressive_gamma'] and 
                    vega[i] >= th['expressive_vega'] and
                    curvature[i] >= th['expressive_curvature']):
                    # Check if RUPTURE_CANDIDATE (EXPRESSIVE + high fragility/amplification)
                    if expressivity[i] > 0.7:  # High expressivity = rupture candidate
                        mask_states[i] = 'RUPTURE_CANDIDATE'
                    else:
                        mask_states[i] = 'EXPRESSIVE'
                    continue
                
                # Default: SENSITIVE
                mask_states[i] = 'SENSITIVE'
            
            return mask_states
        except Exception as e:
            self.logger.warning(f"Error assigning mask state: {e}")
            n = len(expressivity) if hasattr(expressivity, '__len__') else 0
            return np.full(n, 'DORMANT', dtype=object) if n > 0 else np.array([], dtype=object)
    
    def compute_masks(self, df_calls, df_puts, spot_price, current_date, 
                     execution_quality, regime, structure_family, gate_state=None,
                     flip_level=None, walls=None):
        """
        Main entry point: compute all mask features and return DataFrame + summary.
        
        Returns:
            df_masks: DataFrame with one row per contract (calls)
            masks_summary: Dict with top lists and counts
        """
        try:
            # Use calls as primary (can extend to puts later)
            df = df_calls.copy()
            n = len(df)
            
            # Coerce numeric columns at the start
            numeric_cols = ['IV', 'Delta', 'Gamma', 'Vega', 'Theta', 'Bid', 'Ask', 'Strike']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Compute structural position first (needed for eligibility)
            structural = self.compute_structural_position(df, spot_price, flip_level, walls)
            
            # Compute sensitivity profile (needed for eligibility)
            sensitivity = self.compute_sensitivity_profile(df)
            
            # Get IV for eligibility check
            iv_col = 'IV' if 'IV' in df.columns else None
            if iv_col:
                df[iv_col] = pd.to_numeric(df[iv_col], errors='coerce')
                iv_local = df[iv_col].values / 100.0
            else:
                iv_local = np.full(n, np.nan)
            
            # Compute eligibility filter BEFORE geometry
            eligible_mask, ineligible_reasons = self.compute_eligibility_filter(
                df, spot_price, current_date, structural, {'iv_local': iv_local}, sensitivity
            )
            
            # Check if any eligible rows exist (use .any() not scalar if)
            if not eligible_mask.any():
                self.logger.warning("No eligible contracts found for geometry computation")
                # Return defaults for all contracts
                volatility = self.compute_volatility_geometry(df, spot_price, eligible_mask=None)
            else:
                # Compute volatility geometry (only on eligible contracts)
                volatility = self.compute_volatility_geometry(df, spot_price, eligible_mask=eligible_mask)
            volatility['iv_local'] = iv_local  # Use coerced IV
            
            # Compute local skew (requires both calls and puts)
            if df_puts is not None and len(df_puts) > 0:
                volatility['skew_local'] = self.compute_local_skew(df_calls, df_puts, spot_price)
            
            # Compute expressivity score (with gate_state for damping)
            expressivity = self.compute_expressivity_score(
                structural, volatility, sensitivity, execution_quality, regime, structure_family,
                gate_state=gate_state, eligible_mask=eligible_mask
            )
            
            # For ineligible contracts: force low expressivity and DORMANT state
            expressivity[~eligible_mask] = 0.05  # Near zero but not exactly 0
            
            # Assign mask states
            mask_states = self.assign_mask_state(
                structural, volatility, sensitivity, execution_quality, expressivity
            )
            
            # Force ineligible contracts to DORMANT
            mask_states[~eligible_mask] = 'DORMANT'
            
            # Build DataFrame
            premium_mid = (pd.to_numeric(df.get('Bid', 0), errors='coerce') + 
                          pd.to_numeric(df.get('Ask', 0), errors='coerce')) / 2.0
            
            df_masks = pd.DataFrame({
                'contract_type': 'CALL',
                'strike': pd.to_numeric(df['Strike'], errors='coerce') if 'Strike' in df.columns else np.full(n, np.nan),
                'expiration': df.get('Expiration Date', pd.Series([None] * n)),
                'premium_mid': premium_mid,
                'iv': volatility['iv_local'],
                'delta': sensitivity['delta'],
                'gamma': sensitivity['gamma'],
                'vega': sensitivity['vega'],
                'theta': sensitivity['theta'],
                'dS': structural['dS'],
                'abs_dS': structural['abs_dS'],
                'log_moneyness': structural['log_moneyness'],
                'abs_dFlip': structural['abs_dFlip'],
                'wall_distance': structural['wall_distance'],
                'moneyness_bucket': structural['moneyness_bucket'],
                'iv_slope_local': volatility['iv_slope_local'],
                'iv_curvature_local': volatility['iv_curvature_local'],
                'term_slope': volatility['term_slope'],
                'skew_local': volatility['skew_local'],
                'expressivity_score': expressivity,
                'mask_state': mask_states,
                'eligible_mask': eligible_mask,
                'ineligible_reason': ineligible_reasons
            })
            
            # Compute summary statistics
            masks_summary = self._compute_summary(df_masks, execution_quality, eligible_mask)
            
            return df_masks, masks_summary
        except Exception as e:
            self.logger.error(f"Error computing masks: {e}", exc_info=True)
            # Return empty DataFrame and default summary
            return pd.DataFrame(), {
                'top_by_expressivity': [],
                'top_by_gamma': [],
                'top_by_vega': [],
                'top_by_curvature': [],
                'top_by_execution_quality': [],
                'recommended_watchlist': [],
                'mask_state_counts': {}
            }
    
    def _compute_summary(self, df_masks, execution_quality, eligible_mask=None):
        """Compute summary statistics and top lists."""
        try:
            if len(df_masks) == 0:
                return {
                    'top_by_expressivity': [],
                    'top_by_gamma': [],
                    'top_by_vega': [],
                    'top_by_curvature': [],
                    'top_by_execution_quality': [],
                    'recommended_watchlist': [],
                    'mask_state_counts': {},
                    'mask_state_counts_eligible': {},
                    'nearest_expiry_cluster': None,
                    'top_strike_cluster': None
                }
            
            # Filter to eligible for top lists
            if eligible_mask is not None and 'eligible_mask' in df_masks.columns:
                df_eligible = df_masks[df_masks['eligible_mask']].copy()
            else:
                df_eligible = df_masks.copy()
            
            # Top by expressivity (eligible only)
            if len(df_eligible) > 0:
                top_expressivity = df_eligible.nlargest(10, 'expressivity_score')[
                    ['strike', 'expiration', 'contract_type', 'premium_mid', 'iv', 
                     'delta', 'gamma', 'vega', 'theta', 'mask_state', 'expressivity_score',
                     'iv_slope_local', 'iv_curvature_local', 'term_slope', 'skew_local']
                ].to_dict('records')
                
                # Top by |gamma| (eligible only)
                top_gamma = df_eligible.reindex(
                    df_eligible['gamma'].abs().nlargest(10).index
                )[['strike', 'expiration', 'contract_type', 'gamma', 'mask_state', 'expressivity_score']].to_dict('records')
                
                # Top by vega (eligible only)
                top_vega = df_eligible.nlargest(10, 'vega')[
                    ['strike', 'expiration', 'contract_type', 'vega', 'mask_state', 'expressivity_score']
                ].to_dict('records')
                
                # Top by |curvature| (eligible only)
                top_curvature = df_eligible.reindex(
                    df_eligible['iv_curvature_local'].abs().nlargest(10).index
                )[['strike', 'expiration', 'contract_type', 'iv_curvature_local', 'mask_state', 'expressivity_score']].to_dict('records')
                
                # Top by execution quality (eligible only)
                top_execution = df_eligible.nlargest(10, 'expressivity_score')[
                    ['strike', 'expiration', 'contract_type', 'expressivity_score', 'mask_state']
                ].to_dict('records')
                
                # Recommended watchlist: top expressivity AND not TOXIC (eligible only)
                recommended = df_eligible[
                    (df_eligible['mask_state'] != 'TOXIC') & 
                    (df_eligible['expressivity_score'] > 0.5)
                ].nlargest(10, 'expressivity_score')[
                    ['strike', 'expiration', 'contract_type', 'expressivity_score', 'mask_state']
                ].to_dict('records')
            else:
                top_expressivity = []
                top_gamma = []
                top_vega = []
                top_curvature = []
                top_execution = []
                recommended = []
            
            # Mask state counts (all contracts)
            mask_state_counts = df_masks['mask_state'].value_counts().to_dict()
            
            # Mask state counts (eligible only)
            if len(df_eligible) > 0:
                mask_state_counts_eligible = df_eligible['mask_state'].value_counts().to_dict()
            else:
                mask_state_counts_eligible = {}
            
            # Nearest expiry cluster (eligible only)
            nearest_expiry_cluster = None
            if len(df_eligible) > 0 and 'expiration' in df_eligible.columns:
                expiry_counts = df_eligible['expiration'].value_counts()
                if len(expiry_counts) > 0:
                    nearest_expiry_cluster = str(expiry_counts.index[0])
            
            # Top strike cluster (within ±5 points of spot, eligible only)
            top_strike_cluster = None
            if len(df_eligible) > 0 and 'strike' in df_eligible.columns and 'abs_dS' in df_eligible.columns:
                near_spot = df_eligible[df_eligible['abs_dS'] <= 5.0]
                if len(near_spot) > 0:
                    # Group by strike, get avg expressivity
                    strike_groups = near_spot.groupby('strike')['expressivity_score'].mean().sort_values(ascending=False)
                    if len(strike_groups) > 0:
                        top_strike_cluster = float(strike_groups.index[0])
            
            # Convert NaN to None for JSON serialization
            def clean_dict(d):
                if isinstance(d, dict):
                    return {k: clean_dict(v) for k, v in d.items()}
                elif isinstance(d, list):
                    return [clean_dict(item) for item in d]
                elif isinstance(d, (np.integer, np.floating)):
                    return float(d) if not np.isnan(d) else None
                elif pd.isna(d):
                    return None
                return d
            
            return {
                'top_by_expressivity': clean_dict(top_expressivity),
                'top_by_gamma': clean_dict(top_gamma),
                'top_by_vega': clean_dict(top_vega),
                'top_by_curvature': clean_dict(top_curvature),
                'top_by_execution_quality': clean_dict(top_execution),
                'recommended_watchlist': clean_dict(recommended),
                'mask_state_counts': clean_dict(mask_state_counts),
                'mask_state_counts_eligible': clean_dict(mask_state_counts_eligible),
                'nearest_expiry_cluster': nearest_expiry_cluster,
                'top_strike_cluster': top_strike_cluster
            }
        except Exception as e:
            self.logger.warning(f"Error computing summary: {e}")
            return {
                'top_by_expressivity': [],
                'top_by_gamma': [],
                'top_by_vega': [],
                'top_by_curvature': [],
                'top_by_execution_quality': [],
                'recommended_watchlist': [],
                'mask_state_counts': {},
                'mask_state_counts_eligible': {},
                'nearest_expiry_cluster': None,
                'top_strike_cluster': None
            }
