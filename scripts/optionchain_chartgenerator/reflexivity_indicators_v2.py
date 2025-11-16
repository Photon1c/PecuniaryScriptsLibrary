"""
Reflexivity Index Calculator - Modernized Version
Calculates reflexivity metrics using local CSV files for stock and options data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import the data loader
from data_loader import load_stock_data, load_option_chain_data, get_most_recent_option_date

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

# Component weights (tunable)
GPD_WEIGHT = 0.50  # Gamma Pressure Differential weight
TENSION_WEIGHT = 0.50  # Dealer Tension weight
MOMENTUM_WEIGHT = 0.20  # IV Momentum weight

# GPD sensitivity parameters
GPD_EPSILON = 0.002  # Minimum GPD floor when spot is very close to flip
GPD_WALL_THRESHOLD = 1.0  # When abs(spot - flip) < this, use wall distance
GPD_NONLINEAR_SCALE = 3.0  # Scaling factor for sigmoid mapping

# Component non-linear mapping
USE_NONLINEAR_MAPPING = True  # Enable sigmoid/tanh mapping on components
COMPONENT_DEAD_ZONE = 0.01  # Dead zone threshold before non-linear ramp

# Comparison feature
COMPARE_TO_PREV = True  # Compare to previous CSV if available


class ReflexivityCalculator:
    """Professional-grade reflexivity index calculator"""
    
    def __init__(self, ticker="SPY", stock_base_dir="F:/inputs/stocks", 
                 options_base_dir="F:/inputs/options/log"):
        self.ticker = ticker
        self.stock_base_dir = stock_base_dir
        self.options_base_dir = options_base_dir
        self.stock_data = None
        self.options_data = None
        self.results = {}
    
    def _apply_nonlinear_mapping(self, value, dead_zone=COMPONENT_DEAD_ZONE, scale=GPD_NONLINEAR_SCALE):
        """
        Apply sigmoid/tanh non-linear mapping to component values
        to ramp faster once they leave the dead zone
        """
        if not USE_NONLINEAR_MAPPING:
            return value
        
        # Apply dead zone - values below threshold remain unchanged
        if abs(value) < dead_zone:
            return value
        
        # Apply tanh mapping for smooth ramping
        # Scale the input to control the steepness of the ramp
        scaled_value = value * scale
        mapped = np.tanh(scaled_value)
        
        # Scale back to appropriate range
        return mapped / scale
    
    def _calculate_improved_gpd(self, spot, flip_point, call_wall, put_wall):
        """
        Calculate Gamma Pressure Differential with improved sensitivity
        - Adds epsilon floor when spot is very close to flip
        - Uses distance to nearest wall when abs(spot - flip) < threshold
        """
        flip_distance = abs(spot - flip_point)
        
        # When very close to flip, use distance to nearest wall instead
        if flip_distance < GPD_WALL_THRESHOLD:
            dist_to_call_wall = abs(spot - call_wall) / spot
            dist_to_put_wall = abs(spot - put_wall) / spot
            wall_distance = min(dist_to_call_wall, dist_to_put_wall)
            # Use wall distance with minimum floor
            gpd = max(wall_distance, GPD_EPSILON)
        else:
            # Normal calculation with epsilon floor
            gpd = max(flip_distance / spot, GPD_EPSILON)
        
        # Apply non-linear mapping for better sensitivity
        gpd_mapped = self._apply_nonlinear_mapping(gpd, dead_zone=GPD_EPSILON)
        
        return gpd_mapped
        
    def load_data(self, options_date=None):
        """Load stock and options data"""
        print(f"\n{'='*60}")
        print(f"Loading data for {self.ticker.upper()}")
        print(f"{'='*60}")
        
        # Load stock data
        try:
            self.stock_data = load_stock_data(self.ticker, self.stock_base_dir)
            print(f"✓ Loaded {len(self.stock_data)} days of stock data")
        except Exception as e:
            print(f"✗ Error loading stock data: {e}")
            return False
        
        # Load options data
        try:
            if options_date is None:
                options_date = get_most_recent_option_date(
                    self.ticker, self.options_base_dir
                )
            
            self.options_data = load_option_chain_data(
                self.ticker, options_date, self.options_base_dir
            )
            print(f"✓ Loaded options data with {len(self.options_data)} rows")
            
            # Clean column names
            self.options_data.columns = self.options_data.columns.str.strip()
            
            # Verify column structure
            print(f"\n📋 Options data structure check:")
            if len(self.options_data) > 0:
                sample_row = self.options_data.iloc[0]
                print(f"   Total columns: {len(sample_row)}")
                if 'Strike' in sample_row.index:
                    strike_idx = sample_row.index.get_loc('Strike')
                    print(f"   Strike column index: {strike_idx}")
                    print(f"   Columns before Strike: {list(sample_row.index[:min(strike_idx+1, len(sample_row.index))])}")
                    print(f"   Columns after Strike: {list(sample_row.index[strike_idx+1:min(strike_idx+12, len(sample_row.index))])}")
                else:
                    print(f"   ⚠ Warning: 'Strike' column not found!")
                    print(f"   Available columns: {list(sample_row.index[:20])}")
            
            return True
        except Exception as e:
            print(f"✗ Error loading options data: {e}")
            return False
    
    def get_available_expirations(self):
        """Get list of available expiration dates, sorted chronologically"""
        if self.options_data is None:
            return []
        
        expirations = self.options_data['Expiration Date'].unique()
        # Filter out NaN values
        valid_expirations = [exp for exp in expirations if pd.notna(exp)]
        
        # Convert to datetime for proper chronological sorting
        try:
            # Try parsing as pandas datetime
            exp_dates = pd.to_datetime(valid_expirations, errors='coerce')
            # Sort by date
            sorted_indices = exp_dates.argsort()
            sorted_expirations = [valid_expirations[i] for i in sorted_indices if pd.notna(exp_dates[i])]
            return sorted_expirations
        except Exception:
            # Fallback to string sort if parsing fails
            return sorted(valid_expirations)
    
    def select_expiration(self, expiration=None):
        """Filter options data by expiration date"""
        available = self.get_available_expirations()
        
        if not available:
            print("✗ No expiration dates found in data")
            return None
        
        if expiration is None:
            # Use nearest expiration
            expiration = available[0]
        
        filtered = self.options_data[
            self.options_data['Expiration Date'] == expiration
        ].copy()
        
        print(f"   {len(filtered)} options contracts for expiration: {expiration}")
        
        return filtered
    
    def calculate_historical_volatility(self, period=21):
        """Calculate annualized historical volatility"""
        if self.stock_data is None:
            return None
        
        # Get close price - handle both 'Close' and 'Close/Last' columns
        if 'Close/Last' in self.stock_data.columns:
            prices = self.stock_data['Close/Last'].copy()
            # Remove $ and convert to float if needed
            if prices.dtype == 'object':
                prices = prices.str.replace('$', '').str.replace(',', '').astype(float)
        else:
            prices = self.stock_data['Close'].copy()
        
        # Calculate log returns
        log_returns = np.log(prices / prices.shift(1))
        
        # Annualized volatility
        hv = log_returns.rolling(window=period).std() * np.sqrt(252)
        
        return hv.iloc[-1] * 100  # Return as percentage
    
    def calculate_implied_volatility(self, options_df, spot_price):
        """
        Calculate IV using professional dealer methodology:
        - Focus on ATM options (within 3% of spot)
        - Weight by open interest and gamma
        - Average call and put IV
        """
        atm_range = 0.03  # 3% around spot
        lower_bound = spot_price * (1 - atm_range)
        upper_bound = spot_price * (1 + atm_range)
        
        # Filter for ATM options
        atm_options = options_df[
            (options_df['Strike'] >= lower_bound) & 
            (options_df['Strike'] <= upper_bound)
        ].copy()
        
        if len(atm_options) == 0:
            print("⚠ Warning: No ATM options found, using all available strikes")
            atm_options = options_df.copy()
        
        # Separate calls and puts
        iv_values = []
        weights = []
        
        for _, row in atm_options.iterrows():
            # Process calls
            call_iv = row.get('IV')
            call_oi = row.get('Open Interest')
            call_gamma = abs(row.get('Gamma', 0))
            
            if pd.notna(call_iv) and call_iv > 0 and pd.notna(call_oi):
                # Weight by OI * Gamma (dealer's risk metric)
                weight = call_oi * (1 + call_gamma * 100)
                iv_values.append(call_iv)
                weights.append(weight)
        
        if not iv_values:
            print("⚠ Warning: No valid IV values found, returning None")
            return None
        
        # Calculate weighted average IV
        weighted_iv = np.average(iv_values, weights=weights)
        
        # If IV is in decimal form (0.18), convert to percentage
        if weighted_iv < 1:
            weighted_iv *= 100
        
        return weighted_iv
    
    def calculate_gamma_metrics(self, options_df, spot_price):
        """
        Calculate gamma flip, call wall, and put wall
        Returns both file-based and computed values
        """
        results = {
            'file_based': {},
            'computed': {}
        }
        
        # Group by strike
        strikes = sorted(options_df['Strike'].unique())
        
        # Debug: Print column names once to help diagnose issues
        print(f"\n📋 Available columns in options data:")
        print(f"   {list(options_df.columns[:20])}")  # Print first 20 columns
        if len(options_df.columns) > 20:
            print(f"   ... and {len(options_df.columns) - 20} more columns")
        
        # File-based calculation (using provided Delta/Gamma)
        file_gamma_profile = []
        
        # Computed calculation (using OI)
        computed_gamma_profile = []
        
        # Track if we found put columns (for debugging)
        found_put_oi = False
        found_put_gamma = False
        
        for strike in strikes:
            strike_data = options_df[options_df['Strike'] == strike]
            
            if len(strike_data) == 0:
                continue
            
            row = strike_data.iloc[0]
            
            # Get call values
            call_oi = row.get('Open Interest', 0)
            if pd.isna(call_oi) or call_oi == '':
                call_oi = 0
            else:
                try:
                    call_oi = float(call_oi)
                except:
                    call_oi = 0
            
            call_gamma = row.get('Gamma', 0)
            call_gamma_raw = call_gamma  # Store raw value for debugging
            if pd.isna(call_gamma) or call_gamma == '':
                call_gamma = 0
            else:
                try:
                    call_gamma = abs(float(call_gamma)) * 100  # Scale by 100 (per contract standard)
                except Exception as e:
                    call_gamma = 0
                    if strike == strikes[0]:  # First strike for debug visibility
                        print(f"⚠ Failed to convert call gamma to float at strike {strike}: {call_gamma_raw} (error: {e})")
            
            # Get put values using positional indexing
            # CSV structure: ... Strike, Puts, Last Sale, Net, Bid, Ask, Volume, IV, Delta, Gamma, Open Interest
            put_oi = 0
            put_gamma = 0
            
            try:
                # Find Strike column index
                if 'Strike' in row.index:
                    strike_idx = row.index.get_loc('Strike')
                    
                    # Put data starts after Strike: Puts label (+1), then Last Sale, Net, Bid, Ask, Volume, IV, Delta (+8) = Gamma
                    # Put Open Interest is after Gamma (+1 more)
                    put_gamma_idx = strike_idx + 1 + 8  # Strike + Puts label + 8 columns to Gamma
                    put_oi_idx = strike_idx + 1 + 9     # Strike + Puts label + 9 columns to Open Interest
                    
                    # Extract using positional indexing
                    if put_gamma_idx < len(row):
                        put_gamma_val = row.iloc[put_gamma_idx]
                        put_gamma_raw = put_gamma_val  # Store raw value for debugging
                        if pd.notna(put_gamma_val) and put_gamma_val != '':
                            try:
                                put_gamma = abs(float(put_gamma_val)) * 100  # Scale by 100 (per contract standard)
                                if not found_put_gamma:
                                    found_put_gamma = True
                                    col_name = row.index[put_gamma_idx] if put_gamma_idx < len(row.index) else f"index_{put_gamma_idx}"
                                    print(f"   ✓ Found Put Gamma at index {put_gamma_idx} (column: '{col_name}')")
                            except Exception as e:
                                if strike == strikes[0]:  # First strike for debug visibility
                                    print(f"⚠ Failed to convert put gamma to float at strike {strike}: {put_gamma_raw} (error: {e})")
                                pass
                    
                    if put_oi_idx < len(row):
                        put_oi_val = row.iloc[put_oi_idx]
                        if pd.notna(put_oi_val) and put_oi_val != '':
                            try:
                                put_oi = float(put_oi_val)
                                if not found_put_oi:
                                    found_put_oi = True
                                    col_name = row.index[put_oi_idx] if put_oi_idx < len(row.index) else f"index_{put_oi_idx}"
                                    print(f"   ✓ Found Put OI at index {put_oi_idx} (column: '{col_name}')")
                            except:
                                pass
                
                # Fallback: Try named columns if positional indexing didn't work
                if put_oi == 0:
                    put_oi_cols = ['Open Interest.1', 'Open Interest.2', 'Open Interest 1']
                    for col_name in put_oi_cols:
                        if col_name in row.index:
                            put_oi_val = row.get(col_name, 0)
                            if pd.notna(put_oi_val) and put_oi_val != '':
                                try:
                                    put_oi = float(put_oi_val)
                                    if not found_put_oi:
                                        found_put_oi = True
                                        print(f"   ✓ Found Put OI column: '{col_name}'")
                                    break
                                except:
                                    pass
                
                if put_gamma == 0:
                    put_gamma_cols = ['Gamma.1', 'Gamma.2', 'Gamma 1']
                    for col_name in put_gamma_cols:
                        if col_name in row.index:
                            put_gamma_val = row.get(col_name, 0)
                            if pd.notna(put_gamma_val) and put_gamma_val != '':
                                try:
                                    put_gamma = abs(float(put_gamma_val)) * 100  # Scale by 100 (per contract standard)
                                    if not found_put_gamma:
                                        found_put_gamma = True
                                        print(f"   ✓ Found Put Gamma column: '{col_name}'")
                                    break
                                except:
                                    pass
                    
            except Exception as e:
                # If positional indexing fails, try fallback methods
                pass
            
            # Debug: Print gamma values for first strike and ATM strikes
            if strike == strikes[0]:
                print(f"\n   📊 First Strike ({strike}) Gamma Values:")
                print(f"      Call gamma: {call_gamma}")
                print(f"      Put gamma: {put_gamma}")
                print(f"      Call OI: {call_oi}")
                print(f"      Put OI: {put_oi}")
            
            # Debug: Print gamma values for ATM strikes
            if strike in [670.0, 671.0, 672.0]:
                print(f"   DEBUG Strike {strike}: call_gamma={call_gamma}, put_gamma={put_gamma}, call_oi={call_oi}, put_oi={put_oi}")
            
            # Store original values for debug
            call_gamma_orig = call_gamma
            put_gamma_orig = put_gamma
            
            # File-based: Use actual Gamma * OI
            # Dealers are short calls (negative gamma) and long puts (positive gamma)
            # Net gamma exposure = put_oi * put_gamma - call_oi * call_gamma
            calculation_method = ""
            if put_gamma > 0 and call_gamma > 0:
                file_gamma_exposure = (put_oi * put_gamma) - (call_oi * call_gamma)
                calculation_method = "both_gamma"
            elif put_gamma > 0:
                # Only put gamma available, use it for both
                file_gamma_exposure = (put_oi - call_oi) * put_gamma
                calculation_method = "put_gamma_only"
            elif call_gamma > 0:
                # Only call gamma available, use it for both
                file_gamma_exposure = (put_oi - call_oi) * call_gamma
                calculation_method = "call_gamma_only"
            else:
                # No gamma data available - estimate from moneyness
                # Gamma is highest at-the-money and decreases with distance
                moneyness = abs(strike - spot_price) / spot_price
                if moneyness < 0.05:
                    # Very close to ATM, high gamma
                    estimated_gamma = 100
                elif moneyness < 0.10:
                    # Moderately close, medium gamma
                    estimated_gamma = 50
                elif moneyness < 0.20:
                    # Further out, low gamma
                    estimated_gamma = 10
                else:
                    # Deep OTM/ITM, very low gamma
                    estimated_gamma = 1
                file_gamma_exposure = (put_oi - call_oi) * estimated_gamma
                calculation_method = f"estimated_{estimated_gamma}"
                if strike == strikes[0] or strike in [670.0, 671.0, 672.0]:
                    print(f"      ⚠ Strike {strike}: No gamma data - using estimated gamma: {estimated_gamma} (moneyness: {moneyness:.3f})")
            
            # Debug: Show calculation method for ATM strikes
            if strike in [670.0, 671.0, 672.0]:
                print(f"      Strike {strike} file-based calc: method={calculation_method}, result={file_gamma_exposure:,.0f}")
            
            file_gamma_profile.append((strike, file_gamma_exposure, call_oi, put_oi, call_gamma, put_gamma))  # Store OI and gamma for tracking
            
            # Computed: Simple OI-based
            computed_gamma_exposure = put_oi - call_oi
            computed_gamma_profile.append((strike, computed_gamma_exposure, call_oi, put_oi))
            
            # Debug: Show computed calculation for ATM strikes
            if strike in [670.0, 671.0, 672.0]:
                print(f"      Strike {strike} computed calc: result={computed_gamma_exposure:,.0f} (put_oi - call_oi)")
        
        # Diagnostic logging
        print(f"\n🔍 Gamma Calculation Debug:")
        print(f"   Processed {len(file_gamma_profile)} strikes")
        if file_gamma_profile:
            sample = file_gamma_profile[0]
            print(f"   Sample strike: {sample[0]}")
            print(f"   Sample call OI: {sample[2]}")
            print(f"   Sample put OI: {sample[3]}")
            print(f"   Sample call gamma: {sample[4]}")
            print(f"   Sample put gamma: {sample[5]}")
            print(f"   Sample file-based net gamma: {sample[1]:,.0f}")
        if computed_gamma_profile:
            sample_comp = computed_gamma_profile[0]
            print(f"   Sample computed net gamma: {sample_comp[1]:,.0f}")
        
        total_call_oi = sum(x[2] for x in file_gamma_profile)
        total_put_oi = sum(x[3] for x in file_gamma_profile)
        print(f"   Total call OI: {total_call_oi:,.0f}")
        print(f"   Total put OI: {total_put_oi:,.0f}")
        
        # Count non-zero gamma values
        non_zero_call_gamma = sum(1 for x in file_gamma_profile if x[4] != 0)
        non_zero_put_gamma = sum(1 for x in file_gamma_profile if x[5] != 0)
        print(f"   Strikes with non-zero call gamma: {non_zero_call_gamma} / {len(file_gamma_profile)}")
        print(f"   Strikes with non-zero put gamma: {non_zero_put_gamma} / {len(file_gamma_profile)}")
        
        if file_gamma_profile:
            max_net_gamma_file = max(abs(x[1]) for x in file_gamma_profile)
            print(f"   Max net gamma (file-based): {max_net_gamma_file:,.0f}")
        if computed_gamma_profile:
            max_net_gamma_comp = max(abs(x[1]) for x in computed_gamma_profile)
            print(f"   Max net gamma (computed): {max_net_gamma_comp:,.0f}")
        
        # Debug: Find and compare ATM strike
        if file_gamma_profile and computed_gamma_profile:
            try:
                atm_strike_idx = min(range(len(file_gamma_profile)), 
                                     key=lambda i: abs(file_gamma_profile[i][0] - spot_price))
                atm_data = file_gamma_profile[atm_strike_idx]
                comp_data = computed_gamma_profile[atm_strike_idx]
                print(f"\n   🎯 ATM Strike ({atm_data[0]}):")
                print(f"      File net gamma: {atm_data[1]:,.0f}")
                print(f"      Computed net gamma: {comp_data[1]:,.0f}")
                print(f"      Difference: {abs(atm_data[1] - comp_data[1]):,.0f}")
                print(f"      Call gamma: {atm_data[4]}, Put gamma: {atm_data[5]}")
                print(f"      Call OI: {atm_data[2]}, Put OI: {atm_data[3]}")
            except Exception as e:
                print(f"   ⚠ Could not find ATM strike for comparison: {e}")
        
        # Warn if put columns weren't found
        if not found_put_oi:
            print(f"\n⚠ Warning: Could not find Put Open Interest column. Available columns:")
            put_oi_candidates = [col for col in options_df.columns if 'interest' in str(col).lower() or 'oi' in str(col).lower()]
            if put_oi_candidates:
                print(f"   Candidates: {put_oi_candidates}")
            else:
                print(f"   No Open Interest columns found")
        
        if not found_put_gamma:
            print(f"\n⚠ Warning: Could not find Put Gamma column. Available columns:")
            put_gamma_candidates = [col for col in options_df.columns if 'gamma' in str(col).lower()]
            if put_gamma_candidates:
                print(f"   Candidates: {put_gamma_candidates}")
            else:
                print(f"   No Gamma columns found")
            print(f"   File-based gamma will fall back to computed method (OI-based)")
        
        # Convert to DataFrames for analysis
        # Extract only first 4 values for DataFrame (strike, net_gamma, call_oi, put_oi)
        file_gamma_profile_df = [(x[0], x[1], x[2], x[3]) for x in file_gamma_profile]
        file_df = pd.DataFrame(
            file_gamma_profile_df, 
            columns=['Strike', 'NetGamma', 'CallOI', 'PutOI']
        )
        
        computed_df = pd.DataFrame(
            computed_gamma_profile,
            columns=['Strike', 'NetGamma', 'CallOI', 'PutOI']
        )
        
        # Find call walls (max call OI)
        results['file_based']['call_wall'] = file_df.loc[file_df['CallOI'].idxmax(), 'Strike']
        results['computed']['call_wall'] = computed_df.loc[computed_df['CallOI'].idxmax(), 'Strike']
        
        # Find put walls (max put OI)
        results['file_based']['put_wall'] = file_df.loc[file_df['PutOI'].idxmax(), 'Strike']
        results['computed']['put_wall'] = computed_df.loc[computed_df['PutOI'].idxmax(), 'Strike']
        
        # Find gamma flip (zero crossing of net gamma)
        results['file_based']['flip_point'] = self._find_zero_crossing(
            file_df, spot_price
        )
        results['computed']['flip_point'] = self._find_zero_crossing(
            computed_df, spot_price
        )
        
        # Debug: Show NetGamma values around flip points for both methods
        print(f"\n   🔍 Flip Point Analysis:")
        print(f"      File-based flip: ${results['file_based']['flip_point']:.2f}")
        print(f"      Computed flip: ${results['computed']['flip_point']:.2f}")
        
        # Find strikes near flip points
        for method_name, flip_point, df in [('File-based', results['file_based']['flip_point'], file_df),
                                            ('Computed', results['computed']['flip_point'], computed_df)]:
            # Find closest strikes to flip point
            df_sorted = df.sort_values('Strike').reset_index(drop=True)
            closest_pos = (df_sorted['Strike'] - flip_point).abs().idxmin()
            closest_strike = df_sorted.loc[closest_pos, 'Strike']
            closest_gamma = df_sorted.loc[closest_pos, 'NetGamma']
            
            # Find surrounding strikes
            if closest_pos > 0 and closest_pos < len(df_sorted) - 1:
                prev_strike = df_sorted.loc[closest_pos - 1, 'Strike']
                prev_gamma = df_sorted.loc[closest_pos - 1, 'NetGamma']
                next_strike = df_sorted.loc[closest_pos + 1, 'Strike']
                next_gamma = df_sorted.loc[closest_pos + 1, 'NetGamma']
                
                print(f"      {method_name} around flip:")
                print(f"         Strike {prev_strike:.0f}: NetGamma = {prev_gamma:,.0f}")
                print(f"         Strike {closest_strike:.0f}: NetGamma = {closest_gamma:,.0f}")
                print(f"         Strike {next_strike:.0f}: NetGamma = {next_gamma:,.0f}")
                
                # Check if zero crossing exists between these strikes
                if prev_gamma * closest_gamma < 0:
                    print(f"         ✓ Zero crossing between {prev_strike:.0f} and {closest_strike:.0f}")
                elif closest_gamma * next_gamma < 0:
                    print(f"         ✓ Zero crossing between {closest_strike:.0f} and {next_strike:.0f}")
                else:
                    print(f"         ⚠ No zero crossing found in this range")
        
        # Store full profiles for plotting
        results['file_based']['profile'] = file_df
        results['computed']['profile'] = computed_df
        
        return results
    
    def _find_zero_crossing(self, gamma_df, default_price):
        """Find where net gamma crosses zero"""
        # Ensure DataFrame is sorted by strike
        gamma_df_sorted = gamma_df.sort_values('Strike').reset_index(drop=True)
        
        for i in range(len(gamma_df_sorted) - 1):
            current = gamma_df_sorted.iloc[i]
            next_row = gamma_df_sorted.iloc[i + 1]
            
            if current['NetGamma'] * next_row['NetGamma'] < 0:
                # Linear interpolation
                x1, y1 = current['Strike'], current['NetGamma']
                x2, y2 = next_row['Strike'], next_row['NetGamma']
                
                if y2 - y1 != 0:
                    flip = x1 - y1 * (x2 - x1) / (y2 - y1)
                    return flip
        
        return default_price
    
    def calculate_reflexivity_index(self, expiration=None, method='file_based'):
        """
        Calculate the full reflexivity index
        
        Parameters:
        -----------
        expiration : str
            Specific expiration date to use
        method : str
            'file_based' or 'computed' for gamma calculations
        """
        print(f"\n{'='*60}")
        print("CALCULATING REFLEXIVITY INDEX")
        print(f"{'='*60}")
        
        # Select expiration
        options_df = self.select_expiration(expiration)
        if options_df is None:
            return None
        
        # Get current price
        if 'Close/Last' in self.stock_data.columns:
            spot = self.stock_data['Close/Last'].iloc[-1]
            if isinstance(spot, str):
                spot = float(spot.replace('$', '').replace(',', ''))
        else:
            spot = self.stock_data['Close'].iloc[-1]
        
        print(f"\n📊 Current Price: ${spot:.2f}")
        
        # Calculate HV
        hv = self.calculate_historical_volatility()
        print(f"📈 Historical Vol (21d): {hv:.2f}%")
        
        # Calculate IV
        iv = self.calculate_implied_volatility(options_df, spot)
        if iv is None:
            print("✗ Could not calculate IV")
            return None
        print(f"📉 Implied Vol (ATM): {iv:.2f}%")
        
        # Calculate gamma metrics
        gamma_metrics = self.calculate_gamma_metrics(options_df, spot)
        
        selected_metrics = gamma_metrics[method]
        
        print(f"\n🎯 Gamma Analysis ({method}):")
        print(f"   Gamma Flip: ${selected_metrics['flip_point']:.2f}")
        print(f"   Call Wall:  ${selected_metrics['call_wall']:.2f}")
        print(f"   Put Wall:   ${selected_metrics['put_wall']:.2f}")
        
        # Calculate reflexivity components
        base_ratio = iv / hv
        
        # 1. Gamma Pressure Differential (improved with epsilon floor and wall distance)
        gpd = self._calculate_improved_gpd(
            spot, 
            selected_metrics['flip_point'],
            selected_metrics['call_wall'],
            selected_metrics['put_wall']
        )
        # gpd already has non-linear mapping applied in _calculate_improved_gpd
        gpd_mapped = gpd
        gpd_weighted = GPD_WEIGHT * gpd_mapped
        
        # 2. Dealer Tension
        tension = (iv - hv) / (iv + hv + 1e-9)
        tension_mapped = self._apply_nonlinear_mapping(tension)
        tension_weighted = TENSION_WEIGHT * tension_mapped
        
        # 3. IV Momentum (ATR-based)
        # Calculate ATR
        high = self.stock_data['High'].iloc[-14:]
        low = self.stock_data['Low'].iloc[-14:]
        close = self.stock_data['Close/Last'].iloc[-14:] if 'Close/Last' in self.stock_data.columns else self.stock_data['Close'].iloc[-14:]
        
        if isinstance(high.iloc[0], str):
            high = high.str.replace('$', '').str.replace(',', '').astype(float)
            low = low.str.replace('$', '').str.replace(',', '').astype(float)
            close = close.str.replace('$', '').str.replace(',', '').astype(float)
        
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        
        atr = tr.mean()
        atr_pct = atr / spot
        
        # ATR momentum (rate of change)
        if len(close) > 5:
            atr_pct_prev = tr.iloc[:-5].mean() / close.iloc[-6]
            atr_mom = (atr_pct - atr_pct_prev) / (atr_pct_prev + 1e-9)
            atr_mom_clamped = np.clip(atr_mom, -0.5, 0.5)
            
            # Hyperbolic tangent normalization
            ex = np.exp(atr_mom_clamped * 5.0)
            atr_mom_norm = (ex - 1/ex) / (ex + 1/ex + 1e-9)
        else:
            atr_mom_norm = 0
        
        # Apply non-linear mapping to momentum
        atr_mom_mapped = self._apply_nonlinear_mapping(atr_mom_norm)
        mom_weighted = MOMENTUM_WEIGHT * atr_mom_mapped
        
        # Composite Score
        score = base_ratio * (1.0 + gpd_weighted + tension_weighted + mom_weighted)
        
        # Determine regime (expanded range: 0.7-1.4)
        if score >= 1.20:
            regime = "RINSE (High Risk)"
            regime_emoji = "🔴"
        elif score >= 1.00:
            regime = "NEUTRAL"
            regime_emoji = "🟡"
        elif score >= 0.85:
            regime = "CALM (Low Risk)"
            regime_emoji = "🟢"
        else:
            regime = "DRY (Very Low Risk)"
            regime_emoji = "🟢"
        
        # Store results
        self.results = {
            'score': score,
            'regime': regime,
            'regime_emoji': regime_emoji,
            'spot': spot,
            'iv': iv,
            'hv': hv,
            'base_ratio': base_ratio,
            'gpd': gpd,
            'gpd_mapped': gpd_mapped,
            'gpd_weighted': gpd_weighted,
            'tension': tension,
            'tension_mapped': tension_mapped,
            'tension_weighted': tension_weighted,
            'atr_mom_norm': atr_mom_norm,
            'atr_mom_mapped': atr_mom_mapped,
            'mom_weighted': mom_weighted,
            'gamma_metrics': gamma_metrics,
            'selected_method': method,
            'expiration': expiration
        }
        
        # Print results
        print(f"\n{'='*60}")
        print("REFLEXIVITY INDEX RESULTS")
        print(f"{'='*60}")
        print(f"Score: {score:.3f}")
        print(f"Regime: {regime_emoji} {regime}")
        print(f"IV/HV Ratio: {base_ratio:.2f}")
        print(f"\nComponent Breakdown:")
        print(f"  Gamma Pressure: {gpd_weighted:.4f}")
        print(f"  Dealer Tension: {tension_weighted:.4f}")
        print(f"  IV Momentum: {mom_weighted:.4f}")
        print(f"{'='*60}")
        
        return self.results
    
    def _load_previous_results(self, output_dir):
        """Load previous CSV results for comparison"""
        if not COMPARE_TO_PREV:
            return None
        
        output_path = Path(output_dir)
        if not output_path.exists():
            return None
        
        # Find all CSV files matching the pattern
        csv_files = list(output_path.glob(f"{self.ticker}_reflexivity_*.csv"))
        if not csv_files:
            return None
        
        # Get the most recent CSV file
        csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_csv = csv_files[0]
        
        try:
            prev_df = pd.read_csv(latest_csv)
            if len(prev_df) > 0:
                return prev_df.iloc[-1]  # Return last row
        except Exception as e:
            print(f"⚠ Could not load previous results: {e}")
        
        return None
    
    def _calculate_deltas(self, prev_results, has_both_scores):
        """Calculate deltas between current and previous results"""
        try:
            deltas = {}
            
            # Determine which columns to use based on whether both methods were used
            prev_cols = prev_results.index.tolist() if hasattr(prev_results, 'index') else list(prev_results.keys())
            
            if has_both_scores:
                # Try to find file-based score column
                if 'Reflexivity_Score_File_Based' in prev_cols:
                    prev_score_file = prev_results['Reflexivity_Score_File_Based']
                    deltas['delta_score_file'] = self.results['score_file_based'] - prev_score_file
                elif 'Reflexivity_Score' in prev_cols:
                    prev_score_file = prev_results['Reflexivity_Score']
                    deltas['delta_score_file'] = self.results['score_file_based'] - prev_score_file
                
                # Computed score column
                if 'Reflexivity_Score_Computed' in prev_cols:
                    prev_score_comp = prev_results['Reflexivity_Score_Computed']
                    deltas['delta_score_comp'] = self.results['score_computed'] - prev_score_comp
            else:
                # Single method
                if 'Reflexivity_Score' in prev_cols:
                    prev_score = prev_results['Reflexivity_Score']
                    deltas['delta_score'] = self.results['score'] - prev_score
            
            # GPD delta
            if has_both_scores and 'GPD_Weighted_File_Based' in prev_cols:
                prev_gpd = prev_results['GPD_Weighted_File_Based']
                current_gpd = self.results['gpd_weighted']
            elif 'GPD_Weighted' in prev_cols:
                prev_gpd = prev_results['GPD_Weighted']
                current_gpd = self.results['gpd_weighted']
            else:
                prev_gpd = None
            
            if prev_gpd is not None:
                deltas['delta_gpd'] = current_gpd - prev_gpd
            
            # Tension delta
            if 'Tension_Weighted' in prev_cols:
                prev_tension = prev_results['Tension_Weighted']
                deltas['delta_tension'] = self.results['tension_weighted'] - prev_tension
            
            # IV/HV ratio delta
            if 'IV_HV_Ratio' in prev_cols:
                prev_iv_hv = prev_results['IV_HV_Ratio']
                deltas['delta_iv_hv'] = self.results['base_ratio'] - prev_iv_hv
            
            return deltas if deltas else None
            
        except Exception as e:
            print(f"⚠ Error calculating deltas: {e}")
            return None
    
    def save_results(self, output_dir="reflexivity_output"):
        """Save results to CSV and generate charts"""
        if not self.results:
            print("✗ No results to save. Run calculate_reflexivity_index() first.")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load previous results for comparison
        prev_results = self._load_previous_results(output_dir)
        
        # Check if both methods were calculated
        has_both_scores = 'score_file_based' in self.results and 'score_computed' in self.results
        
        # Build summary dict
        summary_dict = {
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Ticker': self.ticker,
            'Spot_Price': self.results['spot'],
            'IV': self.results['iv'],
            'HV': self.results['hv'],
            'IV_HV_Ratio': self.results['base_ratio'],
            'Gamma_Flip_File': self.results['gamma_metrics']['file_based']['flip_point'],
            'Gamma_Flip_Computed': self.results['gamma_metrics']['computed']['flip_point'],
            'Call_Wall_File': self.results['gamma_metrics']['file_based']['call_wall'],
            'Call_Wall_Computed': self.results['gamma_metrics']['computed']['call_wall'],
            'Put_Wall_File': self.results['gamma_metrics']['file_based']['put_wall'],
            'Put_Wall_Computed': self.results['gamma_metrics']['computed']['put_wall'],
            'Tension_Weighted': self.results['tension_weighted'],
            'Momentum_Weighted': self.results['mom_weighted'],
            'Expiration': self.results['expiration']
        }
        
        if has_both_scores:
            # Include both scores
            summary_dict['Reflexivity_Score_File_Based'] = self.results['score_file_based']
            summary_dict['Reflexivity_Score_Computed'] = self.results['score_computed']
            summary_dict['GPD_Weighted_File_Based'] = self.results['gpd_weighted']
            summary_dict['GPD_Weighted_Computed'] = self.results.get('gpd_weighted_computed', 0)
            summary_dict['GPD_File_Based'] = self.results.get('gpd', 0)
            summary_dict['GPD_Computed'] = self.results.get('gpd_computed', 0)
            # Regime for both methods
            summary_dict['Regime_File_Based'] = self.results['regime']
            summary_dict['Regime_Computed'] = self.results.get('regime_computed', '')
        else:
            # Single method
            summary_dict['Reflexivity_Score'] = self.results['score']
            summary_dict['Regime'] = self.results['regime']
            summary_dict['GPD_Weighted'] = self.results['gpd_weighted']
        
        summary_df = pd.DataFrame([summary_dict])
        
        csv_path = output_path / f"{self.ticker}_reflexivity_{timestamp}.csv"
        summary_df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved summary to: {csv_path}")
        
        # Calculate and display comparison deltas if previous results available
        deltas = None
        if prev_results is not None:
            deltas = self._calculate_deltas(prev_results, has_both_scores)
            if deltas:
                print(f"\n{'='*60}")
                print("COMPARISON TO PREVIOUS RUN")
                print(f"{'='*60}")
                if has_both_scores:
                    print(f"ΔScore (File-Based): {deltas.get('delta_score_file', 0):+.4f}")
                    print(f"ΔScore (Computed): {deltas.get('delta_score_comp', 0):+.4f}")
                else:
                    print(f"ΔScore: {deltas.get('delta_score', 0):+.4f}")
                print(f"ΔGPD: {deltas.get('delta_gpd', 0):+.4f}")
                print(f"ΔTension: {deltas.get('delta_tension', 0):+.4f}")
                print(f"ΔIV/HV: {deltas.get('delta_iv_hv', 0):+.4f}")
        
        # Generate charts
        self._generate_charts(output_path, timestamp, prev_results=prev_results, deltas=deltas)
    
    def _generate_charts(self, output_path, timestamp, prev_results=None, deltas=None):
        """Generate visualization charts"""
        
        # Check if both methods were calculated
        has_both_scores = 'score_file_based' in self.results and 'score_computed' in self.results
        method_label = 'Both Methods' if has_both_scores else self.results.get('selected_method', 'Unknown').replace('_', '-').title()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.ticker} Reflexivity Analysis ({method_label}) - {timestamp}', 
                     fontsize=16, fontweight='bold')
        
        # Chart 1: Reflexivity Score
        ax1 = axes[0, 0]
        score = self.results['score']
        colors = ['green', 'yellow', 'red']
        
        # Determine max score for y-axis scaling (expanded range: 0.7-1.4)
        max_score = score
        if has_both_scores:
            max_score = max(score, self.results['score_computed'])
        
        # Expanded zones for better sensitivity
        min_score = min(0.7, min(score, self.results['score_computed']) if has_both_scores else score)
        zones = [0.7, 0.85, 1.0, 1.2, max(1.4, max_score + 0.1)]
        zone_colors = ['darkgreen', 'green', 'yellow', 'red']
        
        for i in range(len(zones)-1):
            ax1.axhspan(zones[i], zones[i+1], alpha=0.3, color=zone_colors[i])
        
        # Reference lines for zones
        ax1.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Neutral')
        ax1.axhline(y=1.2, color='red', linestyle='--', linewidth=1, label='Risk Zone (1.2)')
        ax1.axhline(y=0.85, color='green', linestyle='--', linewidth=1, label='Calm Zone (0.85)')
        ax1.axhline(y=0.7, color='darkgreen', linestyle='--', linewidth=1, label='Dry Zone (0.7)')
        
        ax1.set_ylim(min(0.65, min_score - 0.05), max(1.5, max_score + 0.1))
        
        # Create readable x-axis with method labels or expiration date
        expiration = self.results.get('expiration', 'Unknown')
        
        if has_both_scores:
            # Show method labels on x-axis for both methods
            method_positions = [0.25, 0.75]
            method_labels = ['File-Based', 'Computed']
            method_scores = [self.results['score_file_based'], self.results['score_computed']]
            method_colors = ['blue', 'orange']
            
            ax1.set_xlim(0, 1)
            ax1.set_xticks(method_positions)
            ax1.set_xticklabels(method_labels, fontsize=10, fontweight='bold')
            ax1.set_xlabel(f'Expiration: {expiration}', fontsize=10, style='italic')
            
            # Draw horizontal lines for each method at their x-position
            for i, (pos, score_val, color) in enumerate(zip(method_positions, method_scores, method_colors)):
                ax1.plot([pos - 0.15, pos + 0.15], [score_val, score_val], 
                        linewidth=4, color=color,
                        label=f'{method_labels[i]}: {score_val:.3f}')
        else:
            # Single method - show expiration date as x-axis label
            ax1.set_xlim(0, 1)
            ax1.set_xticks([0.5])
            ax1.set_xticklabels([expiration], fontsize=10)
            ax1.set_xlabel('Expiration Date', fontsize=10)
            
            # Show single score line
            ax1.axhline(y=score, color='blue', linewidth=4, label=f'Score: {score:.3f}')
        
        # Create title
        title_text = 'Reflexivity Score'
        ax1.set_title(title_text, fontweight='bold', fontsize=12)
        
        ax1.set_ylabel('Score', fontsize=10)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Add comparison text box if deltas available
        if deltas:
            delta_text = "Δ vs Previous:\n"
            if has_both_scores:
                if 'delta_score_file' in deltas:
                    delta_text += f"Score (File): {deltas['delta_score_file']:+.3f}\n"
                if 'delta_score_comp' in deltas:
                    delta_text += f"Score (Comp): {deltas['delta_score_comp']:+.3f}\n"
            else:
                if 'delta_score' in deltas:
                    delta_text += f"Score: {deltas['delta_score']:+.3f}\n"
            if 'delta_gpd' in deltas:
                delta_text += f"GPD: {deltas['delta_gpd']:+.4f}\n"
            if 'delta_tension' in deltas:
                delta_text += f"Tension: {deltas['delta_tension']:+.4f}\n"
            if 'delta_iv_hv' in deltas:
                delta_text += f"IV/HV: {deltas['delta_iv_hv']:+.4f}"
            
            ax1.text(0.02, 0.98, delta_text.strip(), transform=ax1.transAxes,
                    fontsize=8, verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Chart 2: Gamma Profile Comparison
        ax2 = axes[0, 1]
        file_profile = self.results['gamma_metrics']['file_based']['profile']
        computed_profile = self.results['gamma_metrics']['computed']['profile']
        spot = self.results['spot']
        
        # Get walls and flip points for shading
        file_put_wall = self.results['gamma_metrics']['file_based']['put_wall']
        file_call_wall = self.results['gamma_metrics']['file_based']['call_wall']
        file_flip = self.results['gamma_metrics']['file_based']['flip_point']
        comp_put_wall = self.results['gamma_metrics']['computed']['put_wall']
        comp_call_wall = self.results['gamma_metrics']['computed']['call_wall']
        comp_flip = self.results['gamma_metrics']['computed']['flip_point']
        
        # Calculate symmetric y-axis range based on max abs(NetGamma) from both profiles
        max_gamma_file = abs(file_profile['NetGamma']).max()
        max_gamma_comp = abs(computed_profile['NetGamma']).max()
        max_gamma = max(max_gamma_file, max_gamma_comp)
        y_margin = max_gamma * 0.1  # 10% margin
        y_range = [-max_gamma - y_margin, max_gamma + y_margin]
        
        # Add vertical shaded bands for garage wash phases
        # P2 expansion: put wall → flip
        ax2.axvspan(min(file_put_wall, comp_put_wall), max(file_flip, comp_flip),
                   alpha=0.15, color='orange', label='P2 Expansion')
        # P3 IV fade/rinse: flip → call wall
        ax2.axvspan(min(file_flip, comp_flip), max(file_call_wall, comp_call_wall),
                   alpha=0.15, color='red', label='P3 IV Fade/Rinse')
        
        # Plot gamma profiles
        ax2.plot(file_profile['Strike'], file_profile['NetGamma'], 
                label='File-Based Gamma', linewidth=2.5, color='blue', marker='o', 
                markersize=2, markevery=10)
        ax2.plot(computed_profile['Strike'], computed_profile['NetGamma'], 
                label='Computed Gamma', linewidth=2, color='orange', linestyle='--',
                marker='s', markersize=2, markevery=10)
        
        # Zero line
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
        
        # Spot price
        ax2.axvline(x=spot, color='green', linestyle='--', linewidth=2.5, 
                   alpha=0.8, label=f'Spot: ${spot:.2f}')
        
        # Mark flip points with clear markers
        ax2.axvline(x=file_flip, color='blue', linestyle=':', linewidth=2.5, 
                   alpha=0.8, label=f"File Flip: ${file_flip:.2f}")
        ax2.plot(file_flip, 0, marker='o', markersize=10, color='blue', 
                markeredgecolor='white', markeredgewidth=1.5, zorder=10)
        
        ax2.axvline(x=comp_flip, color='orange', linestyle=':', linewidth=2.5, 
                   alpha=0.8, label=f"Comp Flip: ${comp_flip:.2f}")
        ax2.plot(comp_flip, 0, marker='s', markersize=10, color='orange', 
                markeredgecolor='white', markeredgewidth=1.5, zorder=10)
        
        # Mark walls with vertical lines
        ax2.axvline(x=file_put_wall, color='green', linestyle='-.', linewidth=1.5, 
                   alpha=0.6, label=f"File Put Wall: ${file_put_wall:.2f}")
        ax2.axvline(x=file_call_wall, color='red', linestyle='-.', linewidth=1.5, 
                   alpha=0.6, label=f"File Call Wall: ${file_call_wall:.2f}")
        
        ax2.set_title('Gamma Profile Comparison', fontweight='bold')
        ax2.set_xlabel('Strike Price', fontsize=10)
        ax2.set_ylabel('Net Gamma Exposure', fontsize=10)
        ax2.set_ylim(y_range)  # Symmetric y-axis
        ax2.legend(loc='best', fontsize=8, ncol=2)
        ax2.grid(True, alpha=0.3)
        
        # Chart 3: Open Interest Distribution
        ax3 = axes[1, 0]
        width = (file_profile['Strike'].iloc[1] - file_profile['Strike'].iloc[0]) * 0.4
        ax3.bar(file_profile['Strike'] - width/2, file_profile['CallOI'], 
               width=width, label='Call OI', color='green', alpha=0.6)
        ax3.bar(file_profile['Strike'] + width/2, file_profile['PutOI'], 
               width=width, label='Put OI', color='red', alpha=0.6)
        ax3.axvline(x=spot, color='blue', linestyle='--', linewidth=2, label=f'Spot: ${spot:.2f}')
        
        ax3.set_title('Open Interest Distribution', fontweight='bold')
        ax3.set_xlabel('Strike Price')
        ax3.set_ylabel('Open Interest')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Chart 4: Component Breakdown
        ax4 = axes[1, 1]
        
        if has_both_scores:
            # Grouped bars for both methods
            component_names = ['IV/HV\nBase', 'Gamma\nPressure', 'Dealer\nTension', 'IV\nMomentum']
            x_pos = np.arange(len(component_names))
            width = 0.35
            
            # File-based values (base_ratio is same for both)
            file_values = [
                self.results['base_ratio'],
                self.results['gpd_weighted'],
                self.results['tension_weighted'],
                self.results['mom_weighted']
            ]
            
            # Computed values
            comp_values = [
                self.results['base_ratio'],  # Same for both
                self.results.get('gpd_weighted_computed', 0),
                self.results['tension_weighted'],  # Same for both
                self.results['mom_weighted']  # Same for both
            ]
            
            bars1 = ax4.bar(x_pos - width/2, file_values, width, 
                          label='File-Based', color='blue', alpha=0.7)
            bars2 = ax4.bar(x_pos + width/2, comp_values, width,
                          label='Computed', color='orange', alpha=0.7)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if abs(height) > 0.001:  # Only show if significant
                        ax4.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.3f}', ha='center', 
                                va='bottom' if height >= 0 else 'top', 
                                fontsize=8, fontweight='bold')
            
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(component_names, fontsize=9)
            ax4.legend(fontsize=9)
        else:
            # Single method - simple bars
            components = {
                'IV/HV Base': self.results['base_ratio'],
                'Gamma\nPressure': self.results['gpd_weighted'],
                'Dealer\nTension': self.results['tension_weighted'],
                'IV\nMomentum': self.results['mom_weighted']
            }
            
            colors_comp = ['skyblue', 'purple', 'orange', 'pink']
            bars = ax4.bar(components.keys(), components.values(), color=colors_comp, alpha=0.7)
            
            for bar, val in zip(bars, components.values()):
                height = bar.get_height()
                if abs(height) > 0.001:  # Only show if significant
                    ax4.text(bar.get_x() + bar.get_width()/2., height,
                            f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax4.set_title('Component Breakdown', fontweight='bold')
        ax4.set_ylabel('Value')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        chart_path = output_path / f"{self.ticker}_reflexivity_charts_{timestamp}.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved charts to: {chart_path}")
        
        plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Configuration
    TICKER = "SPY"
    STOCK_DIR = "F:/inputs/stocks"
    OPTIONS_DIR = "F:/inputs/options/log"
    OUTPUT_DIR = "reflexivity_output"
    # Calculation method: 'file_based', 'computed', or 'both'
    CALCULATION_METHOD = 'both'  # Change to 'computed' or 'both' as needed
    
    print(f"\n{'='*60}")
    print("REFLEXIVITY INDEX CALCULATOR - MODERNIZED")
    print(f"{'='*60}")
    
    # Initialize calculator
    calc = ReflexivityCalculator(
        ticker=TICKER,
        stock_base_dir=STOCK_DIR,
        options_base_dir=OPTIONS_DIR
    )
    
    # Load data
    if not calc.load_data():
        print("\n✗ Failed to load data. Exiting.")
        exit(1)
    
    # Get available expirations and let user choose
    expirations = calc.get_available_expirations()
    
    if not expirations:
        print("\n✗ No expirations found. Exiting.")
        exit(1)
    
    # Interactive expiration selection
    print(f"\n{'='*60}")
    print("SELECT EXPIRATION DATE")
    print(f"{'='*60}")
    for i, exp in enumerate(expirations, 1):
        print(f"{i}. {exp}")
    
    while True:
        try:
            choice = input(f"\nSelect expiration (1-{len(expirations)}) or press Enter for nearest: ").strip()
            if choice == "":
                selected_exp = expirations[0]
                print(f"✓ Using nearest: {selected_exp}")
                break
            choice_num = int(choice)
            if 1 <= choice_num <= len(expirations):
                selected_exp = expirations[choice_num - 1]
                print(f"✓ Selected: {selected_exp}")
                break
            else:
                print(f"⚠ Please enter a number between 1 and {len(expirations)}")
        except ValueError:
            print("⚠ Invalid input. Please enter a number or press Enter.")
    
    # Handle calculation method
    if CALCULATION_METHOD == 'both':
        # When running both, calculate once and compute scores for both methods
        print("\n" + "="*60)
        print("ANALYSIS USING BOTH METHODS (FILE-BASED & COMPUTED)")
        print("="*60)
        
        # Calculate using file_based (gamma_metrics will contain both methods)
        results = calc.calculate_reflexivity_index(
            expiration=selected_exp,
            method='file_based'
        )
        
        if results:
            # Calculate computed score using the same gamma_metrics
            gamma_metrics = results['gamma_metrics']
            spot = results['spot']
            iv = results['iv']
            hv = results['hv']
            base_ratio = results['base_ratio']
            
            # Calculate computed method components
            computed_metrics = gamma_metrics['computed']
            file_based_metrics = gamma_metrics['file_based']
            
            # Get flip points for comparison
            flip_file = file_based_metrics['flip_point']
            flip_computed = computed_metrics['flip_point']
            
            # Calculate GPD for computed method (using improved calculation)
            gpd_computed = calc._calculate_improved_gpd(
                spot,
                flip_computed,
                computed_metrics['call_wall'],
                computed_metrics['put_wall']
            )
            # gpd_computed already has non-linear mapping applied
            gpd_mapped_computed = gpd_computed
            gpd_weighted_computed = GPD_WEIGHT * gpd_mapped_computed
            
            # File-based GPD (already calculated)
            gpd_file = results['gpd']
            gpd_weighted_file = results['gpd_weighted']
            
            # Tension and momentum are the same for both methods
            tension = results['tension']
            tension_weighted = results['tension_weighted']
            atr_mom_norm = results['atr_mom_norm']
            mom_weighted = results['mom_weighted']
            
            # Compute score for computed method
            score_computed = base_ratio * (1.0 + gpd_weighted_computed + tension_weighted + mom_weighted)
            
            # Determine regime for computed method (expanded range: 0.7-1.4)
            if score_computed >= 1.20:
                regime_computed = "RINSE (High Risk)"
                regime_emoji_computed = "🔴"
            elif score_computed >= 1.00:
                regime_computed = "NEUTRAL"
                regime_emoji_computed = "🟡"
            elif score_computed >= 0.85:
                regime_computed = "CALM (Low Risk)"
                regime_emoji_computed = "🟢"
            else:
                regime_computed = "DRY (Very Low Risk)"
                regime_emoji_computed = "🟢"
            
            # Store both scores in results
            results['score_file_based'] = results['score']
            results['score_computed'] = score_computed
            results['gpd_weighted_computed'] = gpd_weighted_computed
            results['gpd_computed'] = gpd_computed
            results['regime_computed'] = regime_computed
            results['regime_emoji_computed'] = regime_emoji_computed
            
            # Save combined results
            calc.save_results(OUTPUT_DIR)
            
            # Print comparison of both methods
            print(f"\n{'='*60}")
            print("COMPARISON: FILE-BASED vs COMPUTED")
            print(f"{'='*60}")
            print(f"\n📊 File-Based Method:")
            print(f"   Score: {results['score']:.3f}")
            print(f"   Regime: {results['regime_emoji']} {results['regime']}")
            print(f"   Gamma Flip: ${flip_file:.2f}")
            print(f"   GPD: {gpd_file:.4f} ({gpd_weighted_file:.4f} weighted)")
            print(f"\n📊 Computed Method:")
            print(f"   Score: {score_computed:.3f}")
            print(f"   Regime: {regime_emoji_computed} {regime_computed}")
            print(f"   Gamma Flip: ${flip_computed:.2f}")
            print(f"   GPD: {gpd_computed:.4f} ({gpd_weighted_computed:.4f} weighted)")
            print(f"\n🔍 Difference:")
            print(f"   Score diff: {abs(results['score'] - score_computed):.3f}")
            print(f"   Flip point diff: ${abs(flip_file - flip_computed):.2f}")
            print(f"   GPD diff: {abs(gpd_file - gpd_computed):.4f}")
            
            print(f"\n{'='*60}")
            print("✓ Analysis complete!")
            print(f"{'='*60}")
            print(f"\nResults saved to: ./{OUTPUT_DIR}/")
            print(f"  - Summary CSV")
            print(f"  - Visualization charts (PNG)")
        else:
            print("\n✗ Analysis failed.")
    
    elif CALCULATION_METHOD in ['file_based', 'computed']:
        # Single method calculation
        print("\n" + "="*60)
        print(f"ANALYSIS USING {CALCULATION_METHOD.upper().replace('_', '-')} GAMMA")
        print("="*60)
        
        results = calc.calculate_reflexivity_index(
            expiration=selected_exp,
            method=CALCULATION_METHOD
        )
        
        if results:
            calc.save_results(OUTPUT_DIR)
            
            print(f"\n{'='*60}")
            print("✓ Analysis complete!")
            print(f"{'='*60}")
            print(f"\nResults saved to: ./{OUTPUT_DIR}/")
            print(f"  - Summary CSV")
            print(f"  - Visualization charts (PNG)")
        else:
            print("\n✗ Analysis failed.")
    else:
        print("\n✗ Invalid CALCULATION_METHOD. Must be 'file_based', 'computed', or 'both'.")
        exit(1)