"""
Reflexivity Index 1.0 - Automated Python Implementation
Automatically fetches stock data and calculates options metrics

!!! yfinance module is now deprecated, this script is only here for reference.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ReflexivityIndex:
    def __init__(self, ticker="SPY", lookback_days=90):
        """
        Initialize Reflexivity Index calculator
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol (default: SPY)
        lookback_days : int
            Days of historical data to fetch
        """
        self.ticker = ticker
        self.lookback_days = lookback_days
        self.stock = yf.Ticker(ticker)
        self.data = None
        self.options_data = None
        
    def fetch_stock_data(self):
        """Fetch historical stock price data"""
        print(f"Fetching {self.lookback_days} days of data for {self.ticker}...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        
        self.data = self.stock.history(start=start_date, end=end_date)
        
        if self.data.empty:
            raise ValueError(f"No data retrieved for {self.ticker}")
        
        print(f"✓ Retrieved {len(self.data)} trading days")
        return self.data
    
    def calculate_historical_volatility(self, window=21):
        """
        Calculate Historical Volatility (HV)
        
        Parameters:
        -----------
        window : int
            Rolling window for volatility calculation (default: 21 days)
        """
        if self.data is None:
            self.fetch_stock_data()
        
        # Calculate log returns
        log_returns = np.log(self.data['Close'] / self.data['Close'].shift(1))
        
        # Annualized volatility
        hv = log_returns.rolling(window=window).std() * np.sqrt(252)
        
        self.data['HV'] = hv * 100  # Convert to percentage
        
        return self.data['HV'].iloc[-1]
    
    def calculate_implied_volatility(self):
        """
        Calculate Implied Volatility from options chain
        Uses VIX as proxy for SPY, or calculates from ATM options
        """
        try:
            # For SPY, we can use VIX as IV proxy
            if self.ticker.upper() in ['SPY', 'SPX']:
                vix = yf.Ticker("^VIX")
                vix_data = vix.history(period="5d")
                iv = vix_data['Close'].iloc[-1]
                print(f"✓ Using VIX as IV proxy: {iv:.2f}%")
                return iv
            
            # For other tickers, calculate from options chain
            expirations = self.stock.options
            if not expirations:
                print("⚠ No options data available, estimating IV from HV")
                return self.calculate_historical_volatility() * 1.2
            
            # Get nearest expiration
            nearest_exp = expirations[0]
            options = self.stock.option_chain(nearest_exp)
            
            # Get current stock price
            current_price = self.data['Close'].iloc[-1]
            
            # Find ATM options
            calls = options.calls
            puts = options.puts
            
            # Get ATM implied volatility (average of call and put)
            atm_calls = calls[abs(calls['strike'] - current_price) < current_price * 0.02]
            atm_puts = puts[abs(puts['strike'] - current_price) < current_price * 0.02]
            
            iv_values = []
            if not atm_calls.empty and 'impliedVolatility' in atm_calls.columns:
                iv_values.extend(atm_calls['impliedVolatility'].dropna().tolist())
            if not atm_puts.empty and 'impliedVolatility' in atm_puts.columns:
                iv_values.extend(atm_puts['impliedVolatility'].dropna().tolist())
            
            if iv_values:
                iv = np.mean(iv_values) * 100  # Convert to percentage
                print(f"✓ Calculated IV from options: {iv:.2f}%")
                return iv
            else:
                print("⚠ Could not extract IV, estimating from HV")
                return self.calculate_historical_volatility() * 1.2
                
        except Exception as e:
            print(f"⚠ Error fetching IV: {e}")
            print("  Using HV * 1.2 as IV estimate")
            return self.calculate_historical_volatility() * 1.2
    
    def calculate_gamma_levels(self):
        """
        Calculate put/call walls and gamma flip point
        Uses open interest and delta to estimate dealer positioning
        """
        try:
            expirations = self.stock.options
            if not expirations:
                print("⚠ No options data available for gamma calculation")
                return None, None, None
            
            # Get options for nearest 2-3 expirations
            all_calls = []
            all_puts = []
            
            for exp in expirations[:min(3, len(expirations))]:
                chain = self.stock.option_chain(exp)
                all_calls.append(chain.calls)
                all_puts.append(chain.puts)
            
            calls_df = pd.concat(all_calls, ignore_index=True)
            puts_df = pd.concat(all_puts, ignore_index=True)
            
            # Calculate gamma exposure by strike
            current_price = self.data['Close'].iloc[-1]
            
            # Group by strike and sum open interest
            call_oi = calls_df.groupby('strike')['openInterest'].sum()
            put_oi = puts_df.groupby('strike')['openInterest'].sum()
            
            # Find maximum OI levels (walls)
            call_wall = call_oi.idxmax() if not call_oi.empty else current_price * 1.05
            put_wall = put_oi.idxmax() if not put_oi.empty else current_price * 0.95
            
            # Estimate gamma flip (simplified: weighted average of put/call OI)
            strikes = sorted(set(call_oi.index) | set(put_oi.index))
            
            net_gamma = []
            for strike in strikes:
                call_gamma = call_oi.get(strike, 0)
                put_gamma = put_oi.get(strike, 0)
                # Calls are negative gamma for dealers, puts are positive
                net = put_gamma - call_gamma
                net_gamma.append((strike, net))
            
            # Flip point is where net gamma crosses zero
            net_gamma_df = pd.DataFrame(net_gamma, columns=['strike', 'net_gamma'])
            
            # Find zero crossing
            flip_point = current_price  # Default to current price
            
            for i in range(len(net_gamma_df) - 1):
                if (net_gamma_df.iloc[i]['net_gamma'] * net_gamma_df.iloc[i+1]['net_gamma']) < 0:
                    # Linear interpolation for zero crossing
                    x1, y1 = net_gamma_df.iloc[i]['strike'], net_gamma_df.iloc[i]['net_gamma']
                    x2, y2 = net_gamma_df.iloc[i+1]['strike'], net_gamma_df.iloc[i+1]['net_gamma']
                    flip_point = x1 - y1 * (x2 - x1) / (y2 - y1)
                    break
            
            print(f"✓ Call Wall: ${call_wall:.2f}")
            print(f"✓ Put Wall: ${put_wall:.2f}")
            print(f"✓ Gamma Flip: ${flip_point:.2f}")
            
            return call_wall, put_wall, flip_point
            
        except Exception as e:
            print(f"⚠ Error calculating gamma levels: {e}")
            current_price = self.data['Close'].iloc[-1]
            return current_price * 1.05, current_price * 0.95, current_price
    
    def tanh(self, x):
        """Hyperbolic tangent function"""
        ex = np.exp(np.clip(x, -10, 10))  # Clip to prevent overflow
        exn = 1.0 / ex
        return (ex - exn) / (ex + exn + 1e-9)
    
    def calculate_atr(self, period=14):
        """Calculate Average True Range"""
        if self.data is None:
            self.fetch_stock_data()
        
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        self.data['ATR'] = atr
        return atr
    
    def calculate_reflexivity_index(self, w_gpd=0.50, w_tension=0.50, w_mom=0.20, smooth_period=3):
        """
        Calculate the Reflexivity Index
        
        Parameters:
        -----------
        w_gpd : float
            Weight for Gamma Pressure Differential
        w_tension : float
            Weight for Dealer Tension
        w_mom : float
            Weight for IV Momentum
        smooth_period : int
            EMA smoothing period
        """
        # Ensure we have data
        if self.data is None:
            self.fetch_stock_data()
        
        # Calculate all components
        print("\n" + "="*60)
        print("CALCULATING REFLEXIVITY INDEX COMPONENTS")
        print("="*60)
        
        hv = self.calculate_historical_volatility()
        iv = self.calculate_implied_volatility()
        call_wall, put_wall, flip_point = self.calculate_gamma_levels()
        
        current_price = self.data['Close'].iloc[-1]
        
        print(f"\n📊 Current Price: ${current_price:.2f}")
        print(f"📈 Historical Vol: {hv:.2f}%")
        print(f"📉 Implied Vol: {iv:.2f}%")
        
        # Calculate components for entire history
        scores = []
        
        for i in range(len(self.data)):
            if i < 20:  # Skip first 20 bars for warm-up
                scores.append(np.nan)
                continue
            
            spot = self.data['Close'].iloc[i]
            iv_val = iv / 100.0
            hv_val = self.data['HV'].iloc[i] / 100.0 if not np.isnan(self.data['HV'].iloc[i]) else hv / 100.0
            
            # Base ratio
            base = iv_val / (hv_val + 1e-9)
            
            # 1) Gamma Pressure Differential
            gpd_raw = (spot - flip_point) / spot
            gpd = abs(gpd_raw)
            gpd_weighted = w_gpd * gpd
            
            # 2) Dealer Tension
            tension = (iv_val - hv_val) / (iv_val + hv_val + 1e-9)
            tension_weighted = w_tension * tension
            
            # 3) IV Momentum (ATR acceleration)
            if i >= 25:
                atr_pct = self.data['ATR'].iloc[i] / spot
                atr_pct_prev = self.data['ATR'].iloc[i-5] / self.data['Close'].iloc[i-5]
                atr_mom = (atr_pct - atr_pct_prev) / (atr_pct_prev + 1e-9)
                atr_mom_clamped = np.clip(atr_mom, -0.5, 0.5)
                atr_mom_norm = self.tanh(atr_mom_clamped * 5.0)
            else:
                atr_mom_norm = 0
            
            mom_weighted = w_mom * atr_mom_norm
            
            # Composite Score
            score_raw = base * (1.0 + gpd_weighted + tension_weighted + mom_weighted)
            scores.append(score_raw)
        
        # Apply EMA smoothing
        self.data['ReflexivityScore'] = pd.Series(scores).ewm(span=smooth_period, adjust=False).mean()
        
        # Get latest values
        latest_score = self.data['ReflexivityScore'].iloc[-1]
        
        # Determine regime
        if latest_score >= 1.20:
            regime = "RINSE (High Risk)"
            regime_color = "🔴"
        elif latest_score >= 1.00:
            regime = "NEUTRAL"
            regime_color = "🟡"
        else:
            regime = "DRY (Low Risk)"
            regime_color = "🟢"
        
        print("\n" + "="*60)
        print("REFLEXIVITY INDEX RESULTS")
        print("="*60)
        print(f"Score: {latest_score:.3f}")
        print(f"Regime: {regime_color} {regime}")
        print(f"IV/HV Ratio: {(iv/hv):.2f}")
        print("="*60)
        
        return {
            'score': latest_score,
            'regime': regime,
            'iv': iv,
            'hv': hv,
            'flip_point': flip_point,
            'call_wall': call_wall,
            'put_wall': put_wall,
            'current_price': current_price,
            'data': self.data
        }


# ═════════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLE
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Initialize with ticker (default: SPY)
    ticker = "SPY"  # Change to any ticker with options
    
    print(f"\n{'='*60}")
    print(f"REFLEXIVITY INDEX 1.0 - AUTOMATED")
    print(f"Ticker: {ticker}")
    print(f"{'='*60}\n")
    
    # Create calculator
    calc = ReflexivityIndex(ticker=ticker, lookback_days=90)
    
    # Calculate ATR first
    calc.fetch_stock_data()
    calc.calculate_atr()
    
    # Calculate reflexivity index
    results = calc.calculate_reflexivity_index(
        w_gpd=0.50,
        w_tension=0.50,
        w_mom=0.20,
        smooth_period=3
    )
    
    # Display recent history
    print("\n📊 Recent Reflexivity Scores:")
    print(results['data'][['Close', 'ReflexivityScore']].tail(10).to_string())
    
    # Alert conditions
    score = results['score']
    if score >= 1.20:
        print("\n⚠️  ALERT: Entering RINSE zone - High reflexivity risk!")
    elif score < 0.90:
        print("\n✅ ALERT: Entering DRY zone - Low reflexivity environment")