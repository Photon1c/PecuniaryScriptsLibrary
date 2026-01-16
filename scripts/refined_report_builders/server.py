from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import numpy as np
import math
import traceback
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
import os

app = FastAPI()

# Enable CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@dataclass
class HarvestPolicy:
    max_hold_bars: int = 24
    min_gamma: float = 0.05
    min_delta: float = 0.20
    max_delta: float = 0.55
    min_delta_slope_to_hold: float = 0.00
    take_profit_r: float = 2.0
    stop_loss_r: float = 1.0
    stall_bars: int = 3
    entry_spread_bps: float = 8.0
    exit_spread_bps: float = 8.0
    fee_per_contract: float = 0.65
    risk_per_trade_dollars: float = 100.0
    contracts: int = 1

def _sigmoid(x: float) -> float:
    x = max(min(x, 60.0), -60.0)
    return 1.0 / (1.0 + math.exp(-x))

def estimate_greeks_proxy(close: float, close_prev: float, vol_proxy: float) -> Dict[str, float]:
    dS = close - close_prev
    vol_floor = 1e-4
    x = dS / max(vol_proxy, vol_floor)
    delta = _sigmoid(x)
    gamma = abs(delta * (1 - delta)) * 4.0
    return {"delta": delta, "gamma": gamma, "iv": vol_proxy}

def run_simulation(df: pd.DataFrame, policy: HarvestPolicy):
    dfx = df.copy()
    
    if "_delta" not in dfx.columns:
        rets = dfx["close"].pct_change().fillna(0.0)
        dfx["_vol"] = rets.rolling(20, min_periods=2).std().bfill().fillna(0.01)
        deltas, gammas, ivs = [0.5], [0.0], [dfx["_vol"].iloc[0]]
        for i in range(1, len(dfx)):
            g = estimate_greeks_proxy(dfx["close"].iloc[i], dfx["close"].iloc[i - 1], dfx["_vol"].iloc[i])
            deltas.append(g["delta"])
            gammas.append(g["gamma"])
            ivs.append(g["iv"])
        dfx["_delta"] = deltas
        dfx["_gamma"] = gammas
        dfx["_iv"] = ivs
    
    dfx["_delta_slope"] = dfx["_delta"].diff().fillna(0.0)
    rets = dfx["close"].pct_change().fillna(0.0)

    trades, equity = [], 0.0
    equity_curve = [0.0] * len(dfx)
    in_trade, entry_idx, entry_price, stall_count = False, -1, 0.0, 0

    for i in range(1, len(dfx)):
        base = 1.0 + 50.0 * abs(rets.iloc[i]) * dfx["_delta"].iloc[i] + 10.0 * dfx["_iv"].iloc[i]
        
        if not in_trade:
            if i >= 10 and dfx["_gamma"].iloc[i] >= policy.min_gamma and policy.min_delta <= dfx["_delta"].iloc[i] <= policy.max_delta:
                entry_price = base + (base * (policy.entry_spread_bps / 10000.0))
                entry_idx, in_trade, stall_count = i, True, 0
        else:
            exit_mark = max(0.01, base - (base * (policy.exit_spread_bps / 10000.0)))
            pnl = (exit_mark - entry_price) * 100.0 * policy.contracts - (policy.fee_per_contract * policy.contracts * 2)
            r = pnl / max(policy.risk_per_trade_dollars, 1e-9)
            
            held = i - entry_idx
            reason = ""
            if r >= policy.take_profit_r: reason = "take_profit"
            elif r <= -policy.stop_loss_r: reason = "stop_loss"
            elif held >= policy.max_hold_bars: reason = "timebox_expired"
            elif dfx["_delta_slope"].iloc[i] <= policy.min_delta_slope_to_hold:
                stall_count += 1
                if stall_count >= policy.stall_bars: reason = "delta_stall"
            else: stall_count = 0

            if reason:
                equity += pnl
                trades.append({"entry_idx": int(entry_idx), "exit_idx": int(i), "pnl": float(pnl), "r": float(r), "reason": reason, "equity": float(equity)})
                in_trade = False
        
        cur_pnl = 0
        if in_trade:
            cur_pnl = (max(0.01, base - (base * (policy.exit_spread_bps / 10000.0))) - entry_price) * 100.0 * policy.contracts - (policy.fee_per_contract * policy.contracts * 2)
        equity_curve[i] = equity + cur_pnl

    return dfx, trades, equity_curve

# --- API ROUTES ---

@app.get("/api/health")
async def health():
    return {"status": "ok"}

@app.get("/api/expiries")
async def get_expiries():
    try:
        csv_path = os.path.join(os.path.dirname(__file__), "..", "reference", "spy_quotedata.csv")
        print(f"DEBUG: Looking for CSV at {csv_path}")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, skiprows=3, on_bad_lines='skip', engine='python')
            df.columns = [c.strip() for c in df.columns]
            if "Expiration Date" in df.columns:
                expiries = sorted(df["Expiration Date"].dropna().unique().tolist())
                print(f"DEBUG: Found {len(expiries)} expiries")
                return {"expiries": expiries}
            else:
                print(f"DEBUG: 'Expiration Date' column not found. Available: {df.columns.tolist()}")
        else:
            print(f"DEBUG: CSV file not found at {csv_path}")
        return {"expiries": []}
    except Exception as e:
        print(f"ERROR in get_expiries: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.get("/api/simulate")
async def simulate_endpoint(
    use_real_data: bool = False, 
    periods: int = 400,
    expiry: Optional[str] = None,
    option_type: str = "Calls",
    max_hold_bars: int = 24,
    min_gamma: float = 0.05,
    min_delta: float = 0.20,
    max_delta: float = 0.55,
    min_delta_slope_to_hold: float = 0.00,
    take_profit_r: float = 2.0,
    stop_loss_r: float = 1.0,
    stall_bars: int = 3,
    entry_spread_bps: float = 8.0,
    exit_spread_bps: float = 8.0,
    fee_per_contract: float = 0.65,
    risk_per_trade_dollars: float = 100.0,
    contracts: int = 1
):
    try:
        df = pd.DataFrame()
        if use_real_data:
            csv_path = os.path.join(os.path.dirname(__file__), "..", "reference", "spy_quotedata.csv")
            if os.path.exists(csv_path):
                # We need the underlying price to find ATM strikes
                # The file has underlying price in line 2 or 3
                underlying_price = 683.0 # Default fallback
                try:
                    with open(csv_path, 'r') as f:
                        for _ in range(3):
                            line = f.readline()
                            if "Last:" in line:
                                parts = line.split("Last:")
                                if len(parts) > 1:
                                    underlying_price = float(parts[1].split(",")[0].strip())
                                    break
                except: pass

                df = pd.read_csv(csv_path, skiprows=3, on_bad_lines='skip', engine='python')
                df.columns = [c.strip() for c in df.columns]
                
                if expiry:
                    df = df[df["Expiration Date"] == expiry]
                
                # identify price column (Strike)
                df["close"] = pd.to_numeric(df["Strike"], errors='coerce')
                df = df.dropna(subset=["close"]).sort_values("close")

                # Filter for strikes around ATM to see Delta transitions
                # We want a range of strikes, e.g. underlying_price +/- 100
                df = df[(df["close"] >= underlying_price - 150) & (df["close"] <= underlying_price + 150)]

                delta_col = "Delta" if option_type == "Calls" else "Delta.1"
                gamma_col = "Gamma" if option_type == "Calls" else "Gamma.1"
                iv_col = "IV" if option_type == "Calls" else "IV.1"
                
                if delta_col in df.columns:
                    df["_delta"] = pd.to_numeric(df[delta_col], errors='coerce').fillna(0.5)
                if gamma_col in df.columns:
                    df["_gamma"] = pd.to_numeric(df[gamma_col], errors='coerce').fillna(0.0)
                if iv_col in df.columns:
                    df["_iv"] = pd.to_numeric(df[iv_col], errors='coerce').fillna(0.01)
                    
        if df.empty or "close" not in df.columns:
            df = pd.DataFrame({"close": 500 * (1 + np.cumsum(np.random.normal(0, 0.001, size=periods)))})
            
        df = df.head(periods)
        
        policy = HarvestPolicy(
            max_hold_bars=max_hold_bars, min_gamma=min_gamma, min_delta=min_delta, max_delta=max_delta,
            min_delta_slope_to_hold=min_delta_slope_to_hold, take_profit_r=take_profit_r, stop_loss_r=stop_loss_r,
            stall_bars=stall_bars, entry_spread_bps=entry_spread_bps, exit_spread_bps=exit_spread_bps,
            fee_per_contract=fee_per_contract, risk_per_trade_dollars=risk_per_trade_dollars, contracts=contracts
        )
        
        df_res, trades, equity_curve = run_simulation(df, policy)
        return {
            "prices": df_res["close"].tolist(), 
            "deltas": df_res["_delta"].tolist(), 
            "gammas": df_res["_gamma"].tolist(), 
            "equity": equity_curve, 
            "trades": trades
        }
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": str(e)})

# --- FILE ROUTES ---

@app.get("/")
async def get_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "fincap.html"))

@app.get("/fincap.html")
async def get_fincap():
    return FileResponse(os.path.join(os.path.dirname(__file__), "fincap.html"))

@app.get("/fincap.js")
async def get_js():
    return FileResponse(os.path.join(os.path.dirname(__file__), "fincap.js"))

app.mount("/reference", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "reference")), name="reference")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
