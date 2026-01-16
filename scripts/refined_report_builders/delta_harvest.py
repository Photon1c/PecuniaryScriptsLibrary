# Delta Harvest
# timeboxed_delta_harvest.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_trade_analysis_plotly(out: pd.DataFrame, filename: str = "delta_harvest_analysis.html") -> None:
    if out is None or out.empty:
        print("No trades to plot.")
        return

    # Create subplots: 1 row, 2 columns
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Entry Delta vs PnL", "Entry Gamma vs PnL")
    )

    # Delta vs P/L
    fig.add_trace(
        go.Scatter(
            x=out["meta_entry_delta"],
            y=out["pnl"],
            mode='markers',
            marker=dict(color='blue', opacity=0.6),
            name="Delta vs PnL"
        ),
        row=1, col=1
    )

    # Gamma vs P/L
    fig.add_trace(
        go.Scatter(
            x=out["meta_entry_gamma"],
            y=out["pnl"],
            mode='markers',
            marker=dict(color='green', opacity=0.6),
            name="Gamma vs PnL"
        ),
        row=1, col=2
    )

    fig.update_layout(
        title_text="Delta Harvest Operational Analysis",
        height=500,
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Entry Delta (proxy)", row=1, col=1)
    fig.update_yaxes(title_text="PnL ($)", row=1, col=1)
    fig.update_xaxes(title_text="Entry Gamma (proxy)", row=1, col=2)
    fig.update_yaxes(title_text="PnL ($)", row=1, col=2)

    fig.write_html(filename)
    print(f"[SUCCESS] Analysis charts saved to: {filename}")


# Harness
def score_operational_fitness(out: pd.DataFrame) -> Dict[str, float]:
    if out.empty:
        return {
            "trades": 0,
            "avg_R": 0.0,
            "win_rate": 0.0,
            "timeout_rate": 0.0,
            "avg_MAE_$": 0.0,
            "avg_MFE_$": 0.0,
            "fitness": 0.0,
        }

    trades = len(out)
    avg_R = float(out["r_multiple"].mean())
    win_rate = float((out["pnl"] > 0).mean())
    timeout_rate = float((out["exit_reason"] == "timebox_expired").mean())
    avg_mae = float(out["mae"].mean())
    avg_mfe = float(out["mfe"].mean())

    # Fitness: reward expectancy + discipline, penalize timeouts
    fitness = (avg_R * 100.0) + (win_rate * 10.0) - (timeout_rate * 20.0)

    return {
        "trades": trades,
        "avg_R": avg_R,
        "win_rate": win_rate,
        "timeout_rate": timeout_rate,
        "avg_MAE_$": avg_mae,
        "avg_MFE_$": avg_mfe,
        "fitness": float(fitness),
    }


def run_rehearsals(
    df: pd.DataFrame,
    policy: HarvestPolicy,
    *,
    n_windows: int = 200,
    window_bars: int = 200,
    stride: int = 25,
    price_col: str = "close",
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Slices df into rolling windows and runs the sim per window.
    This is your 'months of rehearsal' generator.
    """
    all_trades = []
    start = 0
    windows = 0

    while windows < n_windows and (start + window_bars) <= len(df):
        dwin = df.iloc[start : start + window_bars].copy()
        trades = simulate_timeboxed_delta_harvest(dwin, policy, price_col=price_col, verbose=verbose)
        if trades:
            all_trades.extend(trades)
        start += stride
        windows += 1

    out = trades_to_frame(all_trades)
    fitness = score_operational_fitness(out)

    return {"out": out, "fitness": fitness}


# ----------------------------
# Config / Policy
# ----------------------------
@dataclass
class HarvestPolicy:
    # timebox in bars (e.g., if 5-min candles and you want 2 hours -> 24 bars)
    max_hold_bars: int = 72

    # entry filters (operate on proxies for now)
    min_gamma: float = 0.05
    min_delta: float = 0.20
    max_delta: float = 0.55

    # "delta acceleration" proxy threshold: delta_slope must exceed this to keep holding
    min_delta_slope_to_hold: float = 0.00

    # exit triggers
    take_profit_r: float = 2.0   # take profit when P/L >= 2R
    stop_loss_r: float = 1.0     # stop at -1R
    stall_bars: int = 4          # consecutive bars of weak delta slope => exit

    # execution assumptions (slippage + spread)
    entry_spread_bps: float = 8.0   # pay up at ask
    exit_spread_bps: float = 8.0    # hit bid
    fee_per_contract: float = 0.65

    # risk model
    risk_per_trade_dollars: float = 100.0  # defines "1R"
    contracts: int = 1


@dataclass
class SimTrade:
    entry_idx: int
    exit_idx: int
    entry_time: Any
    exit_time: Any
    entry_price: float
    exit_price: float
    pnl: float
    r_multiple: float
    mfe: float
    mae: float
    exit_reason: str
    meta: Dict[str, Any]


# ----------------------------
# Option proxy model (swap later)
# ----------------------------
def _sigmoid(x: float) -> float:
    # Prevent exp overflow when x is huge (common when vol_proxy ~ 0)
    x = max(min(x, 60.0), -60.0)
    return 1.0 / (1.0 + math.exp(-x))



def estimate_greeks_proxy(
    close: float,
    close_prev: float,
    vol_proxy: float,
) -> Dict[str, float]:
    """
    Stand-in for delta/gamma/iv. Replace this with:
      - real chain data lookup (delta/gamma/iv at that timestamp)
      - or a BSM estimation using implied vol + strike + DTE.

    For now:
      delta ~ sigmoid(price change / vol)
      gamma ~ sensitivity of delta to change (crudely tied to vol_proxy)
      iv ~ vol_proxy (already)
    """
    # work in return space so x doesn't blow up
    ret = 0.0 if close_prev == 0 else (close / close_prev) - 1.0

    # vol_proxy is already rolling std of returns; keep floors small but realistic
    vol_floor = 1e-5
    x = ret / max(vol_proxy, vol_floor)

    delta = _sigmoid(x)  # stays near 0.5 most of the time, deviates on impulse

    gamma = abs(delta * (1 - delta)) * 4.0  # peak near delta=0.5
    iv = vol_proxy
    return {"delta": delta, "gamma": gamma, "iv": iv}


# ----------------------------
# Engine
# ----------------------------
def simulate_timeboxed_delta_harvest(
    df: pd.DataFrame,
    policy: HarvestPolicy,
    *,
    price_col: str = "close",
    time_col: Optional[str] = None,
    verbose: bool = False,
) -> List[SimTrade]:
    """
    df must have a monotonic index (or time column) and at least `price_col`.
    If you have intraday candles, each row = one bar.

    This sim:
      - looks for entries when gamma & delta are in range
      - holds while delta slope is positive enough
      - exits on stop/takeprofit/timebox/stall
    """
    if price_col not in df.columns:
        raise ValueError(f"df missing required column: {price_col}")

    dfx = df.copy()
    dfx["_close"] = dfx[price_col].astype(float)

    # vol proxy: rolling std of returns
    rets = dfx["_close"].pct_change().fillna(0.0)
    dfx["_vol"] = (
        rets.rolling(20, min_periods=5).std()
        .bfill()
        .fillna(0.0)
    )

    # compute proxy greeks + delta slope
    deltas = [0.5]
    gammas = [0.0]
    ivs = [dfx["_vol"].iloc[0]]
    for i in range(1, len(dfx)):
        g = estimate_greeks_proxy(
            close=dfx["_close"].iloc[i],
            close_prev=dfx["_close"].iloc[i - 1],
            vol_proxy=dfx["_vol"].iloc[i],
        )
        deltas.append(g["delta"])
        gammas.append(g["gamma"])
        ivs.append(g["iv"])

    dfx["_delta"] = deltas
    dfx["_gamma"] = gammas
    dfx["_iv"] = ivs
    dfx["_delta_slope"] = dfx["_delta"].diff().fillna(0.0)

    if verbose:
        print("delta range:", dfx["_delta"].min(), dfx["_delta"].max())
        print("gamma range:", dfx["_gamma"].min(), dfx["_gamma"].max())
        print("pct gamma >= min_gamma:", (dfx["_gamma"] >= policy.min_gamma).mean())
        print("pct delta in band:", ((dfx["_delta"]>=policy.min_delta)&(dfx["_delta"]<=policy.max_delta)).mean())


    trades: List[SimTrade] = []
    in_trade = False
    entry_idx = -1
    entry_price = 0.0
    stall_count = 0
    mfe = 0.0
    mae = 0.0

    # helper for timestamp
    def _t(i: int):
        if time_col and time_col in dfx.columns:
            return dfx[time_col].iloc[i]
        return dfx.index[i]

    # simplistic option price proxy: map delta * underlying move + iv term
    # Replace with real mid/ask/bid from chain later.
    def option_price_proxy(i: int) -> float:
        # baseline premium proxy
        base = 1.00
        move_term = 50.0 * abs(rets.iloc[i]) * dfx["_delta"].iloc[i]
        iv_term = 10.0 * dfx["_iv"].iloc[i]
        return max(0.05, base + move_term + iv_term)

    def apply_spread(price: float, bps: float, side: str) -> float:
        # side: "buy" pays up, "sell" receives less
        adj = price * (bps / 10_000.0)
        return price + adj if side == "buy" else max(0.01, price - adj)

    i = 20  # start after warmup
    while i < len(dfx):
        if not in_trade:
            # ENTRY LOGIC
            delta = dfx["_delta"].iloc[i]
            gamma = dfx["_gamma"].iloc[i]

            delta_slope = dfx["_delta_slope"].iloc[i]
            if (
                gamma >= policy.min_gamma
                and policy.min_delta <= delta <= policy.max_delta
                and delta_slope > 0.0  # entry gating: only enter when delta is starting to lift
            ):

                # enter at ask (spread penalty)
                raw = option_price_proxy(i)
                entry_price = apply_spread(raw, policy.entry_spread_bps, "buy")
                entry_idx = i
                in_trade = True
                stall_count = 0
                mfe = 0.0
                mae = 0.0
        else:
            # UPDATE P/L
            raw_now = option_price_proxy(i)
            exit_mark = apply_spread(raw_now, policy.exit_spread_bps, "sell")

            # P/L per contract
            pnl_gross = (exit_mark - entry_price) * 100.0 * policy.contracts
            fees = policy.fee_per_contract * policy.contracts * 2  # in+out
            pnl = pnl_gross - fees

            mfe = max(mfe, pnl)
            mae = min(mae, pnl)

            # convert to R
            r = pnl / max(policy.risk_per_trade_dollars, 1e-9)

            # EXIT CONDITIONS
            held = i - entry_idx
            delta_slope = dfx["_delta_slope"].iloc[i]

            if r >= policy.take_profit_r:
                reason = "take_profit"
            elif r <= -policy.stop_loss_r:
                reason = "stop_loss"
            elif held >= policy.max_hold_bars:
                reason = "timebox_expired"
            else:
                # stall logic
                if delta_slope <= policy.min_delta_slope_to_hold:
                    stall_count += 1
                else:
                    stall_count = 0

                if stall_count >= policy.stall_bars:
                    reason = "delta_stall"
                else:
                    reason = ""

            if reason:
                trades.append(
                    SimTrade(
                        entry_idx=entry_idx,
                        exit_idx=i,
                        entry_time=_t(entry_idx),
                        exit_time=_t(i),
                        entry_price=entry_price,
                        exit_price=exit_mark,
                        pnl=pnl,
                        r_multiple=r,
                        mfe=mfe,
                        mae=mae,
                        exit_reason=reason,
                        meta={
                            "entry_delta": float(dfx["_delta"].iloc[entry_idx]),
                            "entry_gamma": float(dfx["_gamma"].iloc[entry_idx]),
                            "exit_delta": float(dfx["_delta"].iloc[i]),
                            "exit_gamma": float(dfx["_gamma"].iloc[i]),
                            "held_bars": held,
                        },
                    )
                )
                in_trade = False

        i += 1


        # --- FORCE EXIT AT END OF DATA (prevents empty trades when still in a position)
        if in_trade and entry_idx >= 0:
            last = len(dfx) - 1
            raw_now = option_price_proxy(last)
            exit_mark = apply_spread(raw_now, policy.exit_spread_bps, "sell")

            pnl_gross = (exit_mark - entry_price) * 100.0 * policy.contracts
            fees = policy.fee_per_contract * policy.contracts * 2
            pnl = pnl_gross - fees
            r = pnl / max(policy.risk_per_trade_dollars, 1e-9)

            trades.append(
                SimTrade(
                    entry_idx=entry_idx,
                    exit_idx=last,
                    entry_time=_t(entry_idx),
                    exit_time=_t(last),
                    entry_price=entry_price,
                    exit_price=exit_mark,
                    pnl=pnl,
                    r_multiple=r,
                    mfe=max(mfe, pnl),
                    mae=min(mae, pnl),
                    exit_reason="eod_forced",
                    meta={
                        "entry_delta": float(dfx["_delta"].iloc[entry_idx]),
                        "entry_gamma": float(dfx["_gamma"].iloc[entry_idx]),
                        "exit_delta": float(dfx["_delta"].iloc[last]),
                        "exit_gamma": float(dfx["_gamma"].iloc[last]),
                        "held_bars": last - entry_idx,
                    },
                )
            )


    return trades


def trades_to_frame(trades: List[SimTrade]) -> pd.DataFrame:
    rows = []
    for t in trades:
        d = asdict(t)
        meta = d.pop("meta", {})
        d.update({f"meta_{k}": v for k, v in meta.items()})
        rows.append(d)
    return pd.DataFrame(rows)


import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_trade_analysis_plotly(out: pd.DataFrame, filename: str = "delta_harvest_analysis.html") -> None:
    if out is None or out.empty:
        print("No trades to plot.")
        return

    # Create subplots: 1 row, 2 columns
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Entry Delta vs PnL", "Entry Gamma vs PnL")
    )

    # Delta vs P/L
    fig.add_trace(
        go.Scatter(
            x=out["meta_entry_delta"],
            y=out["pnl"],
            mode='markers',
            marker=dict(color='blue', opacity=0.6),
            name="Delta vs PnL"
        ),
        row=1, col=1
    )

    # Gamma vs P/L
    fig.add_trace(
        go.Scatter(
            x=out["meta_entry_gamma"],
            y=out["pnl"],
            mode='markers',
            marker=dict(color='green', opacity=0.6),
            name="Gamma vs PnL"
        ),
        row=1, col=2
    )

    fig.update_layout(
        title_text="Delta Harvest Operational Analysis",
        height=500,
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Entry Delta (proxy)", row=1, col=1)
    fig.update_yaxes(title_text="PnL ($)", row=1, col=1)
    fig.update_xaxes(title_text="Entry Gamma (proxy)", row=1, col=2)
    fig.update_yaxes(title_text="PnL ($)", row=1, col=2)

    fig.write_html(filename)
    print(f"[SUCCESS] Analysis charts saved to: {filename}")

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    import numpy as np
    
    # 1. Setup synthetic data
    idx = pd.date_range("2026-01-12 09:30", periods=500, freq="5min")
    price = 120 * (1 + np.cumsum(np.random.normal(0, 0.0025, size=len(idx))))
    df = pd.DataFrame({"close": price}, index=idx)

    # 2. Define Policy
    policy = HarvestPolicy(max_hold_bars=24, take_profit_r=2.0, stop_loss_r=1.0)

    # 3. Run Rehearsals (Comprehensive Run)
    print("Running operational rehearsals...")
    result = run_rehearsals(df, policy, n_windows=100, window_bars=100, stride=10, verbose=False)
    out = result["out"]
    fitness = result["fitness"]

    # 4. Output Summary Table
    print("\n--- REHEARSAL SUMMARY ---")
    for k, v in fitness.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    if not out.empty:
        print("\nLast 5 Trades:")
        cols_to_show = ["entry_time", "exit_time", "pnl", "r_multiple", "exit_reason"]
        print(out[cols_to_show].tail(5))
        
        # 5. Generate and save Plotly charts
        plot_trade_analysis_plotly(out, "delta_harvest_analysis.html")
    else:
        print("\nNo trades were generated during the rehearsal.")



