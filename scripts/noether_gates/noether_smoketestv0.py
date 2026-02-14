#Noether Smoke Test
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress



# Read CSV with Date and Close/Last; compute LogReturn
df = pd.read_csv("F:/inputs/stocks/SPY")
# Use Date and Close/Last; clean Close/Last (e.g. "$123.45" -> float)
df = df[["Date", "Close/Last"]].copy()
df["Date"] = pd.to_datetime(df["Date"])
df["Close"] = df["Close/Last"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)
df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
df = df.dropna(subset=["LogReturn"]).set_index("Date")


def compute_rolling_stats(returns, win_mu=252, win_sigma=252, win_slope=30):
    mu_log = returns.rolling(win_mu).mean()
    sigma2 = returns.rolling(win_sigma).var()
    return mu_log, sigma2

def compute_q_dot(Q, win_slope=30):
    # Slope of linear fit on last win_slope points
    slopes = np.full(len(Q), np.nan)
    for i in range(win_slope-1, len(Q)):
        y = Q.iloc[i-win_slope+1:i+1].values
        x = np.arange(len(y))
        if np.isnan(y).any(): continue
        slope, _, _, _, _ = linregress(x, y)
        slopes[i] = slope
    return pd.Series(slopes, index=Q.index)

# Parameters
lambda_risk = 1.0      # λ in Q = mu - λ sigma^2
kappa = 100.0          # Gate strength; try 50–200
mom_window = 20        # For simple momentum signal
kelly_win = 252        # ~1 year for stats
slope_win = 30         # For Q̇ estimate

# Simple momentum signal: long if close > SMA(20)
df['SMA20'] = df['Close'].rolling(mom_window).mean()
df['Signal'] = np.where(df['Close'] > df['SMA20'], 1.0, 0.0)

# Rolling stats for Kelly
mu_log, sigma2 = compute_rolling_stats(df['LogReturn'], kelly_win, kelly_win)

# Kelly fraction (basic form: f = mu / sigma^2, clamped)
f_kelly = mu_log / sigma2
f_kelly = f_kelly.clip(0, 2.0)  # Rough cap to avoid explosion

# Q_t = risk-adjusted log-growth proxy
Q_t = mu_log - lambda_risk * sigma2

# Q̇_t
Q_dot = compute_q_dot(Q_t, slope_win)

# Gated fraction
f_gated = f_kelly * np.exp(-kappa * np.abs(Q_dot.fillna(0)))

# Strategy returns
strat_ret_kelly = df['Signal'].shift(1) * df['LogReturn'] * f_kelly.shift(1)
strat_ret_gated = df['Signal'].shift(1) * df['LogReturn'] * f_gated.shift(1)

# Cumulative equity (assume starting capital 1, log → exp)
eq_kelly = np.exp(strat_ret_kelly.cumsum()).rename('Kelly')
eq_gated = np.exp(strat_ret_gated.cumsum()).rename('Gated')
eq_buyhold = df['Close'] / df['Close'].iloc[0]

# Metrics function
def metrics(eq_curve):
    ret = eq_curve.pct_change().dropna()
    cagr = (eq_curve.iloc[-1] ** (252 / len(eq_curve)) - 1) * 100
    max_dd = (eq_curve / eq_curve.cummax() - 1).min() * 100
    ulcer = np.sqrt((eq_curve / eq_curve.cummax() - 1).pow(2).mean()) * 100  # approx
    mar = cagr / -max_dd if max_dd != 0 else np.nan
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() != 0 else np.nan
    return {'CAGR (%)': round(cagr, 2), 'Max DD (%)': round(max_dd, 2),
            'MAR': round(mar, 2), 'Sharpe': round(sharpe, 2), 'Ulcer (%)': round(ulcer, 2)}

print("Buy & Hold:", metrics(eq_buyhold))
print("Pure Rolling Kelly:", metrics(eq_kelly))
print("Q̇-Gated Kelly:", metrics(eq_gated))

# Plots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
eq_buyhold.plot(ax=ax1, label='Buy&Hold')
eq_kelly.plot(ax=ax1, label='Kelly')
eq_gated.plot(ax=ax1, label='Gated')
ax1.legend(); ax1.set_title('Equity Curves')

Q_t.plot(ax=ax2, label='Q_t'); ax2.legend(); ax2.set_title('Risk-Adj Log-Growth (Q)')
Q_dot.plot(ax=ax3, label='Q̇_t (slope)'); ax3.legend(); ax3.set_title('Q̇ (Symmetry Break Proxy)')

f_gated.plot(ax=ax3.twinx(), color='green', alpha=0.4, label='Gated f')
ax3.legend(loc='upper left')
plt.tight_layout()
plt.show()
