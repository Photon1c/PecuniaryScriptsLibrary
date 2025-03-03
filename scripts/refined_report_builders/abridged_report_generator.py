#Generate a Perfunctory Report on Selected Ticker
#Requires CBOE and Nasdaq data manually curated and maintained
import os
import glob
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from datetime import datetime

# 🎯 Define paths
ticker = "SPY"
stock_file_path = r"X:\inputs\stocks\SPY.csv"  # Update with the correct path
ticker_options_dir = r"X:\inputs\options\log"

### 📌 1️⃣ Function to Find Latest Option Data File
def get_latest_option_file(ticker_options_dir, ticker):
    """Find the most recent options data CSV file for a given ticker, sorted by actual date."""
    try:
        option_folders = [
            f for f in glob.glob(os.path.join(ticker_options_dir, ticker.lower(), "*"))
            if os.path.isdir(f) and os.path.basename(f).count("_") == 2
        ]

        option_folders.sort(key=lambda f: datetime.strptime(os.path.basename(f), "%m_%d_%Y"), reverse=True)

        if not option_folders:
            return None, "⚠️ No valid date folders found"

        latest_folder = option_folders[0]
        csv_file = os.path.join(latest_folder, f"{ticker.lower()}_quotedata.csv")

        return (csv_file, None) if os.path.exists(csv_file) else (None, "❌ CSV file not found")
    
    except Exception as e:
        return None, f"⚠️ Error in get_latest_option_file(): {e}"

### 📌 2️⃣ Compute RSI
def compute_rsi(series, period=14):
    """Compute RSI using EMA smoothing."""
    delta = series.diff().dropna()
    gain = (delta.where(delta > 0, 0)).ewm(span=period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=period, adjust=False).mean()
    return 100 - (100 / (1 + gain / loss))

### 📌 3️⃣ Main Ticker Analysis Function
def analyze_ticker(ticker, stock_file_path, ticker_options_dir=None):
    """Perform financial analysis on a stock ticker."""
    try:
        # 🏦 Load stock data
        stock_df = pd.read_csv(stock_file_path)
        stock_df.columns = [col.lower().replace("/", "_") for col in stock_df.columns]

        # 🔍 Identify essential columns
        price_col = next((c for c in stock_df.columns if "close" in c or "last" in c), None)
        volume_col = next((c for c in stock_df.columns if "volume" in c), None)
        date_col = next((c for c in stock_df.columns if "date" in c), None)

        if not all([price_col, volume_col, date_col]):
            return {"error": "❌ Missing essential columns"}

        # 🗓 Convert to datetime and sort
        stock_df[date_col] = pd.to_datetime(stock_df[date_col])
        stock_df = stock_df.sort_values(by=date_col).dropna(subset=[price_col, volume_col])

        if len(stock_df) < 50:
            return {"error": "⚠️ Insufficient data (<50 rows)"}

        # 📊 Compute Key Technical Indicators
        stock_df["ma_20"] = stock_df[price_col].rolling(20).mean()
        stock_df["ma_50"] = stock_df[price_col].rolling(50).mean()
        stock_df["rsi"] = compute_rsi(stock_df[price_col])
        stock_df["macd"] = stock_df[price_col].ewm(span=12).mean() - stock_df[price_col].ewm(span=26).mean()
        stock_df["atr"] = (stock_df["high"] - stock_df["low"]).rolling(14).mean() if "high" in stock_df and "low" in stock_df else np.nan
        stock_df["bollinger_width"] = (stock_df[price_col].rolling(20).std() * 2) / stock_df[price_col].rolling(20).mean()

        # 📈 Trend Detection via Linear Regression Slope
        recent_prices = stock_df[price_col].tail(30).dropna()
        trend_slope = (
            LinearRegression().fit(np.arange(len(recent_prices)).reshape(-1, 1), recent_prices).coef_[0]
            if len(recent_prices) >= 20 else None
        )
        trend_slope = round(trend_slope / stock_df[price_col].mean() * 100, 2) if trend_slope else "Unavailable (Data issue)"

        # 📉 Stationarity Test
        try:
            is_stationary = adfuller(stock_df[price_col].dropna().tail(90))[1] < 0.05
        except:
            is_stationary = "Unavailable (ADF test failed)"

        # 🔮 Short-Term Forecast via ARIMA (Predict 5 Business Days Ahead)
        forecast_pct_change = "Unavailable (ARIMA model error)"
        try:
            if len(recent_prices) >= 30:
                # ✅ Ensure the dataset has a valid datetime index
                if stock_df.index.name != date_col:
                    stock_df = stock_df.set_index(date_col)

                # ✅ Create a complete Business Day index
                all_dates = pd.date_range(start=stock_df.index.min(), end=stock_df.index.max(), freq="B")
                stock_df = stock_df.reindex(all_dates, method="ffill")  # Fill missing business days

                recent_prices = stock_df[price_col].dropna().tail(90)  # Use last 90 days

                # ✅ Store the last observed price before differencing
                last_observed_price = recent_prices.iloc[-1]

                # ✅ Perform differencing to make data stationary
                differenced_series = recent_prices.diff().dropna()

                print(f"📊 Debug: Differenced Series Before ARIMA - Last 5 Rows:\n{differenced_series.tail()}")  # Debug

                # ✅ Run ARIMA Model with stabilization
                model = ARIMA(differenced_series, order=(1,1,1), enforce_stationarity=False, enforce_invertibility=False)
                model_fit = model.fit()

                # ✅ Forecasting 5 Business Days Ahead
                forecast_steps = 5
                forecast_differenced = model_fit.forecast(steps=forecast_steps)

                # ✅ Generate explicit forecast dates (next 5 business days)
                forecast_dates = pd.date_range(start=recent_prices.index[-1], periods=forecast_steps + 1, freq="B")[1:]

                # ✅ Convert forecasted value back to original scale (undo differencing)
                forecasted_prices = last_observed_price + forecast_differenced.cumsum()
                forecasted_series = pd.Series(forecasted_prices.values, index=forecast_dates)

                # ✅ Compute percentage change from last observed price
                forecast_pct_change = round(((forecasted_series.iloc[-1] - last_observed_price) / last_observed_price) * 100, 2)

                print(f"🔮 Forecasted Prices:\n{forecasted_series}")  # Debug
                print(f"📊 Forecasted Change for {forecast_dates[-1].strftime('%Y-%m-%d')}: {forecast_pct_change}%")  # Debug
        except Exception as e:
            forecast_pct_change = f"Unavailable ({str(e)})"

        print(f"🔍 Final Short-Term Forecast: {forecast_pct_change}")  # Debug








        # 📊 Retrieve Put-Call Ratio from Options Data
        csv_file, option_error = get_latest_option_file(ticker_options_dir, ticker)
        put_call_ratio = "Unavailable"
        if csv_file:
            try:
                options_df = pd.read_csv(csv_file, skiprows=3)
                options_df.columns = [col.lower().replace(" ", "_") for col in options_df.columns]

                call_volume_col, put_volume_col = "volume", "volume.1"
                if call_volume_col in options_df and put_volume_col in options_df:
                    put_volume, call_volume = options_df[put_volume_col].sum(), options_df[call_volume_col].sum()
                    put_call_ratio = round(put_volume / call_volume, 2) if call_volume > 0 else "Unavailable (Call volume is zero)"
            except Exception as e:
                put_call_ratio = f"Unavailable ({str(e)})"
        else:
            put_call_ratio = f"Unavailable ({option_error})"

        # 📝 Build Final Report Table
        report_table = pd.DataFrame({
            "Metric": [
                "📌 Current Price", "📈 Momentum (Slope)", "📊 Stationary Prices", "🔮 Short-Term Forecast (%)",
                "📉 20-Day MA", "📉 50-Day MA", "📊 RSI", "📊 MACD", "📊 ATR", "📊 Bollinger Width", "📊 Put-Call Ratio"
            ],
            "Value": [
                round(stock_df[price_col].iloc[-1], 2), trend_slope, is_stationary, forecast_pct_change,
                round(stock_df["ma_20"].iloc[-1], 2), round(stock_df["ma_50"].iloc[-1], 2),
                round(stock_df["rsi"].iloc[-1], 2), round(stock_df["macd"].iloc[-1], 4),
                round(stock_df["atr"].iloc[-1], 2), round(stock_df["bollinger_width"].iloc[-1], 2),
                put_call_ratio
            ]
        })

        return report_table

    except Exception as e:
        return {"error": f"⚠️ Analysis failed: {e}"}

# 🚀 Run the analysis
report_table = analyze_ticker(ticker, stock_file_path, ticker_options_dir)
report_table.to_html("spy_analysis_report.html", index=False)

print("📊 Report saved as spy_analysis_report.html")
print(report_table)
