"""
wyckoff_labeler.py

Upgrades:
- Computes compact features from price/volume
- Calls OpenAI Responses API with Structured Outputs (json_schema) to get:
    [{"label","start_date","end_date","confidence","notes","events":[...]}]
- Plots shaded Wyckoff phases + optional volume panel
- Saves phases to JSON/CSV and chart PNG

CSV requirements:
- Date column named 'Date'
- Price column named 'Close/Last'
- Optional: 'Volume'

Env:
- set OPENAI_API_KEY in your environment
- pip install openai pandas matplotlib numpy

Docs:
- Structured outputs: json_schema :contentReference[oaicite:3]{index=3}
- Responses API :contentReference[oaicite:4]{index=4}
"""

from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

#client = OpenAI()

#response = client.responses.create(
#    model="gpt-5.2",
#    input="Write a short bedtime story about a unicorn."
#)

#print(response.output_text)



# -------------------------
# Config
# -------------------------

DEFAULT_MODEL = "gpt-4.1"  # pick your preferred model; keep stable in your own env
OUT_DIR = r"D:\wyckoff_output"



OUT_JSON  = os.path.join(OUT_DIR, "wyckoff_phases.json")
OUT_CSV   = os.path.join(OUT_DIR, "wyckoff_phases.csv")
OUT_CHART = os.path.join(OUT_DIR, "wyckoff_chart.png")

os.makedirs(OUT_DIR, exist_ok=True)


# -------------------------
# Data utilities
# -------------------------

def load_stock_csv(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")

    if "Close/Last" not in df.columns:
        raise ValueError("CSV must have 'Close/Last' column.")

    # numeric cleanup
    df["Close/Last"] = pd.to_numeric(df["Close/Last"], errors="coerce")
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    df = df.dropna(subset=["Close/Last"])
    return df


def build_features(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Build compact, model-friendly features.
    Avoid sending full raw series if you don't need it.
    """
    out = pd.DataFrame(index=df.index)
    px = df["Close/Last"].astype(float)

    out["close"] = px
    out["ret_1d"] = px.pct_change()
    out["logret_1d"] = np.log(px).diff()

    # trend proxy: slope of rolling linear regression on log price
    # (kept simple + deterministic)
    def rolling_slope(x: np.ndarray) -> float:
        if np.any(~np.isfinite(x)):
            return np.nan
        n = len(x)
        t = np.arange(n, dtype=float)
        t = (t - t.mean())
        y = x - x.mean()
        denom = np.dot(t, t)
        if denom == 0:
            return 0.0
        return float(np.dot(t, y) / denom)

    logp = np.log(px)
    out["trend_slope"] = (
        logp.rolling(lookback)
        .apply(lambda s: rolling_slope(s.values), raw=False)
    )

    out["volatility"] = out["logret_1d"].rolling(lookback).std()
    out["range_pct"] = (px.rolling(lookback).max() - px.rolling(lookback).min()) / px.rolling(lookback).mean()

    if "Volume" in df.columns:
        v = df["Volume"].astype(float)
        out["vol"] = v
        out["vol_z"] = (v - v.rolling(lookback).mean()) / (v.rolling(lookback).std() + 1e-9)
        out["vol_trend"] = v.pct_change().rolling(lookback).mean()
    else:
        out["vol"] = np.nan
        out["vol_z"] = np.nan
        out["vol_trend"] = np.nan

    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def compress_for_model(feat: pd.DataFrame, max_points: int = 260) -> List[Dict[str, Any]]:
    """
    Downsample to a max number of points (keeps token usage sane).
    We keep end-points and roughly uniform spacing.
    """
    if len(feat) <= max_points:
        use = feat
    else:
        idx = np.linspace(0, len(feat) - 1, max_points).round().astype(int)
        use = feat.iloc[idx]

    rows: List[Dict[str, Any]] = []
    for dt, r in use.iterrows():
        rows.append({
            "date": dt.strftime("%Y-%m-%d"),
            "close": round(float(r["close"]), 4),
            "ret_1d": None if pd.isna(r["ret_1d"]) else round(float(r["ret_1d"]), 6),
            "trend_slope": None if pd.isna(r["trend_slope"]) else round(float(r["trend_slope"]), 6),
            "volatility": None if pd.isna(r["volatility"]) else round(float(r["volatility"]), 6),
            "range_pct": None if pd.isna(r["range_pct"]) else round(float(r["range_pct"]), 6),
            "vol_z": None if pd.isna(r["vol_z"]) else round(float(r["vol_z"]), 4),
        })
    return rows


# -------------------------
# OpenAI labeling
# -------------------------

WYCKOFF_SCHEMA: Dict[str, Any] = {
    "name": "wyckoff_phase_labels",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ticker": {"type": "string"},
            "timeframe": {"type": "string"},
            "phases": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "label": {
                            "type": "string",
                            "enum": [
                                "Accumulation",
                                "Markup",
                                "Distribution",
                                "Markdown",
                                "Re-accumulation",
                                "Re-distribution",
                                "Transition/Unclear"
                            ]
                        },
                        "start_date": {"type": "string"},
                        "end_date": {"type": "string"},
                        "confidence": {"type": "number"},
                        "notes": {"type": "string"},
                        "events": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "date": {"type": "string"},
                                    "tag": {
                                        "type": "string",
                                        "enum": [
                                            "PS", "SC", "AR", "ST",
                                            "Spring", "Test", "SOS", "LPS",
                                            "UT", "UTAD", "SOW", "LPSY",
                                            "Breakout", "Breakdown"
                                        ]
                                    },
                                    "notes": {"type": "string"}
                                },
                                "required": ["date", "tag", "notes"]
                            }
                        }
                    },
                    "required": ["label", "start_date", "end_date", "confidence", "notes", "events"]
                }
            }
        },
        "required": ["ticker", "timeframe", "phases"]
    },
    "strict": True
}


def call_openai_for_phases(
    series_payload: List[Dict[str, Any]],
    ticker: str = "UNKNOWN",
    timeframe: str = "1D",
    model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    Uses Responses API + Structured Outputs json_schema to guarantee valid JSON.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing OPENAI_API_KEY env var.")

    client = OpenAI(api_key=api_key)

    system_msg = (
        "You are a Wyckoff analyst.\n"
        "Return ONLY valid JSON matching this schema exactly:\n"
        f"{json.dumps(WYCKOFF_SCHEMA['schema'])}\n"
        "No prose, no markdown, no commentary.\n"
        "If uncertain, use label 'Transition/Unclear' with lower confidence.\n"
    )


    user_msg = {
        "ticker": ticker,
        "timeframe": timeframe,
        "series": series_payload,
        "task": (
            "Return Wyckoff phases that cover the provided date range. "
            "Each phase has start_date/end_date (inclusive) and confidence in [0,1]. "
            "Optionally include key Wyckoff events (PS, SC, AR, ST, Spring, Test, SOS, LPS, UT, UTAD, SOW, LPSY, Breakout, Breakdown) "
            "with dates. Notes must be short and evidence-based (trend_slope, volatility, range_pct, vol_z, returns)."
        )
    }
    
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": json.dumps(user_msg)}
        ]
    )


    # The SDK returns parsed JSON under output_text as a string for many models;
    # keep it robust: find the first JSON object.
    txt = resp.output_text
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        # fallback: attempt to extract JSON object boundaries
        start = txt.find("{")
        end = txt.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Model did not return JSON parseable output.")
        data = json.loads(txt[start:end+1])

    return data


# -------------------------
# Plotting + persistence
# -------------------------

def phases_to_dataframe(phases_payload: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for ph in phases_payload.get("phases", []):
        rows.append({
            "label": ph["label"],
            "start_date": ph["start_date"],
            "end_date": ph["end_date"],
            "confidence": ph["confidence"],
            "notes": ph["notes"],
            "events": json.dumps(ph.get("events", []))
        })
    return pd.DataFrame(rows)


def plot_wyckoff_chart(
    df: pd.DataFrame,
    phases_payload: Dict[str, Any],
    title: str = "Stock Price with Wyckoff Phases",
    out_png: str = OUT_CHART
) -> None:
    has_volume = "Volume" in df.columns

    if has_volume:
        fig, axes = plt.subplots(
            2, 1, figsize=(14, 10), sharex=True,
            height_ratios=[3, 1]
        )
        ax_price, ax_vol = axes
    else:
        fig, ax_price = plt.subplots(figsize=(14, 8))
        ax_vol = None

    # price
    ax_price.plot(df.index, df["Close/Last"], linewidth=1.5, label="Close/Last")
    ax_price.set_title(title)
    ax_price.set_ylabel("Price")
    ax_price.grid(True, linestyle="--", alpha=0.6)

    # volume
    if ax_vol is not None:
        ax_vol.bar(df.index, df["Volume"], alpha=0.5)
        ax_vol.set_ylabel("Volume")
        ax_vol.grid(True, linestyle="--", alpha=0.6)

    # consistent colors by label (simple + readable)
    color_map = {
        "Accumulation": "lightgreen",
        "Markup": "palegreen",
        "Distribution": "moccasin",
        "Markdown": "lightcoral",
        "Re-accumulation": "honeydew",
        "Re-distribution": "navajowhite",
        "Transition/Unclear": "lightgrey"
    }

    # shaded phases + labels
    for ph in phases_payload.get("phases", []):
        start = pd.to_datetime(ph["start_date"])
        end = pd.to_datetime(ph["end_date"])
        label = ph["label"]
        conf = ph["confidence"]
        color = color_map.get(label, "lightgrey")

        ax_price.axvspan(start, end, alpha=0.25, color=color)

        # label centered vertically on visible range
        mid = start + (end - start) / 2
        ymid = np.mean(ax_price.get_ylim())
        ax_price.text(
            mid, ymid,
            f"{label}\n{conf:.2f}",
            ha="center", va="center",
            fontsize=9, fontweight="bold",
            rotation=90
        )

        # plot events as markers
        for ev in ph.get("events", []):
            try:
                ev_date = pd.to_datetime(ev["date"])
            except Exception:
                continue
            # place marker on close price if date exists in df
            if ev_date in df.index:
                y = float(df.loc[ev_date, "Close/Last"])
                ax_price.scatter([ev_date], [y], s=30)
                ax_price.text(ev_date, y, f" {ev['tag']}", fontsize=8, va="bottom")

    # x-axis formatting
    ax_price.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    ax_price.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.show()


def save_outputs(phases_payload: Dict[str, Any], out_json: str = OUT_JSON, out_csv: str = OUT_CSV) -> None:
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(phases_payload, f, indent=2)

    pdf = phases_to_dataframe(phases_payload)
    pdf.to_csv(out_csv, index=False)


# -------------------------
# Main
# -------------------------

def run(
    csv_file: str,
    ticker: str = "SPY",
    timeframe: str = "1D",
    title: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    lookback: int = 20,
    max_points: int = 260
) -> None:
    df = load_stock_csv(csv_file)
    feat = build_features(df, lookback=lookback)

    series_payload = compress_for_model(feat, max_points=max_points)

    phases_payload = call_openai_for_phases(
        series_payload=series_payload,
        ticker=ticker,
        timeframe=timeframe,
        model=model
    )

    save_outputs(phases_payload, OUT_JSON, OUT_CSV)

    plot_title = title or f"{ticker} Wyckoff Phase Labeling ({timeframe})"
    plot_wyckoff_chart(df, phases_payload, title=plot_title, out_png=OUT_CHART)

    print(f"Saved: {OUT_JSON}, {OUT_CSV}, {OUT_CHART}")


if __name__ == "__main__":
    # Example:
    # python wyckoff_labeler.py
    # (adjust path + ticker)
    run(
        csv_file="F:/inputs/stocks/SPY.csv",
        ticker="SPY",
        timeframe="1D",
        title="SPY Wyckoff Phases (OpenAI-labeled)",
        model=DEFAULT_MODEL,
        lookback=20,
        max_points=260
    )
