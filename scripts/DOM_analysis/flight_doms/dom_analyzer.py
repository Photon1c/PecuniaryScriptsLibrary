"""
Live DOM Dashboard v3
---------------------

What this provides
- Live Cboe DOM scrape (bid/ask depth + last trade) with a pixel heatmap
- Bubble overlay for instantaneous depth (green=bid, red=ask)
- Axis-integrated, time–decayed bid/ask bars for per-price accumulation
- Thermo map (flip/walls/spot + simple BS pricing and greeks)

Install
    pip install flask selenium webdriver-manager scipy

Usage
- Live (headless):
    python metascripts/DOM_analysis/DOM_analyzer_dash_v3.py --default-ticker SPY --poll-interval 0.5 --port 8021 --headless

- Replay newest CSV in a directory:
    python metascripts/DOM_analysis/DOM_analyzer_dash_v3.py --replay-latest --replay-dir metascripts/DOM_analysis/data --poll-interval 0.05 --port 8021

Key flags
- --default-ticker        Default symbol (e.g., SPY)
- --poll-interval         Seconds between polls; supports fractional values
- --host/--port           Bind address/port (use 0.0.0.0 to expose on LAN)
- --headless              Run Chrome in headless mode
- --replay-latest/--replay-file  Backtest from CSV instead of live scrape

UI notes
- Bubbles drift right→left over time; newest column is at the far right
- Right axis shows integrated green/red bars aligned to price labels
- Thermo flip/put/call default to near-spot when inputs left blank/"auto"

Open in browser
    http://127.0.0.1:8021
"""

from __future__ import annotations
#from intraday_flight_phase import IntradayFlightClassifier
import atexit
import threading
import time
import argparse
from datetime import datetime
from typing import Dict, Optional, List, Tuple
import os
import glob
import csv

from flask import Flask, jsonify, request, Response

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# NEW: simple BS price + greeks
from math import log, sqrt, exp
from scipy.stats import norm  # requires scipy

# --- Optional intraday flight classifier ---
#
# This module is imported both as a script (historical usage) and as a package
# module via `flight_doms_app.adapter_dom` (`from flight_doms import dom_analyzer`).
# In the package-import case, intra-package imports must be relative.
try:
    # Preferred when `flight_doms` is imported as a package.
    from .intraday_flight_phase import IntradayFlightClassifier  # type: ignore
    FLIGHT_CLASSIFIER_AVAILABLE = True
except Exception as e_rel:
    try:
        # Fallback for legacy "run as script" workflows.
        from intraday_flight_phase import IntradayFlightClassifier  # type: ignore
        FLIGHT_CLASSIFIER_AVAILABLE = True
    except Exception as e_abs:
        print("[WARN] Flight classifier disabled:", e_rel, "|", e_abs)
        IntradayFlightClassifier = None  # type: ignore[assignment]
        FLIGHT_CLASSIFIER_AVAILABLE = False

# Create Classifier Instance
flight_clf = None
if FLIGHT_CLASSIFIER_AVAILABLE and IntradayFlightClassifier is not None:
    flight_clf = IntradayFlightClassifier(horizon_secs=600)

def bs_all(s: float, k: float, t_years: float, r: float, sigma: float, opt_type: str = "call") -> Dict[str, float]:
    """Return Black–Scholes price and greeks for a European call/put.

    Note: theta returned is per-year; UI can scale to per-day if desired.
    """
    t = max(t_years, 0.0)
    if t == 0 or sigma <= 0:
        intrinsic = max(0.0, s - k) if opt_type == "call" else max(0.0, k - s)
        return {
            "price": float(intrinsic),
            "delta": 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "rho": 0.0,
        }
    d1 = (log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt(t))
    d2 = d1 - sigma * sqrt(t)
    if opt_type == "call":
        price = s * norm.cdf(d1) - k * exp(-r * t) * norm.cdf(d2)
        delta = norm.cdf(d1)
        rho = k * t * exp(-r * t) * norm.cdf(d2)
    else:
        price = k * exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)
        delta = -norm.cdf(-d1)
        rho = -k * t * exp(-r * t) * norm.cdf(-d2)
    gamma = norm.pdf(d1) / (s * sigma * sqrt(t))
    vega = s * norm.pdf(d1) * sqrt(t)
    theta = -(
        (s * norm.pdf(d1) * sigma) / (2 * sqrt(t))
        + (r * k * exp(-r * t) * (norm.cdf(d2) if opt_type == "call" else norm.cdf(-d2)))
    )
    return {
        "price": float(price),
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
        "vega": float(vega),
        "rho": float(rho),
    }


# Direct per-symbol Book Viewer pages
CBOE_BOOK_URL_TEMPLATE = "https://www.cboe.com/us/equities/market_statistics/book/{symbol}/"


def _to_float(value: str) -> Optional[float]:
    try:
        s = (value or "").replace(",", "").replace("$", "").strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def _to_int(value: str) -> Optional[int]:
    try:
        s = (value or "").replace(",", "").strip()
        if s == "":
            return None
        return int(float(s))  # sizes sometimes come as floats; coerce
    except Exception:
        return None


class TickerScraper:
    """
    Singleton-like scraper that maintains one Chrome session for the entire app.
    Thread-safe via lock.
    """

    def __init__(self, headless: bool = True, timeout: int = 30) -> None:
        self._lock = threading.Lock()
        self._driver = self._create_driver(headless)
        self._wait = WebDriverWait(self._driver, timeout)
        self._last_ticker: Optional[str] = None
        self._container = None
        self._navigate()
        atexit.register(self.shutdown)

    def _create_driver(self, headless: bool) -> webdriver.Chrome:
        def build_options(use_new_headless: bool) -> Options:
            o = Options()
            if headless:
                o.add_argument("--headless=new" if use_new_headless else "--headless")
            o.add_argument("--no-sandbox")
            o.add_argument("--disable-gpu")
            o.add_argument("--disable-dev-shm-usage")
            o.add_argument("--window-size=1600,1200")
            o.add_argument("--log-level=3")
            o.add_experimental_option("excludeSwitches", ["enable-automation"])
            o.add_experimental_option("useAutomationExtension", False)
            return o

        try:
            return webdriver.Chrome(options=build_options(use_new_headless=True))
        except Exception:
            return webdriver.Chrome(options=build_options(use_new_headless=False))

    def _navigate(self) -> None:
        # Open any Cboe page to accept cookies; symbol navigation happens in ensure_ticker
        self._driver.get("https://www.cboe.com/us/equities/market_statistics/")
        time.sleep(3)
        self._accept_cookies_if_present()
        try:
            self._driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 3);")
        except Exception:
            pass

    def _accept_cookies_if_present(self) -> None:
        xpaths = [
            "/html/body/div[2]/div[2]/div/div[1]/div/div[2]/div/button[3]",
            "//button[contains(., 'Accept')]",
            "//button[@mode='accept']",
        ]
        for xpath in xpaths:
            try:
                btn = WebDriverWait(self._driver, 3).until(
                    EC.element_to_be_clickable((By.XPATH, xpath))
                )
                self._driver.execute_script("arguments[0].click();", btn)
                time.sleep(0.5)
                return
            except Exception:
                continue

    def ensure_ticker(self, ticker: str) -> None:
        ticker = (ticker or "").strip().upper() or "SPY"
        with self._lock:
            if ticker == self._last_ticker and self._container is not None:
                return

            try:
                url = CBOE_BOOK_URL_TEMPLATE.format(symbol=ticker.lower())
                self._driver.get(url)
                self._container = self._driver
                self._wait.until(
                    lambda d: (
                        len(d.find_elements(By.CLASS_NAME, "book-viewer__bid-price")) > 0
                        or len(d.find_elements(By.XPATH, "//*[contains(text(), 'Top of Book')]")) > 0
                    )
                )
                self._last_ticker = ticker
            except Exception as e:
                print(f"Direct symbol navigation failed for {ticker}: {e}")
                self._last_ticker = ticker

    def get_depth_snapshot(self, max_levels: int = 15) -> Dict:
        with self._lock:
            try:
                assert self._container is not None

                bid_price_els = self._container.find_elements(By.CLASS_NAME, "book-viewer__bid-price")
                bid_size_els = self._container.find_elements(By.CLASS_NAME, "book-viewer__bid-shares")
                ask_price_els = self._container.find_elements(By.CLASS_NAME, "book-viewer__ask-price")
                ask_size_els = self._container.find_elements(By.CLASS_NAME, "book-viewer__ask-shares")
                trade_price_els = self._container.find_elements(By.CLASS_NAME, "book-viewer__trades-price")
                trade_size_els = self._container.find_elements(By.CLASS_NAME, "book-viewer__trades-shares")

                def collect(els: List) -> List[str]:
                    out: List[str] = []
                    for el in els[:max_levels]:
                        try:
                            out.append(el.text.strip())
                        except Exception:
                            out.append("")
                    return out

                raw_bids_p = collect(bid_price_els)
                raw_bids_s = collect(bid_size_els)
                raw_asks_p = collect(ask_price_els)
                raw_asks_s = collect(ask_size_els)

                bids: List[Tuple[Optional[float], Optional[int]]] = []
                asks: List[Tuple[Optional[float], Optional[int]]] = []

                for i in range(min(len(raw_bids_p), len(raw_bids_s), max_levels)):
                    bids.append((_to_float(raw_bids_p[i]), _to_int(raw_bids_s[i])))
                for i in range(min(len(raw_asks_p), len(raw_asks_s), max_levels)):
                    asks.append((_to_float(raw_asks_p[i]), _to_int(raw_asks_s[i])))

                last_price = None
                if trade_price_els:
                    try:
                        last_price = _to_float(trade_price_els[0].text)
                    except Exception:
                        last_price = None

                now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                return {
                    "timestamp": now,
                    "symbol": self._last_ticker or "",
                    "last_price": last_price,
                    "bids": bids,  # list of [price, size]
                    "asks": asks,  # list of [price, size]
                }              
                               
                
                
            except Exception as e:
                print(f"Depth scraping error: {e}")
                now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                return {
                    "timestamp": now,
                    "symbol": self._last_ticker or "",
                    "last_price": None,
                    "bids": [],
                    "asks": [],
                }

    def shutdown(self) -> None:
        with self._lock:
            if hasattr(self, "_driver") and self._driver:
                self._driver.quit()


# Global scraper (singleton)
_scraper: Optional[TickerScraper] = None
_scraper_lock = threading.Lock()


def get_global_scraper(headless: bool = True) -> TickerScraper:
    global _scraper
    if _scraper is None:
        with _scraper_lock:
            if _scraper is None:
                _scraper = TickerScraper(headless=headless)
    return _scraper


def create_app(default_ticker: str = "SPY", poll_interval_seconds: int = 1, headless: bool = True, replayer: Optional["CSVReplayer"] = None) -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health() -> Response:
        return Response("ok", status=200, mimetype="text/plain")

    @app.get("/api/depth")
    def api_depth() -> Response:
        ticker = request.args.get("ticker", default_ticker).strip().upper()
        levels = request.args.get("levels", "15")

        try:
            max_levels = max(1, min(50, int(levels)))
        except Exception:
            max_levels = 15

        try:
            # --- get basic depth snapshot (replayer OR live) ---
            if replayer is not None:
                data = replayer.get_depth_snapshot(max_levels=max_levels)
            else:
                scraper = get_global_scraper(headless=headless)
                scraper.ensure_ticker(ticker)
                data = scraper.get_depth_snapshot(max_levels=max_levels)
                # Retry once if depth is empty
                if not data.get("bids") and not data.get("asks"):
                    time.sleep(0.5)
                    scraper.ensure_ticker(ticker)
                    data = scraper.get_depth_snapshot(max_levels=max_levels)

            # --- compute flight phase (always) ---
            bids = data.get("bids") or []
            asks = data.get("asks") or []

            last_price = data.get("last_price")
            if last_price is None:
                # Fallback: compute a simple mid from top-of-book so the phase
                # classifier can run even when the "last trade" cell is blank.
                try:
                    best_bid = next((p for p, _ in bids if isinstance(p, (int, float))), None)
                    best_ask = next((p for p, _ in asks if isinstance(p, (int, float))), None)
                    if best_bid is not None and best_ask is not None:
                        last_price = (float(best_bid) + float(best_ask)) / 2.0
                        data["last_price"] = last_price
                except Exception:
                    pass

            # sum sizes; ignore None
            total_bid_size = sum((sz or 0.0) for _, sz in bids)
            total_ask_size = sum((sz or 0.0) for _, sz in asks)

            phase_snapshot = None
            if flight_clf is not None and last_price is not None:
                flight_clf.add_sample(
                    t=time.time(),
                    price=float(last_price),
                    bid_size=total_bid_size,
                    ask_size=total_ask_size,
                )
                phase_snapshot = flight_clf.snapshot()

            payload = {
                "ok": True,
                "data": data,
                "flight_phase": phase_snapshot,
            }
            return jsonify(payload)

        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    # NEW: Option pricing + greeks API using live spot
    @app.get("/api/option")
    def api_option() -> Response:
        """
        Compute live BS price + greeks using the most recent DOM last_price
        (or mid of best bid/ask if last missing).
        Query params:
          type=call|put, strike=645, entry=0.35, iv=0.16, dte=7, r=0.02,
          flip=640.67, put_wall=635, call_wall=645, regime=auto|positive|negative
          ticker=SPY
        """
        try:
            q = request.args
            opt_type = (q.get("type", "call")).lower()
            strike = float(q.get("strike", "645"))
            entry = float(q.get("entry", "0.35"))
            iv = float(q.get("iv", "0.16"))
            dte = float(q.get("dte", "7"))
            r = float(q.get("r", "0.02"))
            # defer flip/walls defaults until after we have spot
            raw_flip = q.get("flip", "").strip()
            raw_put = q.get("put_wall", "").strip()
            raw_call = q.get("call_wall", "").strip()
            regime = (q.get("regime", "auto")).lower()
            ticker = (q.get("ticker", "SPY")).upper()

            if replayer is not None:
                depth = replayer.get_depth_snapshot(max_levels=5)
            else:
                depth = None

            if depth is None:
              scraper = get_global_scraper(headless=headless)
              scraper.ensure_ticker(ticker)
              depth = scraper.get_depth_snapshot(max_levels=5)
            s = depth.get("last_price")
            if s is None:
                # fallback to simple mid from top of book
                try:
                    best_bid = depth["bids"][0][0]
                    best_ask = depth["asks"][0][0]
                    if best_bid and best_ask:
                        s = (best_bid + best_ask) / 2.0
                except Exception:
                    pass

            if s is None:
                return jsonify({"ok": False, "error": "no live price"}), 503

            # compute thermo defaults around spot if inputs were blank/auto
            def _auto_or_value(raw: str, val: float) -> float:
                try:
                    if raw and raw.lower() != "auto":
                        return float(raw)
                except Exception:
                    pass
                return float(val)

            if s is not None:
                flip = _auto_or_value(raw_flip, float(s) * 0.98)
                put_wall = _auto_or_value(raw_put, float(s) * 0.96)
                call_wall = _auto_or_value(raw_call, float(s) * 1.04)
            else:
                # fallback if no spot
                flip = _auto_or_value(raw_flip, 0.0)
                put_wall = _auto_or_value(raw_put, 0.0)
                call_wall = _auto_or_value(raw_call, 0.0)

            t_years = max(dte, 0.0) / 365.0
            res = bs_all(float(s), strike, t_years, r, iv, opt_type)
            pnl = res["price"] - entry

            # regime if 'auto'
            if regime == "auto":
                regime = "positive" if float(s) >= flip else "negative"

            # ---------------- Flight DOMS: flight rules + envelopes ----------------
            # NOTE: These are deliberately "minimal but high-impact" heuristics based
            # on currently-available inputs (DTE + BS greeks + flip/walls + spot).
            mode_raw = (q.get("mode", "AUTO") or "AUTO").strip().upper()
            if mode_raw not in ("AUTO", "VFR", "IFR", "IMC"):
                mode_raw = "AUTO"

            # Mode recommendation from DTE
            if dte >= 45:
                recommended_mode = "IMC"
            elif dte >= 30:
                recommended_mode = "IFR"
            elif dte <= 14:
                recommended_mode = "VFR"
            else:
                # "transition band" between VFR and IFR
                recommended_mode = "VFR"

            selected_mode = recommended_mode if mode_raw == "AUTO" else mode_raw

            # Gate constraints (contract class)
            gate_ok = True
            gate_reasons: List[str] = []
            if selected_mode == "VFR" and dte > 14:
                gate_ok = False
                gate_reasons.append("VFR requires DTE ≤ 14")
            if selected_mode == "IFR" and dte < 30:
                gate_ok = False
                gate_reasons.append("IFR requires DTE ≥ 30")
            if selected_mode == "IMC" and dte < 45:
                gate_ok = False
                gate_reasons.append("IMC requires DTE ≥ 45")

            # Weekend risk multiplier (timezone-agnostic heuristic)
            # Fri/Sat/Sun penalized to reflect gap/illiquidity risk.
            now_utc = datetime.utcnow()
            weekend_risk_multiplier = 2.5 if now_utc.weekday() in (4, 5, 6) else 1.0

            # Stall envelope: compare daily theta burn vs expected daily convexity gain.
            theta_per_day = abs(float(res["theta"])) / 365.0
            expected_move_1d = float(s) * float(iv) * sqrt(1.0 / 365.0)
            convexity_gain_1d = 0.5 * abs(float(res["gamma"])) * (expected_move_1d ** 2)
            denom = max(convexity_gain_1d, 1e-12)
            stall_ratio = (theta_per_day * weekend_risk_multiplier) / denom
            stall_margin_pct = (1.0 - stall_ratio) * 100.0

            # Lift envelope: Delta × Vega (use vega per 1 vol point for scale)
            delta_abs = abs(float(res["delta"]))
            vega_per_1vol = abs(float(res["vega"])) * 0.01
            lift_score = delta_abs * vega_per_1vol
            lift_ok = (delta_abs >= 0.15) and (vega_per_1vol >= 0.05)

            # Structural envelope (proxy): gamma near expiry vs "dead wing" far OTM
            gamma_abs = abs(float(res["gamma"]))
            structural_flutter_risk = (dte <= 7) and (gamma_abs >= 0.02)
            dead_wing_risk = (dte >= 30) and (delta_abs <= 0.10)
            structural_ok = not (structural_flutter_risk or dead_wing_risk)

            # Recovery altitude: distance to flip in trade direction (pts)
            if opt_type == "call":
                recovery_altitude_pts = float(s) - float(flip)
            else:
                recovery_altitude_pts = float(flip) - float(s)

            # Overnight safety: VFR never overnight; IFR/IMC require positive margins.
            overnight_safe = (
                selected_mode in ("IFR", "IMC")
                and gate_ok
                and (stall_margin_pct >= 0.0)
                and (recovery_altitude_pts > 0.0)
                and structural_ok
                and lift_ok
            )

            warnings: List[str] = []
            if not gate_ok:
                warnings.append("FLIGHT RULE GATE FAIL")
            if stall_margin_pct < 0:
                warnings.append("STRUCTURAL STALL RISK — NOT OVERNIGHT SAFE")
            if recovery_altitude_pts <= 0:
                warnings.append("RECOVERY ALTITUDE ≤ 0 — PROBE ONLY")
            if structural_flutter_risk:
                warnings.append("GAMMA FLUTTER RISK (near-expiry)")
            if dead_wing_risk:
                warnings.append("DEAD WING RISK (low delta far horizon)")
            if not lift_ok:
                warnings.append("LOW LIFT (delta×vega) — poor regime sensitivity")

            return jsonify({
                "ok": True,
                "data": {
                    "spot": float(s),
                    "flip": flip,
                    "put_wall": put_wall,
                    "call_wall": call_wall,
                    "regime": regime,
                    "option": {
                        "type": opt_type,
                        "strike": strike,
                        "entry": entry,
                        "iv": iv,
                        "dte": dte,
                        "r": r,
                        "price": res["price"],
                        "pnl": pnl,
                        "delta": res["delta"],
                        "gamma": res["gamma"],
                        "theta": res["theta"],
                        "vega": res["vega"],
                        "rho": res["rho"],
                    },
                    "flight_doms": {
                        "mode_selected": selected_mode,
                        "mode_recommended": recommended_mode,
                        "gate_ok": gate_ok,
                        "gate_reasons": gate_reasons,
                        "overnight_safe": overnight_safe,
                        "stall_margin_pct": stall_margin_pct,
                        "weekend_risk_multiplier": weekend_risk_multiplier,
                        "recovery_altitude_pts": recovery_altitude_pts,
                        "envelopes": {
                            "lift_ok": lift_ok,
                            "lift_score": lift_score,
                            "stall_ratio": stall_ratio,
                            "structural_ok": structural_ok,
                            "structural_flutter_risk": structural_flutter_risk,
                            "dead_wing_risk": dead_wing_risk,
                        },
                        "warnings": warnings,
                    },
                },
            })
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.get("/")
    def index() -> Response:
        html = """
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Live DOM Dashboard v3</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {{
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      margin: 0; padding: 0; background: #0b1221; color: #e6edf3;
    }}
    header {{
      padding: 12px 16px; background: #0e1730; border-bottom: 1px solid #1f2a44;
      display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
      position: relative; z-index: 10;
    }}
    h1 {{ margin: 0; font-size: 16px; font-weight: 600; }}
    .ticker-input {{ display: flex; align-items: center; gap: 8px; background: #0b1221; border: 1px solid #223154; border-radius: 8px; padding: 6px 8px; }}
    .ticker-input input, .ticker-input select {{ background: transparent; border: none; outline: none; color: #e6edf3; font-size: 14px; min-width: 60px; text-transform: uppercase; }}
    .button {{ background: #3b82f6; color: white; border: none; border-radius: 8px; padding: 6px 10px; cursor: pointer; font-weight: 600; }}
    .status {{ margin-left: auto; color: #9fb0d4; font-size: 13px; }}

    main {{ padding: 12px; position: relative; z-index: 1; }}
    /* Two-column cockpit layout to eliminate unused whitespace */
    .layout {{ display: grid; grid-template-columns: 1fr 360px; gap: 12px; align-items: start; }}
    .panel {{ background: #0e1730; border: 1px solid #1f2a44; border-radius: 8px; padding: 6px; }}
    .panel.tight {{ padding: 0; }}
    .center {{ display: grid; grid-template-rows: 300px 360px; gap: 12px; }}
    .right {{ display: grid; grid-template-rows: auto auto; gap: 12px; }}
    .panel-title {{ font-size: 12px; font-weight: 700; letter-spacing: 0.06em; color: #9fb0d4; margin: 2px 2px 8px; text-transform: uppercase; }}
    .kv {{ display: grid; grid-template-columns: 1fr auto; gap: 10px; padding: 6px 6px; border-top: 1px solid #1f2a44; }}
    .kv:first-of-type {{ border-top: none; }}
    .k {{ color: #9fb0d4; font-size: 12px; }}
    .v {{ font-weight: 700; font-size: 13px; }}
    .pill {{ padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 800; border: 1px solid #223154; }}
    .ok {{ color: #bbf7d0; background: rgba(34,197,94,0.10); border-color: rgba(34,197,94,0.35); }}
    .bad {{ color: #fecaca; background: rgba(239,68,68,0.10); border-color: rgba(239,68,68,0.35); }}
    .warnbox {{ margin-top: 8px; padding: 8px; border-radius: 8px; border: 1px dashed #334155; background: rgba(15,23,42,0.35); color: #e6edf3; font-size: 12px; line-height: 1.35; }}
    .stack {{ position: relative; width: 100%; height: 100%; }}
    .stack canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; image-rendering: pixelated; }
    #heatmap { z-index: 1; }
    #bubbles { z-index: 2; pointer-events: none; }
    #priceChartCanvas { z-index: 3; pointer-events: none; background: transparent; }
    canvas {{ display: block; width: 100%; height: 100%; image-rendering: pixelated; }}
    .row2 { display: grid; grid-template-columns: 1fr; gap: 10px; align-items: stretch; position: relative; z-index: 1; min-height: 360px; }
  </style>
  </head>
  <body>
    <header>
      <h1>Live DOM Dashboard v3</h1>
      <div class=\"ticker-input\"> 
        <span>Ticker</span>
        <input id=\"ticker\" value=\"__DEFAULT_TICKER__\" maxlength=\"10\" />
        <button id=\"apply\" class=\"button\">Apply</button>
      </div>

      <!-- NEW: regime & walls -->
      <div class=\"ticker-input\"> 
        <span>Regime</span>
        <select id=\"regime\"> 
          <option value=\"auto\" selected>Auto</option>
          <option value=\"positive\">Positive</option>
          <option value=\"negative\">Negative</option>
        </select>
        <span>Flip</span><input id=\"flip\" placeholder=\"auto\" size=\"6\" />
        <span>Put</span><input id=\"putwall\" placeholder=\"auto\" size=\"5\" />
        <span>Call</span><input id=\"callwall\" placeholder=\"auto\" size=\"5\" />
      </div>

      <!-- NEW: option tracker -->
      <div class=\"ticker-input\"> 
        <span>Opt</span>
        <select id=\"otype\"><option>call</option><option>put</option></select>
        <span>K</span><input id=\"strike\" placeholder=\"auto\" size=\"5\" />
        <span>Entry</span><input id=\"entry\" value=\"0.35\" size=\"5\" />
        <span>IV</span><input id=\"iv\" value=\"0.16\" size=\"4\" />
        <span>DTE</span><input id=\"dte\" value=\"7\" size=\"3\" />
        <span>Mode</span>
        <select id=\"flightmode\">
          <option value=\"AUTO\" selected>AUTO</option>
          <option value=\"VFR\">VFR</option>
          <option value=\"IFR\">IFR</option>
          <option value=\"IMC\">IMC</option>
        </select>
      </div>

      <div class=\"status\" id=\"status\">Idle</div>
    </header>

    <main>
      <div class=\"layout\">
        <div class=\"center\">
          <div class=\"panel\"><canvas id=\"thermo\" height=\"220\"></canvas></div>
          <div class=\"row2\"> 
          <div class=\"panel tight\">
            <div class=\"stack\">
              <canvas id=\"heatmap\" width=\"900\" height=\"360\"></canvas>
                <canvas id=\"bubbles\" width=\"900\" height=\"360\"></canvas>
              <canvas id=\"priceChartCanvas\" width=\"900\" height=\"360\"></canvas>
            </div>
          </div>
        </div>
        </div>
        <div class=\"right\">
          <div class=\"panel\" id=\"cockpit\">
            <div class=\"panel-title\">Flight Rules Gate</div>
            <div class=\"kv\"><div class=\"k\">PHASE</div><div class=\"v\" id=\"fd_phase\">N/A</div></div>
            <div class=\"kv\"><div class=\"k\">FLIGHT_RULE</div><div class=\"v\" id=\"fd_rule\">—</div></div>
            <div class=\"kv\"><div class=\"k\">OVERNIGHT_SAFE</div><div class=\"v\"><span class=\"pill\" id=\"fd_overnight\">—</span></div></div>
            <div class=\"kv\"><div class=\"k\">STALL_MARGIN</div><div class=\"v\" id=\"fd_stall\">—</div></div>
            <div class=\"kv\"><div class=\"k\">WEEKEND_RISK_MULT</div><div class=\"v\" id=\"fd_weekend\">—</div></div>
            <div class=\"kv\"><div class=\"k\">RECOVERY_ALTITUDE</div><div class=\"v\" id=\"fd_recovery\">—</div></div>
            <div class=\"warnbox\" id=\"fd_warnings\">—</div>
          </div>
          <div class=\"panel\" id=\"envelopes\">
            <div class=\"panel-title\">Envelope Flags</div>
            <div class=\"kv\"><div class=\"k\">LIFT (Δ×V)</div><div class=\"v\"><span class=\"pill\" id=\"fd_lift\">—</span></div></div>
            <div class=\"kv\"><div class=\"k\">STRUCTURAL</div><div class=\"v\"><span class=\"pill\" id=\"fd_struct\">—</span></div></div>
            <div class=\"kv\"><div class=\"k\">GATE</div><div class=\"v\"><span class=\"pill\" id=\"fd_gate\">—</span></div></div>
            <div class=\"warnbox\" style=\"opacity:0.9\">
              VFR: ≤14 DTE (≤48h). IFR: ≥30 DTE. IMC: ≥45 DTE. If NOT overnight-safe, treat as probe only.
            </div>
          </div>
        </div>
      </div>
    </main>

    <script>
      const statusEl = document.getElementById('status');
      const tickerEl = document.getElementById('ticker');
      const applyEl = document.getElementById('apply');
      const heatmapCanvas = document.getElementById('heatmap');
      const heatmapCtx = heatmapCanvas.getContext('2d');
      const bubblesCanvas = document.getElementById('bubbles');
      const bubblesCtx = bubblesCanvas.getContext('2d');
      // Sidebar bar canvases removed (keep axis bars only)
      const priceChartCanvas = document.getElementById('priceChartCanvas');
      const priceChartCtx = priceChartCanvas.getContext('2d');
      // Custom price trace aligned to heatmap/bubbles (no Chart.js)
      function drawPriceTrace(last) {
        const H = priceChartCanvas.height|0;
        const usableW = priceChartCanvas.width - AXIS_RIGHT_PAD; // avoid drawing in axis band
        // shift left by 1 px within usable area
        const img = priceChartCtx.getImageData(1, 0, usableW - 1, H);
        priceChartCtx.putImageData(img, 0, 0);
        // clear the last usable column
        priceChartCtx.clearRect(usableW - 1, 0, 1, H);
        const y = yForPrice(last, H);
        if (y !== null) {
          priceChartCtx.fillStyle = '#00e5ff';
          priceChartCtx.globalAlpha = 1.0;
          priceChartCtx.fillRect(usableW - 1, Math.max(0, y - 1), 1, 3);
        }
      }

      let currentTicker = '__DEFAULT_TICKER__';
      let minPrice = null, maxPrice = null;
      let maxSeenSize = 1;
      let prevPriceY = null; // reserved (not used with custom trace)
      let overlayX = 0;      // reserved

      // Accumulators for vertical area charts (time-decayed)
      let accumBids = new Float32Array(heatmapCanvas.height|0);
      let accumAsks = new Float32Array(heatmapCanvas.height|0);
      let accumMaxBid = 1.0, accumMaxAsk = 1.0;
      const ACCUM_DECAY = 0.985; // exponential decay per poll tick

      // Shared layout constants
      const AXIS_RIGHT_PAD = 68; // must match axis band width used in drawHeatmapColumn

      function syncSidebarHeights() {
        // Match sidebar canvas heights to the main heatmap's rendered height for perfect Y alignment
        // sidebars removed; nothing to sync
      }
      window.addEventListener('resize', () => { syncSidebarHeights(); });

      function clearOverlayAxis() {
        // Prevent overlays (bubbles/price line) from dimming the axis/labels
        const bandW = AXIS_RIGHT_PAD;
        bubblesCtx.clearRect(bubblesCanvas.width - bandW, 0, bandW, bubblesCanvas.height);
        priceChartCtx.clearRect(priceChartCanvas.width - bandW, 0, bandW, priceChartCanvas.height);
      }

      // Particle-based drifting bubbles
      const particles = [];
      const MAX_PARTICLES = 800;
      function spawnParticles(levels, isBid) {
        const now = performance.now();
        const H = bubblesCanvas.height|0;
        for (const [p, s] of levels) {
          const y = yForPrice(p, H);
          if (y === null || !s) continue;
          const r = bubbleRadius(s);
          const vx = -(2.4 + Math.random() * 1.6); // faster drift to cross the full width
          const vy = (Math.random() - 0.5) * 0.3;
          particles.push({
            x: bubblesCanvas.width - 2 - r,
            y,
            r,
            vx, vy,
            colorA: isBid ? 'rgba(56,189,108,0.85)' : 'rgba(239,68,68,0.85)',
            colorB: isBid ? 'rgba(34,197,94,0.10)' : 'rgba(239,68,68,0.10)',
            splitAt: now + 900 + Math.random()*1200,
            isBid,
            born: now,
          });
        }
        while (particles.length > MAX_PARTICLES) particles.shift();
      }

      // NEW: regime & walls and option inputs
      const regimeEl = document.getElementById('regime');
      const flipEl = document.getElementById('flip');
      const putEl = document.getElementById('putwall');
      const callEl = document.getElementById('callwall');

      const otypeEl = document.getElementById('otype');
      const strikeEl = document.getElementById('strike');
      const entryEl = document.getElementById('entry');
      const ivEl = document.getElementById('iv');
      const dteEl = document.getElementById('dte');
      const flightModeEl = document.getElementById('flightmode');

      // Flight DOMS cockpit panel elements
      const fdPhaseEl = document.getElementById('fd_phase');
      const fdRuleEl = document.getElementById('fd_rule');
      const fdOvernightEl = document.getElementById('fd_overnight');
      const fdStallEl = document.getElementById('fd_stall');
      const fdWeekendEl = document.getElementById('fd_weekend');
      const fdRecoveryEl = document.getElementById('fd_recovery');
      const fdWarningsEl = document.getElementById('fd_warnings');
      const fdLiftEl = document.getElementById('fd_lift');
      const fdStructEl = document.getElementById('fd_struct');
      const fdGateEl = document.getElementById('fd_gate');

      function setPill(el, isOk, text) {
        if (!el) return;
        el.textContent = text;
        el.classList.remove('ok', 'bad');
        el.classList.add(isOk ? 'ok' : 'bad');
      }

      function updateFlightDomsPanel(optData, flightPhase) {
        if (!optData) return;
        const fd = optData.flight_doms || null;
        if (!fd) return;

        // Phase (from /api/depth)
        const ph = (flightPhase && typeof flightPhase.phase === 'string') ? flightPhase.phase : 'N/A';
        if (fdPhaseEl) fdPhaseEl.textContent = ph;

        // Core labels
        if (fdRuleEl) fdRuleEl.textContent = `${fd.mode_selected || '—'} (rec: ${fd.mode_recommended || '—'})`;
        setPill(fdOvernightEl, !!fd.overnight_safe, (fd.overnight_safe ? 'YES' : 'NO'));

        const stall = (typeof fd.stall_margin_pct === 'number') ? fd.stall_margin_pct : null;
        if (fdStallEl) fdStallEl.textContent = (stall === null) ? '—' : `${stall.toFixed(1)}%`;

        const wmult = (typeof fd.weekend_risk_multiplier === 'number') ? fd.weekend_risk_multiplier : null;
        if (fdWeekendEl) fdWeekendEl.textContent = (wmult === null) ? '—' : `${wmult.toFixed(2)}x`;

        const rec = (typeof fd.recovery_altitude_pts === 'number') ? fd.recovery_altitude_pts : null;
        if (fdRecoveryEl) fdRecoveryEl.textContent = (rec === null) ? '—' : `${rec.toFixed(2)} pts`;

        // Envelopes
        const env = fd.envelopes || {};
        setPill(fdGateEl, !!fd.gate_ok, fd.gate_ok ? 'PASS' : 'FAIL');
        setPill(fdLiftEl, !!env.lift_ok, env.lift_ok ? 'OK' : 'LOW');
        setPill(fdStructEl, !!env.structural_ok, env.structural_ok ? 'OK' : 'RISK');

        // Warnings
        const warnings = Array.isArray(fd.warnings) ? fd.warnings : [];
        if (fdWarningsEl) {
          fdWarningsEl.textContent = warnings.length ? warnings.join(' • ') : 'No active warnings.';
        }
      }

      const thermoCanvas = document.getElementById('thermo');
      const thermoCtx = thermoCanvas.getContext('2d');

      applyEl.addEventListener('click', () => {
        const next = (tickerEl.value || '').trim().toUpperCase();
        if (!next) return;
        currentTicker = next;
        minPrice = null; maxPrice = null; maxSeenSize = 1; prevPriceY = null;
        heatmapCtx.clearRect(0, 0, heatmapCanvas.width, heatmapCanvas.height);
        bubblesCtx.clearRect(0, 0, bubblesCanvas.width, bubblesCanvas.height);
        // reset price trace overlay (no Chart.js)
        priceChartCtx.clearRect(0, 0, priceChartCanvas.width, priceChartCanvas.height);
        thermoCtx.clearRect(0, 0, thermoCanvas.width, thermoCanvas.height);
        // reset accumulators
        accumBids = new Float32Array(heatmapCanvas.height|0);
        accumAsks = new Float32Array(heatmapCanvas.height|0);
        accumMaxBid = 1.0; accumMaxAsk = 1.0;
        syncSidebarHeights();
      });

      function updatePriceRange(last, bids, asks) {
        const prices = [];
        if (typeof last === 'number') prices.push(last);
        bids.forEach(([p,_]) => { if (typeof p === 'number') prices.push(p); });
        asks.forEach(([p,_]) => { if (typeof p === 'number') prices.push(p); });
        if (!prices.length) return;
        let pmin = Math.min(...prices), pmax = Math.max(...prices);
        if (!isFinite(pmin) || !isFinite(pmax)) return;
        if (pmax - pmin < 0.01) pmax = pmin + 0.5;
        const mid = (pmin + pmax) / 2;
        const pad = Math.max(0.0025 * mid, 0.5); // ~0.25% or 0.5 pts
        const targetMin = pmin - pad;
        const targetMax = pmax + pad;
        if (minPrice === null || maxPrice === null) {
          minPrice = targetMin;
          maxPrice = targetMax;
        } else {
          const alpha = 0.25; // smoothing factor for stability
          minPrice = minPrice * (1 - alpha) + targetMin * alpha;
          maxPrice = maxPrice * (1 - alpha) + targetMax * alpha;
        }
        if (maxPrice - minPrice < 0.1) { maxPrice = minPrice + 0.5; }
      }

      function yForPrice(price, targetHeight) {
        if (minPrice === null || maxPrice === null) return null;
        if (price === null || typeof price !== 'number') return null;
        const H = Math.max(1, targetHeight || heatmapCanvas.height);
        const r = (price - minPrice) / Math.max(1e-9, (maxPrice - minPrice));
        const y = Math.round((1 - r) * (H - 1));
        return Math.max(0, Math.min(H - 1, y));
      }

      function intensity(size) {
        if (!size || size <= 0) return 0;
        maxSeenSize = Math.max(maxSeenSize, size);
        const v = Math.log(size + 1) / Math.log(maxSeenSize + 1);
        return Math.max(0.05, Math.min(1, v));
      }

      function bubbleRadius(size) {
        if (!size || size <= 0) return 0;
        // map size to 3..14 px using log scaling relative to maxSeenSize
        maxSeenSize = Math.max(maxSeenSize, size);
        const f = Math.log(size + 1) / Math.log(maxSeenSize + 1);
        return Math.max(2, Math.round(3 + 12 * f));
      }

      function drawHeatmapColumn(bids, asks, last) {
        // shift only the plot area (exclude the right axis band) by 1 px left
        const plotW = Math.max(1, heatmapCanvas.width - AXIS_RIGHT_PAD);
        const H = heatmapCanvas.height;
        try {
          const imageData = heatmapCtx.getImageData(1, 0, plotW - 1, H);
        heatmapCtx.putImageData(imageData, 0, 0);
          // clear last column within the plot area
          heatmapCtx.clearRect(plotW - 1, 0, 1, H);
        } catch (e) {
          // Fallback: if getImageData fails (e.g., resizing), clear plot area
          heatmapCtx.clearRect(0, 0, plotW, H);
        }

        const x = plotW - 1;
        const drawLevels = (levels, isBid) => {
          for (const [p, s] of levels) {
            const y = yForPrice(p, heatmapCanvas.height);
            if (y === null) continue;
            const a = intensity(s);
            // Use neutral cyan tones on the heatmap so left-side doesn't look like bid/ask bars
            const base = isBid ? [80, 180, 255] : [110, 200, 255];
            heatmapCtx.fillStyle = `rgba(${base[0]},${base[1]},${base[2]},${Math.min(0.9, a*0.85)})`;
            heatmapCtx.fillRect(x, y - 1, 1, 3);
          }
        };
        drawLevels(bids, true);
        drawLevels(asks, false);

        // Update custom price trace (aligned to heatmap, restricted from axis band)
        if (typeof last === 'number') {
          drawPriceTrace(last);
        }

        // Draw Y-axis labels and grid on heatmap (right-side band) and overlay integrated bars
        if (minPrice !== null && maxPrice !== null) {
          const ticks = 8;
          const rightPad = AXIS_RIGHT_PAD; // keep in sync with overlays
          const axisX0 = heatmapCanvas.width - rightPad;
          // background axis band
          heatmapCtx.fillStyle = '#0b1221';
          heatmapCtx.fillRect(axisX0, 0, rightPad, heatmapCanvas.height);
          // grid + labels (brighter for readability)
          heatmapCtx.save();
          heatmapCtx.fillStyle = '#ffffff';
          heatmapCtx.globalAlpha = 1.0; // ensure labels are not faded
          heatmapCtx.shadowColor = 'rgba(0,0,0,0.65)';
          heatmapCtx.shadowBlur = 2;
          heatmapCtx.shadowOffsetX = 0;
          heatmapCtx.shadowOffsetY = 0;
          heatmapCtx.font = '12px system-ui';
          heatmapCtx.textBaseline = 'middle';
          heatmapCtx.textAlign = 'right';
          heatmapCtx.strokeStyle = 'rgba(255,255,255,0.08)';
          heatmapCtx.lineWidth = 1;
          for (let i = 0; i < ticks; i++) {
            const t = i / (ticks - 1);
            const price = minPrice + t * (maxPrice - minPrice);
            const y = Math.round((1 - t) * (heatmapCanvas.height - 1));
            const label = price.toFixed(2);
            // grid line (stop before axis band)
            heatmapCtx.beginPath();
            heatmapCtx.moveTo(0, y);
            heatmapCtx.lineTo(axisX0 - 4, y);
            heatmapCtx.stroke();
            // label
            heatmapCtx.fillText(label, heatmapCanvas.width - 6, y);
          }
          heatmapCtx.restore();

          // integrated bid/ask bars overlayed on price axis
          const H = heatmapCanvas.height|0;
          const barWidth = rightPad - 8; // larger bars for clarity
          const axisLeft = axisX0 + 2;    // start a bit inside the band
          const mid = axisLeft + Math.floor(barWidth * 0.5);
          const maxBid = Math.max(1e-6, accumMaxBid);
          const maxAsk = Math.max(1e-6, accumMaxAsk);
          // ensure bars render above any prior alpha changes
          const prevAlpha = heatmapCtx.globalAlpha; heatmapCtx.globalAlpha = 1.0;
          for (let y = 0; y < H; y++) {
            const fb = Math.min(1, accumBids[y] / maxBid);
            const fa = Math.min(1, accumAsks[y] / maxAsk);
            const wb = Math.round((barWidth * 0.5 - 2) * fb);
            const wa = Math.round((barWidth * 0.5 - 2) * fa);
            // dim base background for contrast exactly under labels
            heatmapCtx.fillStyle = 'rgba(255,255,255,0.06)';
            heatmapCtx.fillRect(axisLeft, y, barWidth, 1);
            // bars
            heatmapCtx.fillStyle = 'rgba(16,185,129,0.98)';
            if (wb > 0) heatmapCtx.fillRect(mid - 3 - wb, y, wb, 1);
            heatmapCtx.fillStyle = 'rgba(244,63,94,0.98)';
            if (wa > 0) heatmapCtx.fillRect(mid + 3, y, wa, 1);
          }
          heatmapCtx.globalAlpha = prevAlpha;

          // headers for clarity
          heatmapCtx.save();
          heatmapCtx.fillStyle = '#ffffff';
          heatmapCtx.globalAlpha = 1.0;
          heatmapCtx.font = '11px system-ui';
          heatmapCtx.textAlign = 'center';
          heatmapCtx.textBaseline = 'top';
          heatmapCtx.fillText('BID', axisLeft + Math.floor(barWidth*0.25) - 4, 2);
          heatmapCtx.fillText('ASK', axisLeft + Math.floor(barWidth*0.75) + 4, 2);
          heatmapCtx.restore();
        }
      }

      function updateAccumulators(bids, asks) {
        // apply decay
        for (let i = 0; i < accumBids.length; i++) { accumBids[i] *= ACCUM_DECAY; }
        for (let i = 0; i < accumAsks.length; i++) { accumAsks[i] *= ACCUM_DECAY; }
        // add current sizes to corresponding price bins
        const H = heatmapCanvas.height|0;
        for (const [p, s] of bids) {
          const y = yForPrice(p, H); if (y === null) continue;
          accumBids[y] += (s||0);
        }
        for (const [p, s] of asks) {
          const y = yForPrice(p, H); if (y === null) continue;
          accumAsks[y] += (s||0);
        }
        // track maxima for scaling
        for (let i = 0; i < H; i++) {
          if (accumBids[i] > accumMaxBid) accumMaxBid = accumBids[i];
          if (accumAsks[i] > accumMaxAsk) accumMaxAsk = accumAsks[i];
        }
      }

      function drawAccumBars(canvas, ctx, isBid, lastPrice) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const H = canvas.height|0;
        const W = canvas.width|0;
        const arr = isBid ? accumBids : accumAsks;
        const maxVal = Math.max(1e-6, isBid ? accumMaxBid : accumMaxAsk);
        ctx.fillStyle = isBid ? '#22c55e' : '#ef4444';
        // vertical area profile aligned to main chart's price scale
        for (let y = 0; y < H; y++) {
          const frac = arr[y] / maxVal;
          if (frac <= 0) continue;
          const w = Math.max(1, Math.round(frac * (W - 8)));
          const x = isBid ? (W - w) : 0;
          ctx.globalAlpha = 0.9;
          ctx.fillRect(x, y, w, 1);
        }
        ctx.globalAlpha = 1.0;
        // Overlay current price line on sidebars
        const yNow = yForPrice(lastPrice, canvas.height);
        if (yNow !== null) {
          ctx.strokeStyle = 'rgba(255,255,255,0.9)';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(0, yNow);
          ctx.lineTo(canvas.width, yNow);
          ctx.stroke();
        }
        // Draw axis labels on sidebars for sanity (match heatmap font/color)
        ctx.fillStyle = '#ffffff';
        ctx.font = '11px system-ui';
        ctx.textBaseline = 'top';
      }

      function drawBubbles(bids, asks) {
        // subtle trail fade; lower alpha so trails persist longer across screen
        bubblesCtx.fillStyle = 'rgba(0,0,0,0.02)';
        bubblesCtx.fillRect(0, 0, bubblesCanvas.width, bubblesCanvas.height);

        // spawn new particles from current book levels
        spawnParticles(bids, true);
        spawnParticles(asks, false);

        // integrate particle motion and draw
        const now = performance.now();
        for (let i = particles.length - 1; i >= 0; i--) {
          const pt = particles[i];
          pt.x += pt.vx; pt.y += pt.vy;
          // single split to simulate mitosis
          if (now > pt.splitAt && pt.r > 3) {
            pt.splitAt = Infinity;
            const child = { ...pt, r: Math.max(2, Math.round(pt.r * 0.7)), vy: -pt.vy, x: pt.x + 2 };
            pt.r = Math.max(2, Math.round(pt.r * 0.7));
            particles.push(child);
          }
          if (pt.x + pt.r < 0) { particles.splice(i,1); continue; }
          // keep particles within the drawable area (exclude axis band width)
          if (pt.x > bubblesCanvas.width - AXIS_RIGHT_PAD - 1) pt.x = bubblesCanvas.width - AXIS_RIGHT_PAD - 1;
          // subtle pulsation to enhance animation
          const pulse = 0.85 + 0.15*Math.sin(now/600 + i*0.3);
          const pr = Math.max(2, pt.r * pulse);
          // colored glow halo (no white)
          bubblesCtx.save();
          bubblesCtx.globalCompositeOperation = 'source-over';
          const haloColor = pt.isBid ? 'rgba(56,189,108,0.12)' : 'rgba(239,68,68,0.12)';
          const halo = bubblesCtx.createRadialGradient(pt.x, pt.y, pr*0.9, pt.x, pt.y, pr*1.8);
          halo.addColorStop(0, haloColor);
          halo.addColorStop(1, 'rgba(0,0,0,0)');
          bubblesCtx.fillStyle = halo;
          bubblesCtx.beginPath();
          bubblesCtx.arc(pt.x, pt.y, pr*1.8, 0, Math.PI * 2);
          bubblesCtx.fill();

          // luminous core
          const grad = bubblesCtx.createRadialGradient(pt.x - pr*0.35, pt.y - pr*0.35, pr*0.18, pt.x, pt.y, pr);
          grad.addColorStop(0, pt.colorA);
          grad.addColorStop(1, pt.colorB);
          bubblesCtx.fillStyle = grad;
          bubblesCtx.beginPath();
          bubblesCtx.arc(pt.x, pt.y, pr, 0, Math.PI * 2);
          bubblesCtx.fill();
          bubblesCtx.restore();
        }
      }


      function drawThermo(spot, flip, putw, callw, regime, opt) {
        const W = thermoCanvas.width, H = thermoCanvas.height;
        thermoCtx.clearRect(0,0,W,H);

        // lighter background gradient for readability
        const grad = thermoCtx.createLinearGradient(0,0,W,0);
        if (regime === 'positive') { grad.addColorStop(0,'#d9ecff'); grad.addColorStop(1,'#c8f3ea'); }
        else { grad.addColorStop(0,'#ffe0e6'); grad.addColorStop(1,'#ffd9cc'); }
        thermoCtx.fillStyle = grad;
        thermoCtx.fillRect(0,0,W,H);

        // scale x to price window currently used by heatmap (clamped to edges)
        function xForPrice(p) {
          if (minPrice===null || maxPrice===null) return null;
          const r = (p - minPrice) / Math.max(1e-9, (maxPrice - minPrice));
          const rc = Math.max(0, Math.min(1, r));
          return Math.round(rc * (W - 1));
        }

        // walls + flip
        const xFlip = xForPrice(flip), xPut = xForPrice(putw), xCall = xForPrice(callw), xSpot = xForPrice(spot);
        thermoCtx.strokeStyle = '#ffffff55'; thermoCtx.setLineDash([6,4]);

        if (xFlip!==null) { thermoCtx.beginPath(); thermoCtx.moveTo(xFlip,0); thermoCtx.lineTo(xFlip,H); thermoCtx.stroke();
          thermoCtx.fillStyle='#e6edf3'; thermoCtx.fillText('Gamma Flip', Math.max(4,xFlip-35), 14); }

        thermoCtx.setLineDash([]);
        // more visible put/call walls: thicker core with colored edge glow
        if (xPut!==null){
          // edge glow
          thermoCtx.fillStyle='rgba(34,197,94,0.20)';
          thermoCtx.fillRect(xPut-3,0,6,H);
          // core
          thermoCtx.fillStyle='#16a34a';
          thermoCtx.fillRect(xPut-2,0,4,H);
          // cap + label
          thermoCtx.fillStyle='#0b1221';
          thermoCtx.fillText('PUT', Math.max(4,xPut-20), 16);
        }
        if (xCall!==null){
          thermoCtx.fillStyle='rgba(244,63,94,0.20)';
          thermoCtx.fillRect(xCall-3,0,6,H);
          thermoCtx.fillStyle='#f43f5e';
          thermoCtx.fillRect(xCall-2,0,4,H);
          thermoCtx.fillStyle='#0b1221';
          thermoCtx.fillText('CALL', Math.max(4,xCall-24), 32);
        }

        // spot
        if (xSpot!==null){ thermoCtx.fillStyle='#0b1221'; thermoCtx.fillRect(xSpot-1,0,2,H); }

        // option box
        const label = `${opt.type.toUpperCase()} K=${opt.strike}  Px=${opt.price.toFixed(2)}  P&L ${(opt.pnl>=0?'+':'')}${opt.pnl.toFixed(2)}  Δ ${opt.delta.toFixed(2)}  Γ ${opt.gamma.toFixed(4)}  Θ/day ${(opt.theta/365).toFixed(3)}`;
        // readable text on light background
        thermoCtx.fillStyle='#0b1221';
        thermoCtx.font = '12px system-ui';
        thermoCtx.fillText(label, 8, 32);
      }

      async function priceAndOption(ticker) {
        // dynamic defaults derived from current spot when inputs are blank
        const resDepth = await fetch(`/api/depth?ticker=${encodeURIComponent(ticker)}&levels=5`, { cache: 'no-store' });
        const depthBody = await resDepth.json();
        let spot = null;
        if (depthBody && depthBody.ok) spot = depthBody.data.last_price;
        // derive -10%/+10% around spot if provided
        // align defaults around current spot if provided
        const flip = Number(flipEl.value) || (spot ? Number((spot * 0.98).toFixed(2)) : 0); // closer to spot
        const putw = Number(putEl.value) || (spot ? Number((spot * 0.96).toFixed(2)) : 0);
        const callw = Number(callEl.value) || (spot ? Number((spot * 1.04).toFixed(2)) : 0);
        const type = otypeEl.value;
        const strike = Number(strikeEl.value) || (spot ? Number(spot.toFixed(2)) : 0);
        const entry = Number(entryEl.value) || 0;
        const iv = Number(ivEl.value) || 0.2;
        const dte = Number(dteEl.value) || 7;

        const params = new URLSearchParams({
          ticker,
          type,
          strike: String(strike), entry: String(entry),
          iv: String(iv), dte: String(dte), r: '0.02',
          mode: flightModeEl ? String(flightModeEl.value) : 'AUTO',
          regime: regimeEl.value,
          flip: String(flip), put_wall: String(putw), call_wall: String(callw)
        });
        const res = await fetch(`/api/option?${params}`, { cache: 'no-store' });
        const body = await res.json();
        if (!body.ok) throw new Error(body.error||'option api error');
        return body.data;
      }

      async function poll() {
        try {
          // Show transient status while we fetch
          statusEl.textContent = 'Updating…';

          const res = await fetch(
            `/api/depth?ticker=${encodeURIComponent(currentTicker)}&levels=15`,
            { cache: 'no-store' }
          );
          const body = await res.json();
          if (!body.ok) throw new Error(body.error || 'Fetch failed');

          // Depth payload from Flask
          const d = body.data || {};
          const last = (typeof d.last_price === 'number') ? d.last_price : null;
          const bids = (d.bids || []).filter(([p, _]) => typeof p === 'number');
          const asks = (d.asks || []).filter(([p, _]) => typeof p === 'number');

          // Core DOM visuals
          updatePriceRange(last, bids, asks);
          drawHeatmapColumn(bids, asks, last);
          drawBubbles(bids, asks);
          clearOverlayAxis();
          updateAccumulators(bids, asks);

          // Live option calc + thermo
          try {
            const opt = await priceAndOption(currentTicker);
            if (opt) {
              drawThermo(
                opt.spot,
                opt.flip,
                opt.put_wall,
                opt.call_wall,
                opt.regime,
                opt.option
              );
              updateFlightDomsPanel(opt, body.flight_phase);
            }
          } catch (e) {
            console.error('Thermo fetch error', e);
          }
           // --- NEW: show flight phase from backend safely ---
          let phase = 'N/A';
          if (body.flight_phase && typeof body.flight_phase.phase === 'string') {
            phase = body.flight_phase.phase;
          }
          if (fdPhaseEl) fdPhaseEl.textContent = phase;
          statusEl.innerHTML = `Live · Phase: <strong>${phase}</strong>`;
         } catch (err) {
          statusEl.innerHTML =
            `<span style="color:#ef4444">${String(err)}</span>`;
          console.error('Poll error', err);
        }
      }
      // Kick off polling on load
      const POLL_INTERVAL = Number(__POLL__) || 1.0;  // backend will inject a number
      setInterval(poll, POLL_INTERVAL * 1000);
      syncSidebarHeights();
      poll();      
      
      
    </script>
  </body>
  </html>
        """
        html = html.replace("__DEFAULT_TICKER__", default_ticker)
        html = html.replace("__POLL__", str(poll_interval_seconds))
        return Response(html, mimetype="text/html")

    @app.get("/favicon.ico")
    def favicon() -> Response:
        # Tiny empty favicon to suppress 404
        return Response(b"", mimetype="image/x-icon", status=200)

    return app


class CSVReplayer:
    """Simple CSV replayer for top-of-book DOM CSV exported by DOM_analyzer_v7.

    Exposes get_depth_snapshot() with the same structure as TickerScraper.
    Iterates one row per call; when reaching the end, it loops back to start.
    """

    def __init__(self, csv_path: str) -> None:
        self._csv_path = csv_path
        self._rows: List[Dict[str, str]] = []
        self._idx: int = 0
        self._symbol: str = ""
        self._load()

    def _load(self) -> None:
        try:
            with open(self._csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self._rows = [row for row in reader]
            if self._rows:
                self._symbol = (self._rows[0].get("symbol") or "").upper()
        except Exception as e:
            print(f"Failed to load CSV replay file {self._csv_path}: {e}")
            self._rows = []

    def _to_float(self, s: Optional[str]) -> Optional[float]:
        try:
            if s is None:
                return None
            s = s.replace(",", "").strip()
            if s == "":
                return None
            return float(s)
        except Exception:
            return None

    def _to_int(self, s: Optional[str]) -> Optional[int]:
        try:
            if s is None:
                return None
            s = s.replace(",", "").strip()
            if s == "":
                return None
            return int(float(s))
        except Exception:
            return None

    def get_depth_snapshot(self, max_levels: int = 15) -> Dict:
        if not self._rows:
            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            return {"timestamp": now, "symbol": self._symbol, "last_price": None, "bids": [], "asks": []}

        row = self._rows[self._idx]
        # step index for next call; loop
        self._idx = (self._idx + 1) % len(self._rows)

        bid_p = self._to_float(row.get("bid_price"))
        bid_s = self._to_int(row.get("bid_size"))
        ask_p = self._to_float(row.get("ask_price"))
        ask_s = self._to_int(row.get("ask_size"))
        last = self._to_float(row.get("trade_price"))

        bids: List[Tuple[Optional[float], Optional[int]]] = []
        asks: List[Tuple[Optional[float], Optional[int]]] = []
        if bid_p is not None:
            bids.append((bid_p, bid_s))
        if ask_p is not None:
            asks.append((ask_p, ask_s))

        ts = (row.get("timestamp") or "") + " UTC"
        return {
            "timestamp": ts.strip(),
            "symbol": self._symbol,
            "last_price": last,
            "bids": bids,
            "asks": asks,
        }


def main():
    parser = argparse.ArgumentParser(description="Live DOM Dashboard v3 with Thermo + Option Tracker")
    parser.add_argument("--default-ticker", default="SPY", help="Default ticker (e.g., SPY, AAPL)")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Poll interval in seconds (can be fractional, e.g., 0.05)")
    parser.add_argument("--port", type=int, default=8021, help="Server port")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (0.0.0.0 for external access)")
    parser.add_argument("--headless", action="store_true", help="Run Chrome in headless mode")
    # Replay/backtest flags
    parser.add_argument("--replay-latest", action="store_true", help="Replay most recent CSV in --replay-dir")
    parser.add_argument("--replay-dir", default="metascripts/DOM_analysis/data", help="Directory containing CSV files to replay")
    parser.add_argument("--replay-file", default=None, help="Explicit CSV file to replay")
    args = parser.parse_args()

    # Determine replay source if requested
    replay_path: Optional[str] = None
    if args.replay_file:
        replay_path = args.replay_file
    elif args.replay_latest:
        # pick the newest *.csv in the directory
        pattern = os.path.join(args.replay_dir, "*.csv")
        candidates = glob.glob(pattern)
        if candidates:
            replay_path = max(candidates, key=os.path.getmtime)

    replayer: Optional[CSVReplayer] = None
    if replay_path:
        if os.path.isfile(replay_path):
            print(f"▶️  Replay mode: {replay_path}")
            replayer = CSVReplayer(replay_path)
        else:
            print(f"⚠️  Replay file not found: {replay_path}")

    app = create_app(
        default_ticker=args.default_ticker,
        poll_interval_seconds=args.poll_interval,
        headless=args.headless,
        replayer=replayer,
    )

    print(f"""
🚀 Live DOM Dashboard v3 Started
🌐 Open: http://{args.host}:{args.port}
📊 Ticker: {args.default_ticker}
🔁 Poll: {args.poll_interval}s
🖥️  Headless: {args.headless}
🎞️  Mode: {'Replay' if replayer is not None else 'Live'}
🛑 Press Ctrl+C to stop.
    """.strip())

    try:
        app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    finally:
        global _scraper
        if _scraper:
            _scraper.shutdown()


if __name__ == "__main__":
    main()

