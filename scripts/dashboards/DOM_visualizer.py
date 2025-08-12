"""
Live DOM Dashboard: Real-time Bid/Ask/Last + Price Chart (via Cboe Equities)

Instructions:
- Install dependencies:
    pip install flask selenium webdriver-manager

- Run:
    python DOM_analyzer_v8.py --default-ticker SPY --poll-interval 2 --port 8000 --headless

- Open in browser:
    http://127.0.0.1:8000
"""

from __future__ import annotations

import atexit
import threading
import time
import argparse
from datetime import datetime
from typing import Dict, Optional

from flask import Flask, jsonify, request, Response

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


CBOE_EQUITIES_BOOK_URL = "https://www.cboe.com/market_data_services/us/equities/#"


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
        options = Options()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1600,1200")
        options.add_argument("--log-level=3")
        # Prevent popups
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=options)

    def _navigate(self) -> None:
        self._driver.get(CBOE_EQUITIES_BOOK_URL)
        time.sleep(3)
        self._accept_cookies_if_present()
        try:
            self._driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 2);")
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
                symbol_input = self._wait.until(EC.presence_of_element_located((By.ID, "symbol0")))
                symbol_input.clear()
                symbol_input.send_keys(ticker)
                search_btn = self._driver.find_element(By.ID, "symbol-search-0")
                self._driver.execute_script("arguments[0].click();", search_btn)

                self._wait.until(EC.presence_of_element_located((By.ID, "bookViewer0")))
                self._container = self._driver.find_element(By.ID, "bookViewer0")
                self._wait.until(
                    lambda d: any(el.text.strip() for el in d.find_elements(By.CLASS_NAME, "book-viewer__bid-price"))
                )
                self._last_ticker = ticker
            except Exception as e:
                print(f"Failed to load ticker {ticker}: {e}")

    def get_snapshot(self) -> Dict[str, str]:
        with self._lock:
            try:
                assert self._container is not None
                bid_prices = self._container.find_elements(By.CLASS_NAME, "book-viewer__bid-price")
                bid_sizes = self._container.find_elements(By.CLASS_NAME, "book-viewer__bid-shares")
                ask_prices = self._container.find_elements(By.CLASS_NAME, "book-viewer__ask-price")
                ask_sizes = self._container.find_elements(By.CLASS_NAME, "book-viewer__ask-shares")
                trade_prices = self._container.find_elements(By.CLASS_NAME, "book-viewer__trades-price")
                trade_sizes = self._container.find_elements(By.CLASS_NAME, "book-viewer__trades-shares")

                now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                return {
                    "timestamp": now,
                    "symbol": self._last_ticker or "",
                    "bid_price": bid_prices[0].text.strip() if bid_prices else "",
                    "bid_size": bid_sizes[0].text.strip() if bid_sizes else "",
                    "ask_price": ask_prices[0].text.strip() if ask_prices else "",
                    "ask_size": ask_sizes[0].text.strip() if ask_sizes else "",
                    "last_price": trade_prices[0].text.strip() if trade_prices else "",
                    "last_size": trade_sizes[0].text.strip() if trade_sizes else "",
                }
            except Exception as e:
                print(f"Scraping error: {e}")
                return {
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "symbol": self._last_ticker or "",
                    "bid_price": "",
                    "bid_size": "",
                    "ask_price": "",
                    "ask_size": "",
                    "last_price": "",
                    "last_size": "",
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


def create_app(default_ticker: str = "SPY", poll_interval_seconds: int = 2, headless: bool = True) -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health() -> Response:
        return Response("ok", status=200, mimetype="text/plain")

    @app.get("/api/quote")
    def api_quote() -> Response:
        ticker = request.args.get("ticker", default_ticker).strip().upper()
        try:
            scraper = get_global_scraper(headless=headless)
            scraper.ensure_ticker(ticker)
            data = scraper.get_snapshot()
            return jsonify({"ok": True, "data": data})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.get("/")
    def index() -> Response:
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Live DOM Dashboard</title>
  <!-- Chart.js -->
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {{
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      margin: 0; padding: 0; background: #0b1221; color: #e6edf3;
    }}
    header {{
      padding: 16px 20px;
      background: #0e1730;
      border-bottom: 1px solid #1f2a44;
      display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
    }}
    h1 {{ margin: 0; font-size: 18px; font-weight: 600; }}
    .ticker-input {{
      display: flex; align-items: center; gap: 8px;
      background: #0b1221; border: 1px solid #223154; border-radius: 8px; padding: 8px 10px;
    }}
    .ticker-input input {{
      background: transparent; border: none; outline: none; color: #e6edf3; font-size: 16px; min-width: 120px; text-transform: uppercase;
    }}
    .button {{
      background: #3b82f6; color: white; border: none; border-radius: 8px; padding: 8px 12px; cursor: pointer; font-weight: 600;
    }}
    .button:disabled {{ opacity: 0.6; cursor: not-allowed; }}

    main {{ padding: 20px; }}
    .grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 16px;
    }}
    .card {{
      background: #0e1730; border: 1px solid #1f2a44; border-radius: 12px; padding: 16px; min-height: 120px;
    }}
    .label {{ color: #93a2bf; font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }}
    .value {{ font-size: 32px; font-weight: 700; }}
    .sub {{ color: #9fb0d4; margin-top: 6px; font-size: 13px; }}

    .chart-container {{
      background: #0e1730; border: 1px solid #1f2a44; border-radius: 12px; padding: 16px;
    }}
    .status {{ margin-left: auto; color: #9fb0d4; font-size: 13px; }}
    .ok {{ color: #22c55e; }}
    .err {{ color: #ef4444; }}
  </style>
</head>
<body>
  <header>
    <h1>Live DOM Dashboard</h1>
    <div class="ticker-input">
      <span>Ticker</span>
      <input id="ticker" value="{default_ticker}" maxlength="10" />
      <button id="apply" class="button">Apply</button>
    </div>
    <div class="status" id="status">Idle</div>
  </header>

  <main>
    <div class="grid">
      <div class="card">
        <div class="label">Bid</div>
        <div class="value" id="bid">—</div>
        <div class="sub" id="bidSize">Size: —</div>
      </div>
      <div class="card">
        <div class="label">Ask</div>
        <div class="value" id="ask">—</div>
        <div class="sub" id="askSize">Size: —</div>
      </div>
      <div class="card">
        <div class="label">Last</div>
        <div class="value" id="last">—</div>
        <div class="sub" id="lastSize">Size: —</div>
      </div>
      <div class="card">
        <div class="label">Meta</div>
        <div class="sub" id="symbol">Symbol: {default_ticker}</div>
        <div class="sub" id="timestamp">Updated: —</div>
      </div>
    </div>

    <!-- Real-Time Price Chart -->
    <div class="chart-container">
      <canvas id="priceChart"></canvas>
    </div>
  </main>

  <script>
    // UI Elements
    const statusEl = document.getElementById('status');
    const tickerEl = document.getElementById('ticker');
    const applyEl = document.getElementById('apply');
    const bidEl = document.getElementById('bid');
    const bidSizeEl = document.getElementById('bidSize');
    const askEl = document.getElementById('ask');
    const askSizeEl = document.getElementById('askSize');
    const lastEl = document.getElementById('last');
    const lastSizeEl = document.getElementById('lastSize');
    const symbolEl = document.getElementById('symbol');
    const timestampEl = document.getElementById('timestamp');

    // Chart
    let priceHistory = [];
    const maxDataPoints = 50;
    const ctx = document.getElementById('priceChart').getContext('2d');
    const priceChart = new Chart(ctx, {{
      type: 'line',
      data: {{
        labels: [],
        datasets: [{{ 
          label: 'Last Price',
          data: [],
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          borderWidth: 2,
          fill: true,
          tension: 0.3
        }}]
      }},
      options: {{
        animation: {{ duration: 300 }},
        scales: {{
          y: {{ beginAtZero: false, grid: {{ color: '#1f2a44' }} }},
          x: {{ grid: {{ color: '#1f2a44' }} }}
        }},
        plugins: {{
          legend: {{ labels: {{ color: '#e6edf3' }} }}
        }},
        backgroundColor: '#0e1730'
      }}
    }});

    let currentTicker = '{default_ticker}';

    // Apply new ticker
    applyEl.addEventListener('click', () => {{
      const next = (tickerEl.value || '').trim().toUpperCase();
      if (!next) return;
      currentTicker = next;
      symbolEl.textContent = `Symbol: ${{next}}`;
      // Reset chart on ticker change
      priceHistory = [];
      priceChart.data.labels = [];
      priceChart.data.datasets[0].data = [];
      priceChart.update();
    }});

    // Poll for data
    async function poll() {{
      try {{
        statusEl.textContent = 'Updating…';
        const res = await fetch(`/api/quote?ticker=${{encodeURIComponent(currentTicker)}}`, {{ cache: 'no-store' }});
        const body = await res.json();
        if (!body.ok) throw new Error(body.error || 'Fetch failed');
        
        const d = body.data;
        const price = parseFloat(d.last_price.replace(/,/g, '')) || null;

        // Update UI
        bidEl.textContent = d.bid_price || '—';
        bidSizeEl.textContent = `Size: ${{d.bid_size || '—'}}`;
        askEl.textContent = d.ask_price || '—';
        askSizeEl.textContent = `Size: ${{d.ask_size || '—'}}`;
        lastEl.textContent = d.last_price || '—';
        lastSizeEl.textContent = `Size: ${{d.last_size || '—'}}`;
        timestampEl.textContent = `Updated: ${{d.timestamp}}`;
        statusEl.innerHTML = '<span class="ok">Live</span>';

        // Update chart
        if (price !== null) {{
          const time = new Date().toLocaleTimeString();
          priceHistory.push({{ time, price }});
          if (priceHistory.length > maxDataPoints) {{
            priceHistory.shift();
          }}
          priceChart.data.labels = priceHistory.map(p => p.time);
          priceChart.data.datasets[0].data = priceHistory.map(p => p.price);
          priceChart.update();
        }}
      }} catch (err) {{
        statusEl.innerHTML = `<span class="err">$${{String(err)}}</span>`;
        console.error("Poll error:", err);
      }}
    }}

    // Start polling
    setInterval(poll, {poll_interval_seconds} * 1000);
    poll(); // Initial poll
  </script>
</body>
</html>
        """
        return Response(html, mimetype="text/html")

    return app


def main():
    parser = argparse.ArgumentParser(description="Live DOM Dashboard with Real-Time Chart")
    parser.add_argument("--default-ticker", default="SPY", help="Default ticker (e.g., SPY, AAPL)")
    parser.add_argument("--poll-interval", type=int, default=2, help="Poll interval in seconds")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (0.0.0.0 for external access)")
    parser.add_argument("--headless", action="store_true", help="Run Chrome in headless mode")
    args = parser.parse_args()

    app = create_app(
        default_ticker=args.default_ticker,
        poll_interval_seconds=args.poll_interval,
        headless=args.headless
    )

    print(f"""
🚀 Live DOM Dashboard Started
🌐 Open: http://{args.host}:{args.port}
📊 Ticker: {args.default_ticker}
🔁 Poll: {args.poll_interval}s
🖥️  Headless: {args.headless}
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
