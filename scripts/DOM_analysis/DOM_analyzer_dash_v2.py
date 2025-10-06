"""
Live DOM Dashboard v2: Real-time Bid/Ask/Last + Depth Heatmap and Vertical Density Bars

Instructions:
- Install dependencies:
    pip install flask selenium webdriver-manager

- Run:
    python metascripts/DOM_analysis/DOM_analyzer_dash_v2.py --default-ticker SPY --poll-interval 1 --port 8020 --headless

- Open in browser:
    http://127.0.0.1:8020
"""

from __future__ import annotations

import atexit
import threading
import time
import argparse
from datetime import datetime
from typing import Dict, Optional, List, Tuple

from flask import Flask, jsonify, request, Response

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


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
                        or len(d.find_elements(By.XPATH, "//*[contains(text(), 'Top of Book')]") ) > 0
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


def create_app(default_ticker: str = "SPY", poll_interval_seconds: int = 1, headless: bool = True) -> Flask:
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
            scraper = get_global_scraper(headless=headless)
            scraper.ensure_ticker(ticker)
            data = scraper.get_depth_snapshot(max_levels=max_levels)
            # Retry once if depth is empty
            if not data.get("bids") and not data.get("asks"):
                time.sleep(0.5)
                scraper.ensure_ticker(ticker)
                data = scraper.get_depth_snapshot(max_levels=max_levels)
            return jsonify({"ok": True, "data": data})
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
  <title>Live DOM Dashboard v2</title>
  <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>
  <style>
    body {{
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      margin: 0; padding: 0; background: #0b1221; color: #e6edf3;
    }}
    header {{
      padding: 12px 16px; background: #0e1730; border-bottom: 1px solid #1f2a44;
      display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
    }}
    h1 {{ margin: 0; font-size: 16px; font-weight: 600; }}
    .ticker-input {{ display: flex; align-items: center; gap: 8px; background: #0b1221; border: 1px solid #223154; border-radius: 8px; padding: 6px 8px; }}
    .ticker-input input {{ background: transparent; border: none; outline: none; color: #e6edf3; font-size: 14px; min-width: 100px; text-transform: uppercase; }}
    .button {{ background: #3b82f6; color: white; border: none; border-radius: 8px; padding: 6px 10px; cursor: pointer; font-weight: 600; }}
    .status {{ margin-left: auto; color: #9fb0d4; font-size: 13px; }}

    main {{ padding: 12px; }}
    .layout {{ display: grid; grid-template-columns: 120px 1fr 120px; gap: 12px; align-items: stretch; }}
    .panel {{ background: #0e1730; border: 1px solid #1f2a44; border-radius: 10px; padding: 8px; }}
    .center {{ display: grid; grid-template-rows: 360px 240px; gap: 12px; }}
    canvas {{ display: block; width: 100%; height: 100%; image-rendering: pixelated; }}
  </style>
  </head>
  <body>
    <header>
      <h1>Live DOM Dashboard v2</h1>
      <div class=\"ticker-input\">
        <span>Ticker</span>
        <input id=\"ticker\" value=\"__DEFAULT_TICKER__\" maxlength=\"10\" />
        <button id=\"apply\" class=\"button\">Apply</button>
      </div>
      <div class=\"status\" id=\"status\">Idle</div>
    </header>

    <main>
      <div class=\"layout\">
        <div class=\"panel\">
          <canvas id=\"bidBars\" width=\"120\" height=\"360\"></canvas>
        </div>
        <div class=\"center\">
          <div class=\"panel\">
            <canvas id=\"heatmap\" width=\"900\" height=\"360\"></canvas>
          </div>
          <div class=\"panel\">
            <canvas id=\"priceChartCanvas\" height=\"240\"></canvas>
          </div>
        </div>
        <div class=\"panel\">
          <canvas id=\"askBars\" width=\"120\" height=\"360\"></canvas>
        </div>
      </div>
    </main>

    <script>
      const statusEl = document.getElementById('status');
      const tickerEl = document.getElementById('ticker');
      const applyEl = document.getElementById('apply');
      const heatmapCanvas = document.getElementById('heatmap');
      const heatmapCtx = heatmapCanvas.getContext('2d');
      const bidBarsCanvas = document.getElementById('bidBars');
      const bidBarsCtx = bidBarsCanvas.getContext('2d');
      const askBarsCanvas = document.getElementById('askBars');
      const askBarsCtx = askBarsCanvas.getContext('2d');

      const chartCtx = document.getElementById('priceChartCanvas').getContext('2d');
      const priceChart = new Chart(chartCtx, {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Last Price', data: [], borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.1)', borderWidth: 2, fill: true, tension: 0.3 }] },
        options: { animation: { duration: 200 }, plugins: { legend: { labels: { color: '#e6edf3' } } }, scales: { y: { grid: { color: '#1f2a44' } }, x: { grid: { color: '#1f2a44' } } } }
      });

      let currentTicker = '__DEFAULT_TICKER__';
      let minPrice = null, maxPrice = null;
      let maxSeenSize = 1;
      const maxChartPoints = 120;

      applyEl.addEventListener('click', () => {
        const next = (tickerEl.value || '').trim().toUpperCase();
        if (!next) return;
        currentTicker = next;
        minPrice = null; maxPrice = null; maxSeenSize = 1;
        priceChart.data.labels = []; priceChart.data.datasets[0].data = []; priceChart.update();
        heatmapCtx.clearRect(0, 0, heatmapCanvas.width, heatmapCanvas.height);
        bidBarsCtx.clearRect(0, 0, bidBarsCanvas.width, bidBarsCanvas.height);
        askBarsCtx.clearRect(0, 0, askBarsCanvas.width, askBarsCanvas.height);
      });

      function updatePriceRange(last, bids, asks) {
        const prices = [];
        if (typeof last === 'number') prices.push(last);
        bids.forEach(([p,_]) => { if (typeof p === 'number') prices.push(p); });
        asks.forEach(([p,_]) => { if (typeof p === 'number') prices.push(p); });
        if (!prices.length) return;
        const pmin = Math.min(...prices), pmax = Math.max(...prices);
        if (minPrice === null || pmin < minPrice) minPrice = pmin;
        if (maxPrice === null || pmax > maxPrice) maxPrice = pmax;
        if (maxPrice - minPrice < 0.01) { maxPrice = minPrice + 0.5; }
      }

      function yForPrice(price) {
        if (minPrice === null || maxPrice === null) return null;
        if (price === null || typeof price !== 'number') return null;
        const r = (price - minPrice) / (maxPrice - minPrice);
        const y = Math.round((1 - r) * (heatmapCanvas.height - 1));
        return Math.max(0, Math.min(heatmapCanvas.height - 1, y));
      }

      function intensity(size) {
        if (!size || size <= 0) return 0;
        maxSeenSize = Math.max(maxSeenSize, size);
        const v = Math.log(size + 1) / Math.log(maxSeenSize + 1);
        return Math.max(0.05, Math.min(1, v));
      }

      function drawHeatmapColumn(bids, asks, last) {
        // shift left by 1 px
        const imageData = heatmapCtx.getImageData(1, 0, heatmapCanvas.width - 1, heatmapCanvas.height);
        heatmapCtx.putImageData(imageData, 0, 0);
        // clear last column
        heatmapCtx.clearRect(heatmapCanvas.width - 1, 0, 1, heatmapCanvas.height);

        const x = heatmapCanvas.width - 1;
        const drawLevels = (levels, isBid) => {
          for (const [p, s] of levels) {
            const y = yForPrice(p);
            if (y === null) continue;
            const a = intensity(s);
            const r = isBid ? 50 : 220;
            const g = isBid ? 220 : 60;
            const b = 80;
            heatmapCtx.fillStyle = `rgba(${r},${g},${b},${a})`;
            heatmapCtx.fillRect(x, y - 1, 1, 3);
          }
        };
        drawLevels(bids, true);
        drawLevels(asks, false);

        // Draw last price cursor
        const ly = yForPrice(last);
        if (ly !== null) {
          heatmapCtx.fillStyle = 'rgba(255,255,255,0.9)';
          heatmapCtx.fillRect(x, ly, 1, 1);
        }
      }

      function drawSideBars(canvas, ctx, levels, isBid) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const maxSize = Math.max(1, ...levels.map(([_, s]) => (s||0)));
        for (const [p, s] of levels) {
          const y = yForPrice(p); if (y === null) continue;
          const w = Math.round(((s||0) / maxSize) * (canvas.width - 10));
          const h = 4;
          const color = isBid ? '#22c55e' : '#ef4444';
          ctx.fillStyle = color;
          const x = isBid ? (canvas.width - w) : 0;
          ctx.fillRect(x, y - Math.floor(h/2), w, h);
        }
      }

      async function poll() {
        try {
          statusEl.textContent = 'Updating…';
          const res = await fetch(`/api/depth?ticker=${encodeURIComponent(currentTicker)}&levels=15`, { cache: 'no-store' });
          const body = await res.json();
          if (!body.ok) throw new Error(body.error || 'Fetch failed');
          const d = body.data;
          const last = typeof d.last_price === 'number' ? d.last_price : null;
          const bids = (d.bids || []).filter(([p,_]) => typeof p === 'number');
          const asks = (d.asks || []).filter(([p,_]) => typeof p === 'number');

          updatePriceRange(last, bids, asks);
          drawHeatmapColumn(bids, asks, last);
          drawSideBars(bidBarsCanvas, bidBarsCtx, bids, true);
          drawSideBars(askBarsCanvas, askBarsCtx, asks, false);

          const price = last;
          if (price !== null) {
            const timeLabel = new Date().toLocaleTimeString();
            priceChart.data.labels.push(timeLabel);
            priceChart.data.datasets[0].data.push(price);
            if (priceChart.data.labels.length > maxChartPoints) { priceChart.data.labels.shift(); priceChart.data.datasets[0].data.shift(); }
            priceChart.update();
          }
          statusEl.innerHTML = '<span style="color:#22c55e">Live</span>';
        } catch (err) {
          statusEl.innerHTML = `<span style='color:#ef4444'>${String(err)}</span>`;
          console.error('Poll error', err);
        }
      }

      setInterval(poll, __POLL__ * 1000);
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


def main():
    parser = argparse.ArgumentParser(description="Live DOM Dashboard v2 with Heatmap")
    parser.add_argument("--default-ticker", default="SPY", help="Default ticker (e.g., SPY, AAPL)")
    parser.add_argument("--poll-interval", type=int, default=1, help="Poll interval in seconds")
    parser.add_argument("--port", type=int, default=8020, help="Server port")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (0.0.0.0 for external access)")
    parser.add_argument("--headless", action="store_true", help="Run Chrome in headless mode")
    args = parser.parse_args()

    app = create_app(
        default_ticker=args.default_ticker,
        poll_interval_seconds=args.poll_interval,
        headless=args.headless
    )

    print(f"""
🚀 Live DOM Dashboard v2 Started
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


