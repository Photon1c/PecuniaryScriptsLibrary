"""
Bulk-download Cboe option chain CSVs for a list of tickers.

Prereqs:
  - pip install selenium
  - Firefox installed
  - geckodriver available on PATH (or set GECKODRIVER env var)

Notes:
  - This mirrors your Selenium IDE clicks:
      - accept cookies
      - open ReactSelect dropdown and pick the first option
      - click "Download" button, then click the CSV item
  - If Cboe changes selectors, update the CSS/ID constants below.
"""

from __future__ import annotations
import os
import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
import shutil
from pathlib import Path
import tempfile

CHROME_PROFILE_BASE = Path(r"F:\inputs\options\chrome_profiles")
CHROME_PROFILE_BASE.mkdir(parents=True, exist_ok=True)

BASE_LOG_DIR = Path(r"F:\inputs\options\log")

# Path to shared tickers.json file (relative to this script's location)
SCRIPT_DIR = Path(__file__).parent
TICKERS_JSON = SCRIPT_DIR.parent / "tickers.json"

def load_tickers() -> list[tuple[str, str]]:
    """Load tickers from shared JSON file."""
    if not TICKERS_JSON.exists():
        raise FileNotFoundError(f"Tickers file not found: {TICKERS_JSON}")
    
    with open(TICKERS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Convert list of lists to list of tuples
    return [tuple(ticker) for ticker in data["tickers"]]

def build_target_path(ticker: str) -> Path:
    today = datetime.now().strftime("%m_%d_%Y")
    t = ticker.lower()
    out_dir = BASE_LOG_DIR / t / today
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{t}_quotedata.csv"

def move_with_retries(src: Path, dst: Path, tries: int = 10, sleep: float = 0.3) -> None:
    """Move file with retries, handling file locks and cross-drive moves."""
    if not src.exists():
        raise FileNotFoundError(f"Source file does not exist: {src}")
    
    if dst.exists():
        dst.unlink()

    last_err = None
    for attempt in range(tries):
        try:
            shutil.move(str(src), str(dst))
            return
        except Exception as e:
            last_err = e
            if attempt < tries - 1:  # Don't sleep on last attempt
                time.sleep(sleep)
    
    raise RuntimeError(f"Failed to move {src} to {dst} after {tries} attempts. Last error: {last_err}")



# -----------------------------
# Config
# -----------------------------
# Load tickers from shared JSON file
# Format: (symbol, type) where type is either 'etf' or 'stocks'
TICKERS = load_tickers()

BASE_URL = "https://www.cboe.com/delayed_quotes/{ticker}/quote_table"

DOWNLOAD_DIR = Path.cwd() / "cboe_downloads"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

WAIT_SEC = 20
POST_CLICK_PAUSE_SEC = 2


# -----------------------------
# Selectors (from your IDE export)
# -----------------------------
COOKIE_ACCEPT_ID = "onetrust-accept-btn-handler"
REACTSELECT_CONTROL_CSS = ".ReactSelect__control--is-focused > .ReactSelect__value-container"
REACTSELECT_FIRST_OPTION_ID = "react-select-4-option-0"
DOWNLOAD_BUTTON_CSS = ".hMfKkd > .Button__TextContainer-cui__sc-1ahwe65-1"
CSV_MENU_ITEM_CSS = ".nQdRD"  # in your export, this is the click after opening the download menu


def make_chrome(download_dir: Path) -> webdriver.Chrome:
    opts = Options()

    # Make a fresh profile dir per run on F: (no Temp usage)
    run_profile = CHROME_PROFILE_BASE / f"selenium_profile_{int(time.time())}"
    run_profile.mkdir(parents=True, exist_ok=True)

    opts.add_argument(f"--user-data-dir={run_profile}")
    opts.add_argument(f"--disk-cache-dir={run_profile / 'cache'}")

    # Optional: reduce cache churn/noise
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-background-networking")
    opts.add_argument("--disable-sync")
    opts.add_argument("--disable-default-apps")
    opts.add_argument("--disable-extensions")
    opts.add_argument("--disable-notifications")
    opts.add_argument("--no-first-run")
    opts.add_argument("--no-default-browser-check")


    prefs = {
        "download.default_directory": str(download_dir),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
    }
    opts.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(options=opts)
    driver.set_window_size(1550, 838)
    return driver




def wait_for_any_new_csv(download_dir: Path, start_snapshot: set[Path], timeout_sec: int) -> Path:
    """Wait until a new .csv appears (and is not a partial .crdownload for Chrome)."""
    end_time = time.time() + timeout_sec
    while time.time() < end_time:
        # Check for Chrome's partial download files
        if any(download_dir.glob("*.crdownload")):
            time.sleep(0.5)
            continue
        
        current = {p for p in download_dir.glob("*.csv")}
        new = list(current - start_snapshot)
        if new:
            # Ensure file is fully written by checking it's not being written to
            newest = max(new, key=lambda p: p.stat().st_mtime)
            # Wait a bit more to ensure download is complete
            time.sleep(0.5)
            # Verify file still exists and is readable
            if newest.exists() and newest.stat().st_size > 0:
                return newest
        
        time.sleep(0.25)

    raise TimeoutError(f"No new CSV detected in {download_dir} within {timeout_sec}s")


def safe_click(wait: WebDriverWait, by: By, selector: str) -> None:
    el = wait.until(EC.element_to_be_clickable((by, selector)))
    el.click()


def download_chain_for_ticker(driver: webdriver.Chrome, ticker: str) -> Path:
    ticker_l = ticker.lower()
    url = BASE_URL.format(ticker=ticker_l)

    wait = WebDriverWait(driver, WAIT_SEC)
    before = {p for p in DOWNLOAD_DIR.glob("*.csv")}

    driver.get(url)

    # Cookie accept (first run). If already accepted, ignore.
    try:
        safe_click(wait, By.ID, COOKIE_ACCEPT_ID)
    except Exception:
        pass

    # These clicks mirror your Selenium IDE script :contentReference[oaicite:1]{index=1}
    # 1) ReactSelect control -> 2) first option -> 3) download -> 4) csv
    try:
        safe_click(wait, By.CSS_SELECTOR, REACTSELECT_CONTROL_CSS)
        safe_click(wait, By.ID, REACTSELECT_FIRST_OPTION_ID)
    except Exception:
        # If the control isn’t focused or IDs differ, you may need to update selectors.
        pass

    time.sleep(POST_CLICK_PAUSE_SEC)
    safe_click(wait, By.CSS_SELECTOR, DOWNLOAD_BUTTON_CSS)
    time.sleep(0.5)
    safe_click(wait, By.CSS_SELECTOR, CSV_MENU_ITEM_CSS)
    
    # Give the download a moment to start
    time.sleep(1.0)

    try:
        csv_path = wait_for_any_new_csv(DOWNLOAD_DIR, before, timeout_sec=WAIT_SEC + 25)
    except TimeoutError as e:
        # Provide more diagnostic info
        current_files = list(DOWNLOAD_DIR.glob("*.csv"))
        crdownload_files = list(DOWNLOAD_DIR.glob("*.crdownload"))
        raise TimeoutError(
            f"{e}\n"
            f"Download dir: {DOWNLOAD_DIR}\n"
            f"Existing CSVs: {[f.name for f in current_files]}\n"
            f"In-progress downloads: {[f.name for f in crdownload_files]}"
        )

    target = build_target_path(ticker)
    
    # Use move_with_retries which handles overwriting and retries
    move_with_retries(csv_path, target)
    return target



def main() -> None:
    driver = make_chrome(DOWNLOAD_DIR)
    print("DOWNLOAD_DIR:", DOWNLOAD_DIR.resolve())


    try:
        for symbol, asset_type in TICKERS:
            try:
                out = download_chain_for_ticker(driver, symbol)
                print(f"[OK] {symbol} ({asset_type}): {out.name}")
            except Exception as e:
                print(f"[FAIL] {symbol} ({asset_type}): {e}")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
