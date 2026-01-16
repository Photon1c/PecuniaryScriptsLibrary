# nasdaq-datadownloader.py
# Purpose: Auto-download 1Y historical CSVs from Nasdaq for a list of ETF tickers.
# Notes: Uses robust selectors (falls back to your absolute XPath), handles cookie banners,
#        waits for the file to fully finish (no .crdownload), and standardizes filenames.

import os, time, random, shutil, glob, datetime as dt
import json
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --------- CONFIG ----------
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

# Format: (symbol, type) where type is either 'etf' or 'stocks'
# Loaded from shared tickers.json file
TICKERS = load_tickers()

DOWNLOAD_DIR = r"F:\inputs\stocks"
HEADLESS = True
# -----------------------------

URL_TMPL = "https://www.nasdaq.com/market-activity/{asset_type}/{ticker}/historical?page=1&rows_per_page=10&timeline=y1"



# Prefer robust selectors; fall back to the user-provided absolute XPath
XPATHS_TO_TRY = [
    # Common patterns seen on Nasdaq
    "//button[.//span[contains(translate(., 'abcdefghijklmnopqrstuvwxyz','ABCDEFGHIJKLMNOPQRSTUVWXYZ'),'DOWNLOAD')]]",
    "//button[contains(@aria-label,'Download') or contains(@data-testid,'download')]",
    # User-provided absolute XPath (last resort; brittle)
    "/html/body/div[2]/div/main/div[2]/article/div/div[2]/div/div[2]/div[2]/div[1]/div[2]/div[1]/div[1]/div[2]/button/span"
]

# OneTrust cookie banner is common on Nasdaq properties
COOKIE_ACCEPT_XPATHS = [
    "//*[@id='onetrust-accept-btn-handler']",
    "//button[contains(@id,'onetrust-accept')]",
    "//button[contains(.,'Accept All') or contains(.,'I Accept')]",
]

def build_driver(download_dir: str, headless: bool):
    download_path = Path(download_dir).resolve()
    download_path.mkdir(parents=True, exist_ok=True)

    def make_options(use_new_headless: bool):
        opts = webdriver.ChromeOptions()
        if headless:
            opts.add_argument("--headless=new" if use_new_headless else "--headless")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--window-size=1400,900")
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_argument("--user-agent=Mozilla/5.0")
        prefs = {
            "download.default_directory": str(download_path),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
        }
        opts.add_experimental_option("prefs", prefs)
        return opts

    errors = []
    # 1) Try Selenium Manager (recommended in modern Selenium)
    for use_new_headless in (True, False):
        try:
            return webdriver.Chrome(options=make_options(use_new_headless))
        except Exception as e:
            errors.append(f"selenium_manager(headless={'new' if use_new_headless else 'legacy'}): {e}")
            continue

    # 2) Fallback: webdriver_manager-managed chromedriver
    for use_new_headless in (True, False):
        try:
            driver_path = ChromeDriverManager().install()
            if not os.path.isfile(driver_path) or not driver_path.lower().endswith(".exe"):
                raise RuntimeError(f"Bad driver path: {driver_path}")
            svc = Service(driver_path)
            return webdriver.Chrome(service=svc, options=make_options(use_new_headless))
        except Exception as e:
            errors.append(f"webdriver_manager(headless={'new' if use_new_headless else 'legacy'}): {e}")
            continue

    raise RuntimeError("Failed to start Chrome WebDriver. Attempts:\n" + "\n".join(errors))

def click_if_present(driver, xpaths, timeout=6):
    """Try multiple XPaths; return True if any was clicked."""
    for xp in xpaths:
        try:
            el = WebDriverWait(driver, timeout).until(EC.element_to_be_clickable((By.XPATH, xp)))
            driver.execute_script("arguments[0].click();", el)
            return True
        except Exception:
            pass
    return False

def wait_for_download_complete(dir_path: str, start_ts: float, timeout=30) -> str:
    """Wait until a new .csv (not .crdownload) appears after start_ts; return file path."""
    end = time.time() + timeout
    newest = ""
    while time.time() < end:
        # Ignore temp files, require modified after navigation time
        candidates = [p for p in glob.glob(os.path.join(dir_path, "*.csv")) if os.path.getmtime(p) >= start_ts]
        if candidates:
            # ensure chrome finished (no matching .crdownload)
            in_progress = glob.glob(os.path.join(dir_path, "*.crdownload"))
            if not in_progress:
                newest = max(candidates, key=os.path.getmtime)
                break
        time.sleep(0.5)
    if not newest:
        raise TimeoutError("CSV download not detected (timeout).")
    return newest

def standardize_filename(src_path: str, ticker: str) -> str:
    """Rename downloaded CSV -> {TICKER}.csv (uppercase); overwrite if exists."""
    dst = Path(src_path).with_name(f"{ticker.upper()}.csv")
    # Overwrite if exists
    if dst.exists():
        dst.unlink()
    shutil.move(src_path, dst)
    return str(dst)

def fetch_one(driver, ticker: str, asset_type: str, download_dir: str) -> str:
    url = URL_TMPL.format(asset_type=asset_type.lower(), ticker=ticker.lower())
    driver.get(url)
    # small jitter to look human + let layout settle
    time.sleep(random.uniform(0.8, 1.6))

    # Clear cookie banner if present
    click_if_present(driver, COOKIE_ACCEPT_XPATHS, timeout=3)

    # Click the download button
    if not click_if_present(driver, XPATHS_TO_TRY, timeout=8):
        # If the last selector is the absolute XPath to the span, try clicking its parent button
        try:
            span = WebDriverWait(driver, 4).until(EC.presence_of_element_located((By.XPATH, XPATHS_TO_TRY[-1])))
            parent_btn = span.find_element(By.XPATH, "./ancestor::button[1]")
            driver.execute_script("arguments[0].click();", parent_btn)
        except Exception as e:
            raise RuntimeError(f"Failed to locate/press download button for {ticker}: {e}")

    # Mark time then wait for the file to land
    start_ts = time.time()
    csv_path = wait_for_download_complete(download_dir, start_ts, timeout=45)
    return standardize_filename(csv_path, ticker)

def main():
    driver = build_driver(DOWNLOAD_DIR, HEADLESS)
    try:
        results = {}
        for symbol, asset_type in TICKERS:
            try:
                print(f"→ ({symbol}, {asset_type}): navigating & downloading…")
                final_path = fetch_one(driver, symbol, asset_type, DOWNLOAD_DIR)
                results[(symbol, asset_type)] = final_path
                print(f"   saved: {final_path}")
                time.sleep(random.uniform(0.8, 1.5))  # polite pacing
            except Exception as e:
                results[(symbol, asset_type)] = f"ERROR: {e}"
                print(f"   ERROR for ({symbol}, {asset_type}): {e}")
        print("\nSummary:")
        for k, v in results.items():
            print(f"{k}: {v}")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
