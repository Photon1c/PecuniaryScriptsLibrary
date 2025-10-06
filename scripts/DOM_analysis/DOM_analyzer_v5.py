"""
CBOE DOM Analyzer v5 (Cookie‑Acceptance Flow)
============================================

Purpose
- Demonstrates loading the CBOE equities page, forcibly accepting cookies,
  typing a ticker, and printing book data to stdout.

What it does
- Scrolls to bring widgets into view, clicks "Accept All Cookies"
- Types the user-provided ticker and clicks Search
- Prints ask/bid ladders and last 10 trades

Inputs/Outputs
- Input: Ticker (via prompt)
- Output: Console printout of current DOM snapshot

Usage
- `python DOM_analyzer_v5.py`

Dependencies
- Selenium, webdriver-manager

Notes
- Intended as a diagnostic/reference version. Prefer v7/v7.1 for production use.
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time

ticker=input("Enter ticker in uppercase letters: ")

def extract_dom_force_accept_cookie(ticker):
    url = "https://www.cboe.com/market_data_services/us/equities/#"

    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--log-level=2")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)

    try:
        time.sleep(5)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        # ✅ Force click "Accept All Cookies"
        try:
            print("🍪 Trying to click 'Accept All Cookies'...")
            accept_btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "/html/body/div[2]/div[2]/div/div[1]/div/div[2]/div/button[3]"))
            )
            driver.execute_script("arguments[0].click();", accept_btn)
            print("✅ Accepted cookies.")
            time.sleep(2)
        except Exception as e:
            print(f"⚠️ Could not click Accept button: {e}")

        # ✅ Proceed with typing ticker
        print(f"🔎 Typing ticker: {ticker}")
        symbol_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "symbol0"))
        )
        symbol_input.clear()
        symbol_input.send_keys(ticker)

        print("✅ Clicking search submit button...")
        submit_button = driver.find_element(By.ID, "symbol-search-0")
        driver.execute_script("arguments[0].click();", submit_button)

        print("⏳ Waiting for bookViewer0 to load...")
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "bookViewer0"))
        )
        time.sleep(3)

        container = driver.find_element(By.ID, "bookViewer0")
        print("📊 DOM loaded. Extracting...")

        # ⏳ Wait until DOM actually contains populated bid prices
        print("⏳ Waiting for bid/ask data to populate...")
        WebDriverWait(driver, 10).until(
            lambda d: any(el.text.strip() for el in d.find_elements(By.CLASS_NAME, "book-viewer__bid-price"))
        )


        # ASK SIDE
        asks = zip(
            container.find_elements(By.CLASS_NAME, "book-viewer__ask-shares"),
            container.find_elements(By.CLASS_NAME, "book-viewer__ask-price")
        )
        print("\n--- ASK SIDE ---")
        for size, price in asks:
            print(f"{size.text:>8} @ {price.text}")

        # BID SIDE
        bids = zip(
            container.find_elements(By.CLASS_NAME, "book-viewer__bid-shares"),
            container.find_elements(By.CLASS_NAME, "book-viewer__bid-price")
        )
        print("\n--- BID SIDE ---")
        for size, price in bids:
            print(f"{size.text:>8} @ {price.text}")

        # LAST 10 TRADES
        times = container.find_elements(By.CLASS_NAME, "book-viewer__trades-time")
        prices = container.find_elements(By.CLASS_NAME, "book-viewer__trades-price")
        volumes = container.find_elements(By.CLASS_NAME, "book-viewer__trades-shares")
        print("\n--- LAST 10 TRADES ---")
        for time_el, price_el, vol_el in zip(times, prices, volumes):
            print(f"{time_el.text:>8} | {vol_el.text:>6} shares @ {price_el.text}")

    except Exception as e:
        print("❌ Error:", e)

    finally:
        driver.quit()

extract_dom_force_accept_cookie(ticker)
