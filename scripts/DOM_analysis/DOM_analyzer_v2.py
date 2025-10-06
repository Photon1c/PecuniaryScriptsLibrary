"""
CBOE DOM Analyzer v2 (Container‑First Scrape)
============================================

Purpose
- Minimal example that locates the book viewer container and prints ladders
  and recent trades without interacting with the search widget.

What it does
- Scrolls and waits for `#bookViewer0`, then scrapes bid/ask/trade sections

Usage
- `python DOM_analyzer_v2.py`

Dependencies
- Selenium, webdriver-manager

Notes
- Sensitive to page structure changes; selectors may require maintenance.
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

def extract_full_dom_snapshot(ticker="SPY"):
    url = "https://www.cboe.com/market_data_services/us/equities/#"

    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--log-level=3")


    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)

    try:
        time.sleep(5)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)

        print("✅ Searching for bookViewer0...")
        container = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "bookViewer0"))
        )

        print("📊 Scraping full DOM...")

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

        # LAST TRADES
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


extract_full_dom_snapshot(ticker="SPY")
