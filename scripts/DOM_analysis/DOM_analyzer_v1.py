"""
CBOE DOM Analyzer v1 (First Pass)
=================================

Purpose
- Earliest working prototype to locate the book viewer and print bid/ask rows.

What it does
- Scrolls to reveal DOM, waits for `#bookViewer0`, scrapes bid/ask tables

Usage
- `python DOM_analyzer_v1.py`

Dependencies
- Selenium, webdriver-manager

Notes
- Superseded by later versions (v6, v7, v7.1). Kept for reference.
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

def extract_aapl_dom_snapshot():
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

        print("📊 Found DOM container. Scraping bids and asks...")

        bids = container.find_elements(By.CSS_SELECTOR, "td.book-viewer__bid-price")
        bid_sizes = container.find_elements(By.CSS_SELECTOR, "td.book-viewer__bid-shares")

        asks = container.find_elements(By.CSS_SELECTOR, "td.book-viewer__ask-price")
        ask_sizes = container.find_elements(By.CSS_SELECTOR, "td.book-viewer__ask-shares")

        print("\n--- BID SIDE ---")
        for price, size in zip(bids, bid_sizes):
            print(f"{size.text:>5} @ {price.text}")

        print("\n--- ASK SIDE ---")
        for price, size in zip(asks, ask_sizes):
            print(f"{size.text:>5} @ {price.text}")

    except Exception as e:
        print("❌ Error:", e)

    finally:
        driver.quit()


extract_aapl_dom_snapshot()
