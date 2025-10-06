"""
CBOE DOM Analyzer v0 (Very First Prototype)
==========================================

Purpose
- Minimal example to locate the DOM container and print ladder tables.

Notes
- Superseded by later analyzers; kept for historical reference.
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

symbol = input("Enter the symbol: ")

def extract_aapl_dom_snapshot():
    #url = "https://www.cboe.com/market_data_services/us/equities/#"
    #direct book viewer url
    url = "https://www.cboe.com/us/equities/market_statistics/book/{symbol}/"

    options = Options()
    options.add_argument("--start-maximized")

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
