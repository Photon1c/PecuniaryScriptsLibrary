"""
CBOE DOM Analyzer v3 (Legacy Autocomplete Approach)
===================================================

Status
- Historical artifact demonstrating use of the autocomplete list.
- Known to break because the input locator and suggestion list can change.

Purpose
- Explore typed-search and autosuggest selection before scraping DOM.

Notes
- Prefer v7/v7.1 direct URLs for stability. Use this file as a reference for
  autosuggest flows if reintroducing that UX.
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

def extract_true_dom_snapshot(ticker="SPY"):
    url = "https://www.cboe.com/market_data_services/us/equities/#"

    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--log-level=3")


    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)

    try:
        time.sleep(5)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        print(f"🔎 Typing ticker: {ticker}")
        input_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'input[placeholder="Search symbol"]'))
        )
        input_box.clear()
        input_box.send_keys(ticker)
        time.sleep(1)

        # Wait for the autocomplete list
        print("⌛ Waiting for autocomplete suggestions...")
        suggestions_list = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "ul.react-autosuggest__suggestions-list"))
        )

        suggestions = suggestions_list.find_elements(By.TAG_NAME, "li")
        if not suggestions:
            raise Exception("No autocomplete suggestions appeared.")

        print(f"✅ Clicking first suggestion: {suggestions[0].text}")
        suggestions[0].click()

        print("⏳ Waiting for bookViewer0 to load...")
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "bookViewer0"))
        )
        time.sleep(3)

        container = driver.find_element(By.ID, "bookViewer0")
        print("📊 DOM loaded. Extracting...")

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


extract_true_dom_snapshot(ticker="SPY")