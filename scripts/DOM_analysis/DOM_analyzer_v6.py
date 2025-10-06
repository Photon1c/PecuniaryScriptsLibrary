"""
CBOE DOM Logger v6 (Main Page Workflow)
======================================

Purpose
- Logs DOM (Depth of Market) data from the CBOE market data services main page.

What it does
- Loads the landing page, accepts cookies, scrolls into view
- Robustly clears the search field (clear + Ctrl+A + Delete)
- Enters the desired ticker and streams top-of-book levels to CSV

Inputs/Outputs
- Inputs: `ticker`, `poll_interval`, `max_cycles`
- Output: `dom_ticks.csv`

Usage
- `python DOM_analyzer_v6.py` (edit the example call at bottom or import and call `listen_and_log_dom`)

Dependencies
- Selenium, webdriver-manager

Notes
- This flow is more brittle than v7 direct-URL approach and may require selector updates when the site changes.
"""
import csv
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import os
from selenium.webdriver.common.keys import Keys

print("Writable:", os.access('.', os.W_OK))

def listen_and_log_dom(ticker="AAPL", poll_interval=5, max_cycles=60, output_file="dom_ticks.csv"):
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

        # Accept cookies if needed
        try:
            accept_btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "/html/body/div[2]/div[2]/div/div[1]/div/div[2]/div/button[3]"))
            )
            driver.execute_script("arguments[0].click();", accept_btn)
            time.sleep(2)
        except:
            pass

        # Enter ticker symbol and search
        symbol_input = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "symbol0")))
        
        # Clear any existing text in the search field
        symbol_input.clear()
        time.sleep(1)  # Wait for clear to take effect
        
        # Double-check the field is empty by selecting all and deleting
        symbol_input.send_keys(Keys.CONTROL + "a")  # Select all text
        symbol_input.send_keys(Keys.DELETE)  # Delete selected text
        time.sleep(0.5)  # Brief pause
        
        # Now enter the new ticker
        symbol_input.send_keys(ticker)
        time.sleep(1)  # Wait for input to register
        
        # Click search button
        driver.find_element(By.ID, "symbol-search-0").click()

        # Wait for book to load
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "bookViewer0")))
        container = driver.find_element(By.ID, "bookViewer0")

        WebDriverWait(driver, 10).until(
            lambda d: any(el.text.strip() for el in d.find_elements(By.CLASS_NAME, "book-viewer__bid-price"))
        )

        with open(output_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "symbol", "bid_price", "bid_size",
                "ask_price", "ask_size", "trade_price", "trade_size"
            ])

            for _ in range(max_cycles):
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                bid_prices = container.find_elements(By.CLASS_NAME, "book-viewer__bid-price")
                bid_sizes = container.find_elements(By.CLASS_NAME, "book-viewer__bid-shares")
                ask_prices = container.find_elements(By.CLASS_NAME, "book-viewer__ask-price")
                ask_sizes = container.find_elements(By.CLASS_NAME, "book-viewer__ask-shares")
                trade_prices = container.find_elements(By.CLASS_NAME, "book-viewer__trades-price")
                trade_sizes = container.find_elements(By.CLASS_NAME, "book-viewer__trades-shares")

                row = [
                    now,
                    ticker,
                    bid_prices[0].text if bid_prices else "",
                    bid_sizes[0].text if bid_sizes else "",
                    ask_prices[0].text if ask_prices else "",
                    ask_sizes[0].text if ask_sizes else "",
                    trade_prices[0].text if trade_prices else "",
                    trade_sizes[0].text if trade_sizes else ""
                ]

                print(f"📈 {row}")
                writer.writerow(row)
                time.sleep(poll_interval)

    except Exception as e:
        print("❌ Error:", e)

    finally:
        driver.quit()

# Example usage
listen_and_log_dom(ticker="SPY", poll_interval=2, max_cycles=100)
