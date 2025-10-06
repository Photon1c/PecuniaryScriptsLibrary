"""
CBOE Direct Book Viewer Data Collector v7
=========================================

Purpose
- Collects real-time DOM (Depth of Market) data directly from the CBOE book viewer
  using the per-symbol URL format for faster, more reliable access.

What it does
- Navigates to `https://www.cboe.com/us/equities/market_statistics/book/{symbol}/`
- Extracts top-of-book bid/ask levels and recent trade info
- Streams rows to a CSV with timestamps for later analysis/visualization

Inputs/Outputs
- Inputs: Ticker symbol (prompted), polling interval, number of cycles
- Output: CSV file `dom_data_{TICKER}_{YYYYMMDD_HHMMSS}.csv`

Usage
- Run directly: `python DOM_analyzer_v7.py`
- Follow prompts for symbol and settings

Dependencies
- Selenium, webdriver-manager

Notes
- Designed for quick/direct access; if the page structure changes, selectors may need updates.
"""
import csv
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

print("Writable:", os.access('.', os.W_OK))

def listen_and_log_dom_direct(ticker="AAPL", poll_interval=5, max_cycles=60, output_file="dom_ticks_direct.csv"):
    """
    Direct access to CBOE book viewer using the specific URL format.
    More efficient than navigating through the main page.
    """
    # Construct the direct URL for the ticker
    url = f"https://www.cboe.com/us/equities/market_statistics/book/{ticker.lower()}/"
    
    print(f"🌐 Accessing direct book viewer URL: {url}")
    
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--log-level=3")
    # Add additional options for better stability
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    # Execute script to remove webdriver property
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    try:
        # Navigate directly to the book viewer page
        driver.get(url)
        print(f"📊 Loading book viewer for {ticker.upper()}...")
        
        # Wait for page to load
        time.sleep(5)
        
        # Wait for the book viewer to be present
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME, "book-viewer"))
        )
        
        print(f"✅ Book viewer loaded successfully for {ticker.upper()}")
        
        # Wait for data to populate
        WebDriverWait(driver, 10).until(
            lambda d: len(d.find_elements(By.CLASS_NAME, "book-viewer__bid-price")) > 0 or
                     len(d.find_elements(By.CLASS_NAME, "book-viewer__ask-price")) > 0
        )
        
        # Create or append to CSV file
        file_exists = os.path.exists(output_file)
        with open(output_file, mode="a", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header only if file is new
            if not file_exists:
                writer.writerow([
                    "timestamp", "symbol", "bid_price", "bid_size",
                    "ask_price", "ask_size", "trade_price", "trade_size",
                    "last_updated"
                ])
                print(f"📝 Created new data file: {output_file}")
            
            print(f"🔄 Starting data collection for {ticker.upper()}...")
            print(f"⏱️  Polling every {poll_interval} seconds for {max_cycles} cycles")
            
            for cycle in range(max_cycles):
                try:
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Find bid data
                    bid_prices = driver.find_elements(By.CLASS_NAME, "book-viewer__bid-price")
                    bid_sizes = driver.find_elements(By.CLASS_NAME, "book-viewer__bid-shares")
                    
                    # Find ask data
                    ask_prices = driver.find_elements(By.CLASS_NAME, "book-viewer__ask-price")
                    ask_sizes = driver.find_elements(By.CLASS_NAME, "book-viewer__ask-shares")
                    
                    # Find trade data
                    trade_prices = driver.find_elements(By.CLASS_NAME, "book-viewer__trades-price")
                    trade_sizes = driver.find_elements(By.CLASS_NAME, "book-viewer__trades-shares")
                    
                    # Get last updated timestamp if available
                    last_updated = ""
                    try:
                        last_updated_elem = driver.find_element(By.CLASS_NAME, "last-updated")
                        last_updated = last_updated_elem.text.strip()
                    except:
                        pass
                    
                    # Extract first level data (top of book)
                    bid_price = bid_prices[0].text.strip() if bid_prices else ""
                    bid_size = bid_sizes[0].text.strip() if bid_sizes else ""
                    ask_price = ask_prices[0].text.strip() if ask_prices else ""
                    ask_size = ask_sizes[0].text.strip() if ask_sizes else ""
                    trade_price = trade_prices[0].text.strip() if trade_prices else ""
                    trade_size = trade_sizes[0].text.strip() if trade_sizes else ""
                    
                    row = [
                        now,
                        ticker.upper(),
                        bid_price,
                        bid_size,
                        ask_price,
                        ask_size,
                        trade_price,
                        trade_size,
                        last_updated
                    ]
                    
                    # Log the data
                    print(f"📈 Cycle {cycle + 1}/{max_cycles}: {ticker.upper()} - Bid: {bid_price} ({bid_size}) | Ask: {ask_price} ({ask_size}) | Trade: {trade_price} ({trade_size})")
                    
                    # Write to CSV
                    writer.writerow(row)
                    f.flush()  # Ensure data is written immediately
                    
                    # Wait before next poll
                    time.sleep(poll_interval)
                    
                except Exception as e:
                    print(f"⚠️ Error in cycle {cycle + 1}: {e}")
                    time.sleep(poll_interval)
                    continue
                    
        print(f"✅ Data collection completed. Total cycles: {max_cycles}")
        print(f"📁 Data saved to: {output_file}")

    except Exception as e:
        print(f"❌ Error accessing book viewer: {e}")
        print("💡 Possible issues:")
        print("   - Ticker symbol not found on CBOE")
        print("   - Network connectivity issues")
        print("   - Page structure may have changed")

    finally:
        driver.quit()
        print("🔚 Browser session closed")

def get_user_ticker():
    """Get ticker symbol from user input."""
    while True:
        ticker = input("Enter ticker symbol (e.g., AAPL, SPY, TSLA): ").strip().upper()
        if ticker and len(ticker) <= 5:  # Basic validation
            return ticker
        else:
            print("❌ Please enter a valid ticker symbol (1-5 characters)")

def get_polling_settings():
    """Get polling settings from user."""
    try:
        poll_interval = int(input("Enter polling interval in seconds (default 5): ") or "5")
        max_cycles = int(input("Enter number of polling cycles (default 60): ") or "60")
        return poll_interval, max_cycles
    except ValueError:
        print("⚠️ Using default values due to invalid input")
        return 5, 60

if __name__ == "__main__":
    print("🚀 CBOE Direct Book Viewer Data Collector v7")
    print("=" * 50)
    
    # Get user input
    ticker = get_user_ticker()
    poll_interval, max_cycles = get_polling_settings()
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"dom_data_{ticker}_{timestamp}.csv"
    
    print(f"\n🎯 Configuration:")
    print(f"   Ticker: {ticker}")
    print(f"   Poll Interval: {poll_interval} seconds")
    print(f"   Max Cycles: {max_cycles}")
    print(f"   Output File: {output_file}")
    print(f"\n⏳ Starting data collection...")
    
    # Start data collection
    listen_and_log_dom_direct(
        ticker=ticker,
        poll_interval=poll_interval,
        max_cycles=max_cycles,
        output_file=output_file
    ) 