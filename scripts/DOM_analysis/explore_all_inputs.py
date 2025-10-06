"""
CBOE Input Element Explorer
===========================

Purpose
- Debugging utility to enumerate and log details about all `<input>` elements
  on the CBOE equities page.

What it does
- Prints visibility, location, size, partial outerHTML, and parent text for
  each input element.

Usage
- `python explore_all_inputs.py`

Dependencies
- Selenium, webdriver-manager
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

from selenium.webdriver.common.action_chains import ActionChains

def explore_all_inputs():
    url = "https://www.cboe.com/market_data_services/us/equities/#"
    options = Options()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)

    try:
        time.sleep(5)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(4)

        inputs = driver.find_elements(By.TAG_NAME, "input")
        print(f"🔍 Found {len(inputs)} <input> elements on page.\n")

        for i, inp in enumerate(inputs):
            try:
                box = inp.get_attribute("outerHTML")
                parent = inp.find_element(By.XPATH, "./..")
                parent_text = parent.text.strip().replace("\n", " ")
                is_displayed = inp.is_displayed()
                loc = inp.location
                size = inp.size
                print(f"[{i}] Visible: {is_displayed}, Loc: {loc}, Size: {size}")
                print(f"     OuterHTML: {box[:100]}...")
                print(f"     Parent Text: {parent_text[:100]}\n")
            except Exception as e:
                print(f"[{i}] Skipped due to error: {e}")

    except Exception as e:
        print("❌ Error during input scan:", e)
    finally:
        driver.quit()

explore_all_inputs()
