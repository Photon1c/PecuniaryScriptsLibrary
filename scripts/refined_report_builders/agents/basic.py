# Requires Microsoft Tinytroupe
# Deploys sample conversation between two agents around an SPY chart.
import os
import time
import json
from dotenv import load_dotenv
from openai import OpenAI
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from imagekitio import ImageKit
from PIL import Image
import tinytroupe
from tinytroupe.agent import TinyPerson
from tinytroupe.environment import TinyWorld
from typing import List, Dict, Optional

# + add (near load_dotenv)
HEADLESS = os.getenv("HEADLESS", "1") not in {"0", "false", "False"}

# REPLACE agent setup with a configurable roster and a required 'residence' field
AGENT_PROFILES: List[Dict] = [
    {
        "name": "Fred",
        "age": 75,
        "occupation": "Baker, Mechanic, Accountant",
        "residence": "Trading Room",  # <- prevents KeyError: 'residence'
        "personality": {
            "traits": [
                "Patient and analytical.",
                "Enjoys explaining solutions to problems with numbers.",
                "Friendly and a good active listener."
            ]
        }
    },
    {
        "name": "Greg",
        "age": 45,
        "occupation": "Engineer, Chemist, Gymnast",
        "residence": "Trading Room",  # <- prevents KeyError: 'residence'
        "personality": {
            "traits": [
                "Curious and analytical.",
                "Loves cooking healthy food.",
                "Very focused and productivity centered."
            ]
        }
    }
]




# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
IMAGEKIT_PRIVATE_KEY = os.getenv("IMAGEKIT_PRIVATE_KEY")
IMAGEKIT_PUBLIC_KEY = os.getenv("IMAGEKIT_PUBLIC_KEY")
IMAGEKIT_URL_ENDPOINT = os.getenv("IMAGEKIT_URL_ENDPOINT")
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize ImageKit
imagekit = ImageKit(
    private_key=IMAGEKIT_PRIVATE_KEY,
    public_key=IMAGEKIT_PUBLIC_KEY,
    url_endpoint=IMAGEKIT_URL_ENDPOINT
)

# Define directories
screenshot_dir = "screenshots"
os.makedirs(screenshot_dir, exist_ok=True)
memory_file = "agent_memory.json"



agents: List[TinyPerson] = []
for a in AGENT_PROFILES:
    p = TinyPerson(a["name"])
    # minimal, avoids surprises if TinyPerson validates keys internally
    p.define("age", a["age"])
    p.define("occupation", a["occupation"])
    p.define("residence", a["residence"])   # <-- required by tinytroupe world logic
    p.define("personality", a["personality"])
    agents.append(p)

environment = TinyWorld("Trading Room", agents)
environment.make_everyone_accessible()

# Function to query OpenAI's GPT-4o
def get_gpt_response(prompt):
    completions = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a stock market expert analyzing candlestick charts. Generate a brief summary about the candle wick and body structures. Your analysis will be used by other agents to refine a medium term trading strategy with minimal risk exposure."},
            {"role": "user", "content": prompt}  # Just text now
        ]
    )
    return completions.choices[0].message.content.strip()

def _new_chrome():
    options = Options()
    options.add_argument("--window-size=1920x1080")
    if HEADLESS:
        options.add_argument("--headless=new")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)

def capture_page_screenshot(driver, url: str, out_name: str) -> Optional[str]:
    try:
        driver.get(url)
        time.sleep(5)  # simple wait; replace with explicit waits if you like
        path = os.path.join(screenshot_dir, out_name)
        driver.save_screenshot(path)
        return path if os.path.exists(path) else None
    except Exception as e:
        print(f"[screenshot] {url} -> {e}")
        return None


# Function to capture TradingView chart
def capture_tradingview_chart(symbol):
    try:
        options = Options()
        options.add_argument("--window-size=1920x1080")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.get(f"https://www.tradingview.com/chart/?symbol={symbol}")
        time.sleep(5)
        screenshot_path = os.path.join(screenshot_dir, f"{symbol}_chart.png")
        driver.save_screenshot(screenshot_path)
        driver.quit()
        return screenshot_path
    except Exception as e:
        print(f"Error capturing chart: {e}")
        return None

# Function to upload image to ImageKit
def upload_image_to_imagekit(image_path):
    try:
        if not os.path.exists(image_path):
            return "Error: Image file not found."
        
        print(f"Uploading {image_path} to ImageKit...")
        with open(image_path, "rb") as img_file:
            upload = imagekit.upload(
                file=img_file,
                file_name=os.path.basename(image_path)
            )
        return upload.response_metadata.raw['url']
    except Exception as e:
        print(f"Error uploading image: {e}")
        return None

def upload_many_to_imagekit(paths: List[str]) -> List[str]:
    urls: List[str] = []
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        try:
            with open(p, "rb") as f:
                up = imagekit.upload(file=f, file_name=os.path.basename(p))
            urls.append(up.response_metadata.raw["url"])
        except Exception as e:
            print(f"[imagekit] {p} -> {e}")
    return urls


# Function to analyze chart using GPT-4o Vision
def analyze_charts_with_gpt4o(image_urls: List[str]) -> str:
    if not image_urls:
        return "No images available for analysis."
    try:
        # Build a multi-part message with several images
        user_content: List[Dict] = [
            {"type": "text", "text": (
                "You will analyze several related screenshots for SPY:\n"
                "1) TradingView price chart (context)\n"
                "2) Barchart Gamma Exposure (dealers' gamma risk)\n"
                "3) Barchart Max Pain chart (OI-based pain point)\n"
                "4) Barchart Volatility charts (IV trends)\n\n"
                "Task: synthesize trends, wick/body behavior, IV regime, gamma posture, "
                "and propose entry/exit **ranges** for the next 3 sessions with caveats. "
                "Return:\n- A brief narrative\n- A GitHub Markdown table of entries/exits "
                "(low/likely/high) for each day\n- Key risks to invalidate the plan"
            )}
        ]
        for u in image_urls:
            user_content.append({"type": "image_url", "image_url": {"url": u}})

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Analyze stock context, gamma, max-pain, and IV coherently."},
                {"role": "user", "content": user_content}
            ]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error analyzing charts: {e}"


# Execute stock chart analysis
def analyze_stock_chart(symbol: str):
    print("\n=== Capturing pages ===")
    urls = {
        "tradingview": f"https://www.tradingview.com/chart/?symbol={symbol}",
        "gex": "https://www.barchart.com/etfs-funds/quotes/SPY/gamma-exposure",
        "maxpain": "https://www.barchart.com/etfs-funds/quotes/SPY/max-pain-chart",
        "vol": "https://www.barchart.com/etfs-funds/quotes/SPY/volatility-charts",
    }

    driver = _new_chrome()
    shots: List[str] = []
    try:
        shots.append(capture_page_screenshot(driver, urls["tradingview"], f"{symbol}_price.png"))
        shots.append(capture_page_screenshot(driver, urls["gex"],        f"{symbol}_gex.png"))
        shots.append(capture_page_screenshot(driver, urls["maxpain"],    f"{symbol}_maxpain.png"))
        shots.append(capture_page_screenshot(driver, urls["vol"],        f"{symbol}_vol.png"))
    finally:
        driver.quit()

    shots = [p for p in shots if p]  # drop Nones
    if not shots:
        return "Screenshot capture failed."

    print("\n=== Uploading to ImageKit ===")
    image_urls = upload_many_to_imagekit(shots)
    if not image_urls:
        return "Image upload failed."

    print("\n=== AI Synthesis (vision over multiple images) ===")
    ai_summary = analyze_charts_with_gpt4o(image_urls)
    print(ai_summary)

    print("\n=== Agents Responding ===")
    # Fred drafts, Tiffany critiques/refines. Agents read the multi-image synthesis.
    fred_resp = get_gpt_response(
        "Fred: Using the synthesis below, write a succinct GitHub Markdown table of entry/exit ranges "
        "for the next 3 sessions. Include a one-paragraph rationale referencing price context, GEX posture, "
        "max-pain pull, and IV regime. Keep it practical.\n\n" + ai_summary
    )
    tiffany_resp = get_gpt_response(
        "Tiffany: Improve Fred's table if needed, add brief color on risks (IV crush/expansion, "
        "gamma flip proximity, macro catalysts). Return final table only if it improves clarity. "
        "Stop when done.\n\n" + fred_resp
    )

    # Feed the agents and run a short scene
    for a, msg in [(agents[0], fred_resp), (agents[1], tiffany_resp)]:
        a.listen(msg)

    try:
        environment.run(4)  # short round; tweakable
    except Exception as e:
        print(f"[Trading Room] run error: {e}")

    return ai_summary

    
    return ai_analysis

# Run for a sample stock
analyze_stock_chart("SPY")