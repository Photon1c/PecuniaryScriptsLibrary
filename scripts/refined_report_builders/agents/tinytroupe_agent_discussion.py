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

# Initialize agents
fred = TinyPerson("fred")
fred.define("age", 75)
fred.define("occupation", "Baker, Mechanic, Accountant")
fred.define("personality", {"traits": [
    "Patient and analytical.",
    "Enjoys explaining solutions to problems with numbers.",
    "Friendly and a good active listener."
]})

tiffany = TinyPerson("tiffany")
tiffany.define("age", 21)
tiffany.define("occupation", "Model, Chemist, Gymnast")
tiffany.define("personality", {"traits": [
    "Curious and analytical.",
    "Loves cooking healthy food.",
    "Very playful and energetic."
]})

# Create world
environment = TinyWorld("Trading Room", [fred, tiffany])
environment.make_everyone_accessible()

# Function to query OpenAI's GPT-4o
def get_gpt_response(prompt):
    completions = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a stock market expert analyzing candlestick charts."},
            {"role": "user", "content": prompt}  # Just text now
        ]
    )
    return completions.choices[0].message.content.strip()

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

# Function to analyze chart using GPT-4o Vision
def analyze_chart_with_gpt4o(image_url):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Analyze the stock chart image and provide insights."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Describe trends, volatility, and possible price action. Study the wick and candle body sizes."},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error analyzing chart: {e}"

# Execute stock chart analysis
def analyze_stock_chart(symbol):
    print("\n=== Capturing TradingView Chart ===")
    chart_path = capture_tradingview_chart(symbol)
    if not chart_path:
        return "Chart capture failed."
    
    print("\n=== Uploading Image to ImageKit ===")
    image_url = upload_image_to_imagekit(chart_path)
    if not image_url or "Error" in image_url:
        return "Image upload failed."
    
    print("\n=== AI Analysis ===")
    ai_analysis = analyze_chart_with_gpt4o(image_url)
    print(f"AI Analysis: {ai_analysis}")
    
    print("\n=== Agents Responding ===")
    fred_response = get_gpt_response(f"Fred, provide an expert opinion on the AI's analysis, generate a summary GitHub markdown table formatted containing suggested entry and exit prices for the next three days: {ai_analysis}")
    tiffany_response = get_gpt_response(f"Tiffany, respond to fred's thoughts and work with him to improve his Github markdown table recommending entry and exit ranges for the next three days. Stop when done: {fred_response}")
    
    fred.listen(fred_response)
    tiffany.listen(tiffany_response)
    environment.run(4)
    
    return ai_analysis

# Run for a sample stock
analyze_stock_chart("SPY")
