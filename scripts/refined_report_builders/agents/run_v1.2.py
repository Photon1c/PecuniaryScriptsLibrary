from discuss_v12 import run_discussion_v1_2
from agent_loader import load_preset
from dotenv import load_dotenv

load_dotenv()

# Load agents from preset (default: "markets")
# You can change the preset name here or use the CLI: python discuss_v12.py --preset minimal
preset_name = "markets"
preset_config = load_preset(preset_name)

result = run_discussion_v1_2(
    images=[
        "https://www.barchart.com/etfs-funds/quotes/SPY/gamma-exposure",
        "https://www.barchart.com/etfs-funds/quotes/SPY/volatility-charts"
    ],
    agents=preset_config["agents"],
    backend_name="openai",
    topic_hint=preset_config["topic_hint"] or "SPY dealer gamma vs vol regime for Monday open",
    rounds=2,
)
print(result["transcript"])
