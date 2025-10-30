from discuss_v12 import run_discussion_v1_2, AgentConfig
from adapters.agent_backend import AgentBackend, get_backend
from adapters.prompts import get_role_prompt, get_system_prompt, ModerationDecision
from dotenv import load_dotenv

load_dotenv()
# Example (keep out of prod file if you like):
agents = [
    AgentConfig(name="Ava", role="quant", model_id="gpt-4o-mini", domain="markets"),
    AgentConfig(name="Blake", role="risk_manager", model_id="gpt-4o-mini", domain="markets"),
    AgentConfig(name="Casey", role="skeptic", model_id="gpt-4o-mini", domain="markets"),
]
result = run_discussion_v1_2(
    images=[
        "https://www.barchart.com/etfs-funds/quotes/SPY/gamma-exposure",
        "https://www.barchart.com/etfs-funds/quotes/SPY/volatility-charts"
    ],
    agents=agents,
    backend_name="openai",
    topic_hint="SPY dealer gamma vs vol regime for Monday open",
    rounds=2,
)
print(result["transcript"])
