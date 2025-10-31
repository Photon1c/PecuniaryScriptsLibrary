# Requires Microsoft Tinytroupe
# Deploys sample conversation between two agents around an SPY chart.
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
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Iterable
import json, time, random, os, pathlib, sys

from adapters.agent_backend import AgentBackend, get_backend
from adapters.prompts import get_role_prompt, get_system_prompt, ModerationDecision
from agent_loader import load_preset, load_agents_from_preset

# Settings validation
def validate_settings() -> Dict[str, Any]:
    """Validate settings against schema.json at startup."""
    schema_path = pathlib.Path(__file__).parent / "settings.schema.json"
    if not schema_path.exists():
        print(f"[WARN] settings.schema.json not found at {schema_path}")
        return {}
    
    try:
        import jsonschema
    except ImportError:
        print("[WARN] jsonschema not installed; skipping validation")
        return {}
    
    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    
    # Build settings from env + defaults
    settings = {
        "backend_name": os.getenv("TT_BACKEND", "openai"),
        "model_id": os.getenv("TT_MODEL", "gpt-4o-mini"),
        "rounds": int(os.getenv("TT_ROUNDS", "3")),
        "seed": int(os.getenv("TT_SEED", "42")),
        "soft_guardrails": os.getenv("TT_GUARDRAILS", "1") not in {"0", "false", "False"},
        "turn_timeout_s": int(os.getenv("TT_TIMEOUT", "40")),
        "global_max_tokens": int(os.getenv("TT_MAX_TOKENS", "3000")),
        "headless": os.getenv("HEADLESS", "1") not in {"0", "false", "False"},
    }
    
    try:
        jsonschema.validate(instance=settings, schema=schema)
        print("[OK] Settings validated against schema")
        return settings
    except jsonschema.ValidationError as e:
        print(f"[ERROR] Settings validation failed: {e.message}")
        raise

DEFAULT_SEED = int(os.environ.get("TT_SEED", "42"))
TRANSCRIPTS_DIR = pathlib.Path(os.environ.get("TT_TRANSCRIPTS_DIR", "./transcripts"))
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

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

# NEW: v1.2 config
@dataclass
class AgentConfig:
    name: str
    role: str  # e.g., "quant", "radiologist", "auditor"
    model_id: str  # e.g., "qwen2.5-vl", "gpt-4.1-mini", "claude-3.7-sonnet"
    temperature: float = 0.2
    max_tokens: int = 800
    tools: Optional[List[Dict[str, Any]]] = None
    # Optional per-agent constraints
    domain: Optional[str] = None  # "markets" | "medical" | "accounting"
    speak_once: bool = False      # for short rounds

# NEW: Orchestrator for multi-image multi-agent talk
class MultiImageDiscussionOrchestrator:
    """
    v1.2: Orchestrates agent discussion across one or more images with role-specialized prompts.
    Keeps state, transcripts, and applies moderation/guardrails before each turn.
    """
    def __init__(
        self,
        backend: AgentBackend,
        agents: List[AgentConfig],
        images: List[str],
        topic_hint: str = "",
        seed: int = DEFAULT_SEED,
        round_count: int = 3,
        turn_timeout_s: int = 40,
        global_max_tokens: int = 3000,
        soft_guardrails: bool = True,
        transcript_tag: str = "tinytroupe_v1_2",
    ):
        self.backend = backend
        self.agents = agents
        self.images = images
        self.topic_hint = topic_hint.strip()
        self.rng = random.Random(seed)
        self.round_count = round_count
        self.turn_timeout_s = turn_timeout_s
        self.global_max_tokens = global_max_tokens
        self.soft_guardrails = soft_guardrails
        self.transcript_path = TRANSCRIPTS_DIR / f"{int(time.time())}_{transcript_tag}.jsonl"
        self._token_budget = global_max_tokens
        self._rounds_completed = 0
        self._budget_exhausted = False

    # NEW: core run
    def run(self) -> Dict[str, Any]:
        # Ensure images are proper image URLs (upload local files, screenshot web pages)
        try:
            self.images = self._ensure_image_urls(self.images)
        except Exception as _e:
            # Non-fatal: fall back to originals; downstream may still work if URLs are already images
            pass

        conversation: List[Dict[str, Any]] = []
        # System message once (deterministic)
        system_msg = get_system_prompt(self.topic_hint)
        conversation.append({"role": "system", "content": system_msg})

        # Pre-flight moderation on inputs (filenames/urls only)
        if self.soft_guardrails:
            mod = self.backend.moderate_input({"images": self.images, "topic": self.topic_hint})
            self._log({"type": "moderation_input", "decision": mod.name})
            if mod is ModerationDecision.block:
                return {"status": "blocked", "reason": "Input moderation failed"}

        # Each round: every agent speaks once
        for r in range(self.round_count):
            round_complete = True
            for cfg in self.agents:
                if self._token_budget <= 0:
                    self._budget_exhausted = True
                    round_complete = False
                    break
                turn = self._agent_turn(cfg=cfg, conversation=conversation)
                if turn:
                    conversation.append(turn)
                    self._append_jsonl({"round": r, "turn": turn})
            if self._budget_exhausted:
                break
            if round_complete:
                self._rounds_completed = r + 1

        # Final: backend summary (only if budget allows)
        summary = None
        if self._token_budget > 120:
            summary = self._final_summary(conversation)
            if summary:
                conversation.append(summary)
                self._append_jsonl({"round": "final", "turn": summary})

        status = "ok"
        if self._budget_exhausted and self._rounds_completed < self.round_count:
            status = "partial"

        return {
            "status": status,
            "conversation": conversation,
            "transcript": str(self.transcript_path),
            "rounds_completed": self._rounds_completed,
            "budget_exhausted": self._budget_exhausted,
        }

    # NEW: single agent turn with guardrails + timeout
    def _agent_turn(self, cfg: AgentConfig, conversation: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        role_prompt = get_role_prompt(cfg.role, domain=cfg.domain)
        images_payload = [{"type": "image", "path_or_url": p} for p in self.images]

        content = [
            {"type": "text", "text": role_prompt},
            {"type": "text", "text": self._short_topic_hint()},
            *images_payload,
        ]

        # Guardrails (per-turn)
        if self.soft_guardrails:
            mod = self.backend.moderate_turn({"agent": cfg.name, "content_meta": [i["type"] for i in content]})
            self._log({"type": "moderation_turn", "agent": cfg.name, "decision": mod.name})
            if mod is ModerationDecision.block:
                return {"role": "assistant", "name": cfg.name, "content": "[Turn skipped by moderation]"}

        # Budgeted call
        budget = min(cfg.max_tokens, self._token_budget)
        start = time.time()
        try:
            msg = self.backend.complete(
                model_id=cfg.model_id,
                messages=conversation + [{"role": "user", "content": content}],
                temperature=cfg.temperature,
                max_tokens=budget,
                timeout_s=self.turn_timeout_s,
                seed=self.rng.randint(0, 2**31 - 1),
                tools=cfg.tools or [],
            )
        except TimeoutError:
            return {"role": "assistant", "name": cfg.name, "content": "[Timed out]"}

        elapsed = time.time() - start
        used = msg.get("usage", {}).get("completion_tokens", budget // 2)
        self._token_budget -= used
        self._log({"type": "latency", "agent": cfg.name, "elapsed": round(elapsed, 2), "used_tokens": used})

        return {"role": "assistant", "name": cfg.name, "content": msg["content"]}

    # NEW: final collation/summary
    def _final_summary(self, conversation: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if self._token_budget <= 120:
            return None
        msg = self.backend.complete(
            model_id=self.agents[0].model_id,
            messages=conversation + [{
                "role": "user",
                "content": [{"type": "text", "text": "Produce a brief, structured consensus summary with: (1) agreements, (2) disagreements, (3) uncertainties, (4) next steps, (5) risk flags, (6) domain-specific cautions."}]
            }],
            temperature=0.2,
            max_tokens=min(600, self._token_budget),
            timeout_s=30,
            seed=DEFAULT_SEED + 1,
        )
        self._token_budget -= msg.get("usage", {}).get("completion_tokens", 300)
        return {"role": "assistant", "name": "moderator", "content": msg["content"]}

    def _short_topic_hint(self) -> str:
        return f"Topic hint: {self.topic_hint}" if self.topic_hint else "Analyze the images carefully before concluding."

    def _append_jsonl(self, obj: Dict[str, Any]) -> None:
        with self.transcript_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _log(self, obj: Dict[str, Any]) -> None:
        # Lightweight console logger without spam
        print(f"[TT v1.2] {obj}")

    # Normalize inputs into vision-friendly image URLs.
    # - Local file paths -> upload to ImageKit
    # - Web pages (HTML) -> screenshot then upload
    # - Direct image URLs/data URLs -> pass through
    def _ensure_image_urls(self, inputs: List[str]) -> List[str]:
        def looks_like_image_url(u: str) -> bool:
            if not isinstance(u, str):
                return False
            if u.startswith("data:image"):
                return True
            if not (u.startswith("http://") or u.startswith("https://")):
                return False
            base = u.split("?", 1)[0].lower()
            return base.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif"))

        # Collect web pages to screenshot in one browser session
        web_pages: List[str] = []
        uploads: List[str] = []
        results: List[str] = []
        for p in inputs:
            try:
                if not isinstance(p, str) or not p:
                    continue
                if looks_like_image_url(p):
                    results.append(p)
                    continue
                if p.startswith("http://") or p.startswith("https://"):
                    web_pages.append(p)
                    continue
                # Local file path
                if os.path.exists(p):
                    up = upload_image_to_imagekit(p)
                    if up and isinstance(up, str):
                        results.append(up)
                    else:
                        # fallback to original path (may not work with API)
                        results.append(p)
                    continue
                # Unknown type, pass through
                results.append(p)
            except Exception:
                results.append(p)

        if web_pages:
            # Single browser to capture all page screenshots
            try:
                driver = _new_chrome()
            except Exception as _e:
                # If browser fails, pass through the pages as-is
                results.extend(web_pages)
                return results
            try:
                for idx, url in enumerate(web_pages):
                    out_name = f"webpage_{int(time.time())}_{idx}.png"
                    shot = capture_page_screenshot(driver, url, out_name)
                    if shot and os.path.exists(shot):
                        up = upload_image_to_imagekit(shot)
                        if up and isinstance(up, str):
                            uploads.append(up)
                        else:
                            uploads.append(url)
                    else:
                        uploads.append(url)
            finally:
                try:
                    driver.quit()
                except Exception:
                    pass

            results.extend(uploads)

        # De-duplicate while preserving order
        seen = set()
        deduped: List[str] = []
        for u in results:
            if u not in seen:
                seen.add(u)
                deduped.append(u)
        return deduped
      
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


def run_discussion_v1_2(
    images: List[str],
    agents: List[AgentConfig],
    backend_name: str = "openai",  # "openai" | "anthropic" | "google" | "qwen" | "deepseek" | "xai"
    topic_hint: str = "",
    rounds: int = 3,
    seed: int = DEFAULT_SEED,
    soft_guardrails: bool = True,
) -> Dict[str, Any]:
    backend = get_backend(backend_name)
    orch = MultiImageDiscussionOrchestrator(
        backend=backend,
        agents=agents,
        images=images,
        topic_hint=topic_hint,
        seed=seed,
        round_count=rounds,
        soft_guardrails=soft_guardrails,
    )
    return orch.run()

if __name__ == "__main__":
    import argparse
    
    # Validate settings at startup
    try:
        validate_settings()
    except Exception as e:
        print(f"[WARN] Settings validation issue: {e}")
        print("[WARN] Continuing anyway...")
    
    p = argparse.ArgumentParser(description="TinyTroupe v1.2 Multi-Image Discussion")
    p.add_argument("--images", nargs="+", help="Paths/URLs to images")
    p.add_argument("--preset", default="markets",
                   help="Use preset agent configuration (e.g., markets, minimal, balanced, critical)")
    p.add_argument("--backend", default="openai")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--topic", default="", help="Short topic hint (overrides preset topic_hint)")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate config and show what would run, but don't execute")
    args = p.parse_args()

    # Load preset configuration
    try:
        preset_config = load_preset(args.preset, default_model=os.getenv("TT_MODEL", "gpt-4o-mini"))
        agents = preset_config["agents"]
        preset_topic = preset_config["topic_hint"]
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Use topic from CLI if provided, otherwise from preset
    topic_hint = args.topic if args.topic else preset_topic
    
    # If using preset but no images provided, use default
    if args.images:
        images = args.images
    else:
        images = [
            "https://www.barchart.com/etfs-funds/quotes/SPY/gamma-exposure",
            "https://www.barchart.com/etfs-funds/quotes/SPY/volatility-charts"
        ]

    if args.dry_run:
        print("[DRY RUN] Configuration:")
        print(f"  Preset: {args.preset}")
        print(f"  Agents: {[a.name for a in agents]}")
        print(f"  Images: {len(images)}")
        print(f"  Backend: {args.backend}")
        print(f"  Rounds: {args.rounds}")
        print(f"  Topic: {topic_hint or '(none)'}")
        print("[DRY RUN] Would execute, but --dry-run flag set.")
        sys.exit(0)

    result = run_discussion_v1_2(
        images=images,
        agents=agents,
        backend_name=args.backend,
        topic_hint=topic_hint,
        rounds=args.rounds,
        seed=args.seed,
    )
    print(json.dumps({"status": result["status"], "transcript": result.get("transcript")}, indent=2))

# Run for a sample stock
# analyze_stock_chart("SPY")