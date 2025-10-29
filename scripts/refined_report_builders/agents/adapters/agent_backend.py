"" *Not an entry file, used as support
""
# adapters/agent_backend.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
import os, time

class TimeoutError(Exception): ...

class AgentBackend:
    def complete(
        self,
        model_id: str,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        timeout_s: int,
        seed: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def moderate_input(self, payload: Dict[str, Any]):
        return ModerationDecision.allow

    def moderate_turn(self, payload: Dict[str, Any]):
        return ModerationDecision.allow

# Simple enum without importing extra libs
class ModerationDecision:
    allow = type("Allow", (), {"name": "allow"})()
    block = type("Block", (), {"name": "block"})()

# --- Example OpenAI backend (replace with your SDK of choice) ---
class OpenAIBackend(AgentBackend):
    def __init__(self):
        # Lazily import to avoid hard dependency
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def complete(self, model_id, messages, temperature, max_tokens, timeout_s, seed=None, tools=None):
        # Translate messages into OpenAI format (supports image urls or file paths).
        # Only text+image; keep minimal.
        oai_msgs = []
        for m in messages:
            role = "assistant" if m["role"] == "assistant" else m["role"]
            content_items = []
            c = m["content"]
            if isinstance(c, list):
                for part in c:
                    if part.get("type") == "text":
                        content_items.append({"type": "text", "text": part["text"]})
                    elif part.get("type") == "image":
                        # If local path, you can convert to data URL or upload; here we just pass the path/URL
                        content_items.append({"type": "input_image", "image_url": part["path_or_url"]})
            else:
                content_items.append({"type": "text", "text": str(c)})
            oai_msgs.append({"role": role, "content": content_items})

        start = time.time()
        try:
            rsp = self.client.chat.completions.create(
                model=model_id,
                messages=oai_msgs,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                timeout=timeout_s,
            )
        except Exception as e:
            if "timeout" in str(e).lower():
                raise TimeoutError(str(e))
            raise

        content = rsp.choices[0].message.content
        usage = getattr(rsp, "usage", None) or {}
        return {"content": content, "usage": {"completion_tokens": usage.get("completion_tokens", max_tokens//2)}}

def get_backend(name: str) -> AgentBackend:
    name = (name or "").lower()
    if name in ("openai", "oai", "gpt"):
        return OpenAIBackend()
    # TODO: add AnthropicBackend, GoogleBackend, QwenBackend, DeepSeekBackend, xAIBackend
    # Each should implement .complete(...) with the same signature.
    raise ValueError(f"Unsupported backend: {name}")
