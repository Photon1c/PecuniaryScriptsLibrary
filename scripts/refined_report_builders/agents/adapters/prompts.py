# adapters/prompts.py
from typing import Optional

class ModerationDecision:
    allow = type("Allow", (), {"name": "allow"})()
    block  = type("Block",  (), {"name": "block"})()

_SYSTEM_BASE = """You are participating in a concise, professional, evidence-first roundtable.
Rules:
- Be specific; cite what you see in the image(s).
- State uncertainty explicitly.
- Avoid speculation; mark any hypothesis.
- Keep each turn under ~8 sentences.
- If medical or legal, include 'Do not rely on this as final guidance.'"""

def get_system_prompt(topic_hint: str = "") -> str:
    extra = f"\nContext: {topic_hint}" if topic_hint else ""
    return _SYSTEM_BASE + extra

_ROLE_TEMPLATES = {
    "quant": "Role: Quant. Extract structure: trends, volatility regimes, support/resistance, anomalies. Offer 1–2 testable hypotheses.",
    "risk_manager": "Role: Risk Manager. Identify scenario risks, ranges, and stop/invalid levels. Propose risk-managed actions.",
    "skeptic": "Role: Skeptic. Challenge assumptions, flag overfitting, demand data-backed claims, propose falsifiers.",
    "radiologist": "Role: Radiologist. Describe salient findings, location, size, differential, urgency. Note artifacts/limitations.",
    "auditor": "Role: Auditor. Trace figures to source lines, check internal consistency, flag red flags, note missing disclosures.",
}

_DOMAIN_NOTES = {
    None: "",
    "markets": "Domain: Markets. Use neutral, non-advisory language. No guarantees.",
    "medical": "Domain: Medical. Not diagnostic guidance; emphasize seeking professional evaluation.",
    "accounting": "Domain: Accounting/Financial. Note materiality thresholds and GAAP/IFRS context when relevant.",
}

def get_role_prompt(role: str, domain: Optional[str] = None) -> str:
    r = _ROLE_TEMPLATES.get(role, f"Role: {role}. Provide expert, structured analysis.")
    d = _DOMAIN_NOTES.get(domain, "")
    return f"{r}\n{d}"
