"""
Agent loader: decouples agent construction from entrypoints.
Takes validated presets from settings and returns AgentConfig objects.
"""
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path
import os

if TYPE_CHECKING:
    from discuss_v12 import AgentConfig
else:
    # Defer import to avoid circular dependency
    AgentConfig = None


# Built-in presets (fallback if settings.json not found)
_BUILTIN_PRESETS = {
    "markets": {
        "topic_hint": "Market analysis",
        "agents": [
            {"name": "Ava", "role": "quant", "domain": "markets"},
            {"name": "Blake", "role": "risk_manager", "domain": "markets"},
            {"name": "Casey", "role": "skeptic", "domain": "markets"},
        ],
    },
    "minimal": {
        "topic_hint": "Quick analysis",
        "agents": [
            {"name": "Analyst", "role": "quant", "domain": "markets"},
        ],
    },
    "balanced": {
        "topic_hint": "Balanced analysis",
        "agents": [
            {"name": "Quant", "role": "quant", "domain": "markets"},
            {"name": "Risk", "role": "risk_manager", "domain": "markets"},
        ],
    },
    "critical": {
        "topic_hint": "Critical analysis",
        "agents": [
            {"name": "Skeptic1", "role": "skeptic", "domain": "markets", "temperature": 0.1},
            {"name": "Skeptic2", "role": "skeptic", "domain": "markets", "temperature": 0.2},
        ],
    },
}


def _load_settings_if_available() -> Optional[Dict[str, Any]]:
    """Try to load settings.json, return None if not found."""
    try:
        from settings_loader import load_settings
        settings_path = Path(__file__).parent / "settings.json"
        if settings_path.exists():
            return load_settings(str(settings_path))
    except (FileNotFoundError, ImportError, SystemExit):
        pass
    return None


def _agent_dict_to_config(agent_dict: Dict[str, Any], default_model: str, default_temperature: float = 0.2):
    """Convert agent dict from preset to AgentConfig object."""
    # Import here to avoid circular dependency
    from discuss_v12 import AgentConfig
    return AgentConfig(
        name=agent_dict["name"],
        role=agent_dict["role"],
        model_id=agent_dict.get("model_id", default_model),
        domain=agent_dict.get("domain"),
        temperature=agent_dict.get("temperature", default_temperature),
        max_tokens=agent_dict.get("max_tokens", 800),
        tools=agent_dict.get("tools"),
    )


def load_preset(preset_name: str, default_model: Optional[str] = None, default_backend: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a preset configuration (agents + topic_hint).
    
    Args:
        preset_name: Name of preset (e.g., "markets", "minimal")
        default_model: Default model ID (from settings or env, fallback: "gpt-4o-mini")
        default_backend: Default backend (unused here, kept for consistency)
    
    Returns:
        Dict with keys:
            - "agents": List[AgentConfig]
            - "topic_hint": str
    
    Raises:
        ValueError: If preset not found
    """
    # Try to load from settings first
    settings = _load_settings_if_available()
    
    # Get default model from settings or env or fallback
    if default_model is None:
        if settings:
            default_model = settings.get("default_model", "gpt-4o-mini")
        else:
            default_model = os.getenv("TT_MODEL", "gpt-4o-mini")
    
    # Load preset from settings or use built-in
    preset_dict = None
    if settings and "presets" in settings and preset_name in settings["presets"]:
        preset_dict = settings["presets"][preset_name]
    elif preset_name in _BUILTIN_PRESETS:
        preset_dict = _BUILTIN_PRESETS[preset_name]
    else:
        available = list(_BUILTIN_PRESETS.keys())
        if settings and "presets" in settings:
            available.extend(list(settings["presets"].keys()))
        raise ValueError(
            f"Preset '{preset_name}' not found. Available presets: {', '.join(set(available))}"
        )
    
    # Validate preset structure
    if "agents" not in preset_dict:
        raise ValueError(f"Preset '{preset_name}' missing 'agents' field")
    if not isinstance(preset_dict["agents"], list) or len(preset_dict["agents"]) == 0:
        raise ValueError(f"Preset '{preset_name}' has invalid 'agents' (must be non-empty list)")
    
    # Convert agent dicts to AgentConfig objects
    agents = []
    for agent_dict in preset_dict["agents"]:
        if "name" not in agent_dict or "role" not in agent_dict:
            raise ValueError(f"Preset '{preset_name}' has agent missing 'name' or 'role'")
        agents.append(_agent_dict_to_config(agent_dict, default_model))
    
    # Get topic hint
    topic_hint = preset_dict.get("topic_hint", "")
    
    return {
        "agents": agents,
        "topic_hint": topic_hint,
    }


def load_agents_from_preset(preset_name: str, default_model: Optional[str] = None) -> List:
    """
    Convenience function: load just the agents from a preset.
    
    Args:
        preset_name: Name of preset
        default_model: Default model ID (optional)
    
    Returns:
        List of AgentConfig objects
    """
    result = load_preset(preset_name, default_model=default_model)
    return result["agents"]


def get_preset_topic_hint(preset_name: str) -> str:
    """
    Get the topic hint for a preset.
    
    Args:
        preset_name: Name of preset
    
    Returns:
        Topic hint string
    """
    result = load_preset(preset_name)
    return result["topic_hint"]

