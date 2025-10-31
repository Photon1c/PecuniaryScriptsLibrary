"""
Deterministic eval harness for agent discussions.
Produces diffable JSON results, no UI/HTML.
"""
import json
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import the discussion runner
sys.path.insert(0, str(Path(__file__).parent.parent))
from discuss_v12 import run_discussion_v1_2, AgentConfig
from adapters.agent_backend import AgentBackend


def load_fixture(fixture_path: Path) -> Dict[str, Any]:
    """Load a fixture JSON file."""
    with fixture_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def agents_from_fixture(fixture: Dict[str, Any], default_model: str = "gpt-4o-mini") -> List[AgentConfig]:
    """Convert fixture agent configs to AgentConfig objects."""
    agents = []
    for cfg in fixture.get("agents", []):
        agents.append(AgentConfig(
            name=cfg["name"],
            role=cfg["role"],
            model_id=cfg.get("model_id", default_model),
            domain=cfg.get("domain"),
            temperature=cfg.get("temperature", 0.2),
            max_tokens=cfg.get("max_tokens", 800),
        ))
    return agents


class MockBackend(AgentBackend):
    """Mock backend for --offline mode."""
    def __init__(self):
        self.fake_content = "Mock agent response for offline testing."
        self.fake_tokens = 150
    
    def complete(self, model_id, messages, temperature, max_tokens, timeout_s, seed=None, tools=None):
        time.sleep(0.1)  # Simulate latency
        return {
            "content": self.fake_content,
            "usage": {"completion_tokens": self.fake_tokens}
        }
    
    def moderate_input(self, payload):
        from adapters.prompts import ModerationDecision
        return ModerationDecision.allow
    
    def moderate_turn(self, payload):
        from adapters.prompts import ModerationDecision
        return ModerationDecision.allow


def check_backend_vision_support(backend: AgentBackend, images: List[str], backend_name: str) -> bool:
    """Check if backend supports vision (has images in messages)."""
    if not images:
        return True  # No images needed
    
    # Known vision-capable backends
    vision_capable = {"openai", "oai", "gpt", "anthropic", "google"}
    if backend_name.lower() in vision_capable:
        return True
    
    # For unknown backends, assume they don't support vision unless proven
    # (This is conservative - better to skip than fail)
    return False


def parse_transcript_for_metrics(transcript_path: Path) -> Dict[str, Any]:
    """Parse transcript JSONL to extract metrics."""
    if not transcript_path.exists():
        return {"turns": 0, "rounds": set(), "agents": set(), "latencies": [], "token_usages": []}
    
    rounds = set()
    agents = set()
    latencies = []
    token_usages = []
    turn_count = 0
    
    with transcript_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if "round" in obj:
                    if isinstance(obj["round"], int):
                        rounds.add(obj["round"])
                if "turn" in obj and isinstance(obj["turn"], dict):
                    turn_count += 1
                    if "name" in obj["turn"]:
                        agents.add(obj["turn"]["name"])
                # Look for latency logs
                if obj.get("type") == "latency":
                    if "elapsed" in obj:
                        latencies.append(obj["elapsed"])
                    if "used_tokens" in obj:
                        token_usages.append(obj["used_tokens"])
            except json.JSONDecodeError:
                continue
    
    return {
        "turns": turn_count,
        "rounds": sorted(rounds),
        "agents": sorted(agents),
        "latencies": latencies,
        "token_usages": token_usages,
    }


def calculate_percentiles(values: List[float], percentiles: List[int] = [50, 95]) -> Dict[str, float]:
    """Calculate percentiles from a list of values."""
    if not values:
        return {f"p{p}": None for p in percentiles}
    sorted_vals = sorted(values)
    result = {}
    for p in percentiles:
        idx = int((p / 100.0) * (len(sorted_vals) - 1))
        result[f"p{p}"] = round(sorted_vals[idx], 2)
    return result


def run_eval(
    fixture_path: Path,
    backend_name: str,
    rounds: int,
    seed: int,
    offline: bool,
    save: bool,
    results_dir: Path,
) -> Dict[str, Any]:
    """Run a single fixture evaluation."""
    fixture = load_fixture(fixture_path)
    fixture_name = fixture.get("name") or fixture_path.stem
    
    # Override backend if offline
    original_get_backend = None
    if offline:
        from discuss_v12 import get_backend
        # Replace backend with mock
        original_get_backend = get_backend
        mock = MockBackend()
        # Monkey-patch for this run
        import discuss_v12
        discuss_v12.get_backend = lambda _: mock
        backend_name_display = "mock"
    else:
        backend_name_display = backend_name
    
    # Check vision support
    images = fixture.get("images", [])
    if images and offline:
        vision_supported = True  # Mock always supports it
    elif images:
        from adapters.agent_backend import get_backend
        backend = get_backend(backend_name)
        vision_supported = check_backend_vision_support(backend, images, backend_name)
        if not vision_supported:
            return {
                "fixture": fixture_name,
                "status": "skipped",
                "reason": "backend_does_not_support_vision",
                "backend": backend_name_display,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
    
    # Prepare agents
    agents = agents_from_fixture(fixture)
    agent_names = [a.name for a in agents]
    
    # Run discussion
    start_time = time.time()
    try:
        result = run_discussion_v1_2(
            images=images,
            agents=agents,
            backend_name=backend_name if not offline else "openai",  # Will use mock
            topic_hint=fixture.get("topic_hint", ""),
            rounds=rounds,
            seed=seed,
            soft_guardrails=True,
        )
        elapsed_total = time.time() - start_time
        
        # Parse metrics from transcript
        transcript_path = Path(result.get("transcript", ""))
        metrics = parse_transcript_for_metrics(transcript_path)
        
        # Determine status (use result status if available, otherwise infer)
        status = result.get("status", "unknown")
        if status == "unknown":
            expected_rounds = rounds
            actual_rounds = result.get("rounds_completed") or len(metrics["rounds"])
            if actual_rounds < expected_rounds:
                status = "partial"  # Budget exhausted early
            else:
                status = "ok"
        
        # Calculate token usage
        token_usages = metrics.get("token_usages", [])
        total_completion_tokens = sum(token_usages) if token_usages else None
        
        # Calculate latency percentiles
        latencies = metrics.get("latencies", [])
        if not latencies and elapsed_total > 0:
            # Fallback: use total time as single latency point
            latencies = [elapsed_total / max(metrics.get("turns", 1), 1)]
        latency_percentiles = calculate_percentiles(latencies)
        
        # Build result
        # Get rounds_completed from result if available (more accurate)
        rounds_completed = result.get("rounds_completed") or len(metrics["rounds"])
        
        eval_result = {
            "fixture": fixture_name,
            "status": status,
            "backend": backend_name_display,
            "rounds_completed": rounds_completed,
            "agents": agent_names,
            "token_usage": {
                "total_completion": total_completion_tokens,
            },
            "latency": latency_percentiles,
            "transcript": str(transcript_path.relative_to(Path.cwd())) if transcript_path.exists() else None,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        
        # Restore original backend if offline
        if offline:
            import discuss_v12
            discuss_v12.get_backend = original_get_backend
        
        # Save if requested
        if save:
            timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            result_file = results_dir / f"{fixture_name}_{timestamp_str}.json"
            results_dir.mkdir(parents=True, exist_ok=True)
            with result_file.open("w", encoding="utf-8") as f:
                json.dump(eval_result, f, indent=2)
            eval_result["_saved_to"] = str(result_file)
        
        return eval_result
        
    except Exception as e:
        return {
            "fixture": fixture_name,
            "status": "error",
            "error": str(e),
            "backend": backend_name_display,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }


def main():
    parser = argparse.ArgumentParser(description="Deterministic agent discussion eval harness")
    parser.add_argument("--preset", help="Preset name (for fixture selection)")
    parser.add_argument("--fixture", help="Specific fixture file name (e.g., markets_1)")
    parser.add_argument("--rounds", type=int, default=2, help="Number of rounds")
    parser.add_argument("--backend", default="openai", help="Backend name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--offline", action="store_true", help="Use mock backend (no API calls)")
    parser.add_argument("--save", action="store_true", help="Save results to evals/results/")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle seed per fixture (default: fixed)")
    args = parser.parse_args()
    
    # Locate fixtures
    evals_dir = Path(__file__).parent
    fixtures_dir = evals_dir / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = evals_dir / "results"
    
    # Find fixture(s)
    if args.fixture:
        fixture_files = list(fixtures_dir.glob(f"{args.fixture}.json"))
        if not fixture_files:
            print(f"Fixture '{args.fixture}' not found in {fixtures_dir}")
            sys.exit(1)
    elif args.preset:
        # Find fixtures matching preset
        fixture_files = list(fixtures_dir.glob(f"{args.preset}_*.json"))
        if not fixture_files:
            print(f"No fixtures found for preset '{args.preset}' in {fixtures_dir}")
            sys.exit(1)
    else:
        # Run all fixtures
        fixture_files = sorted(fixtures_dir.glob("*.json"))
        if not fixture_files:
            print(f"No fixtures found in {fixtures_dir}")
            sys.exit(1)
    
    # Run evals
    results = []
    seed = args.seed
    for fixture_path in fixture_files:
        if args.shuffle:
            import random
            seed = random.randint(0, 2**31 - 1)
        
        result = run_eval(
            fixture_path=fixture_path,
            backend_name=args.backend,
            rounds=args.rounds,
            seed=seed,
            offline=args.offline,
            save=args.save,
            results_dir=results_dir,
        )
        results.append(result)
    
    # Output JSON (deterministic, diffable)
    print(json.dumps(results, indent=2))
    
    # Exit code based on results
    if any(r.get("status") in ("error", "skipped") for r in results):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

