"""
One-file evaluation runner for agent discussion fixtures.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Import the discussion runner
sys.path.insert(0, str(Path(__file__).parent.parent))
from discuss_v12 import run_discussion_v1_2, AgentConfig


def load_fixture(fixture_path: Path) -> Dict[str, Any]:
    """Load a fixture JSON file."""
    with fixture_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def agents_from_fixture(fixture: Dict[str, Any]) -> List[AgentConfig]:
    """Convert fixture agent configs to AgentConfig objects."""
    agents = []
    for cfg in fixture.get("agents", []):
        agents.append(AgentConfig(
            name=cfg["name"],
            role=cfg["role"],
            model_id=cfg.get("model_id", "gpt-4o-mini"),
            domain=cfg.get("domain"),
            temperature=cfg.get("temperature", 0.2),
            max_tokens=cfg.get("max_tokens", 800),
        ))
    return agents


def validate_result(result: Dict[str, Any], fixture: Dict[str, Any]) -> tuple:
    """Validate that result meets fixture expectations."""
    errors = []
    
    # Load the transcript to count turns
    transcript_path = Path(result.get("transcript", ""))
    if not transcript_path.exists():
        errors.append(f"Transcript not found: {transcript_path}")
        return False, errors
    
    # Count turns from JSONL
    turn_count = 0
    agents_seen = set()
    with transcript_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                if "turn" in obj and isinstance(obj["turn"], dict):
                    turn_count += 1
                    if "name" in obj["turn"]:
                        agents_seen.add(obj["turn"]["name"])
    
    # Check minimum turns
    expected_min = fixture.get("expected_min_turns", 0)
    if turn_count < expected_min:
        errors.append(f"Expected at least {expected_min} turns, got {turn_count}")
    
    # Check expected agents
    expected_agents = set(fixture.get("expected_agents", []))
    if expected_agents and not expected_agents.issubset(agents_seen):
        missing = expected_agents - agents_seen
        errors.append(f"Missing expected agents: {missing}")
    
    return len(errors) == 0, errors


def run_eval(fixture_path: Path, dry_run: bool = False) -> Dict[str, Any]:
    """Run a single fixture evaluation."""
    print(f"\n{'=' * 60}")
    print(f"Running: {fixture_path.name}")
    print("=" * 60)
    
    fixture = load_fixture(fixture_path)
    print(f"Description: {fixture.get('description', 'N/A')}")
    print(f"Agents: {len(fixture.get('agents', []))}")
    print(f"Images: {len(fixture.get('images', []))}")
    print(f"Rounds: {fixture.get('rounds', 0)}")
    
    if dry_run:
        print("[DRY RUN] Skipping actual execution")
        return {"status": "dry_run", "fixture": fixture_path.name}
    
    try:
        agents = agents_from_fixture(fixture)
        result = run_discussion_v1_2(
            images=fixture.get("images", []),
            agents=agents,
            backend_name="openai",
            topic_hint=fixture.get("topic_hint", ""),
            rounds=fixture.get("rounds", 3),
            seed=42,
        )
        
        print(f"Status: {result.get('status')}")
        print(f"Transcript: {result.get('transcript')}")
        
        # Validate
        is_valid, errors = validate_result(result, fixture)
        if is_valid:
            print("✅ Validation passed")
        else:
            print("❌ Validation failed:")
            for err in errors:
                print(f"   - {err}")
        
        return {
            "status": "ok" if is_valid else "validation_failed",
            "fixture": fixture_path.name,
            "result": result,
            "errors": errors if not is_valid else [],
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "fixture": fixture_path.name,
            "error": str(e),
        }


def main():
    """Run all fixtures in evals/ directory."""
    evals_dir = Path(__file__).parent
    fixture_files = sorted(evals_dir.glob("fixture_*.json"))
    
    if not fixture_files:
        print("No fixtures found in evals/")
        return
    
    print(f"Found {len(fixture_files)} fixtures")
    
    import argparse
    parser = argparse.ArgumentParser(description="Run agent discussion evaluations")
    parser.add_argument("--dry-run", action="store_true", help="Skip actual execution")
    parser.add_argument("--fixture", help="Run only specific fixture (name)")
    args = parser.parse_args()
    
    if args.fixture:
        fixture_files = [f for f in fixture_files if args.fixture in f.name]
        if not fixture_files:
            print(f"No fixture matching '{args.fixture}'")
            return
    
    results = []
    for fixture_path in fixture_files:
        result = run_eval(fixture_path, dry_run=args.dry_run)
        results.append(result)
    
    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print("=" * 60)
    passed = sum(1 for r in results if r.get("status") == "ok")
    failed = sum(1 for r in results if r.get("status") in ("validation_failed", "error"))
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {len(results)}")
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

