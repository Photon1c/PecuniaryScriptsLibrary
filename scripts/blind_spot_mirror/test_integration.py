#!/usr/bin/env python3
"""
Test script for Aerotrader + BSM integration.

This script verifies that:
1. Aerotrader can be invoked successfully
2. JSON output is properly formatted
3. Snapshot schema validation works
4. Signal computation functions correctly
"""

import sys
import json
import subprocess
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ingest import run_aerotrader
from signals import compute_signals
from schemas import Snapshot, FlightData

def test_aerotrader_stdout():
    """Test that aerotrader produces valid JSON output."""
    print("=" * 60)
    print("TEST 1: Aerotrader stdout JSON output")
    print("=" * 60)
    
    try:
        from config import AEROTRADER_CMD, AEROTRADER_DIR
        
        cmd = AEROTRADER_CMD + ["--symbol", "SPY", "--mode", "daily", "--output", "stdout"]
        print(f"Running: {' '.join(cmd)}")
        print(f"Working dir: {AEROTRADER_DIR}")
        
        result = subprocess.run(
            cmd,
            cwd=AEROTRADER_DIR,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print(f"❌ FAILED: Process returned {result.returncode}")
            print(f"stderr: {result.stderr}")
            return False
        
        # Try to parse JSON
        data = json.loads(result.stdout)
        print(f"✅ PASSED: Valid JSON output")
        print(f"   Symbol: {data.get('symbol')}")
        print(f"   Mode: {data.get('mode')}")
        print(f"   Spot: {data.get('spot')}")
        if 'flight_data' in data:
            fd = data['flight_data']
            print(f"   Net Gain: {fd.get('net_gain')}%")
            print(f"   Phase: {fd.get('latest_phase')}")
        
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ FAILED: Command timed out")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ FAILED: Invalid JSON - {e}")
        print(f"stdout: {result.stdout[:500]}")
        return False
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_snapshot_validation():
    """Test that JSON data validates against Snapshot schema."""
    print("\n" + "=" * 60)
    print("TEST 2: Snapshot schema validation")
    print("=" * 60)
    
    try:
        # Sample data
        test_data = {
            "symbol": "SPY",
            "spot": 450.25,
            "date": "2024-01-15",
            "mode": "Macro Cruise (Daily)",
            "book": {"latest_close": 450.25, "latest_volume": 85000000},
            "chain": [],
            "flight_data": {
                "net_gain": 2.5,
                "max_altitude": 3.2,
                "fuel_remaining": 75.0,
                "stall_events": 0,
                "turbulence_heavy": 1,
                "turbulence_moderate": 2,
                "latest_phase": "Thrust",
                "telemetry": [
                    {
                        "time": "09:30",
                        "altitude": 0.5,
                        "fuel": 100.0,
                        "stall": False,
                        "turbulence": "Calm",
                        "phase": "Thrust"
                    }
                ]
            }
        }
        
        snapshot = Snapshot(**test_data)
        print(f"✅ PASSED: Snapshot validation successful")
        print(f"   Symbol: {snapshot.symbol}")
        print(f"   Spot: {snapshot.spot}")
        if snapshot.flight_data:
            print(f"   Net Gain: {snapshot.flight_data.net_gain}%")
            print(f"   Phase: {snapshot.flight_data.latest_phase}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ingest():
    """Test the ingest.py run_aerotrader function."""
    print("\n" + "=" * 60)
    print("TEST 3: Ingest function (run_aerotrader)")
    print("=" * 60)
    
    try:
        print("Calling run_aerotrader('SPY', 'daily')...")
        snapshot = run_aerotrader("SPY", "daily")
        
        print(f"✅ PASSED: Got Snapshot")
        print(f"   Symbol: {snapshot.symbol}")
        print(f"   Spot: {snapshot.spot}")
        if snapshot.flight_data:
            fd = snapshot.flight_data
            print(f"   Net Gain: {fd.net_gain:+.2f}%")
            print(f"   Max Altitude: {fd.max_altitude:+.2f}%")
            print(f"   Fuel: {fd.fuel_remaining:.1f}%")
            print(f"   Stalls: {fd.stall_events}")
            print(f"   Phase: {fd.latest_phase}")
        
        return True
        
    except FileNotFoundError as e:
        print(f"❌ FAILED: Data files not found - {e}")
        print("   Make sure stock/option data is available")
        return False
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_signals():
    """Test signal computation from flight data."""
    print("\n" + "=" * 60)
    print("TEST 4: Signal computation")
    print("=" * 60)
    
    try:
        # Create test snapshot
        test_data = {
            "symbol": "SPY",
            "spot": 450.25,
            "book": {"latest_close": 450.25},
            "chain": [],
            "flight_data": {
                "net_gain": 2.5,
                "max_altitude": 3.2,
                "fuel_remaining": 75.0,
                "stall_events": 0,
                "turbulence_heavy": 0,
                "turbulence_moderate": 1,
                "latest_phase": "Thrust",
                "telemetry": []
            }
        }
        
        snapshot = Snapshot(**test_data)
        signals = compute_signals(snapshot)
        
        print(f"✅ PASSED: Signal computation successful")
        print(f"   Source: {signals.get('source')}")
        print(f"   Count: {signals.get('count')}")
        
        if signals.get('best'):
            best = signals['best']
            print(f"   Direction: {best.get('direction')}")
            print(f"   Confidence: {best.get('confidence')}")
            print(f"   Phase: {best.get('phase')}")
        else:
            print(f"   No signals generated")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n🧪 AEROTRADER + BSM INTEGRATION TEST SUITE\n")
    
    results = {
        "Aerotrader stdout": test_aerotrader_stdout(),
        "Schema validation": test_snapshot_validation(),
        "Ingest function": test_ingest(),
        "Signal computation": test_signals(),
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Integration is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

