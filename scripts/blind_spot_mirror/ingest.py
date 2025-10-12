import json, subprocess, os
from schemas import Snapshot
from config import AEROTRADER_CMD, AEROTRADER_DIR

def run_aerotrader(symbol: str, mode: str = "daily") -> Snapshot:
    """
    Run aerotrader flight simulation and return snapshot with flight data.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'SPY', 'AAPL')
        mode: Simulation mode - 'daily' or 'intraday'
    
    Returns:
        Snapshot with flight simulation telemetry
    """
    # Build command: python entry.py --symbol SPY --mode daily --output stdout
    cmd = AEROTRADER_CMD + ["--symbol", symbol, "--mode", mode, "--output", "stdout"]
    
    # Run from aerotrader directory to ensure correct imports
    out = subprocess.check_output(
        cmd, 
        text=True, 
        cwd=AEROTRADER_DIR,
        stderr=subprocess.DEVNULL  # Suppress debug output
    )
    
    data = json.loads(out)
    return Snapshot(**data)
