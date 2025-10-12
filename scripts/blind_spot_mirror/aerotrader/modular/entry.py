# entry.py
"""
Main entry point for the Aerotrader simulation suite.
Runs the core flight simulation engine.
"""
import sys
import subprocess

if __name__ == "__main__":
    # Pass through all CLI arguments to the core engine
    args = [sys.executable, '-m', 'core.flight_sim_engine'] + sys.argv[1:]
    subprocess.run(args)
