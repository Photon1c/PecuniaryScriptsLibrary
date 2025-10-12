import argparse
from watcher import watch
from ingest import run_aerotrader
from signals import compute_signals
from planner import reason_play
from risk_officer import enforce
from scribe import write_jsonl, write_flight_plan

def main():
    ap = argparse.ArgumentParser("bsm", description="Blind Spot Mirror - Flight-based trading system")
    ap.add_argument("mode", choices=["plan","watch"])
    ap.add_argument("--symbol", default="SPY", help="Stock ticker symbol")
    ap.add_argument("--flight-mode", default="daily", choices=["daily", "intraday"], 
                    help="Aerotrader simulation mode (daily or intraday)")
    args = ap.parse_args()

    if args.mode == "watch":
        watch(args.symbol, args.flight_mode); return
    # one-shot plan:
    ss = run_aerotrader(args.symbol, args.flight_mode)
    sig = compute_signals(ss)
    card = enforce(reason_play(ss, sig))
    write_jsonl(card)
    
    # Display snapshot info for verification
    print(f"\n📊 Data Verification:")
    print(f"   Symbol: {ss.symbol}")
    print(f"   Spot Price: ${ss.spot:.2f} (from CSV last row)")
    print(f"   Data Date: {ss.date}")
    print(f"   Mode: {ss.mode}")
    
    plan_path = write_flight_plan(card)
    print(f"\n✈️ Flight plan written to: {plan_path}")

if __name__ == "__main__":
    main()
