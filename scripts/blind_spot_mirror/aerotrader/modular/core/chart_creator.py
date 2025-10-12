import argparse
import json
import os
import matplotlib.pyplot as plt

# --- Chart Creator for FlightOpsCore Telemetry ---
def load_log(log_path):
    with open(log_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def plot_flight_telemetry(flight, out_dir):
    telemetry = flight.get('telemetry', [])
    if not telemetry:
        print(f"No telemetry for flight {flight.get('id')}")
        return
    ticks = [snap['tick'] for snap in telemetry]
    altitude = [snap['altitude'] for snap in telemetry]
    velocity = [snap['velocity'] for snap in telemetry]
    price = [snap.get('price', None) for snap in telemetry]
    phase = [snap['phase'] for snap in telemetry]
    stall = [snap['status_flags'].get('stall', False) for snap in telemetry]
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(ticks, altitude, label='Altitude', color='tab:blue')
    ax1.set_xlabel('Tick')
    ax1.set_ylabel('Altitude', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.plot(ticks, velocity, label='Velocity', color='tab:orange')
    ax2.set_ylabel('Velocity', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    
    if any(p is not None for p in price):
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(ticks, price, label='Price', color='tab:green')
        ax3.set_ylabel('Price', color='tab:green')
        ax3.tick_params(axis='y', labelcolor='tab:green')
    
    # Mark stall events
    for i, s in enumerate(stall):
        if s:
            ax1.axvline(ticks[i], color='red', linestyle='--', alpha=0.3)
    
    plt.title(f"Flight {flight.get('id')} Telemetry")
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{flight.get('id')}_telemetry.png")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved chart for {flight.get('id')} to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Create charts from FlightOpsCore telemetry logs.")
    parser.add_argument('--log', type=str, required=True, help='Path to the JSON log file.')
    parser.add_argument('--out', type=str, default='modular/logs/', help='Output directory for charts.')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data = load_log(args.log)
    # Expecting a list of flights or a dict with 'flights' key
    flights = data if isinstance(data, list) else data.get('flights', [])
    for flight in flights:
        plot_flight_telemetry(flight, args.out)

if __name__ == "__main__":
    main() 