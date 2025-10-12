from dataclasses import dataclass, field
from typing import List, Optional, Dict
import random
import json

# --- Shared Schema: FlightState ---
@dataclass
class FlightState:
    id: str
    mode: str  # "market", "aircraft", "traffic"
    symbol: Optional[str] = None  # SPY, TSLA, or ICAO
    origin: str = "UNKNOWN"
    destination: str = "UNKNOWN"
    
    price: float = 0.0
    altitude: float = 0.0
    velocity: float = 0.0
    heading: float = 0.0
    phase: str = "Ground_Taxi"
    
    position: Optional[Dict[str, float]] = field(default_factory=dict)
    telemetry: List[Dict] = field(default_factory=list)
    status_flags: Dict[str, bool] = field(default_factory=lambda: {"stall": False, "turbulence": False})

# --- Mocked external functions for demonstration ---
def generate_market_flights(symbol_list):
    # Returns a list of FlightState objects for market flights
    return [
        FlightState(id=f"MKT_{sym}", mode="market", symbol=sym, price=100.0 + i*10, altitude=0.0, velocity=0.0)
        for i, sym in enumerate(symbol_list)
    ]

def spawn_aircraft(flight_plans):
    # Returns a list of FlightState objects for aircraft
    return [
        FlightState(id=plan["id"], mode="aircraft", origin=plan["origin"], destination=plan["dest"], altitude=1000.0, velocity=250.0)
        for plan in flight_plans
    ]

def generate_flight_schedule(airport_layout):
    # Returns a list of FlightState objects for traffic
    return [
        FlightState(id=f"TRF_{airport_layout}_1", mode="traffic", origin=airport_layout, destination="KSEA", altitude=0.0, velocity=0.0)
    ]

def update_market_flight(f: FlightState):
    # Simulate random walk for price
    if not hasattr(f, 'tick'):
        f.tick = 0
    f.tick += 1
    f.price += random.uniform(-1.5, 1.0)  # allow price to drop below 100
    f.altitude = f.price * 10
    f.velocity = 100.0
    f.phase = "Cruise"
    # Trigger stall if price < 100
    if f.price < 100:
        f.status_flags["stall"] = True
    else:
        f.status_flags["stall"] = False

def update_physical_flight(f: FlightState):
    # Simple mock: climb, then cruise
    if f.altitude < 10000:
        f.altitude += 500
        f.phase = "Climb"
    else:
        f.phase = "Cruise"
    f.velocity = 250.0

def update_traffic_logic(f: FlightState):
    # Simple mock: traffic moves
    if not hasattr(f, 'tick'):
        f.tick = 0
    f.tick += 1
    if f.tick == 4:
        f.phase = "GoAround"
    elif hasattr(f, 'cleared_to_land') and f.cleared_to_land:
        f.phase = "ClearedToLand"
    else:
        f.phase = "Taxi"
    f.velocity = 20.0

# --- Core Engine Logic ---
class FlightOpsCore:
    def __init__(self, config: Dict):
        self.config = config
        self.flight_objects = []  # List[FlightState]
        self.timestep = 0
        self.airspace_map = {}  # could link to ATC zones

    def load_market_flights(self, symbol_list):
        self.flight_objects += generate_market_flights(symbol_list)

    def load_aircraft(self, flight_plans):
        self.flight_objects += spawn_aircraft(flight_plans)

    def load_airtraffic(self, airport_layout):
        self.flight_objects += generate_flight_schedule(airport_layout)

    def update(self):
        for f in self.flight_objects:
            self._update_flight(f)
            # --- Telemetry/history buffer ---
            # Store a snapshot of the last 10 ticks
            snap = {
                'tick': self.timestep,
                'altitude': f.altitude,
                'velocity': f.velocity,
                'phase': f.phase,
                'status_flags': dict(f.status_flags),
                'price': f.price,
            }
            f.telemetry.append(snap)
            if len(f.telemetry) > 10:
                f.telemetry.pop(0)
        self.timestep += 1

        # --- Cross-domain event triggers ---
        # Simple mapping: first market flight <-> first aircraft, first traffic <-> first aircraft
        market_flights = [f for f in self.flight_objects if f.mode == "market"]
        aircraft_flights = [f for f in self.flight_objects if f.mode == "aircraft"]
        traffic_flights = [f for f in self.flight_objects if f.mode == "traffic"]

        # 1. Market stall triggers aircraft stall
        if market_flights and aircraft_flights:
            for i, mkt in enumerate(market_flights):
                if mkt.status_flags.get("stall"):
                    idx = min(i, len(aircraft_flights)-1)
                    ac = aircraft_flights[idx]
                    if not ac.status_flags.get("stall"):
                        ac.status_flags["stall"] = True
                        print(f"[TRIGGER] Market flight {mkt.id} is stalled. Aircraft {ac.id} set to stall.")
        # 2. Traffic GoAround triggers aircraft holding
        if traffic_flights and aircraft_flights:
            for i, trf in enumerate(traffic_flights):
                if trf.phase == "GoAround":
                    idx = min(i, len(aircraft_flights)-1)
                    ac = aircraft_flights[idx]
                    if ac.phase != "Holding":
                        ac.phase = "Holding"
                        # Track how long in holding
                        ac.holding_ticks = 1
                        print(f"[TRIGGER] Traffic flight {trf.id} in GoAround. Aircraft {ac.id} set to Holding phase.")
                    elif hasattr(ac, 'holding_ticks'):
                        ac.holding_ticks += 1
                else:
                    # Reset holding_ticks if not in holding
                    idx = min(i, len(aircraft_flights)-1)
                    ac = aircraft_flights[idx]
                    if hasattr(ac, 'holding_ticks'):
                        del ac.holding_ticks
        # 3. Aircraft in Holding >2 ticks triggers traffic ClearedToLand
        if aircraft_flights and traffic_flights:
            for i, ac in enumerate(aircraft_flights):
                if hasattr(ac, 'holding_ticks') and ac.holding_ticks > 2:
                    idx = min(i, len(traffic_flights)-1)
                    trf = traffic_flights[idx]
                    if not hasattr(trf, 'cleared_to_land') or not trf.cleared_to_land:
                        trf.cleared_to_land = True
                        print(f"[TRIGGER] Aircraft {ac.id} held >2 ticks. Traffic {trf.id} set to ClearedToLand.")

    def _update_flight(self, f: FlightState):
        # Route to appropriate logic engine
        if f.mode == "market":
            update_market_flight(f)
        elif f.mode == "aircraft":
            update_physical_flight(f)
        elif f.mode == "traffic":
            update_traffic_logic(f)

        # Shared stall logic
        if f.altitude > 0 and f.velocity < 50:
            f.status_flags["stall"] = True
        else:
            f.status_flags["stall"] = False

    def export_results(self, out_path):
        # Export all flight objects as dicts (including telemetry)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump([f.__dict__ for f in self.flight_objects], f, ensure_ascii=False, indent=2)
        print(f"Exported results to {out_path}")

# --- Example Entry Script ---
if __name__ == "__main__":
    config = {
        "mode": "multi",
        "symbols": ["SPY", "QQQ"],
        "region": "PNW",
        "aircraft_enabled": True,
        "controller_ai": "basic",
    }

    ops = FlightOpsCore(config)
    ops.load_market_flights(config["symbols"])
    ops.load_aircraft([{"id": "AC001", "origin": "KSEA", "dest": "KPDX"}])
    ops.load_airtraffic("PNW")

    print("Initial Flight States:")
    for f in ops.flight_objects:
        print(f)

    for tick in range(5):
        ops.update()
        print(f"\nTick {tick+1}:")
        for f in ops.flight_objects:
            print(f)
    # Export results for charting
    ops.export_results("modular/logs/results.json") 