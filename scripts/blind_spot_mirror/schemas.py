from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class FlightTelemetry(BaseModel):
    time: str
    altitude: float
    fuel: float
    stall: bool
    turbulence: str
    phase: str

class FlightData(BaseModel):
    net_gain: float
    max_altitude: float
    fuel_remaining: float
    stall_events: int
    turbulence_heavy: int
    turbulence_moderate: int
    latest_phase: str
    telemetry: List[FlightTelemetry]

class Snapshot(BaseModel):
    symbol: str
    spot: float
    book: Dict[str, Any]   # best bid/ask, vwap, etc.
    chain: List[Dict[str, Any]] = Field(default_factory=list)  # flattened rows: exp, strike, bid, ask, iv, oi,…
    date: Optional[str] = None
    mode: Optional[str] = None
    flight_data: Optional[FlightData] = None  # Aerotrader flight simulation data

class PlayCard(BaseModel):
    # (use the JSON schema shown above)
    symbol: str; direction: str; horizon: str
    entry: Dict[str, Any]; stop: Dict[str, Any]
    targets: List[float]; risk_reward: float; confidence: float
    preconditions: List[str]; alerts: List[str]; notes: str
    audit: Dict[str, Any]
