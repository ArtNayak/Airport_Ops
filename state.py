from typing import Optional, Literal
from pydantic import BaseModel, Field
from enum import Enum
import json
import os
 
 
class RunwayStatus(str, Enum):
    FREE = "free"
    OCCUPIED = "occupied"
    CLOSED = "closed"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"
 
 
class GateStatus(str, Enum):
    FREE = "free"
    OCCUPIED = "occupied"
    RESERVED = "reserved"
 
 
class GateType(str, Enum):
    PAX = "pax"
    CARGO = "cargo"
    MEDICAL = "medical"
    ISOLATION = "isolation"
 
 
class RunwayType(str, Enum):
    LANDING = "landing"
    TAKEOFF = "takeoff"
    DUAL = "dual"
 
 
class Runway(BaseModel):
    id: str
    type: RunwayType
    status: RunwayStatus = RunwayStatus.FREE
    assigned_flight: Optional[str] = None
 
 
class Gate(BaseModel):
    id: str
    type: GateType
    status: GateStatus = GateStatus.FREE
    assigned_flight: Optional[str] = None
 
 
class Taxiway(BaseModel):
    id: str
    clear: bool = True
    blocked_by: Optional[str] = None
 
 
class GroundUnits(BaseModel):
    ambulances: int
    fire_trucks: int
    security_teams: int
 
 
class CrisisEvent(BaseModel):
    type: Literal[
        "medical_emergency", "bomb_threat", "hijacking", "runway_fire", "fuel_emergency"
    ]
    flight_id: str
    activate_step: int
    target_id: Optional[str] = None
    resolved: bool = False
 
 
class Flight(BaseModel):
    flight_id: str
    flight_type: Literal["army", "medevac", "government", "commercial", "cargo"]
    status: Literal[
        "requesting_landing", "requesting_takeoff", "at_gate", "holding", "diverted"
    ]
    fuel_remaining_mins: int
    passengers: int
    crisis: Optional[Literal["hijack", "bomb_threat", "fire", "medical_onboard"]] = None
    current_location: Optional[str] = None
    steps_waiting: int = 0
    assigned_runway: Optional[str] = None
    assigned_gate: Optional[str] = None
 
 
class TimeContext(BaseModel):
    time_of_day: str
    day_of_week: str
    is_holiday: bool
 

class AirportContext(BaseModel):
    airport_id: str
    airport_name: str
    iata: str
    icao: str
    city: str
    country: str = "India"
    operator: str
    profile_note: str
    runway_aliases: dict[str, str] = Field(default_factory=dict)
    source_urls: list[str] = Field(default_factory=list)


class AirportState(BaseModel):
    step: int = 0
    airport_context: AirportContext
    time_context: TimeContext
    runways: dict[str, Runway] = Field(default_factory=dict)
    gates: dict[str, Gate] = Field(default_factory=dict)
    taxiways: dict[str, Taxiway] = Field(default_factory=dict)
    flights: dict[str, Flight] = Field(default_factory=dict)
    ground_units: GroundUnits
    active_crises: list[CrisisEvent] = Field(default_factory=list)
    state_history: list[dict] = Field(default_factory=list)
 
    def get_time_bucket(self) -> str:
        try:
            hour = int(self.time_context.time_of_day.split(":")[0])
        except (ValueError, IndexError):
            hour = 12
        if self.time_context.is_holiday:
            return "holiday"
        if self.time_context.day_of_week in ("Saturday", "Sunday"):
            return "weekend"
        if 7 <= hour <= 10 or 17 <= hour <= 20:
            return "rush_hour_weekday"
        return "off_peak_weekday"
 
    def get_flight_priority(self, flight_id: str) -> int:
        """Priority 0 = highest (fuel emergency). 1=army, 2=medevac ... 5=cargo."""
        if flight_id not in self.flights:
            return 99
        flight = self.flights[flight_id]
        if flight.fuel_remaining_mins < 10:
            return 0  # beats everyone including army
        return {"army": 1, "medevac": 2, "government": 3, "commercial": 4, "cargo": 5}.get(
            flight.flight_type, 6
        )
 
    def get_available_runways(self) -> list[str]:
        return [rid for rid, r in self.runways.items() if r.status == RunwayStatus.FREE]
 
    def get_available_gates(self, gate_type: Optional[GateType] = None) -> list[str]:
        gates = [gid for gid, g in self.gates.items() if g.status == GateStatus.FREE]
        if gate_type:
            gates = [g for g in gates if self.gates[g].type == gate_type]
        return gates
 
 
class AirportStateMachine:
    def __init__(self, scenario_path: str):
        with open(scenario_path, "r") as f:
            scenario = json.load(f)
        self._load_scenario(scenario)
        self._load_eta_table()
 
    # ── ETA table ─────────────────────────────────────────────────────── #
 
    def _load_eta_table(self) -> None:
        """Load eta_table.json. Stored at data/eta_table.json relative to this file."""
        eta_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "eta_table.json")
        try:
            with open(eta_path, "r") as f:
                self._eta_table: dict = json.load(f)
        except FileNotFoundError:
            self._eta_table = {}
 
    def get_eta(self, runway_id: str, gate_id: str, flight_type: str = "commercial") -> Optional[float]:
        """
        Return taxi time in minutes for runway→gate given current time bucket.
        Returns None if data is missing (Fix #8: caller must NOT treat None as optimal).
        """
        bucket = self.state.get_time_bucket()
        try:
            base: float = (
                self._eta_table["time_buckets"][bucket]["runway_to_gate"][runway_id][gate_id]
            )
            speed: float = self._eta_table.get("flight_type_taxi_speed", {}).get(flight_type, 1.0)
            return round(base * speed, 1)
        except (KeyError, TypeError):
            return None  # explicit None — caller decides fallback
 
    def score_eta_optimality(
        self,
        runway_id: str,
        gate_id: str,
        flight_type: str = "commercial",
        candidate_runways: Optional[list[str]] = None,
    ) -> float:
        """
        Score 0.0–1.0: 1.0 = chosen runway is fastest to gate_id, 0.0 = slowest.
        Fix #8: returns 0.5 (neutral) when ETA data is missing, never 1.0.
        """
        available = candidate_runways or self.state.get_available_runways()
        if not available:
            return 0.5  # no choice to score
 
        chosen_eta = self.get_eta(runway_id, gate_id, flight_type)
        if chosen_eta is None:
            return 0.5  # data missing — neutral, not optimal
 
        all_etas = [self.get_eta(r, gate_id, flight_type) for r in available]
        valid_etas = [e for e in all_etas if e is not None]
        if not valid_etas:
            return 0.5
 
        min_eta = min(valid_etas)
        max_eta = max(valid_etas)
 
        if max_eta == min_eta:
            return 1.0  # all runways equidistant — any choice is optimal
 
        return round(1.0 - (chosen_eta - min_eta) / (max_eta - min_eta), 4)
 
    # ── Scenario loading ───────────────────────────────────────────────── #
 
    def _load_scenario(self, scenario: dict) -> None:
        time_ctx = TimeContext(**scenario["time_context"])
        airport_ctx = AirportContext(
            **scenario.get(
                "airport_context",
                {
                    "airport_id": "generic",
                    "airport_name": "Generic Airport Ops Benchmark",
                    "iata": "GEN",
                    "icao": "VGEN",
                    "city": "Simulation",
                    "operator": "AirportOpsEnv",
                    "profile_note": "Fallback benchmark profile.",
                },
            )
        )
        self.state = AirportState(
            airport_context=airport_ctx,
            time_context=time_ctx,
            ground_units=GroundUnits(**scenario["ground_units"]),
        )
        for r in scenario["runways"]:
            status_map = {"maintenance": RunwayStatus.MAINTENANCE, "occupied": RunwayStatus.OCCUPIED}
            self.state.runways[r["id"]] = Runway(
                id=r["id"],
                type=RunwayType(r["type"]),
                status=status_map.get(r.get("status", "free"), RunwayStatus.FREE),
            )
        for g in scenario["gates"]:
            self.state.gates[g["id"]] = Gate(
                id=g["id"],
                type=GateType(g["type"]),
                status=GateStatus.OCCUPIED if g.get("status") == "occupied" else GateStatus.FREE,
            )
        self.state.taxiways = {f"tw{i}": Taxiway(id=f"tw{i}") for i in range(1, 5)}
        for f in scenario["flights"]:
            self.state.flights[f["flight_id"]] = Flight(**f)
        for c in scenario.get("crises", []):
            self.state.active_crises.append(CrisisEvent(**c))
 
    # ── State mutations ────────────────────────────────────────────────── #
 
    def assign_runway(self, flight_id: str, runway_id: str) -> bool:
        if runway_id not in self.state.runways or flight_id not in self.state.flights:
            return False
        flight = self.state.flights[flight_id]
        runway = self.state.runways[runway_id]
        if runway.status != RunwayStatus.FREE:
            return False
        if flight.status in ("at_gate", "diverted"):
            return False
        if flight.assigned_runway or flight.assigned_gate:
            return False
        runway.status = RunwayStatus.OCCUPIED
        runway.assigned_flight = flight_id
        flight.assigned_runway = runway_id
        flight.current_location = runway_id
        return True
 
    def vacate_runway(self, runway_id: str) -> bool:
        if runway_id not in self.state.runways:
            return False
        runway = self.state.runways[runway_id]
        if runway.assigned_flight:
            flight = self.state.flights.get(runway.assigned_flight)
            if flight:
                flight.assigned_runway = None
                flight.current_location = None
        runway.status = RunwayStatus.FREE
        runway.assigned_flight = None
        return True

    def _release_gate(self, gate_id: str) -> bool:
        if gate_id not in self.state.gates:
            return False
        gate = self.state.gates[gate_id]
        if gate.assigned_flight:
            flight = self.state.flights.get(gate.assigned_flight)
            if flight:
                flight.assigned_gate = None
                if flight.current_location == gate_id:
                    flight.current_location = None
        gate.status = GateStatus.FREE
        gate.assigned_flight = None
        return True

    def assign_gate(self, flight_id: str, gate_id: str) -> bool:
        """Also frees the runway this flight was occupying (runway leak fix)."""
        if gate_id not in self.state.gates or flight_id not in self.state.flights:
            return False
        gate = self.state.gates[gate_id]
        if gate.status != GateStatus.FREE:
            return False
        flight = self.state.flights[flight_id]
        if flight.status in ("at_gate", "diverted"):
            return False
        if not flight.assigned_runway:
            return False
        if flight.assigned_gate:
            return False
        # Free the runway now the aircraft is taxiing to gate
        if flight.assigned_runway:
            self.vacate_runway(flight.assigned_runway)
        gate.status = GateStatus.OCCUPIED
        gate.assigned_flight = flight_id
        flight.assigned_gate = gate_id
        flight.current_location = gate_id
        flight.status = "at_gate"
        return True
 
    def hold_flight(self, flight_id: str) -> bool:
        if flight_id not in self.state.flights:
            return False
        flight = self.state.flights[flight_id]
        if flight.status in ("at_gate", "diverted"):
            return False
        if flight.assigned_runway:
            return False
        if flight.assigned_gate:
            return False
        flight.status = "holding"
        return True
 
    def divert_flight(self, flight_id: str) -> bool:
        if flight_id not in self.state.flights:
            return False
        flight = self.state.flights[flight_id]
        if flight.assigned_runway:
            self.vacate_runway(flight.assigned_runway)
        if flight.assigned_gate:
            self._release_gate(flight.assigned_gate)
        flight.status = "diverted"
        flight.current_location = None
        return True
 
    def close_runway(self, runway_id: str) -> bool:
        if runway_id not in self.state.runways:
            return False
        runway = self.state.runways[runway_id]
        if runway.status == RunwayStatus.MAINTENANCE:
            return False
        runway.status = RunwayStatus.CLOSED
        return True
 
    def scramble_unit(self, unit_type: Literal["ambulance", "fire_truck", "security"]) -> bool:
        units = self.state.ground_units
        if unit_type == "ambulance" and units.ambulances > 0:
            units.ambulances -= 1
            return True
        if unit_type == "fire_truck" and units.fire_trucks > 0:
            units.fire_trucks -= 1
            return True
        if unit_type == "security" and units.security_teams > 0:
            units.security_teams -= 1
            return True
        return False
 
    def resolve_crisis(self, flight_id: str) -> bool:
        """Mark a crisis as resolved by flight_id."""
        for crisis in self.state.active_crises:
            if crisis.flight_id == flight_id and not crisis.resolved:
                crisis.resolved = True
                return True
        return False
 
    def increment_step(self) -> None:
        self.state.step += 1
        for flight in self.state.flights.values():
            if flight.status in ("requesting_landing", "holding"):
                flight.steps_waiting += 1
                if flight.fuel_remaining_mins > 0:
                    flight.fuel_remaining_mins = max(0, flight.fuel_remaining_mins - 1)
        # Snapshot without recursive nesting
        snapshot = self.to_dict()
        snapshot.pop("state_history", None)
        self.state.state_history.append(snapshot)
 
    def to_dict(self) -> dict:
        return {
            "step": self.state.step,
            "airport_context": self.state.airport_context.model_dump(),
            "time_context": self.state.time_context.model_dump(),
            "runways": {k: v.model_dump() for k, v in self.state.runways.items()},
            "gates": {k: v.model_dump() for k, v in self.state.gates.items()},
            "flights": {k: v.model_dump() for k, v in self.state.flights.items()},
            "ground_units": self.state.ground_units.model_dump(),
            "active_crises": [c.model_dump() for c in self.state.active_crises],
            "state_history": self.state.state_history,
        }
 
