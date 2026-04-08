from typing import List, Optional, Literal
from pydantic import BaseModel, Field
 

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
    source_urls: List[str] = Field(default_factory=list)


class ActiveCrisis(BaseModel):
    type: Literal[
        "medical_emergency", "bomb_threat", "hijacking", "runway_fire", "fuel_emergency"
    ]
    flight_id: str
    activate_step: int
    target_id: Optional[str] = None


class FlightInfo(BaseModel):
    flight_id: str
    flight_type: Literal["army", "medevac", "government", "commercial", "cargo"]
    status: Literal[
        "requesting_landing", "requesting_takeoff", "at_gate", "holding", "diverted"
    ]
    fuel_remaining_mins: int
    passengers: int
    crisis: Optional[
        Literal["hijack", "bomb_threat", "fire", "medical_onboard"]
    ] = None
    current_location: Optional[str] = None
    steps_waiting: int = 0
    assigned_runway: Optional[str] = None
    assigned_gate: Optional[str] = None


class Observation(BaseModel):
    step: int
    airport_context: AirportContext
    time_of_day: str
    day_of_week: str
    is_holiday: bool
    flights: List[FlightInfo]
    runways: dict
    gates: dict
    active_crises: List[ActiveCrisis]
    available_runways: List[str] = Field(default_factory=list)
    available_gates: List[str] = Field(default_factory=list)
    ground_units: dict = Field(default_factory=dict)
 
 
class Action(BaseModel):
    flight_id: str
    action_type: Literal[
        "assign_runway",
        "assign_gate",
        "hold",
        "divert",
        "scramble_security",
        "scramble_fire",
        "scramble_medical",
        "close_runway",
        "vacate_runway",
    ]
    target_id: Optional[str] = None
    use_secure_channel: bool = False
    notify_authorities: Optional[List[Literal["security", "fire", "police"]]] = None
 
 
class Reward(BaseModel):
    total: float
    priority_score: float
    resource_match_score: float
    eta_score: float
    penalty: float
    crisis_protocol_score: float
    episode_score: float = 0.0
    progress_delta: float = 0.0
    invalid_action: bool = False
