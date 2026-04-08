from state import AirportState, CrisisEvent, GateStatus, RunwayStatus


class CrisisProtocol:
    REQUIREMENTS: dict[str, list[str]] = {
        "medical_emergency": ["clear_runway", "medical_gate", "scramble_ambulance"],
        "bomb_threat": ["isolation_bay", "halt_movement", "notify_security"],
        "hijacking": ["isolation_gate", "secure_channel", "scramble_security"],
        "runway_fire": ["issue_go_around", "scramble_fire", "close_runway"],
        "fuel_emergency": ["immediate_runway"],
    }

    @staticmethod
    def get_requirements(crisis_type: str) -> list[str]:
        return CrisisProtocol.REQUIREMENTS.get(crisis_type, [])


class EventManager:
    def __init__(self, state: AirportState, crisis_events: list[CrisisEvent]):
        self.state = state
        self.crisis_events = crisis_events
        self._action_log: list[dict] = []
        self._protocol_progress: dict[str, set[str]] = {}

    def get_active_crises(self, current_step: int) -> list[CrisisEvent]:
        return [
            crisis
            for crisis in self.crisis_events
            if crisis.activate_step <= current_step and not crisis.resolved
        ]

    def _matching_steps(
        self, action: dict, crisis: CrisisEvent, current_step: int
    ) -> set[str]:
        flight_id = action.get("flight_id")
        action_type = action.get("action_type")
        target_id = action.get("target_id")
        notify = action.get("notify_authorities") or []
        matched: set[str] = set()

        if crisis.type == "medical_emergency":
            if flight_id == crisis.flight_id and action_type == "assign_runway":
                matched.add("clear_runway")
            if flight_id == crisis.flight_id and action_type == "assign_gate":
                gate = self.state.gates.get(target_id or "")
                if gate and gate.type.value == "medical":
                    matched.add("medical_gate")
            if flight_id == crisis.flight_id and action_type == "scramble_medical":
                matched.add("scramble_ambulance")

        elif crisis.type == "bomb_threat":
            if flight_id == crisis.flight_id and action_type == "assign_gate":
                gate = self.state.gates.get(target_id or "")
                if gate and gate.type.value == "isolation":
                    matched.add("isolation_bay")
            if action_type == "hold":
                matched.add("halt_movement")
            if flight_id == crisis.flight_id and "security" in notify:
                matched.add("notify_security")

        elif crisis.type == "hijacking":
            if flight_id == crisis.flight_id and action_type == "assign_gate":
                gate = self.state.gates.get(target_id or "")
                if gate and gate.type.value == "isolation":
                    matched.add("isolation_gate")
                if action.get("use_secure_channel"):
                    matched.add("secure_channel")
            if flight_id == crisis.flight_id and action_type == "scramble_security":
                matched.add("scramble_security")

        elif crisis.type == "runway_fire":
            if action_type == "hold" and flight_id and flight_id != crisis.flight_id:
                matched.add("issue_go_around")
            if action_type == "scramble_fire":
                matched.add("scramble_fire")
            if action_type == "close_runway" and target_id and (
                crisis.target_id is None or target_id == crisis.target_id
            ):
                matched.add("close_runway")

        elif crisis.type == "fuel_emergency":
            if (
                flight_id == crisis.flight_id
                and action_type == "assign_runway"
                and current_step == crisis.activate_step
            ):
                matched.add("immediate_runway")

        return matched

    def record_protocol_progress(self, action: dict, current_step: int) -> None:
        for crisis in self.get_active_crises(current_step):
            matched = self._matching_steps(action, crisis, current_step)
            if matched:
                self._protocol_progress.setdefault(crisis.flight_id, set()).update(matched)

    def check_protocol_compliance(self, crisis: CrisisEvent) -> tuple[float, list[str]]:
        requirements = CrisisProtocol.get_requirements(crisis.type)
        progress = self._protocol_progress.get(crisis.flight_id, set())
        missing = [req for req in requirements if req not in progress]
        score = (
            len(progress.intersection(set(requirements))) / len(requirements)
            if requirements
            else 1.0
        )
        return round(score, 4), missing

    def check_full_protocol_completion(
        self, crisis: CrisisEvent
    ) -> tuple[float, list[str]]:
        return self.check_protocol_compliance(crisis)

    def validate_action(self, action: dict) -> tuple[bool, str]:
        flight_id = action.get("flight_id")
        action_type = action.get("action_type")
        target_id = action.get("target_id")

        target_actions = {
            "assign_runway",
            "assign_gate",
            "close_runway",
            "vacate_runway",
        }
        if action_type in target_actions and not target_id:
            return False, f"Action {action_type} requires target_id"

        if flight_id and flight_id not in self.state.flights:
            return False, f"Unknown flight: {flight_id}"

        flight = self.state.flights.get(flight_id) if flight_id else None

        if action_type == "assign_runway":
            if target_id not in self.state.runways:
                return False, f"Unknown runway: {target_id}"
            runway = self.state.runways[target_id]
            if runway.status == RunwayStatus.MAINTENANCE:
                return False, f"Runway {target_id} is under maintenance"
            if runway.status == RunwayStatus.CLOSED:
                return False, f"Runway {target_id} is closed"
            if runway.status != RunwayStatus.FREE:
                return False, f"Runway {target_id} is not free (status: {runway.status.value})"
            if not flight:
                return False, "assign_runway requires a valid flight_id"
            if flight.status in ("at_gate", "diverted"):
                return False, f"Flight {flight_id} is already completed"
            if flight.assigned_runway:
                return False, (
                    f"Flight {flight_id} is already assigned to runway {flight.assigned_runway}"
                )
            if flight.assigned_gate:
                return False, f"Flight {flight_id} is already parked at gate {flight.assigned_gate}"

        if action_type == "assign_gate":
            if target_id not in self.state.gates:
                return False, f"Unknown gate: {target_id}"
            gate = self.state.gates[target_id]
            if gate.status != GateStatus.FREE:
                return False, f"Gate {target_id} is not available"
            if not flight:
                return False, "assign_gate requires a valid flight_id"
            if flight.status in ("at_gate", "diverted"):
                return False, f"Flight {flight_id} is already completed"
            if not flight.assigned_runway:
                return False, f"Flight {flight_id} must be assigned a runway before a gate"
            if flight.assigned_gate:
                return False, f"Flight {flight_id} is already assigned to gate {flight.assigned_gate}"

        if action_type == "hold":
            if not flight:
                return False, "hold requires a valid flight_id"
            if flight.status in ("at_gate", "diverted"):
                return False, f"Flight {flight_id} cannot be held in status {flight.status}"
            if flight.assigned_runway:
                return False, (
                    f"Flight {flight_id} is already committed to runway {flight.assigned_runway}"
                )

        if action_type == "divert":
            if not flight:
                return False, "divert requires a valid flight_id"
            if flight.status == "diverted":
                return False, f"Flight {flight_id} is already diverted"
            if flight.status == "at_gate":
                return False, f"Flight {flight_id} is already at a gate"

        if action_type == "scramble_security" and self.state.ground_units.security_teams <= 0:
            return False, "No security teams available"

        if action_type == "scramble_fire" and self.state.ground_units.fire_trucks <= 0:
            return False, "No fire trucks available"

        if action_type == "scramble_medical" and self.state.ground_units.ambulances <= 0:
            return False, "No ambulances available"

        if action_type == "close_runway":
            if target_id not in self.state.runways:
                return False, f"Unknown runway: {target_id}"
            runway = self.state.runways[target_id]
            if runway.status == RunwayStatus.MAINTENANCE:
                return False, f"Runway {target_id} is under maintenance"
            if runway.status == RunwayStatus.CLOSED:
                return False, f"Runway {target_id} is already closed"

        if action_type == "vacate_runway" and target_id not in self.state.runways:
            return False, f"Unknown runway: {target_id}"

        return True, ""

    def log_action(self, action: dict) -> None:
        self._action_log.append(action.copy())

    def get_action_log(self) -> list[dict]:
        return self._action_log.copy()
