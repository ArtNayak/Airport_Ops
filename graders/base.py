from abc import ABC, abstractmethod
 
 
PRIORITY_MAP = {"army": 1, "medevac": 2, "government": 3, "commercial": 4, "cargo": 5}
 
 
class BaseGrader(ABC):
    """
    Base class for all task graders.
 
    Penalty contract (fixes issues #3 and #4):
      check_hard_penalties() → (penalty: float, is_zero_reward: bool)
        is_zero_reward=True  → env.py zeros total immediately, never touches penalty math.
        is_zero_reward=False → penalty 0.0–1.0 fed into 0.10*(1-penalty) in env.py.
                               0.0 = no deduction, 1.0 = 10% component contributes nothing.
    """
 
    def __init__(self):
        self._action_log: list[dict] = []
        self._state_log: list[dict] = []
 
    def record_action(self, action: dict, state: dict) -> None:
        self._action_log.append(action.copy())
        s = state.copy()
        s.pop("state_history", None)  # never store recursive history
        self._state_log.append(s)
 
    @abstractmethod
    def check_hard_penalties(self) -> tuple[float, bool]:
        pass

    def _strict_unit_interval_score(self, score: float) -> float:
        """
        Keep final task scores strictly inside (0, 1).

        The remote OpenEnv submission validator rejects exact boundary values,
        so we clip the externally reported episode score to a tiny epsilon away
        from 0.0 and 1.0 while preserving its ordering.
        """
        return round(min(max(score, 0.0001), 0.9999), 4)
 
    # ── Shared violation detectors ────────────────────────────────────── #
 
    def _used_maintenance_runway(self) -> bool:
        """True if agent ever assigned a flight to a maintenance/closed runway."""
        for i, action in enumerate(self._action_log):
            if action.get("action_type") != "assign_runway":
                continue
            runway_id = action.get("target_id", "")
            state = self._state_log[i] if i < len(self._state_log) else {}
            rw = state.get("runways", {}).get(runway_id, {})
            if rw.get("status") in ("maintenance", "closed"):
                return True
        return False
 
    def _hijack_sent_to_pax_gate(self) -> bool:
        """True if a hijacked flight was assigned to a pax gate (hard zero)."""
        for i, action in enumerate(self._action_log):
            if action.get("action_type") != "assign_gate":
                continue
            state = self._state_log[i] if i < len(self._state_log) else {}
            flight = state.get("flights", {}).get(action.get("flight_id", ""), {})
            gate = state.get("gates", {}).get(action.get("target_id", ""), {})
            if flight.get("crisis") == "hijack" and gate.get("type") == "pax":
                return True
        return False
 
    def _bomb_sent_to_pax_gate(self) -> bool:
        """True if a bomb-threat flight was assigned to a pax gate (hard zero)."""
        for i, action in enumerate(self._action_log):
            if action.get("action_type") != "assign_gate":
                continue
            state = self._state_log[i] if i < len(self._state_log) else {}
            flight = state.get("flights", {}).get(action.get("flight_id", ""), {})
            gate = state.get("gates", {}).get(action.get("target_id", ""), {})
            if flight.get("crisis") == "bomb_threat" and gate.get("type") == "pax":
                return True
        return False
 
    # ── Shared scoring helpers ────────────────────────────────────────── #
 
    def _check_priority_ordering(self) -> float:
        """
        Fraction of consecutive assign pairs where higher-priority flight went first.
        Fuel emergency (fuel < 10) overrides all — treated as priority 0.
        """
        assigns = [
            (i, a) for i, a in enumerate(self._action_log)
            if a.get("action_type") in ("assign_runway", "assign_gate")
        ]
        if not assigns:
            return 0.0
        if len(assigns) == 1:
            return 1.0
 
        correct, total = 0, 0
        for idx in range(1, len(assigns)):
            pi, pa = assigns[idx - 1]
            ci, ca = assigns[idx]
            ps = self._state_log[pi] if pi < len(self._state_log) else {}
            cs = self._state_log[ci] if ci < len(self._state_log) else {}
            pf = ps.get("flights", {}).get(pa.get("flight_id", ""), {})
            cf = cs.get("flights", {}).get(ca.get("flight_id", ""), {})
            p_pri = 0 if pf.get("fuel_remaining_mins", 999) < 10 else PRIORITY_MAP.get(pf.get("flight_type", "commercial"), 6)
            c_pri = 0 if cf.get("fuel_remaining_mins", 999) < 10 else PRIORITY_MAP.get(cf.get("flight_type", "commercial"), 6)
            if p_pri <= c_pri:
                correct += 1
            total += 1
 
        return correct / total if total > 0 else 1.0
