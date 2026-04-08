from graders.base import BaseGrader
 
 
class Task1Grader(BaseGrader):
    """
    Task 1 — Easy.
    Checks: medevac handled within first 2 steps (0.4), gate type correctness (0.4),
            priority ordering (0.2).
    Hard penalties: hijack/bomb to pax gate → zero reward.
    Soft penalty: maintenance runway used → reward capped at 0.2.
    """
 
    def check_hard_penalties(self) -> tuple[float, bool]:
        # Hard zeros — is_zero_reward=True
        if self._hijack_sent_to_pax_gate():
            return 0.0, True
        if self._bomb_sent_to_pax_gate():
            return 0.0, True
        # Soft — maintenance runway: penalty=1.0, is_zero_reward=False
        # env.py will cap total at 0.2 separately when _used_maintenance_runway() is True
        if self._used_maintenance_runway():
            return 1.0, False  # full 10% penalty component zeroed
        return 0.0, False
 
    def grade_episode(self) -> float:
        """Full episode grade — called externally for logging/debugging."""
        penalty, is_zero = self.check_hard_penalties()
        if is_zero:
            return 0.0
        medevac = self._medevac_prompt_response()
        gate = self._gate_type_correctness()
        order = self._check_priority_ordering()
        raw = 0.4 * medevac + 0.4 * gate + 0.2 * order
        if self._used_maintenance_runway():
            raw = min(raw, 0.2)
        return round(min(max(raw, 0.0), 1.0), 4)
 
    def _medevac_prompt_response(self) -> float:
        """FL001 (medevac) must get assign_runway within first 2 actions."""
        for i, action in enumerate(self._action_log):
            if action.get("flight_id") == "FL001" and action.get("action_type") == "assign_runway":
                if i <= 1:
                    return 1.0
                return 0.5  # responded, but late
        return 0.0  # never given runway
 
    def _gate_type_correctness(self) -> float:
        """All gate assignments must match flight type requirements."""
        gate_actions = [
            (i, a) for i, a in enumerate(self._action_log)
            if a.get("action_type") == "assign_gate"
        ]
        if not gate_actions:
            return 0.0
        correct = 0
        for i, action in gate_actions:
            state = self._state_log[i] if i < len(self._state_log) else {}
            flight = state.get("flights", {}).get(action.get("flight_id", ""), {})
            gate = state.get("gates", {}).get(action.get("target_id", ""), {})
            ftype = flight.get("flight_type", "")
            gtype = gate.get("type", "")
            crisis = flight.get("crisis")
            if crisis == "medical_onboard" or ftype == "medevac":
                correct += 1 if gtype == "medical" else 0
            elif crisis in ("hijack", "bomb_threat"):
                correct += 1 if gtype == "isolation" else 0
            elif ftype == "cargo":
                correct += 1 if gtype in ("cargo", "isolation") else 0
            else:
                correct += 1 if gtype == "pax" else 0
        return correct / len(gate_actions)
