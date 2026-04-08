from graders.base import BaseGrader
 
 
class Task2Grader(BaseGrader):
    """
    Task 2 — Medium.
    Key checks: fuel emergency (FL002) overrides army (FL001), bomb threat (FL003)
                → isolation bay, maintenance runway (R3) never used, priority order.
    """
 
    def check_hard_penalties(self) -> tuple[float, bool]:
        # Hard zeros (issue #3 fix: return 0.0 penalty + True for zero reward)
        if self._hijack_sent_to_pax_gate():
            return 0.0, True
        if self._bomb_sent_to_pax_gate():
            return 0.0, True
        # Soft: maintenance runway
        if self._used_maintenance_runway():
            return 1.0, False
        return 0.0, False
 
    def grade_episode(self) -> float:
        penalty, is_zero = self.check_hard_penalties()
        if is_zero:
            return self._strict_unit_interval_score(0.0)
        fuel_score = self._fuel_override_army()
        bomb_score = self._bomb_isolation_protocol()
        order_score = self._check_priority_ordering()
        raw = 0.40 * fuel_score + 0.40 * bomb_score + 0.20 * order_score
        if self._used_maintenance_runway():
            raw = min(raw, 0.2)
        return self._strict_unit_interval_score(raw)
 
    def _fuel_override_army(self) -> float:
        """FL002 (fuel=8 mins) must be assigned runway BEFORE FL001 (army)."""
        fl001_step = None
        fl002_step = None
        for i, action in enumerate(self._action_log):
            if action.get("action_type") == "assign_runway":
                fid = action.get("flight_id")
                if fid == "FL001" and fl001_step is None:
                    fl001_step = i
                if fid == "FL002" and fl002_step is None:
                    fl002_step = i
        if fl002_step is None:
            return 0.0   # fuel flight never given a runway
        if fl001_step is None:
            return 1.0   # army never dispatched (diverted?) — fuel handled fine
        return 1.0 if fl002_step < fl001_step else 0.0
 
    def _bomb_isolation_protocol(self) -> float:
        """FL003 (bomb threat) must be routed to G_ISO."""
        for action in self._action_log:
            if action.get("flight_id") != "FL003":
                continue
            if action.get("action_type") == "assign_gate":
                return 1.0 if action.get("target_id") == "G_ISO" else 0.0
        # Never gate-assigned — partial if at least held
        for action in self._action_log:
            if action.get("flight_id") == "FL003" and action.get("action_type") == "hold":
                return 0.3
        return 0.0
 
