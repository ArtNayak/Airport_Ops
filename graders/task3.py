from graders.base import BaseGrader


class Task3Grader(BaseGrader):
    """
    Task 3 - Hard. 15 flights, Diwali, simultaneous hijack (FL005)
    and runway fire (FL007 at step 3).
    """

    TOTAL_FLIGHTS = 15
    FIRE_ACTIVATE_STEP = 3

    def check_hard_penalties(self) -> tuple[float, bool]:
        if self._hijack_sent_to_pax_gate():
            return 0.0, True
        if self._bomb_sent_to_pax_gate():
            return 0.0, True
        if self._used_maintenance_runway():
            return 1.0, False
        return 0.0, False

    def grade_episode(self) -> float:
        penalty, is_zero = self.check_hard_penalties()
        if is_zero:
            return self._strict_unit_interval_score(0.0)
        raw = (
            0.25 * self._hijack_protocol()
            + 0.25 * self._fire_protocol()
            + 0.20 * self._crisis_isolation()
            + 0.15 * self._throughput()
            + 0.15 * self._cargo_handling()
        )
        if self._used_maintenance_runway():
            raw = min(raw, 0.2)
        return self._strict_unit_interval_score(raw)

    def _hijack_protocol(self) -> float:
        iso_assigned = False
        secure_channel = False
        security_scrambled = False

        for action in self._action_log:
            if action.get("flight_id") == "FL005":
                if action.get("action_type") == "assign_gate" and action.get("target_id") == "G_ISO":
                    iso_assigned = True
                    if action.get("use_secure_channel"):
                        secure_channel = True
            if action.get("action_type") == "scramble_security":
                security_scrambled = True

        score = 0.0
        if iso_assigned:
            score += 0.4
        if secure_channel:
            score += 0.4
        if security_scrambled:
            score += 0.2
        return score

    def _fire_protocol(self) -> float:
        go_around = False
        fire_scrambled = False
        runway_closed = False

        for i, action in enumerate(self._action_log):
            if i < self.FIRE_ACTIVATE_STEP:
                continue
            action_type = action.get("action_type")
            if action_type == "hold":
                go_around = True
            if action_type == "scramble_fire":
                fire_scrambled = True
            if action_type == "close_runway":
                runway_closed = True

        score = 0.0
        if go_around:
            score += 0.35
        if fire_scrambled:
            score += 0.35
        if runway_closed:
            score += 0.30
        return score

    def _crisis_isolation(self) -> float:
        hijack_steps = set()
        fire_steps = set()

        for i, action in enumerate(self._action_log):
            action_type = action.get("action_type", "")
            flight_id = action.get("flight_id", "")
            if flight_id == "FL005" or action_type == "scramble_security":
                hijack_steps.add(i)
            if action_type in ("scramble_fire", "close_runway") or (
                action_type == "hold" and i >= self.FIRE_ACTIVATE_STEP
            ):
                fire_steps.add(i)

        if not hijack_steps and not fire_steps:
            return 0.0
        if not hijack_steps or not fire_steps:
            return 0.3

        max_streak = 0
        hijack_streak = 0
        fire_streak = 0
        for i in range(len(self._action_log)):
            in_hijack = i in hijack_steps
            in_fire = i in fire_steps
            if in_hijack and not in_fire:
                hijack_streak += 1
                fire_streak = 0
            elif in_fire and not in_hijack:
                fire_streak += 1
                hijack_streak = 0
            else:
                hijack_streak = 0
                fire_streak = 0
            max_streak = max(max_streak, hijack_streak, fire_streak)

        if max_streak <= 2:
            return 1.0
        if max_streak <= 4:
            return 0.7
        return 0.4

    def _throughput(self) -> float:
        if not self._state_log:
            return 0.0
        last = self._state_log[-1]
        processed = sum(
            1
            for flight in last.get("flights", {}).values()
            if flight.get("status") in ("at_gate", "diverted")
        )
        return round(processed / self.TOTAL_FLIGHTS, 4)

    def _cargo_handling(self) -> float:
        cargo_ids = {"FL013", "FL014", "FL015"}
        cargo_gate_actions = [
            (i, action)
            for i, action in enumerate(self._action_log)
            if action.get("flight_id") in cargo_ids and action.get("action_type") == "assign_gate"
        ]
        if not cargo_gate_actions:
            return 0.0

        correct = 0
        for i, action in cargo_gate_actions:
            state = self._state_log[i] if i < len(self._state_log) else {}
            gate = state.get("gates", {}).get(action.get("target_id", ""), {})
            if gate.get("type") in ("cargo", "isolation"):
                correct += 1
        return correct / len(cargo_gate_actions)
