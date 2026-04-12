from typing import Optional
from models import Observation, Action, Reward
from state import AirportStateMachine, RunwayStatus, GateType
from events import EventManager
from graders import Task1Grader, Task2Grader, Task3Grader
from graders.base import BaseGrader
 
 
class AirportOpsEnv:
    def __init__(self):
        self.state_machine: Optional[AirportStateMachine] = None
        self.event_manager: Optional[EventManager] = None
        self.grader: Optional[BaseGrader] = None
        self.current_task: Optional[str] = None
        self.max_steps: int = 20
        self.done: bool = False
        self.last_action_error: Optional[str] = None
        self.previous_episode_score: float = 0.0

    def is_ready(self) -> bool:
        return (
            self.state_machine is not None
            and self.event_manager is not None
            and self.grader is not None
        )
 
    def reset(self, task_id: str = "task1") -> Observation:
        scenario_path = f"data/scenarios/{task_id}.json"
        self.state_machine = AirportStateMachine(scenario_path)
        self.event_manager = EventManager(
            self.state_machine.state, self.state_machine.state.active_crises
        )
        self.current_task = task_id
        task_config = {
            "task1": (Task1Grader, 20),
            "task2": (Task2Grader, 40),
            "task3": (Task3Grader, 80),
        }
        grader_cls, max_steps = task_config.get(task_id, (Task1Grader, 20))
        self.grader = grader_cls()
        self.max_steps = max_steps
        self.done = False
        self.last_action_error = None
        self.previous_episode_score = 0.0
        return self._get_observation()
 
    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if not self.is_ready():
            raise RuntimeError("Environment not initialized. Call reset() before step().")

        if self.done:
            return (
                self._get_observation(),
                self._get_zero_reward(),
                True,
                {"error": "Episode already ended. Call reset()."},
            )

        action_dict = action.model_dump()
        pre_step = self.state_machine.state.step
        pre_state = self.state_machine.state.model_copy(deep=True)
        candidate_runways = pre_state.get_available_runways()
        active_crises = self.event_manager.get_active_crises(pre_step)
        pre_crisis_score = self._score_crisis_protocol(active_crises)

        valid, error_msg = self.event_manager.validate_action(action_dict)
        if not valid:
            self.last_action_error = error_msg
            self.state_machine.increment_step()
            self.done = self._check_done()
            return (
                self._get_observation(),
                self._compute_reward(
                    action_dict=action_dict,
                    pre_state=pre_state,
                    candidate_runways=candidate_runways,
                    active_crises=active_crises,
                    episode_score=self.previous_episode_score,
                    progress_delta=0.0,
                    step_total=0.0,
                    invalid_action=True,
                ),
                self.done,
                {
                    "error": error_msg,
                    "last_action_error": error_msg,
                    "episode_score": round(self.previous_episode_score, 4),
                    "progress_delta": 0.0,
                },
            )

        applied, apply_error = self._apply_action(action_dict)
        if not applied:
            self.last_action_error = apply_error
            self.state_machine.increment_step()
            self.done = self._check_done()
            return (
                self._get_observation(),
                self._compute_reward(
                    action_dict=action_dict,
                    pre_state=pre_state,
                    candidate_runways=candidate_runways,
                    active_crises=active_crises,
                    episode_score=self.previous_episode_score,
                    progress_delta=0.0,
                    step_total=0.0,
                    invalid_action=True,
                ),
                self.done,
                {
                    "error": apply_error,
                    "last_action_error": apply_error,
                    "episode_score": round(self.previous_episode_score, 4),
                    "progress_delta": 0.0,
                },
            )

        self.last_action_error = None
        self.event_manager.log_action(action_dict)
        self.event_manager.record_protocol_progress(action_dict, pre_step)
        self.state_machine.increment_step()

        if self.grader:
            self.grader.record_action(action_dict, self.state_machine.to_dict())

        self._auto_resolve_crises()

        episode_score = (
            self.grader.grade_episode() if hasattr(self.grader, "grade_episode") else 0.0
        )
        progress_delta = round(max(0.0, episode_score - self.previous_episode_score), 4)
        post_crisis_score = self._score_crisis_protocol(active_crises)
        crisis_delta = round(max(0.0, post_crisis_score - pre_crisis_score), 4)
        reward = self._compute_reward(
            action_dict=action_dict,
            pre_state=pre_state,
            candidate_runways=candidate_runways,
            active_crises=active_crises,
            episode_score=episode_score,
            progress_delta=progress_delta,
            step_total=max(progress_delta, crisis_delta),
            invalid_action=False,
        )
        self.previous_episode_score = episode_score
        self.done = self._check_done()

        info = {
            "error": None,
            "last_action_error": None,
            "episode_score": round(episode_score, 4),
            "progress_delta": progress_delta,
        }
        if self.done:
            info["final_score"] = round(episode_score, 4)

        return self._get_observation(), reward, self.done, info
 
    def state(self) -> dict:
        if self.state_machine:
            s = self.state_machine.to_dict()
            s.pop("state_history", None)
            s["current_task"] = self.current_task
            s["done"] = self.done
            s["episode_score"] = round(self.previous_episode_score, 4)
            s["last_action_error"] = self.last_action_error
            return s
        return {}

    def grade(self) -> dict:
        if not self.is_ready():
            return {}
        score = self.grader.grade_episode() if hasattr(self.grader, "grade_episode") else 0.0
        return {
            "task_id": self.current_task,
            "episode_score": round(score, 4),
            "done": self.done,
            "max_steps": self.max_steps,
        }
 
    def _apply_action(self, action: dict) -> tuple[bool, str]:
        flight_id = action.get("flight_id")
        action_type = action.get("action_type")
        target_id = action.get("target_id")

        if action_type == "assign_runway" and target_id:
            success = self.state_machine.assign_runway(flight_id, target_id)
            return success, "" if success else f"Failed to assign runway {target_id} to {flight_id}"
        elif action_type == "assign_gate" and target_id:
            success = self.state_machine.assign_gate(flight_id, target_id)
            return success, "" if success else f"Failed to assign gate {target_id} to {flight_id}"
        elif action_type == "hold":
            success = self.state_machine.hold_flight(flight_id)
            return success, "" if success else f"Failed to hold flight {flight_id}"
        elif action_type == "divert":
            success = self.state_machine.divert_flight(flight_id)
            return success, "" if success else f"Failed to divert flight {flight_id}"
        elif action_type == "scramble_security":
            success = self.state_machine.scramble_unit("security")
            return success, "" if success else "Failed to scramble security unit"
        elif action_type == "scramble_fire":
            success = self.state_machine.scramble_unit("fire_truck")
            return success, "" if success else "Failed to scramble fire truck"
        elif action_type == "scramble_medical":
            success = self.state_machine.scramble_unit("ambulance")
            return success, "" if success else "Failed to scramble ambulance"
        elif action_type == "close_runway" and target_id:
            success = self.state_machine.close_runway(target_id)
            return success, "" if success else f"Failed to close runway {target_id}"
        elif action_type == "vacate_runway" and target_id:
            success = self.state_machine.vacate_runway(target_id)
            return success, "" if success else f"Failed to vacate runway {target_id}"
        return False, f"Unsupported or malformed action: {action_type}"
 
    def _auto_resolve_crises(self) -> None:
        """
        Fix #5: calls check_full_protocol_completion() which exists on EventManager.
        Resolves a crisis only when all its required protocol steps are complete.
        """
        active_crises = self.event_manager.get_active_crises(
            self.state_machine.state.step
        )
        for crisis in active_crises:
            score, missing = self.event_manager.check_full_protocol_completion(crisis)
            if score >= 1.0 and not missing:
                crisis.resolved = True
 
    def _compute_reward(
        self,
        action_dict: dict,
        pre_state,
        candidate_runways: list[str],
        active_crises: list,
        episode_score: float,
        progress_delta: float,
        step_total: float,
        invalid_action: bool,
    ) -> Reward:
        if not self.grader:
            return self._get_zero_reward()
 
        
        penalty, is_zero = self.grader.check_hard_penalties()
        if invalid_action or is_zero:
            return Reward(
                total=0.0,
                priority_score=0.0,
                resource_match_score=0.0,
                eta_score=0.0,
                crisis_protocol_score=0.0,
                penalty=1.0,
                episode_score=round(episode_score, 4),
                progress_delta=round(progress_delta, 4),
                invalid_action=invalid_action,
            )
 
        priority_score = self._score_priority(pre_state, action_dict)
        resource_score = self._score_resource_match(pre_state, action_dict)
        eta_score = self._score_eta(pre_state, action_dict, candidate_runways)
        crisis_score = self._score_crisis_protocol(active_crises)
 
        total = (
            0.30 * priority_score
            + 0.25 * resource_score
            + 0.20 * eta_score
            + 0.15 * crisis_score
            + 0.10 * (1.0 - penalty)   
        )
 
        
        if self.grader._used_maintenance_runway():
            total = min(total, 0.2)
 
        return Reward(
            total=round(step_total, 4),
            priority_score=round(priority_score, 4),
            resource_match_score=round(resource_score, 4),
            eta_score=round(eta_score, 4),
            crisis_protocol_score=round(crisis_score, 4),
            penalty=round(penalty, 4),
            episode_score=round(episode_score, 4),
            progress_delta=round(progress_delta, 4),
            invalid_action=False,
        )
 
    def _score_priority(self, pre_state, action: dict) -> float:
        if action.get("action_type") not in ("assign_runway", "assign_gate"):
            return 0.0
        flight_id = action.get("flight_id")
        if not flight_id:
            return 0.0
        waiting = [
            fid for fid, f in pre_state.flights.items()
            if f.status in ("requesting_landing", "holding") and not f.assigned_runway
        ]
        if not waiting:
            return 0.0
        optimal = min(waiting, key=lambda fid: pre_state.get_flight_priority(fid))
        chosen_pri = pre_state.get_flight_priority(flight_id)
        optimal_pri = pre_state.get_flight_priority(optimal)
        if chosen_pri == optimal_pri:
            return 1.0
        gap = abs(chosen_pri - optimal_pri)
        return max(0.0, 1.0 - gap * 0.2)
 
    def _score_resource_match(self, pre_state, action: dict) -> float:
        flight_id = action.get("flight_id")
        target_id = action.get("target_id")
        action_type = action.get("action_type")
        if not flight_id or not target_id or action_type not in ("assign_gate", "assign_runway"):
            return 0.0
        flight = pre_state.flights.get(flight_id)
        if not flight:
            return 0.0
        if action_type == "assign_gate":
            gate = pre_state.gates.get(target_id)
            if not gate:
                return 0.0
            gtype = gate.type
            crisis = flight.crisis
            ftype = flight.flight_type
            if crisis == "hijack" or crisis == "bomb_threat":
                return 1.0 if gtype == GateType.ISOLATION else 0.0
            if ftype == "medevac" or crisis == "medical_onboard":
                return 1.0 if gtype == GateType.MEDICAL else 0.0
            if ftype == "cargo":
                return 1.0 if gtype in (GateType.CARGO, GateType.ISOLATION) else 0.0
            return 1.0 if gtype == GateType.PAX else 0.0
        if action_type == "assign_runway":
            runway = pre_state.runways.get(target_id)
            if not runway:
                return 0.0
            if runway.status in (RunwayStatus.MAINTENANCE, RunwayStatus.CLOSED):
                return 0.0
            return 1.0
        return 0.0
 
    def _score_eta(self, pre_state, action: dict, candidate_runways: list[str]) -> float:
        """
        Fix #8: when ETA lookup fails, return 0.5 (neutral) not 1.0 (optimal).
        ETA only scored on assign_runway — that's the moment the landing path is chosen.
        """
        if action.get("action_type") != "assign_runway":
            return 0.0
        flight_id = action.get("flight_id")
        runway_id = action.get("target_id")
        if not flight_id or not runway_id:
            return 0.0
        flight = pre_state.flights.get(flight_id)
        if not flight:
            return 0.0
 
        
        if flight.crisis in ("hijack", "bomb_threat"):
            target_type = GateType.ISOLATION
        elif flight.flight_type == "medevac" or flight.crisis == "medical_onboard":
            target_type = GateType.MEDICAL
        elif flight.flight_type == "cargo":
            target_type = GateType.CARGO
        else:
            target_type = GateType.PAX
 
        available_gates = pre_state.get_available_gates(target_type)
        gate_id = flight.assigned_gate or (available_gates[0] if available_gates else None)
 
        if not gate_id:
            return 0.5  
 
        score = self.state_machine.score_eta_optimality(
            runway_id,
            gate_id,
            flight.flight_type,
            candidate_runways=candidate_runways,
        )
        return score if score is not None else 0.5
 
    def _score_crisis_protocol(self, active_crises: list) -> float:
        if not active_crises:
            return 0.0
        scores = [
            self.event_manager.check_protocol_compliance(crisis)[0]
            for crisis in active_crises
        ]
        return round(sum(scores) / len(scores), 4) if scores else 0.0
 
    def _get_observation(self) -> Observation:
        from models import ActiveCrisis as ActiveCrisisModel
        from models import AirportContext as AirportContextModel
        from models import FlightInfo as FlightInfoModel
        state = self.state_machine.state
        active_crises = self.event_manager.get_active_crises(state.step)
        flights_data = [
            FlightInfoModel(
                flight_id=f.flight_id,
                flight_type=f.flight_type,
                status=f.status,
                fuel_remaining_mins=f.fuel_remaining_mins,
                passengers=f.passengers,
                crisis=f.crisis,
                current_location=f.current_location,
                steps_waiting=f.steps_waiting,
                assigned_runway=f.assigned_runway,
                assigned_gate=f.assigned_gate,
            )
            for f in state.flights.values()
        ]
        return Observation(
            step=state.step,
            airport_context=AirportContextModel(**state.airport_context.model_dump()),
            time_of_day=state.time_context.time_of_day,
            day_of_week=state.time_context.day_of_week,
            is_holiday=state.time_context.is_holiday,
            flights=flights_data,
            runways={k: v.model_dump() for k, v in state.runways.items()},
            gates={k: v.model_dump() for k, v in state.gates.items()},
            active_crises=[
                ActiveCrisisModel(
                    type=c.type,
                    flight_id=c.flight_id,
                    activate_step=c.activate_step,
                    target_id=c.target_id,
                )
                for c in active_crises
            ],
            available_runways=state.get_available_runways(),
            available_gates=state.get_available_gates(),
            ground_units=state.ground_units.model_dump(),
        )
 
    def _get_zero_reward(self) -> Reward:
        return Reward(
            total=0.0,
            priority_score=0.0,
            resource_match_score=0.0,
            eta_score=0.0,
            crisis_protocol_score=0.0,
            penalty=1.0,
            episode_score=round(self.previous_episode_score, 4),
            progress_delta=0.0,
            invalid_action=False,
        )
 
    def _check_done(self) -> bool:
        if self.state_machine.state.step >= self.max_steps:
            return True
        all_flights_done = all(
            flight.status in ("at_gate", "diverted")
            for flight in self.state_machine.state.flights.values()
        )
        all_crises_resolved = all(
            crisis.resolved for crisis in self.state_machine.state.active_crises
        )
        return all_flights_done and all_crises_resolved
 
