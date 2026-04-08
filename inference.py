import json
import os
from typing import Optional

import requests
from openai import OpenAI


ENV_NAME = "airport-ops-env"
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
SUCCESS_THRESHOLD = 0.60

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = """You are an airport ground operations controller for a realistic Indian airport benchmark.
Return exactly one valid JSON action object per turn.

Rules:
- Prioritize active crises before routine traffic.
- Fuel emergencies must get a runway immediately.
- Medical emergencies should follow: assign_runway -> assign_gate(G_MED) -> scramble_medical.
- Hijacking should follow: assign_runway -> assign_gate(G_ISO, use_secure_channel=true) -> scramble_security.
- Bomb threat should follow: assign_runway -> assign_gate(G_ISO) -> hold nearby traffic / notify security.
- Runway fire should follow: hold other inbound flights -> scramble_fire -> close the affected runway.
- Never assign a gate before a runway.
- Never use a runway that is maintenance or closed.
- Never assign hijack or bomb threat flights to pax gates.
- Use the structured active_crises list in the observation; each item includes the crisis type and target.

Output JSON only.
"""


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _reward_str(value: float) -> str:
    return f"{value:.2f}"


def _error_str(error: Optional[str]) -> str:
    return error if error else "null"


def _action_str(action: dict) -> str:
    return json.dumps(action, separators=(",", ":"), sort_keys=True)


def log_start(task_id: str) -> None:
    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)


def log_step(step: int, action: dict, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={_action_str(action)} reward={_reward_str(reward)} "
        f"done={_bool_str(done)} error={_error_str(error)}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: list[float]) -> None:
    rewards_str = ",".join(_reward_str(reward) for reward in rewards)
    print(
        f"[END] success={_bool_str(success)} steps={steps} rewards={rewards_str}",
        flush=True,
    )


def get_flight_priority(flight: dict) -> int:
    if flight.get("fuel_remaining_mins", 999) < 10:
        return 0
    return {
        "army": 1,
        "medevac": 2,
        "government": 3,
        "commercial": 4,
        "cargo": 5,
    }.get(flight.get("flight_type", "commercial"), 6)


def _route_gate_type(flight: dict, active_types: set[str]) -> str:
    if "hijacking" in active_types or flight.get("crisis") == "hijack":
        return "isolation"
    if "bomb_threat" in active_types or flight.get("crisis") == "bomb_threat":
        return "isolation"
    if "medical_emergency" in active_types or flight.get("crisis") == "medical_onboard":
        return "medical"
    if flight.get("flight_type") == "medevac":
        return "medical"
    if flight.get("flight_type") == "cargo":
        return "cargo"
    return "pax"


def heuristic_action(obs: dict) -> dict:
    flights = obs.get("flights", [])
    active_crises = obs.get("active_crises", [])
    runways = obs.get("runways", {})
    gates = obs.get("gates", {})
    ground_units = obs.get("ground_units", {})

    flight_map = {flight["flight_id"]: flight for flight in flights}
    free_runways = [rid for rid, runway in runways.items() if runway.get("status") == "free"]
    free_by_type: dict[str, list[str]] = {}
    for gate_id, gate in gates.items():
        if gate.get("status") == "free":
            free_by_type.setdefault(gate.get("type", "pax"), []).append(gate_id)

    active_by_flight: dict[str, list[dict]] = {}
    for crisis in active_crises:
        active_by_flight.setdefault(crisis["flight_id"], []).append(crisis)

    crisis_priority = {
        "fuel_emergency": 0,
        "medical_emergency": 1,
        "hijacking": 2,
        "bomb_threat": 3,
        "runway_fire": 4,
    }
    ordered_crises = sorted(
        active_crises,
        key=lambda crisis: (crisis_priority.get(crisis["type"], 99), crisis["activate_step"]),
    )

    for crisis in ordered_crises:
        flight_id = crisis["flight_id"]
        flight = flight_map.get(flight_id)
        if not flight:
            continue

        assigned_runway = flight.get("assigned_runway")
        assigned_gate = flight.get("assigned_gate")
        crisis_type = crisis["type"]

        if crisis_type == "fuel_emergency" and not assigned_runway and free_runways:
            return {
                "flight_id": flight_id,
                "action_type": "assign_runway",
                "target_id": free_runways[0],
                "use_secure_channel": False,
            }

        if crisis_type == "medical_emergency":
            if assigned_gate and ground_units.get("ambulances", 0) > 0:
                return {
                    "flight_id": flight_id,
                    "action_type": "scramble_medical",
                    "use_secure_channel": False,
                }
            if not assigned_runway and not assigned_gate and free_runways:
                return {
                    "flight_id": flight_id,
                    "action_type": "assign_runway",
                    "target_id": free_runways[0],
                    "use_secure_channel": False,
                }
            if assigned_runway and not assigned_gate and free_by_type.get("medical"):
                return {
                    "flight_id": flight_id,
                    "action_type": "assign_gate",
                    "target_id": free_by_type["medical"][0],
                    "use_secure_channel": False,
                }
        if crisis_type == "hijacking":
            if assigned_gate and ground_units.get("security_teams", 0) > 0:
                return {
                    "flight_id": flight_id,
                    "action_type": "scramble_security",
                    "use_secure_channel": False,
                }
            if not assigned_runway and not assigned_gate and free_runways:
                return {
                    "flight_id": flight_id,
                    "action_type": "assign_runway",
                    "target_id": free_runways[0],
                    "use_secure_channel": False,
                }
            if assigned_runway and not assigned_gate and free_by_type.get("isolation"):
                return {
                    "flight_id": flight_id,
                    "action_type": "assign_gate",
                    "target_id": free_by_type["isolation"][0],
                    "use_secure_channel": True,
                }
        if crisis_type == "bomb_threat":
            holding_candidate = next(
                (
                    other["flight_id"]
                    for other in flights
                    if other["flight_id"] != flight_id
                    and other.get("status") in ("requesting_landing", "holding")
                    and not other.get("assigned_runway")
                ),
                None,
            )
            if assigned_gate and holding_candidate:
                return {
                    "flight_id": holding_candidate,
                    "action_type": "hold",
                    "use_secure_channel": False,
                    "notify_authorities": ["security"],
                }
            if not assigned_runway and not assigned_gate and free_runways:
                return {
                    "flight_id": flight_id,
                    "action_type": "assign_runway",
                    "target_id": free_runways[0],
                    "use_secure_channel": False,
                    "notify_authorities": ["security"],
                }
            if assigned_runway and not assigned_gate and free_by_type.get("isolation"):
                return {
                    "flight_id": flight_id,
                    "action_type": "assign_gate",
                    "target_id": free_by_type["isolation"][0],
                    "use_secure_channel": False,
                    "notify_authorities": ["security"],
                }
            if holding_candidate:
                return {
                    "flight_id": holding_candidate,
                    "action_type": "hold",
                    "use_secure_channel": False,
                    "notify_authorities": ["security"],
                }

        if crisis_type == "runway_fire":
            hold_candidate = next(
                (
                    other["flight_id"]
                    for other in sorted(flights, key=get_flight_priority)
                    if other["flight_id"] != flight_id
                    and other.get("status") in ("requesting_landing", "holding")
                    and not other.get("assigned_runway")
                ),
                None,
            )
            if hold_candidate:
                return {
                    "flight_id": hold_candidate,
                    "action_type": "hold",
                    "use_secure_channel": False,
                }
            if ground_units.get("fire_trucks", 0) > 0:
                return {
                    "flight_id": flight_id,
                    "action_type": "scramble_fire",
                    "use_secure_channel": False,
                }
            target_id = crisis.get("target_id") or next(iter(runways.keys()), None)
            if target_id:
                return {
                    "flight_id": flight_id,
                    "action_type": "close_runway",
                    "target_id": target_id,
                    "use_secure_channel": False,
                }

    waiting = sorted(
        [
            flight
            for flight in flights
            if flight.get("status") in ("requesting_landing", "holding")
            and not flight.get("assigned_runway")
        ],
        key=get_flight_priority,
    )
    if waiting and free_runways:
        flight = waiting[0]
        return {
            "flight_id": flight["flight_id"],
            "action_type": "assign_runway",
            "target_id": free_runways[0],
            "use_secure_channel": False,
        }

    taxiing = [
        flight
        for flight in flights
        if flight.get("assigned_runway") and not flight.get("assigned_gate")
    ]
    if taxiing:
        flight = sorted(taxiing, key=get_flight_priority)[0]
        active_types = {crisis["type"] for crisis in active_by_flight.get(flight["flight_id"], [])}
        gate_type = _route_gate_type(flight, active_types)
        if free_by_type.get(gate_type):
            return {
                "flight_id": flight["flight_id"],
                "action_type": "assign_gate",
                "target_id": free_by_type[gate_type][0],
                "use_secure_channel": gate_type == "isolation" and "hijacking" in active_types,
            }

    if waiting:
        return {
            "flight_id": waiting[0]["flight_id"],
            "action_type": "hold",
            "use_secure_channel": False,
        }

    any_flight = flights[0]["flight_id"] if flights else "FL001"
    return {"flight_id": any_flight, "action_type": "hold", "use_secure_channel": False}


def get_llm_action(obs: dict, history: list[dict]) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in history[-3:]:
        messages.append({"role": "user", "content": json.dumps(item["observation"])})
        messages.append({"role": "assistant", "content": json.dumps(item["action"])})
    messages.append(
        {
            "role": "user",
            "content": f"Observation:\n{json.dumps(obs, indent=2)}\n\nReturn the next action JSON.",
        }
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=200,
        )
        raw = (response.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        action = json.loads(raw)
        action.setdefault("use_secure_channel", False)
        return action
    except Exception:
        return heuristic_action(obs)


def fetch_tasks() -> list[str]:
    try:
        response = requests.get(f"{ENV_URL}/tasks", timeout=30)
        response.raise_for_status()
        tasks = response.json().get("tasks", [])
        task_ids = [task["id"] for task in tasks if "id" in task]
        return task_ids or ["task1", "task2", "task3"]
    except Exception:
        return ["task1", "task2", "task3"]


def fetch_grade() -> float:
    try:
        response = requests.get(f"{ENV_URL}/grade", timeout=30)
        response.raise_for_status()
        return float(response.json().get("episode_score", 0.0))
    except Exception:
        return 0.0


def run_task(task_id: str) -> dict:
    log_start(task_id)

    rewards: list[float] = []
    history: list[dict] = []
    step_count = 0
    final_score = 0.0
    done = False
    obs: dict = {}

    try:
        response = requests.post(f"{ENV_URL}/reset", params={"task_id": task_id}, timeout=30)
        response.raise_for_status()
        obs = response.json()

        while not done:
            step_count += 1
            action = get_llm_action(obs, history)
            try:
                response = requests.post(f"{ENV_URL}/step", json=action, timeout=30)
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:
                log_step(step_count, action, 0.0, False, str(exc))
                break

            reward_obj = payload.get("reward", {})
            reward_value = (
                float(reward_obj.get("total", 0.0)) if isinstance(reward_obj, dict) else 0.0
            )
            info = payload.get("info", {}) if isinstance(payload.get("info", {}), dict) else {}
            error = info.get("last_action_error") or info.get("error")
            done = bool(payload.get("done", False))

            rewards.append(reward_value)
            history.append({"observation": obs, "action": action})
            obs = payload.get("observation", obs)
            log_step(step_count, action, reward_value, done, error)

        final_score = fetch_grade()
    finally:
        success = done and final_score >= SUCCESS_THRESHOLD
        log_end(success, step_count, rewards)

    return {
        "task_id": task_id,
        "score": round(final_score, 4),
        "steps": step_count,
        "success": done and final_score >= SUCCESS_THRESHOLD,
    }


def main() -> int:
    results = [run_task(task_id) for task_id in fetch_tasks()]
    return 0 if all(result["success"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
