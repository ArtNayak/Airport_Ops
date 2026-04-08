---
title: Airport Ops Env

colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "1.0.0"
python_version: "3.11"
app_file: app.py
app_port: 8000
pinned: false
---

# AirportOpsEnv

AirportOpsEnv is an OpenEnv-compatible airport ground-operations benchmark where an agent manages inbound traffic, gate allocation, emergency response, and crisis protocols. The environment is designed around realistic controller workflows rather than game mechanics, and the bundled scenarios are now anchored to Delhi Airport-inspired operating context for an Indian aviation domain feel.

## Why This Environment

- Real work simulation: runway assignment, gate assignment, dispatching security/fire/medical units, and managing time-sensitive crises.
- Deterministic grading: each task has a task-specific programmatic grader that produces a reproducible score from `0.0` to `1.0`.
- Dense feedback: step rewards track incremental progress while the final episode score comes from the task grader.
- Hackathon-ready baseline: `inference.py` stays in the repo root, uses the OpenAI client, reads the required environment variables, and emits the guideline-compliant `[START]`, `[STEP]`, and `[END]` lines.

## Observation Space

Each observation contains:

- `step`: current environment step
- `airport_context`: airport metadata, runway aliases, and source links
- `time_of_day`, `day_of_week`, `is_holiday`
- `flights`: typed flight records including `status`, `fuel_remaining_mins`, `crisis`, `current_location`, `assigned_runway`, and `assigned_gate`
- `runways` and `gates`: current resource state
- `active_crises`: structured crisis objects with `type`, `flight_id`, `activate_step`, and optional `target_id`
- `available_runways`, `available_gates`, `ground_units`

## Action Space

Each action is a typed object with:

- `flight_id`
- `action_type`: `assign_runway | assign_gate | hold | divert | scramble_security | scramble_fire | scramble_medical | close_runway | vacate_runway`
- `target_id`: required for target-based actions
- `use_secure_channel`
- `notify_authorities`

## Reward Space

Each step returns:

- `total`: dense progress reward for the step
- `priority_score`
- `resource_match_score`
- `eta_score`
- `crisis_protocol_score`
- `penalty`
- `episode_score`: current deterministic grader score
- `progress_delta`: improvement from the previous step
- `invalid_action`

## Tasks

| Task | Difficulty | Goal |
|------|------------|------|
| `task1` | Easy | Handle a medevac emergency correctly while processing a small inbound bank |
| `task2` | Medium | Resolve a fuel emergency and bomb threat during rush hour without breaking priority rules |
| `task3` | Hard | Manage simultaneous hijacking and runway-fire crises during a Diwali surge while maintaining throughput |

## Indian Airport Context

The scenarios now include Delhi Airport-inspired metadata via `airport_context`. To keep the benchmark deterministic and hackathon-safe, the environment uses stable simulated IDs (`R1`, `R2`, `R3`) mapped to real-world runway aliases rather than making live network calls during evaluation.

Reference inputs used for the airport profile:

- Delhi Airport fact sheet: `https://site.newdelhiairport.in/medias/factsheet/`
- Delhi Airport Terminal 3 overview: `https://site.newdelhiairport.in/terminals/terminal-3`
- Delhi Airport live operations dashboard: `https://dy.newdelhiairport.in/`

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For local runs, you can copy `.env.example` to `.env` and set `HF_TOKEN` there. The repo defaults to the Hugging Face OpenAI-compatible router.

## Run The API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Available endpoints:

- `POST /reset?task_id=task1`
- `POST /step`
- `GET /state`
- `GET /grade`
- `GET /tasks`
- `GET /health`

## Quick Smoke Test

```bash
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/reset?task_id=task1"
curl -X POST "http://localhost:8000/step" ^
  -H "Content-Type: application/json" ^
  -d "{\"flight_id\":\"FL001\",\"action_type\":\"assign_runway\",\"target_id\":\"R1\"}"
curl http://localhost:8000/grade
```

## Inference

`inference.py` is the required hackathon entrypoint. It expects:

- `API_BASE_URL` with a default
- `MODEL_NAME` with a default
- `HF_TOKEN` as a required variable
- `ENV_URL` for the environment server, defaulting to `http://localhost:8000`

Example:

```bash
set HF_TOKEN=your_token_here
python inference.py
```

By default, `inference.py` uses `https://router.huggingface.co/v1` with `MODEL_NAME=Qwen/Qwen2.5-72B-Instruct`.
It also accepts `OPENAI_API_KEY` or `API_KEY` for local use if you point `API_BASE_URL` at another compatible provider.

## Baseline Scores

Local deterministic heuristic-fallback baseline after the fixes in this repo:

| Task | Score |
|------|-------|
| `task1` | `0.9556` |
| `task2` | `0.6000` |
| `task3` | `0.4275` |

These were measured by running the bundled heuristic policy against the environment using the current task graders.

## Notes

- Invalid actions now consume time, return structured errors, and do not grant progress reward.
- Episodes no longer finish early while crises remain unresolved.
- The `/grade` endpoint exposes the current deterministic task score for debugging and baseline evaluation.
