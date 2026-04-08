# AirportOpsEnv - Build Phases

## Overview
AirportOpsEnv is an OpenEnv-compliant environment where an AI agent acts as an Airport Ground Operations Controller, managing flight flow and crisis events.

---

## Phase 1: Foundation
**Status:** Complete ✅

### Files created:
- [x] `models.py` - Pydantic models: `FlightInfo`, `Observation`, `Action`, `Reward`
- [x] `data/` directory
  - [x] `data/eta_table.json` - Pre-computed ETAs by time, day, holiday
  - [x] `data/scenarios/task1.json` - Easy scenario stub
  - [x] `data/scenarios/task2.json` - Medium scenario stub
  - [x] `data/scenarios/task3.json` - Hard scenario stub
- [x] `state.py` - Airport state machine

### Deliverables:
- [x] All 3 Pydantic models import cleanly
- [x] 3 scenario files exist with valid structure
- [x] State machine can instantiate and serialize to dict

---

## Phase 2: Core Environment
**Status:** Complete ✅

### Files created:
- [x] `events.py` - Crisis event triggers and protocols
- [x] `graders/base.py` - BaseGrader class
- [x] `graders/task1.py` - Easy grader (priority order, medevac, gate types)
- [x] `graders/task2.py` - Medium grader (fuel override, bomb threat, maintenance)
- [x] `graders/task3.py` - Hard grader (dual crisis, cargo backlog, throughput)
- [x] `env.py` - `AirportOpsEnv` with `reset()`, `step()`, `state()`

### Deliverables:
- [x] Crises activate at correct step in test
- [x] All 3 graders return float 0.0-1.0 for dummy action sequences
- [x] `reset()`, `step()`, `state()` return correct types

---

## Phase 3: API & Inference
**Status:** Complete ✅

### Files created:
- [x] `app.py` - FastAPI wrapper with `/reset`, `/step`, `/state`, `/health`
- [x] `inference.py` - Enhanced baseline agent with priority logic and crisis detection

### Deliverables:
- [x] `curl localhost:8000/health` returns `{"status": "ok"}`
- [x] Inference script runs all 3 tasks with `[START]`, `[STEP]`, `[END]` logs

---

## Phase 4: Deployment
**Status:** Complete ✅

### Files created:
- [x] `openenv.yaml` - OpenEnv metadata
- [x] `requirements.txt` - Dependencies
- [x] `Dockerfile` - Containerization
- [x] `README.md` - Documentation

### Deliverables:
- [ ] `openenv validate` passes all checks (requires openenv CLI)
- [ ] Docker build and run works cleanly
- [ ] HF Space deployment functional

---

## Priority Hierarchy (Ground Truth)
| Rank | Flight Type | Fuel Override |
|------|-------------|---------------|
| 1 | Army / Defense | N/A |
| 2 | Medevac / Medical | N/A |
| 3 | Government / VIP | → Rank 2 if critical fuel |
| 4 | Commercial | → Rank 1 if < 10 mins |
| 5 | Cargo | → Rank 3 if < 10 mins |

**Fuel emergency (< 10 mins) overrides ALL priorities**

---

## Crisis Protocols
| Crisis | Required Actions |
|--------|-----------------|
| Medical Emergency | Clear nearest runway, assign medical gate, scramble ambulance |
| Bomb Threat | Assign isolation bay, halt nearby movement, notify: Security → Fire → Police |
| Hijacking | Remote stand, `use_secure_channel: true`, scramble security |
| Runway Fire | Issue go-around, scramble fire brigade, close runway |
| Fuel Emergency | Closest runway immediately (may instruct taxiing aircraft to vacate) |

---

## Reward Components
| Component | Weight | Description |
|-----------|--------|-------------|
| priority_score | 30% | Respected priority hierarchy |
| resource_match_score | 25% | Correct gate/runway type for flight type |
| eta_score | 20% | Optimal travel time for assignment |
| crisis_protocol_score | 15% | Crisis protocol followed correctly |
| penalty | 10% | Hard rule violations deducted |

**Hard Penalties:**
- Hijacked plane to passenger gate → total = 0.0
- Using maintenance runway → total capped at 0.2

---

## Build Decisions
- [x] Enhanced inference agent (priority logic + crisis detection)
- [x] Stub scenario data first (populate fully later)

---

## Project Structure
```
airport-ops-env/
├── models.py           # Pydantic models
├── state.py             # Airport state machine
├── events.py            # Crisis event management
├── env.py               # Core OpenEnv class
├── app.py               # FastAPI wrapper
├── inference.py         # Baseline agent
├── openenv.yaml         # OpenEnv metadata
├── requirements.txt     # Dependencies
├── Dockerfile           # Containerization
├── README.md           # Documentation
├── phases.md            # This file
├── data/
│   ├── eta_table.json   # ETA lookup table
│   └── scenarios/
│       ├── task1.json   # Easy scenario
│       ├── task2.json   # Medium scenario
│       └── task3.json   # Hard scenario
└── graders/
    ├── __init__.py
    ├── base.py          # Base grader
    ├── task1.py         # Easy grader
    ├── task2.py         # Medium grader
    └── task3.py         # Hard grader
```
