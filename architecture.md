AIRPORT OPS ENV
Complete Build Manual
OpenEnv Hackathon • Meta × PyTorch × Hugging Face × Scaler

1. What You Are Building
   AirportOpsEnv is an OpenEnv-compliant environment where an AI agent acts as an Airport Ground Operations Controller. The agent manages inbound and outbound flights — assigning runways, gates, and taxiways — while responding to dynamic crisis events like hijackings, medevacs, and runway fires. The environment is fully deterministic, containerised, and deployable to Hugging Face Spaces.

🎯 WHY THIS WINS: Novel domain not seen in OpenEnv. Real humans do this job. Fully deterministic grader. Crisis scenarios challenge frontier models in ways no other submission will.

Scoring Axis How AirportOpsEnv Scores
Real-world utility (30%) Airport controllers manage this exact workflow daily. Immediate training value.
Task & grader quality (25%) 3 tasks with clear difficulty progression. Deterministic partial rewards.
Environment design (20%) Clean state machine, typed Pydantic models, sensible episode boundaries.
Code quality (15%) OpenEnv spec, Dockerfile, HF Space, validated with openenv validate.
Creativity (10%) Crisis overlaid on scheduling — completely novel combination in OpenEnv.

2. System Architecture
   The environment is structured as a single Python package with three logical layers:

Layer What it does Key files
State Layer Tracks all airport resources and active events state.py, models.py
Environment Layer Implements OpenEnv interface env.py, openenv.yaml
Grader Layer Scores agent decisions deterministically graders/task1.py, task2.py, task3.py
Inference Layer Baseline agent using OpenAI client inference.py
Infra Layer Containerisation and deployment Dockerfile, app.py

Folder Structure
airport-ops-env/
├── env.py # AirportOpsEnv — core OpenEnv class
├── models.py # Pydantic models: Observation, Action, Reward
├── state.py # Airport state machine
├── events.py # Crisis event generator
├── data/
│ ├── eta_table.json # Pre-computed ETAs by time, day, holiday
│ └── scenarios/ # JSON files for each task scenario
│ ├── task1.json
│ ├── task2.json
│ └── task3.json
├── graders/
│ ├── base.py # BaseGrader class
│ ├── task1.py # Easy grader
│ ├── task2.py # Medium grader
│ └── task3.py # Hard grader
├── inference.py # Baseline agent (mandatory)
├── openenv.yaml # OpenEnv metadata
├── app.py # FastAPI app exposing endpoints
├── Dockerfile
└── README.md

3. OpenEnv Interface — Where and How to Use It
   📌 CRITICAL: OpenEnv is the judging framework. Every endpoint below is mandatory. The validator will call each one and fail your submission if any are missing or return wrong types.

3.1 openenv.yaml (write this first)
This file declares your environment to the OpenEnv registry. Place it in the project root.

# openenv.yaml

name: airport-ops-env
version: 1.0.0
description: Airport ground operations controller with crisis management
tags: [openenv, airport, dispatch, crisis, scheduling]
tasks:

- id: task1
  name: Basic Priority Landing
  difficulty: easy
  max_steps: 20
- id: task2
  name: Resource Conflict with Emergency
  difficulty: medium
  max_steps: 40
- id: task3
  name: Multi-Crisis Holiday Peak
  difficulty: hard
  max_steps: 80
  observation_space: See models.py:Observation
  action_space: See models.py:Action

  3.2 Pydantic Models (models.py)
  OpenEnv requires typed Observation, Action, and Reward models. Define all three in models.py.

from pydantic import BaseModel
from typing import List, Optional, Literal

class FlightInfo(BaseModel):
flight_id: str
flight_type: Literal['army','medevac','government','commercial','cargo']
status: Literal['requesting_landing','requesting_takeoff','at_gate','holding']
fuel_remaining_mins: int
passengers: int
crisis: Optional[Literal['hijack','bomb_threat','fire','medical_onboard']] = None

class Observation(BaseModel): # <-- OpenEnv requires this
step: int
time_of_day: str # e.g. '08:30'
day_of_week: str # e.g. 'Monday'
is_holiday: bool
flights: List[FlightInfo]
runways: dict # runway_id -> status
gates: dict # gate_id -> status
active_crises: List[str] # list of active crisis flight_ids

class Action(BaseModel): # <-- OpenEnv requires this
flight_id: str
action_type: Literal['assign_runway','assign_gate','hold','divert',
'scramble_security','scramble_fire','scramble_medical']
target_id: Optional[str] = None # runway_id or gate_id
use_secure_channel: bool = False # required for hijack protocol

class Reward(BaseModel): # <-- OpenEnv requires this
total: float # 0.0 - 1.0
priority_score: float
resource_match_score: float
eta_score: float
penalty: float
crisis_protocol_score: float

3.3 AirportOpsEnv Class (env.py)
This is the core class. It must implement reset(), step(), and state() exactly as shown.

class AirportOpsEnv:

    def reset(self, task_id: str) -> Observation:
        # Load scenario from data/scenarios/{task_id}.json
        # Initialise state machine
        # Return clean initial Observation
        ...

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        # 1. Validate action against current state
        # 2. Apply action to state (update runway/gate/flight status)
        # 3. Trigger any new events (crisis escalation, new arrivals)
        # 4. Score the action via the active grader
        # 5. Return (new_observation, reward, done, info)
        ...

    def state(self) -> dict:
        # Return full serialised state for debugging
        return self._state.dict()

3.4 FastAPI App (app.py)
OpenEnv communicates over HTTP. Wrap the env class in a FastAPI app with these exact endpoints.

from fastapi import FastAPI
from models import Observation, Action, Reward
from env import AirportOpsEnv

app = FastAPI()
env = AirportOpsEnv()

@app.post('/reset') # openenv validate calls this
def reset(task_id: str = 'task1') -> Observation:
return env.reset(task_id)

@app.post('/step') # openenv validate calls this
def step(action: Action) -> dict:
obs, reward, done, info = env.step(action)
return {'observation': obs, 'reward': reward, 'done': done, 'info': info}

@app.get('/state') # openenv validate calls this
def state() -> dict:
return env.state()

@app.get('/health') # HF Space ping
def health():
return {'status': 'ok'}

4. State Machine (state.py)
   The state machine tracks every resource and event in the airport. Build it as a Python class that the env.py reads and mutates on each step.

4.1 Airport Resources to Track
Resource Fields to track
Runways (3 total) id, status (free/occupied/closed/emergency), assigned_flight, type (landing/takeoff/dual)
Gates (8 total) id, status, assigned_flight, type (pax/cargo/medical/isolation)
Taxiways (4 total) id, clear (bool), blocked_by
Flights All FlightInfo fields + current_location, steps_waiting, crisis_escalation_timer
Ground units ambulances, fire trucks, security — count available

4.2 Priority Hierarchy (ground truth for grader)
Rank Flight Type Can preempt? Fuel override?
1 Army / Defense Yes — always N/A (already top)
2 Medevac / Medical Yes — over non-critical N/A
3 Government / VIP Yes — over commercial/cargo Bumps to rank 2 if critical
4 Commercial (passengers) Only if fuel critical Bumps to rank 1 if < 10 mins
5 Cargo Never Bumps to rank 3 if < 10 mins

4.3 Time & Holiday Modifiers
Store a pre-computed ETA lookup table in data/eta_table.json keyed by: time_bucket × day_type × route_id.

# eta_table.json structure

{
'rush_hour_weekday': {
'runway1_to_gate_A': 18, # minutes
'runway2_to_gate_B': 22,
...
},
'off_peak_weekday': { ... },
'weekend': { ... },
'holiday': { ... } # Diwali, Republic Day etc — worst ETAs
}

5. Crisis Events (events.py)
   Crises are embedded in the scenario JSON and activate at specific steps. Each crisis type has a required protocol the agent must follow. The grader checks protocol compliance deterministically.

5.1 Crisis Types & Required Protocols
🚨 Medical Emergency (MEDEVAC / Passenger Onboard)
• Agent must clear nearest runway within 1 step
• Must assign a medical-type gate (not cargo or standard pax)
• Must dispatch ambulance unit (action_type: scramble_medical)
• All taxiing aircraft must be held to clear path
• Hard penalty if medical gate not assigned

💣 Bomb Threat
• Flight must be assigned to isolation bay — not any passenger gate
• Nearby ground movement must halt
• Notification sequence: Security → Fire → Police (order matters for grader)
• Other runways must stay operational — airport does not shut down
• Hard penalty: reward = 0.0 if sent to passenger terminal

✈️ Hijacking (Squawk 7500)
• Must assign remote stand away from terminal and other aircraft
• use_secure_channel: true is MANDATORY in the Action — graded explicitly
• Must scramble security (action_type: scramble_security)
• Normal ops must continue for all other flights
• Hard penalty: reward = 0.0 if public gate assigned OR secure channel not used

🔥 Runway Fire / Incursion
• Go-around must be issued to all aircraft on final approach within 1 step
• Fire brigade dispatched (action_type: scramble_fire)
• Affected runway marked closed in state — cannot be used until cleared
• All pending landings rerouted to alternate runways
• Hard penalty if any aircraft lands on the burning runway

⛽ Fuel Emergency (Mayday — < 10 mins fuel)
• Overrides ALL priority hierarchy including Army
• Closest available runway assigned regardless of type
• If no runway free: agent must instruct a taxiing aircraft to vacate
• Hard penalty if mayday flight is made to wait even 1 step

⚠️ DUAL CRISIS (Task 3 only): Two simultaneous crises (e.g. hijack + runway fire) will activate. Agent must execute independent protocols for each without letting one interfere with the other. This is what makes the hard task genuinely hard.

6. Three Tasks — Scenarios & Graders
   Task 1 — Easy: Basic Priority Landing
   Parameter Value
   Flights inbound 5 simultaneous requests
   Crisis events 1 medevac declaring emergency mid-approach
   Resources 3 runways free, 6 gates available
   Time context 10:00 AM Tuesday (off-peak)
   Max steps 20
   What agent must do Queue 4 normal flights correctly by priority; clear runway for medevac immediately

Grader for Task 1:

# graders/task1.py

def grade(actions, ground_truth, state_history) -> float:
score = 0.0 # Check 1: Priority order of non-crisis flights (0.4 weight)
order_score = check_priority_ordering(actions, ground_truth['expected_order']) # Check 2: Medevac runway cleared within 1 step of crisis activation (0.4)
medevac_score = 1.0 if state_history[crisis_step]['runway_cleared'] else 0.0 # Check 3: Correct gate type assigned to each flight (0.2)
gate_score = check_gate_types(actions, ground_truth['gate_assignments'])
score = 0.4*order_score + 0.4*medevac_score + 0.2\*gate_score
return min(max(score, 0.0), 1.0)

Task 2 — Medium: Resource Conflict with Emergency
Parameter Value
Flights inbound 8 simultaneous — 1 medevac, 1 army, 3 commercial, 2 cargo, 1 VIP
Crisis events Fuel emergency on commercial flight (< 10 mins), bomb threat on cargo
Resources 2 runways free (1 under maintenance), 4 gates occupied
Time context 08:15 AM Monday (rush hour)
Max steps 40
What agent must do Resolve priority conflict between army and fuel emergency; handle bomb threat correctly

Grader for Task 2:

# graders/task2.py

def grade(actions, ground_truth, state_history) -> float: # Check 1: Fuel emergency overrides army (0.3) — fuel < 10 mins = priority 1
fuel_override = 1.0 if fuel_flight_landed_before_army else 0.0 # Check 2: Bomb threat flight → isolation bay, not terminal (0.3)
bomb_protocol = 1.0 if isolation_bay_used else 0.0 # else hard penalty = 0 # Check 3: Maintenance runway not used (0.2)
maintenance_penalty = 0.0 if maintenance_runway_used else 1.0 # Check 4: Priority order for remaining flights (0.2)
order_score = check_priority_ordering(remaining_actions)
score = 0.3*fuel_override + 0.3*bomb_protocol + 0.2*maintenance_penalty + 0.2*order_score
return min(max(score, 0.0), 1.0)

Task 3 — Hard: Multi-Crisis Holiday Peak
Parameter Value
Flights inbound 15 simultaneous — army charter, 2 medevacs, VIP, 8 commercial, 2 cargo, 1 hijack
Crisis events Active hijacking + runway fire simultaneously; cargo backlog from curfew
Resources 1 runway under maintenance, 3 gates occupied by delayed flights
Time context 06:00 AM Diwali (holiday peak, night curfew just lifted)
Max steps 80
What agent must do Execute independent hijack + fire protocols; keep airport partially running; clear cargo backlog

# graders/task3.py

def grade(actions, ground_truth, state_history) -> float: # Check 1: Hijack protocol correct (secure channel, remote stand) (0.25)
hijack_score = 1.0 if (secure_channel_used and remote_stand_assigned) else 0.0 # Check 2: Fire protocol correct (go-around issued, fire scrambled) (0.25)
fire_score = check_fire_protocol(actions, state_history) # Check 3: Crises handled independently — neither disrupts the other (0.20)
isolation_score = 1.0 if not cross_crisis_interference else 0.5 # Check 4: Normal ops continued for non-crisis flights (0.15)
throughput_score = flights_processed / total_processable_flights # Check 5: Cargo backlog cleared without blocking pax gates (0.15)
cargo_score = check_cargo_handling(actions, ground_truth)
score = (0.25*hijack_score + 0.25*fire_score + 0.20*isolation_score + 0.15*throughput_score + 0.15\*cargo_score)
return min(max(score, 0.0), 1.0)

7. Reward Function Design
   The reward is computed at every step (not just episode end) so the agent gets continuous learning signal. The Reward Pydantic model is returned from every step() call.

Component Weight Description
priority_score 30% Did agent respect the priority hierarchy for this action?
resource_match_score 25% Was the right gate/runway type assigned to the right flight type?
eta_score 20% How close to optimal travel time was the chosen assignment?
crisis_protocol_score 15% Was the crisis-specific protocol followed correctly?
penalty 10% Deducted for hard rule violations (wrong gate for hijack, etc.)

def compute_reward(action, state, grader) -> Reward:
priority = grader.score_priority(action, state)
resource = grader.score_resource_match(action, state)
eta = grader.score_eta(action, state)
crisis = grader.score_crisis_protocol(action, state)
penalty = grader.compute_penalty(action, state) # always negative or 0

    total = (0.30 * priority + 0.25 * resource +
             0.20 * eta + 0.15 * crisis + 0.10 * (1 - abs(penalty)))

    return Reward(
        total=round(min(max(total, 0.0), 1.0), 4),
        priority_score=priority,
        resource_match_score=resource,
        eta_score=eta,
        crisis_protocol_score=crisis,
        penalty=penalty
    )

💡 HARD PENALTIES: If agent sends hijacked plane to passenger gate → total reward = 0.0 immediately. If agent uses maintenance runway → total reward capped at 0.2. These are non-negotiable protocol violations.

8. Baseline Inference Script (inference.py)
   📌 MANDATORY: The inference.py file must be in the root directory, use OpenAI client, read API_BASE_URL and MODEL_NAME from env vars, and emit [START], [STEP], [END] logs to stdout in exact format.

# inference.py — mandatory file, must be named exactly this

import os, json, requests
from openai import OpenAI

API_BASE_URL = os.environ['API_BASE_URL']
MODEL_NAME = os.environ['MODEL_NAME']
ENV_URL = os.environ.get('ENV_URL', 'http://localhost:8000')

client = OpenAI(base_url=API_BASE_URL, api_key=os.environ.get('OPENAI_API_KEY','x'))

SYSTEM_PROMPT = '''You are an airport ground operations controller.
You receive airport state and must decide one action per step.
Priority order: army > medevac > government > commercial > cargo.
Fuel emergency (< 10 mins) overrides ALL priorities.
Always respond with valid JSON matching the Action schema.'''

def run_task(task_id: str):
obs = requests.post(f'{ENV_URL}/reset', params={'task_id': task_id}).json()
print(json.dumps({'type': '[START]', 'task_id': task_id}))
done = False
while not done:
response = client.chat.completions.create(
model=MODEL_NAME,
messages=[
{'role': 'system', 'content': SYSTEM_PROMPT},
{'role': 'user', 'content': f'Airport state: {json.dumps(obs)}'}
]
)
action = json.loads(response.choices[0].message.content)
result = requests.post(f'{ENV_URL}/step', json=action).json()
print(json.dumps({'type': '[STEP]', 'action': action, 'reward': result['reward']}))
obs = result['observation']
done = result['done']
print(json.dumps({'type': '[END]', 'task_id': task_id, 'final_reward': result['reward']}))

if **name** == '**main**':
for task in ['task1', 'task2', 'task3']:
run_task(task)

9. Dockerfile & Deployment
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 8000
   CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

requirements.txt:
fastapi>=0.110.0
uvicorn>=0.29.0
pydantic>=2.0.0
openai>=1.0.0
requests>=2.31.0

Hugging Face Space Setup 1. Create a new HF Space → Docker SDK → tag with openenv 2. Push your repo to the Space (git remote add space https://huggingface.co/spaces/YOUR/airport-ops-env) 3. Set environment secrets: API_BASE_URL, MODEL_NAME, OPENAI_API_KEY 4. Verify the Space URL returns 200 on GET /health 5. Run: openenv validate --url https://YOUR-SPACE.hf.space

10. Recommended Build Order
    🏗️ FOLLOW THIS ORDER: Each step depends on the previous. Don't start coding inference.py before env.py is validated.

Step What to build Done when...
1 models.py — Pydantic models All 3 models import cleanly
2 data/eta_table.json + scenario JSONs 3 scenario files exist with valid flight data
3 state.py — Airport state machine Can instantiate and call .dict()
4 events.py — Crisis event triggers Crisis activates at correct step in test
5 graders/task1.py Returns float 0.0–1.0 for a dummy action sequence
6 env.py — reset(), step(), state() All 3 methods return correct types
7 app.py — FastAPI wrapper curl localhost:8000/health returns ok
8 graders/task2.py and task3.py All 3 graders tested with dummy sequences
9 inference.py — baseline agent Runs all 3 tasks, produces [START][STEP][END] logs
10 Dockerfile docker build && docker run works cleanly
11 openenv.yaml openenv validate passes all checks
12 Deploy to HF Space Space URL responds, openenv validate --url passes
13 README.md All required sections written

11. Common Pitfalls to Avoid
    ❌ Grader always returns same score The disqualification list explicitly flags this. Make sure partial rewards vary step by step — use the component scores, not just a binary.

❌ Forgetting use_secure_channel on hijack This is the most likely mistake under time pressure. The grader explicitly checks this field. Without it the hijack protocol score is 0.

❌ Episode never ends (done=True never fires) Set clear termination conditions: all flights processed, max_steps exceeded, or hard violation. Missing done=True will fail openenv validate.

❌ Inference script wrong log format The [START], [STEP], [END] format is parsed programmatically by judges. Any deviation in field names or ordering will cause incorrect scoring.

❌ ETA table not used If your eta_score always returns 1.0 because you didn't implement the lookup, judges will notice and score environment design low.

✅ Pro tip: Build task1 end-to-end first (env + grader + inference). Validate it completely. Then layer in task2 and task3 complexity. Don't try to build all 3 tasks at once.

Good luck at the hackathon! ✈️
AirportOpsEnv • OpenEnv Hackathon • Bangalore 2025
AirportOpsEnv • OpenEnv Hackathon • Bangalore 2025
