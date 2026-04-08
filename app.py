from fastapi import FastAPI, HTTPException
from models import Observation, Action
from env import AirportOpsEnv
 
app = FastAPI(
    title="AirportOpsEnv",
    description="OpenEnv-compliant Airport Ground Operations environment",
    version="1.0.0",
)
 
env = AirportOpsEnv()
 
 
@app.post("/reset", response_model=Observation)
def reset(task_id: str = "task1") -> Observation:
    if task_id not in ("task1", "task2", "task3"):
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}. Use task1, task2, or task3.")
    return env.reset(task_id)
 
 
@app.post("/step")
def step(action: Action) -> dict:
    if not env.is_ready():
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }
 
 
@app.get("/state")
def state() -> dict:
    return env.state()
 
 
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
 
 
@app.get("/tasks")
def list_tasks() -> dict:
    """OpenEnv task enumeration endpoint."""
    return {
        "tasks": [
            {"id": "task1", "name": "Basic Priority Landing", "difficulty": "easy", "max_steps": 20},
            {"id": "task2", "name": "Resource Conflict with Emergency", "difficulty": "medium", "max_steps": 40},
            {"id": "task3", "name": "Multi-Crisis Holiday Peak", "difficulty": "hard", "max_steps": 80},
        ]
    }


@app.get("/grade")
def grade() -> dict:
    if not env.is_ready():
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    return env.grade()
 
