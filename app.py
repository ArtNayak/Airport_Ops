from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from models import Observation, Action
from env import AirportOpsEnv
 
app = FastAPI(
    title="AirportOpsEnv",
    description="OpenEnv-compliant Airport Ground Operations environment",
    version="1.0.0",
)
 
env = AirportOpsEnv()


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> str:
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>AirportOpsEnv</title>
      <style>
        :root {
          color-scheme: light;
          --bg: #f4f8fb;
          --card: #ffffff;
          --text: #102033;
          --muted: #536579;
          --line: #d7e3ee;
          --accent: #0c6a8c;
          --accent-2: #118ab2;
        }
        * { box-sizing: border-box; }
        body {
          margin: 0;
          font-family: "Segoe UI", Arial, sans-serif;
          background:
            radial-gradient(circle at top right, rgba(17, 138, 178, 0.12), transparent 32%),
            linear-gradient(180deg, #f8fbfd 0%, var(--bg) 100%);
          color: var(--text);
        }
        main {
          max-width: 860px;
          margin: 0 auto;
          padding: 48px 20px 64px;
        }
        .card {
          background: var(--card);
          border: 1px solid var(--line);
          border-radius: 20px;
          padding: 28px;
          box-shadow: 0 14px 40px rgba(16, 32, 51, 0.08);
        }
        h1 {
          margin: 0 0 10px;
          font-size: clamp(2rem, 4vw, 3rem);
          line-height: 1.05;
        }
        p {
          margin: 0 0 16px;
          color: var(--muted);
          line-height: 1.6;
        }
        .links {
          display: flex;
          flex-wrap: wrap;
          gap: 12px;
          margin: 24px 0 28px;
        }
        .links a {
          text-decoration: none;
          color: white;
          background: linear-gradient(135deg, var(--accent), var(--accent-2));
          padding: 12px 16px;
          border-radius: 999px;
          font-weight: 600;
        }
        .grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
          gap: 14px;
          margin-top: 18px;
        }
        .endpoint {
          border: 1px solid var(--line);
          border-radius: 14px;
          padding: 14px 16px;
          background: #fbfdff;
        }
        .method {
          display: inline-block;
          min-width: 52px;
          margin-right: 8px;
          font-weight: 700;
          color: var(--accent);
        }
        code {
          font-family: Consolas, "Courier New", monospace;
          font-size: 0.95rem;
        }
      </style>
    </head>
    <body>
      <main>
        <section class="card">
          <h1>AirportOpsEnv</h1>
          <p>
            OpenEnv-compatible airport ground-operations environment for the Meta OpenEnv hackathon.
            This Space serves an HTTP API for reset, step, state, grading, and health checks.
          </p>

          <div class="links">
            <a href="/docs">Open API Docs</a>
            <a href="/health">Health Check</a>
            <a href="/openapi.json">OpenAPI Schema</a>
          </div>

          <div class="grid">
            <div class="endpoint"><span class="method">POST</span><code>/reset?task_id=task1</code></div>
            <div class="endpoint"><span class="method">POST</span><code>/step</code></div>
            <div class="endpoint"><span class="method">GET</span><code>/state</code></div>
            <div class="endpoint"><span class="method">GET</span><code>/grade</code></div>
            <div class="endpoint"><span class="method">GET</span><code>/tasks</code></div>
            <div class="endpoint"><span class="method">GET</span><code>/health</code></div>
          </div>
        </section>
      </main>
    </body>
    </html>
    """


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
 
