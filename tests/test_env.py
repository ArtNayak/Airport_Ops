"""
Regression tests for AirportOpsEnv (origin/main API: R1-R3 / G1-G6, env.grade()).
Dependency-free apart from the app's own requirements (stdlib unittest).

Run with:  python -m unittest discover -s tests -v

Covers:
  1. Reward hacking: spamming `hold` must never score well on any task.
  2. Sensible play must clearly beat doing nothing.
  3. ETA lookup resolves for real runway/gate IDs and returns None otherwise.
  4. Maintenance runway is rejected, not silently accepted.
  5. Episode scores stay strictly inside (0, 1) (validator bounds).
  6. HTTP API smoke test (/health, /reset, /step, /grade).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import AirportOpsEnv
from models import Action

TASKS = ("task1", "task2", "task3")
HOLD_SPAM_CEILING = 0.3  # observed: 0.0001 / 0.0001 / 0.1475


def hold_spam(task_id: str) -> tuple[AirportOpsEnv, float]:
    env = AirportOpsEnv()
    obs = env.reset(task_id)
    guard = 0
    while not env.done and guard < 500:
        obs, _, _, _ = env.step(
            Action(flight_id=obs.flights[0].flight_id, action_type="hold")
        )
        guard += 1
    return env, env.grade()["episode_score"]


def greedy_play(task_id: str) -> float:
    """Assign the first free runway, then the first free gate, to each flight."""
    env = AirportOpsEnv()
    obs = env.reset(task_id)
    guard = 0
    while not env.done and guard < 500:
        acted = False
        for f in obs.flights:
            if f.status in ("requesting_landing", "holding") and not f.assigned_runway:
                if obs.available_runways:
                    obs, _, _, _ = env.step(Action(
                        flight_id=f.flight_id, action_type="assign_runway",
                        target_id=obs.available_runways[0]))
                    acted = True
                    break
            elif f.assigned_runway and not f.assigned_gate and obs.available_gates:
                obs, _, _, _ = env.step(Action(
                    flight_id=f.flight_id, action_type="assign_gate",
                    target_id=obs.available_gates[0]))
                acted = True
                break
        if not acted:
            obs, _, _, _ = env.step(
                Action(flight_id=obs.flights[0].flight_id, action_type="hold"))
        guard += 1
    return env.grade()["episode_score"]


class TestEnvReset(unittest.TestCase):
    def test_all_tasks_load(self):
        for t in TASKS:
            env = AirportOpsEnv()
            obs = env.reset(t)
            self.assertTrue(env.is_ready())
            self.assertGreater(len(obs.flights), 0, t)
            self.assertFalse(env.done, t)


class TestRewardHacking(unittest.TestCase):
    def test_hold_spam_scores_low_on_every_task(self):
        for t in TASKS:
            _, score = hold_spam(t)
            self.assertLess(score, HOLD_SPAM_CEILING, f"{t}: hold-spam scored {score}")

    def test_hold_spam_earns_little_step_reward(self):
        env = AirportOpsEnv()
        obs = env.reset("task1")
        total = 0.0
        for _ in range(env.max_steps):
            obs, r, done, _ = env.step(
                Action(flight_id=obs.flights[0].flight_id, action_type="hold"))
            total += r.total
            if done:
                break
        self.assertLess(total, 0.5, f"cumulative hold reward {total}")

    def test_sensible_play_beats_doing_nothing_on_task1(self):
        _, hold_score = hold_spam("task1")
        play_score = greedy_play("task1")
        self.assertGreater(play_score, 0.5)
        self.assertGreater(play_score, hold_score + 0.4)


class TestEtaTable(unittest.TestCase):
    def setUp(self):
        env = AirportOpsEnv()
        env.reset("task1")
        self.sm = env.state_machine

    def test_eta_resolves_for_real_ids(self):
        self.assertIsNotNone(self.sm.get_eta("R1", "G1"))
        self.assertIsNotNone(self.sm.get_eta("R2", "G5"))

    def test_eta_none_for_unknown_ids(self):
        self.assertIsNone(self.sm.get_eta("runway1", "gate_A1"))
        self.assertIsNone(self.sm.get_eta("R1", "G99"))


class TestMaintenanceRunway(unittest.TestCase):
    def test_maintenance_runway_is_rejected(self):
        env = AirportOpsEnv()
        obs = env.reset("task2")
        self.assertNotIn("R3", obs.available_runways)
        fid = next(f.flight_id for f in obs.flights if f.status == "requesting_landing")
        _, reward, _, info = env.step(
            Action(flight_id=fid, action_type="assign_runway", target_id="R3"))
        self.assertTrue(reward.invalid_action)
        self.assertIn("maintenance", info["error"].lower())
        self.assertLess(env.grade()["episode_score"], 0.05)


class TestScoreBounds(unittest.TestCase):
    def test_scores_strictly_inside_unit_interval(self):
        for t in TASKS:
            env = AirportOpsEnv()
            env.reset(t)
            self.assertGreater(env.grade()["episode_score"], 0.0, f"{t} at reset")
            self.assertLess(env.grade()["episode_score"], 1.0, f"{t} at reset")
            _, score = hold_spam(t)
            self.assertGreater(score, 0.0, t)
            self.assertLess(score, 1.0, t)
        for t in TASKS:
            s = greedy_play(t)
            self.assertGreater(s, 0.0, t)
            self.assertLess(s, 1.0, t)


class TestHttpApi(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        import app as app_module
        self.client = TestClient(app_module.app)

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "ok"})

    def test_reset_step_grade_cycle(self):
        for t in TASKS:
            r = self.client.post(f"/reset?task_id={t}")
            self.assertEqual(r.status_code, 200, t)
            fid = r.json()["flights"][0]["flight_id"]
            s = self.client.post(
                "/step", json={"flight_id": fid, "action_type": "hold"})
            self.assertEqual(s.status_code, 200, t)
            g = self.client.get("/grade")
            self.assertEqual(g.status_code, 200, t)
            self.assertEqual(g.json()["task_id"], t)

    def test_tasks_and_homepage(self):
        self.assertEqual(self.client.get("/tasks").status_code, 200)
        self.assertEqual(self.client.get("/").status_code, 200)


if __name__ == "__main__":
    unittest.main()
