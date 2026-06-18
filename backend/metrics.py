"""
Forge Metrics
-------------
In-memory tracking of agent activity for the dashboard:
run history, tool-usage tallies, and server uptime.

Single-process, thread-safe. Resets when the server restarts
(no persistence by design — it's a local app).
"""
import time
from collections import Counter, deque
from threading import Lock


class Metrics:
    def __init__(self, max_runs: int = 50):
        self._lock = Lock()
        self.started_at = time.time()
        self.total_runs = 0
        self.total_tool_calls = 0
        self.tool_counts: Counter = Counter()
        self.runs: deque = deque(maxlen=max_runs)  # most-recent first

    def start_run(self, prompt: str) -> dict:
        """Register a new agent run and return its record."""
        with self._lock:
            self.total_runs += 1
            run = {
                "id": self.total_runs,
                "prompt": (prompt or "").strip()[:140],
                "started_at": time.time(),
                "finished_at": None,
                "status": "running",
                "tool_calls": 0,
                "tools": [],
            }
            self.runs.appendleft(run)
            return run

    def record_tool(self, run: dict, name: str) -> None:
        with self._lock:
            self.tool_counts[name] += 1
            self.total_tool_calls += 1
            run["tool_calls"] += 1
            if name not in run["tools"]:
                run["tools"].append(name)

    def finish_run(self, run: dict, status: str) -> None:
        with self._lock:
            run["finished_at"] = time.time()
            run["status"] = status

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "uptime_seconds": round(time.time() - self.started_at, 1),
                "total_runs": self.total_runs,
                "total_tool_calls": self.total_tool_calls,
                "tool_counts": dict(self.tool_counts),
                "runs": list(self.runs),
            }


# Single shared instance for the server process.
metrics = Metrics()
