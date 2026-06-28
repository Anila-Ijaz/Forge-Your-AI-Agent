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

# Approximate OpenAI pricing in USD per 1M tokens (input, output).
# Local Ollama models are free, so anything not listed costs $0.
PRICING = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimated USD cost for a request. Returns 0 for unknown/local models."""
    rate_in, rate_out = PRICING.get(model, (0.0, 0.0))
    return (prompt_tokens * rate_in + completion_tokens * rate_out) / 1_000_000


class Metrics:
    def __init__(self, max_runs: int = 50):
        self._lock = Lock()
        self.started_at = time.time()
        self.total_runs = 0
        self.total_tool_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
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
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost": 0.0,
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

    def record_usage(self, run: dict, prompt_tokens: int, completion_tokens: int, model: str) -> None:
        """Accumulate token usage and estimated cost for a run."""
        cost = estimate_cost(model, prompt_tokens, completion_tokens)
        with self._lock:
            run["prompt_tokens"] += prompt_tokens
            run["completion_tokens"] += completion_tokens
            run["cost"] += cost
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_cost += cost

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
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
                "total_cost": round(self.total_cost, 4),
                "tool_counts": dict(self.tool_counts),
                "runs": list(self.runs),
            }


# Single shared instance for the server process.
metrics = Metrics()
