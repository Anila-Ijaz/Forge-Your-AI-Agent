"""
Forge API server
----------------
FastAPI backend exposing a streaming chat endpoint and workspace file access.
"""
import json
import os
from pathlib import Path
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import ForgeAgent
from tools import WORKSPACE_DIR
from metrics import metrics


def _load_dotenv():
    """Load KEY=VALUE pairs from a project-root .env file into the environment
    (without overriding values already set). Keeps secrets like OPENAI_API_KEY
    out of the code and out of git (.env is gitignored)."""
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv()

app = FastAPI(title="Forge AI Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# One agent instance per server (single-user local app).
# For multi-user, you'd key by session ID.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PROVIDER = os.environ.get("FORGE_PROVIDER", "openai" if OPENAI_API_KEY else "ollama").lower()
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
agent = ForgeAgent(
    provider=PROVIDER,
    model=os.environ.get("FORGE_MODEL"),
    host=OLLAMA_HOST,
    api_key=OPENAI_API_KEY,
)
# Resolved values (agent fills in provider defaults when unset).
PROVIDER = agent.provider
MODEL = agent.model


class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Stream agent events as Server-Sent Events."""
    def event_stream():
        run = metrics.start_run(req.message)
        status = "done"
        try:
            for event in agent.step(req.message):
                if event.get("type") == "tool_call":
                    metrics.record_tool(run, event.get("name", "unknown"))
                elif event.get("type") == "error":
                    status = "error"
                yield f"data: {json.dumps(event)}\n\n"
            yield "data: {\"type\": \"done\"}\n\n"
        except Exception as e:
            status = "error"
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        finally:
            metrics.finish_run(run, status)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/reset")
def reset():
    agent.reset()
    return {"ok": True}


@app.get("/api/workspace")
def list_workspace():
    """List all files in the workspace."""
    files = []
    for root, _, names in os.walk(WORKSPACE_DIR):
        for n in names:
            full = Path(root) / n
            rel = full.relative_to(WORKSPACE_DIR)
            files.append({"path": str(rel), "size": full.stat().st_size})
    return {"workspace": str(WORKSPACE_DIR), "files": files}


@app.get("/api/workspace/file")
def get_workspace_file(path: str):
    """Download / view a file from the workspace."""
    full = (WORKSPACE_DIR / path).resolve()
    if not str(full).startswith(str(WORKSPACE_DIR)) or not full.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(full)


@app.get("/api/health")
def health():
    return {"status": "ok", "provider": PROVIDER, "model": MODEL, "workspace": str(WORKSPACE_DIR)}


def _provider_status() -> dict:
    """Report whether the active model provider is reachable/configured."""
    if PROVIDER == "openai":
        # We don't spend a request just to ping; a present key means it's configured.
        return {"provider": "openai", "connected": bool(OPENAI_API_KEY), "model": MODEL}
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        resp.raise_for_status()
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        return {"provider": "ollama", "connected": True, "host": OLLAMA_HOST, "models": models}
    except Exception as e:
        return {"provider": "ollama", "connected": False, "host": OLLAMA_HOST, "error": str(e)}


def _workspace_stats() -> dict:
    """Count files and total bytes in the workspace."""
    count, total = 0, 0
    for root, _, names in os.walk(WORKSPACE_DIR):
        for n in names:
            try:
                total += (Path(root) / n).stat().st_size
                count += 1
            except OSError:
                pass
    return {"files": count, "bytes": total}


@app.get("/api/stats")
def stats():
    """Aggregate metrics for the dashboard."""
    snap = metrics.snapshot()
    snap["model"] = MODEL
    snap["provider"] = _provider_status()
    snap["workspace"] = _workspace_stats()
    return snap


# Serve frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    @app.get("/")
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/dashboard")
    def dashboard():
        return FileResponse(FRONTEND_DIR / "dashboard.html")

    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
