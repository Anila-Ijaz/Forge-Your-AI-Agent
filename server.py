"""
Forge API server
----------------
FastAPI backend exposing a streaming chat endpoint and workspace file access.
"""
import json
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import ForgeAgent
from tools import WORKSPACE_DIR

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
MODEL = os.environ.get("FORGE_MODEL", "llama3.1")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
agent = ForgeAgent(model=MODEL, host=OLLAMA_HOST)


class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Stream agent events as Server-Sent Events."""
    def event_stream():
        try:
            for event in agent.step(req.message):
                yield f"data: {json.dumps(event)}\n\n"
            yield "data: {\"type\": \"done\"}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

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
    return {"status": "ok", "model": MODEL, "workspace": str(WORKSPACE_DIR)}


# Serve frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    @app.get("/")
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
