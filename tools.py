"""
Forge Agent Tools
-----------------
Capabilities the AI agent can invoke: file I/O, shell commands, web search/fetch.
All file operations are sandboxed to the WORKSPACE_DIR for safety.
"""
import os
import subprocess
import json
from pathlib import Path
from typing import Any
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

# Sandbox: all file operations happen inside this directory
WORKSPACE_DIR = Path(os.environ.get("FORGE_WORKSPACE", Path(__file__).parent.parent / "workspace")).resolve()
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


def _safe_path(relative_path: str) -> Path:
    """Resolve a path inside the workspace, blocking escape attempts."""
    target = (WORKSPACE_DIR / relative_path).resolve()
    if not str(target).startswith(str(WORKSPACE_DIR)):
        raise ValueError(f"Path escapes workspace: {relative_path}")
    return target


# ---------- File operations ----------

def write_file(path: str, content: str) -> dict:
    """Create or overwrite a file in the workspace."""
    try:
        target = _safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"success": True, "path": str(target.relative_to(WORKSPACE_DIR)), "bytes": len(content)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def read_file(path: str) -> dict:
    """Read a file's contents from the workspace."""
    try:
        target = _safe_path(path)
        if not target.exists():
            return {"success": False, "error": f"File not found: {path}"}
        content = target.read_text(encoding="utf-8", errors="replace")
        return {"success": True, "path": path, "content": content}
    except Exception as e:
        return {"success": False, "error": str(e)}


def list_files(path: str = ".") -> dict:
    """List files and directories within the workspace."""
    try:
        target = _safe_path(path)
        if not target.exists():
            return {"success": False, "error": f"Path not found: {path}"}
        if target.is_file():
            return {"success": True, "type": "file", "path": path}
        items = []
        for item in sorted(target.iterdir()):
            items.append({
                "name": item.name,
                "type": "dir" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else None,
            })
        return {"success": True, "type": "dir", "path": path, "items": items}
    except Exception as e:
        return {"success": False, "error": str(e)}


def delete_file(path: str) -> dict:
    """Delete a file from the workspace."""
    try:
        target = _safe_path(path)
        if not target.exists():
            return {"success": False, "error": f"File not found: {path}"}
        if target.is_file():
            target.unlink()
        else:
            return {"success": False, "error": "Refusing to delete directory"}
        return {"success": True, "path": path}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------- Shell execution ----------

def run_shell(command: str, timeout: int = 60) -> dict:
    """Run a shell command inside the workspace directory."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(WORKSPACE_DIR),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout[-4000:],  # cap output size
            "stderr": result.stderr[-2000:],
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Command timed out after {timeout}s"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------- Web tools ----------

def web_search(query: str, max_results: int = 5) -> dict:
    """Search the web via DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        formatted = [
            {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
            for r in results
        ]
        return {"success": True, "query": query, "results": formatted}
    except Exception as e:
        return {"success": False, "error": str(e)}


def web_fetch(url: str, max_chars: int = 5000) -> dict:
    """Fetch a URL and return readable text content."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Forge AI Agent)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        return {"success": True, "url": url, "content": text[:max_chars], "truncated": len(text) > max_chars}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------- Tool registry ----------
# This schema is sent to the model so it knows which tools exist.

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file in the workspace. Use this to generate code, HTML, configs, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside the workspace, e.g. 'game/index.html'"},
                    "content": {"type": "string", "description": "Full file contents"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file's contents from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Default '.'"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Delete a file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Execute a shell command in the workspace. Use for installing packages, running scripts, building projects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "description": "Seconds, default 60"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information using DuckDuckGo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "description": "Default 5"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch the readable text content of a webpage.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "write_file": write_file,
    "read_file": read_file,
    "list_files": list_files,
    "delete_file": delete_file,
    "run_shell": run_shell,
    "web_search": web_search,
    "web_fetch": web_fetch,
}


def execute_tool(name: str, arguments: dict) -> Any:
    """Dispatch a tool call to its implementation."""
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return {"success": False, "error": f"Unknown tool: {name}"}
    try:
        return func(**arguments)
    except TypeError as e:
        return {"success": False, "error": f"Bad arguments: {e}"}
