"""
Forge Sessions
--------------
Persistent chat history. Each session is one conversation stored as a JSON
file under sessions/, so past chats survive server restarts.

Single-user/local: requests are serialized, so a process-wide lock is enough.
"""
import json
import time
import uuid
from pathlib import Path
from threading import Lock

from agent import SYSTEM_PROMPT

SESSIONS_DIR = Path(__file__).parent.parent / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _new_conversation():
    return [{"role": "system", "content": SYSTEM_PROMPT}]


class SessionStore:
    def __init__(self):
        self._lock = Lock()

    def _path(self, sid: str) -> Path:
        return SESSIONS_DIR / f"{sid}.json"

    def _save(self, data: dict) -> None:
        with self._lock:
            self._path(data["id"]).write_text(json.dumps(data), encoding="utf-8")

    def create(self, title: str = "New chat") -> dict:
        now = time.time()
        data = {
            "id": uuid.uuid4().hex[:12],
            "title": title,
            "created_at": now,
            "updated_at": now,
            "conversation": _new_conversation(),
        }
        self._save(data)
        return data

    def get(self, sid: str):
        path = self._path(sid)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def save_conversation(self, sid: str, conversation: list, title: str = None) -> None:
        data = self.get(sid)
        if not data:
            return
        data["conversation"] = conversation
        data["updated_at"] = time.time()
        if title:
            data["title"] = title[:60]
        self._save(data)

    def delete(self, sid: str) -> bool:
        path = self._path(sid)
        if path.exists():
            path.unlink()
            return True
        return False

    @staticmethod
    def _visible(conversation: list) -> list:
        """Only user/assistant turns with text — what the chat UI shows."""
        return [
            {"role": m["role"], "content": m["content"]}
            for m in conversation
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]

    def messages(self, sid: str):
        data = self.get(sid)
        return None if data is None else self._visible(data["conversation"])

    def list(self) -> list:
        items = []
        for path in SESSIONS_DIR.glob("*.json"):
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            items.append({
                "id": d["id"],
                "title": d.get("title", "New chat"),
                "updated_at": d.get("updated_at", 0),
                "messages": len(self._visible(d.get("conversation", []))),
            })
        items.sort(key=lambda x: x["updated_at"], reverse=True)
        return items


sessions = SessionStore()
