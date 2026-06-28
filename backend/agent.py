"""
Forge Agent Loop
----------------
Runs a multi-step agent: model thinks → calls tools → sees results → repeats
until it produces a final answer or hits the step limit.
"""
import json
import os
from tools import TOOLS_SCHEMA, execute_tool

SYSTEM_PROMPT = """You are Forge, a capable AI agent that can build software, browse the web, and execute commands.

You have access to these tools:
- write_file: create/overwrite files in the workspace
- read_file: read file contents
- list_files: see what's in the workspace
- delete_file: remove a file
- run_shell: execute shell commands (npm, python, etc.) inside the workspace
- web_search: search the web
- web_fetch: get a webpage's text content

Guidelines:
1. When the user asks you to BUILD something (game, website, app, script), CREATE actual files using write_file. Don't just describe — do it.
2. For web projects, prefer a single self-contained index.html when possible (HTML + CSS + JS in one file).
3. Use run_shell to install dependencies or run programs the user requests.
4. After creating files, briefly summarize what you built and how to run it.
5. If you need current information, use web_search and web_fetch.
6. Be efficient: only call tools you actually need. When you've completed the task, give a concise final answer with no further tool calls.
7. Files you create are saved to the workspace and appear in the app's "Workspace Files" panel, where the user can open/download them. To deliver a file, just write it and tell the user its filename — do NOT invent download links like "sandbox:/file" or "[Download](...)"; those do not work here.

WORKING FROM SOURCE MATERIAL (resume, CV, LaTeX, notes, existing text):
- When the user provides content (e.g. a LaTeX CV, a resume, a bio, a list), that content is your SOURCE OF TRUTH. Your job is to TRANSFORM it, not to invent something new.
- Extract EVERY section and detail from the source: name, contact, summary, education, experience, projects, skills, certifications, awards, extracurriculars — all of it. Map each one into the output.
- Use the REAL details verbatim (names, dates, titles, organizations). NEVER replace them with placeholders like "Your Name", "lorem ipsum", "Project 1", or unrelated example code.
- Do NOT fabricate facts that aren't in the source. If something is missing, omit it — don't make it up.
- The output must be COMPLETE: a real, finished page with all the user's content rendered and styled. A page with just a heading and no content is a FAILURE — redo it.

BUILDING A PORTFOLIO / WEBSITE FROM A CV:
- Produce ONE complete, self-contained index.html (HTML + CSS + JS inline) in a SINGLE write_file call.
- Include real, styled sections built from the CV: a header with the person's name and contact links, then About/Summary, Education, Experience, Projects, Skills, and any others present in the source.
- Make it visually polished and responsive (clean layout, readable typography, sensible color scheme) — but content fidelity to the source comes first.
- After writing, summarize the sections you included so the user can verify nothing was dropped.
"""

# Valid tool names, used to validate recovered tool calls.
TOOL_NAMES = {t["function"]["name"] for t in TOOLS_SCHEMA}


def _loads(args):
    """Coerce tool-call arguments (which may be a JSON string or already a dict) into a dict."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return {}
    return {}


def _normalize_tool_obj(obj: dict):
    """Turn a loosely-shaped dict into our internal tool-call format, or None."""
    if not isinstance(obj, dict):
        return None
    # Style A: {"function": {"name": ..., "arguments": {...}}}
    fn = obj.get("function")
    if isinstance(fn, dict) and fn.get("name") in TOOL_NAMES:
        args = fn.get("arguments") or fn.get("parameters") or {}
        return {"function": {"name": fn["name"], "arguments": args}}
    # Style B: {"name": ..., "arguments"|"parameters": {...}}
    name = obj.get("name")
    if isinstance(name, str) and name in TOOL_NAMES:
        args = obj.get("arguments") or obj.get("parameters") or {}
        return {"function": {"name": name, "arguments": args}}
    return None


def _extract_text_tool_calls(text: str):
    """Recover tool calls that a weak model printed as raw JSON in its text
    instead of returning them as structured tool_calls. Scans for balanced
    top-level {...} blocks, parses each, and keeps valid tool calls."""
    calls, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    norm = _normalize_tool_obj(json.loads(text[start:i + 1]))
                except json.JSONDecodeError:
                    norm = None
                if norm:
                    calls.append(norm)
                start = None
    return calls


# Sensible default model per provider.
DEFAULT_MODELS = {"openai": "gpt-4o-mini", "ollama": "qwen2.5-coder:7b"}


class ForgeAgent:
    """Multi-step tool-using agent backed by either OpenAI or a local Ollama model.

    Provider selection:
      - explicit `provider=` argument, else
      - FORGE_PROVIDER env var, else
      - "openai" if an API key is available, otherwise "ollama".
    """

    def __init__(self, provider: str = None, model: str = None,
                 host: str = "http://localhost:11434", api_key: str = None):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._host = host
        provider = (provider or os.environ.get("FORGE_PROVIDER")
                    or ("openai" if self._api_key else "ollama")).lower()
        model = model or os.environ.get("FORGE_MODEL") or DEFAULT_MODELS.get(provider)
        self._configure(provider, model)
        self.conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _configure(self, provider: str, model: str):
        """(Re)build the client for a provider and set the active model."""
        provider = (provider or "").lower()
        if provider == "openai":
            from openai import OpenAI
            if not self._api_key:
                raise ValueError(
                    "OpenAI provider selected but no API key found. "
                    "Set OPENAI_API_KEY (env var or .env file)."
                )
            self.client = OpenAI(api_key=self._api_key)
        elif provider == "ollama":
            import ollama
            self.client = ollama.Client(host=self._host)
        else:
            raise ValueError(f"Unknown provider: {provider!r} (use 'openai' or 'ollama')")
        self.provider = provider
        self.model = model or DEFAULT_MODELS.get(provider)

    def switch(self, provider: str = None, model: str = None) -> dict:
        """Change provider/model at runtime. Resets the conversation when the
        provider changes, since message formats differ between providers."""
        new_provider = (provider or self.provider).lower()
        provider_changed = new_provider != self.provider
        self._configure(new_provider, model if model else (None if provider_changed else self.model))
        if provider_changed:
            self.reset()
        return {"provider": self.provider, "model": self.model}

    def reset(self):
        self.conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

    # ---------- Provider-specific completion ----------

    def _complete(self):
        """Call the model once. Returns (assistant_text, calls, assistant_msg, usage) where
        `calls` is a normalized list of {"id", "name", "arguments"}, `assistant_msg`
        is the turn to append, and `usage` is {"prompt", "completion"} token counts."""
        if self.provider == "openai":
            resp = self.client.chat.completions.create(
                model=self.model, messages=self.conversation, tools=TOOLS_SCHEMA,
            )
            msg = resp.choices[0].message
            text = msg.content or ""
            raw = msg.tool_calls or []
            calls = [{"id": tc.id, "name": tc.function.name,
                      "arguments": _loads(tc.function.arguments)} for tc in raw]
            assistant_msg = {"role": "assistant", "content": text or None}
            if raw:
                assistant_msg["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in raw
                ]
            u = resp.usage
            usage = {"prompt": getattr(u, "prompt_tokens", 0) or 0,
                     "completion": getattr(u, "completion_tokens", 0) or 0}
            return text, calls, assistant_msg, usage

        # ollama
        resp = self.client.chat(model=self.model, messages=self.conversation, tools=TOOLS_SCHEMA)
        msg = resp["message"]
        text = msg.get("content", "") or ""
        raw = msg.get("tool_calls") or []
        calls = [{"id": None, "name": tc["function"]["name"],
                  "arguments": _loads(tc["function"].get("arguments", {}))} for tc in raw]
        assistant_msg = {"role": "assistant", "content": text, "tool_calls": raw or None}
        usage = {"prompt": resp.get("prompt_eval_count", 0) or 0,
                 "completion": resp.get("eval_count", 0) or 0}
        return text, calls, assistant_msg, usage

    # ---------- Agent loop ----------

    def step(self, user_message: str, max_iterations: int = 10):
        """
        Yields events as the agent works:
            {"type": "thinking", "content": "..."}
            {"type": "tool_call", "name": "...", "arguments": {...}}
            {"type": "tool_result", "name": "...", "result": {...}}
            {"type": "final", "content": "..."}
            {"type": "error", "content": "..."}
        """
        self.conversation.append({"role": "user", "content": user_message})

        for iteration in range(max_iterations):
            try:
                assistant_text, calls, assistant_msg, usage = self._complete()
            except Exception as e:
                yield {"type": "error", "content": f"{self.provider} error: {e}"}
                return

            self.conversation.append(assistant_msg)
            yield {"type": "usage", "model": self.model,
                   "prompt": usage["prompt"], "completion": usage["completion"]}

            # Fallback: weak local models sometimes print tool calls as raw JSON
            # text instead of returning structured tool_calls. Recover them.
            recovered = False
            if not calls and assistant_text.strip() and self.provider == "ollama":
                rec = _extract_text_tool_calls(assistant_text)
                calls = [{"id": None, "name": c["function"]["name"],
                          "arguments": c["function"]["arguments"]} for c in rec]
                recovered = bool(calls)

            # Show prose, but not when the text was purely a recovered tool call
            if assistant_text.strip() and not recovered:
                yield {"type": "thinking", "content": assistant_text}

            # No tool calls — the agent is done
            if not calls:
                yield {"type": "final", "content": assistant_text}
                return

            # Execute each requested tool
            for call in calls:
                name, args = call["name"], _loads(call["arguments"])
                yield {"type": "tool_call", "name": name, "arguments": args}
                result = execute_tool(name, args)
                yield {"type": "tool_result", "name": name, "result": result}

                # Feed result back to the model (OpenAI requires the tool_call_id)
                tool_msg = {"role": "tool", "content": json.dumps(result)[:8000]}
                if self.provider == "openai" and call.get("id"):
                    tool_msg["tool_call_id"] = call["id"]
                self.conversation.append(tool_msg)

        yield {"type": "error", "content": f"Reached max iterations ({max_iterations}) without finishing."}
