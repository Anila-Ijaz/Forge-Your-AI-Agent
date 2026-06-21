"""
Forge Agent Loop
----------------
Runs a multi-step agent: model thinks → calls tools → sees results → repeats
until it produces a final answer or hits the step limit.
"""
import json
import ollama
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


class ForgeAgent:
    def __init__(self, model: str = "qwen2.5-coder:7b", host: str = "http://localhost:11434"):
        self.model = model
        self.client = ollama.Client(host=host)
        self.conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

    def reset(self):
        self.conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

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
                response = self.client.chat(
                    model=self.model,
                    messages=self.conversation,
                    tools=TOOLS_SCHEMA,
                )
            except Exception as e:
                yield {"type": "error", "content": f"Ollama error: {e}"}
                return

            msg = response["message"]
            assistant_text = msg.get("content", "") or ""
            tool_calls = msg.get("tool_calls") or []

            # Fallback: some local models emit tool calls as raw JSON in their
            # text instead of as structured tool_calls. Try to recover them.
            recovered = []
            if not tool_calls and assistant_text.strip():
                recovered = _extract_text_tool_calls(assistant_text)

            # Record assistant turn
            self.conversation.append({
                "role": "assistant",
                "content": assistant_text,
                "tool_calls": tool_calls if tool_calls else None,
            })

            # Show prose, but not when the text was purely a recovered tool call
            if assistant_text.strip() and not recovered:
                yield {"type": "thinking", "content": assistant_text}

            calls = tool_calls or recovered

            # No tool calls — the agent is done
            if not calls:
                yield {"type": "final", "content": assistant_text}
                return

            # Execute each requested tool
            for tc in calls:
                fn = tc["function"]
                name = fn["name"]
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                yield {"type": "tool_call", "name": name, "arguments": args}
                result = execute_tool(name, args)
                yield {"type": "tool_result", "name": name, "result": result}

                # Feed result back to the model
                self.conversation.append({
                    "role": "tool",
                    "content": json.dumps(result)[:8000],  # cap tool output
                })

        yield {"type": "error", "content": f"Reached max iterations ({max_iterations}) without finishing."}
