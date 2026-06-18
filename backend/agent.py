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
"""


class ForgeAgent:
    def __init__(self, model: str = "llama3.1", host: str = "http://localhost:11434"):
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

            # Record assistant turn
            self.conversation.append({
                "role": "assistant",
                "content": assistant_text,
                "tool_calls": tool_calls if tool_calls else None,
            })

            if assistant_text.strip():
                yield {"type": "thinking", "content": assistant_text}

            # No tool calls — the agent is done
            if not tool_calls:
                yield {"type": "final", "content": assistant_text}
                return

            # Execute each requested tool
            for tc in tool_calls:
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
