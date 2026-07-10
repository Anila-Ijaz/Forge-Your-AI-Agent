# ⚒ Forge — AI Agent

**Forge is an open-source AI agent** that writes real files, runs real commands, browses the web, and gets things done. Run it on **OpenAI** (GPT-4o / GPT-4.1) for top reliability, or **fully local** with [Ollama](https://ollama.com) so your data never leaves your machine. Think ChatGPT/Claude with full agentic tool use — on your own terms.

![Status](https://img.shields.io/badge/status-ready-success) ![License](https://img.shields.io/badge/license-MIT-blue)

## ✨ Features

- 🔁 **Two providers** — OpenAI *or* local Ollama, switchable from a dropdown in the UI (no restart)
- 🛠 **Real tool use** — write/read/delete files, run shell commands, web search & fetch
- 💬 **Polished chat UI** — markdown rendering with syntax-highlighted code, streaming, dark theme
- 🗂 **Chat history** — conversations persist to disk; revisit, rename, or delete past chats
- 📊 **Live dashboard** — runs, tool usage, **token counts & estimated cost**, workspace stats
- 🔒 **Sandboxed** — all file ops constrained to a `workspace/` directory
- 🌐 **Web access** — DuckDuckGo search (`ddgs`) + page fetching when the agent needs current info

## 🎯 What can it do?

Ask Forge things:

- *"Build a Snake game as a single HTML file"*
- *"Make a portfolio website from this CV"* (paste your resume — it uses your real details)
- *"Write a Python script that fetches Berlin's weather"*
- *"Search for the latest Llama model and summarize it"*
- *"Set up a basic Express.js server with a /hello endpoint and run it"*

## 📋 Prerequisites

- **Python 3.10+**
- **One model provider:**
  - **OpenAI** — an API key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys) *(easiest, most reliable)*, **or**
  - **Ollama** — installed from [ollama.com](https://ollama.com) with a tool-calling model pulled:
    ```bash
    ollama pull qwen2.5-coder:7b      # great for code/agentic tasks
    # or: ollama pull llama3.1 / qwen2.5:14b
    ```

> ⚠️ For local models, pick one that supports **function/tool calling** (Qwen 2.5, Llama 3.1, Mistral Nemo). Small/old models (Llama 2, Phi-2) won't work well, and 7–8B models can be unreliable at tool calls — OpenAI is recommended for best results.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r backend/requirements.txt

# 2. Configure (copy the template and fill in your values)
cp .env.example .env
#   - For OpenAI: set OPENAI_API_KEY
#   - For Ollama: set FORGE_PROVIDER=ollama (and have `ollama serve` running)

# 3. Run Forge
python backend/server.py
```

Then open **http://localhost:8000** (chat) and **http://localhost:8000/dashboard** (metrics).

On Windows you can just double-click **`run.bat`**.

## ⚙️ Configuration

Forge reads a **`.env`** file in the project root (gitignored, so secrets never get committed). See [`.env.example`](.env.example):

```bash
# Provider: "openai" (default when OPENAI_API_KEY is set) or "ollama"
FORGE_PROVIDER=openai

# --- OpenAI ---
OPENAI_API_KEY=sk-proj-your-key-here
FORGE_MODEL=gpt-4o-mini          # or gpt-4o, gpt-4.1-mini, gpt-4.1

# --- Ollama (when FORGE_PROVIDER=ollama) ---
# OLLAMA_HOST=http://localhost:11434
# FORGE_MODEL=qwen2.5-coder:7b

# --- Optional ---
# FORGE_WORKSPACE=/path/to/workspace   # default: ./workspace
```

Any of these can also be set as normal environment variables. The active model is also switchable live from the dropdown in the chat header.

## 📂 Project Structure

```
Forge-Your-AI-Agent/
├── backend/
│   ├── server.py          # FastAPI server: chat (SSE), sessions, models, stats
│   ├── agent.py           # Agent loop + OpenAI/Ollama provider abstraction
│   ├── tools.py           # File ops, shell, web search/fetch (sandboxed)
│   ├── metrics.py         # Run/tool/token/cost tracking for the dashboard
│   ├── sessions.py        # Persistent chat history
│   └── requirements.txt
├── frontend/
│   ├── index.html         # Chat UI (markdown, model switcher, history sidebar)
│   └── dashboard.html     # Live metrics dashboard
├── workspace/             # Where the agent creates files (sandboxed)
├── sessions/              # Saved chat history (gitignored)
├── .env.example           # Config template
├── run.bat                # One-click launcher (Windows)
└── README.md
```

## 🧩 How it works

1. You type a request in the browser; the frontend POSTs to `/api/chat`, which streams **Server-Sent Events**.
2. The backend loads the session's history and calls the active provider (OpenAI or Ollama) with the conversation + tool schemas.
3. If the model returns tool calls, the agent executes them locally and feeds the results back.
4. The loop repeats until the model produces a final answer (no more tool calls).
5. Token usage and cost are recorded per run; the conversation is saved to its session file.
6. The dashboard polls `/api/stats` to show live activity.

### Key API endpoints

| Endpoint | Purpose |
|---|---|
| `POST /api/chat` | Stream an agent run (SSE) |
| `GET/POST /api/sessions`, `GET /api/sessions/{id}/messages`, `DELETE …` | Chat history |
| `GET /api/models`, `POST /api/model` | List / switch the active model |
| `GET /api/stats` | Dashboard metrics (runs, tools, tokens, cost) |
| `GET /api/workspace`, `GET /api/workspace/file` | Browse/download generated files |

## 🛡 Safety notes

- File operations are sandboxed to `workspace/` — paths that try to escape (`../../etc/passwd`) are rejected.
- Shell commands run in `workspace/` as the user running the server. If you don't trust the model, run it in a container or VM — `run_shell` is powerful.
- Web requests go through your machine's network — review fetched content before acting on it.
- Your `OPENAI_API_KEY` lives only in `.env` (gitignored). Never hardcode it in source files.

## 🔧 Extending

Add new tools in `backend/tools.py`:

1. Write a function returning a `dict`.
2. Add a JSON schema entry to `TOOLS_SCHEMA`.
3. Register it in `TOOL_FUNCTIONS`.

The agent discovers and uses it automatically.

## 📜 License

MIT — do whatever you want.

---

**Built with:** FastAPI · OpenAI · Ollama · Vanilla JS · Pure stubbornness 🔥
