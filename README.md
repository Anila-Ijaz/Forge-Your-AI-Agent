# ⚒ Forge — Local AI Agent

**Forge is a fully local, open-source AI agent** powered by Ollama. Ask it to build games, websites, scripts, or anything else — it writes real files, runs real commands, and gets things done. Think ChatGPT/Claude, but running on your own machine with full agentic capabilities.

![Status](https://img.shields.io/badge/status-ready-success) ![License](https://img.shields.io/badge/license-MIT-blue)

## ✨ Features

- 🧠 **Local LLM** — runs on Ollama (Llama 3.1, Qwen, Mistral, etc.) — your data never leaves your machine
- 🛠 **Real tool use** — the agent calls functions to write files, run commands, search the web
- 💬 **Clean chat UI** — dark-themed, streaming responses, live workspace file viewer
- 🔒 **Sandboxed** — all file ops constrained to a `workspace/` directory
- 🌐 **Web access** — DuckDuckGo search + page fetching when the agent needs current info

## 🎯 What can it do?

Ask Forge things like:

- *"Build a Snake game as a single HTML file"*
- *"Create a portfolio website with a dark theme"*
- *"Write a Python script that summarizes a PDF"*
- *"Search for the latest Llama model and tell me about it"*
- *"Set up a basic Express.js server with a /hello endpoint and run it"*

## 📋 Prerequisites

1. **Python 3.9+**
2. **Ollama** — install from [ollama.com](https://ollama.com)
3. A model that supports tool calling. Recommended:
   ```bash
   ollama pull llama3.1
   # or for smaller/faster:
   ollama pull qwen2.5:7b
   # or for stronger reasoning:
   ollama pull qwen2.5:14b
   ```

> ⚠️ **Important:** the model must support function/tool calling. Llama 3.1, Qwen 2.5, and Mistral Nemo work well. Older models (Llama 2, Phi-2) won't.

## 🚀 Quick Start

```bash
# 1. Clone or unzip the project, then:
cd forge

# 2. Install Python dependencies
pip install -r backend/requirements.txt

# 3. Start Ollama (if not already running)
ollama serve &

# 4. Run Forge
python backend/server.py
```

Then open **http://localhost:8000** in your browser.

## ⚙️ Configuration

Set environment variables to customize:

```bash
export FORGE_MODEL="qwen2.5:14b"          # default: llama3.1
export OLLAMA_HOST="http://localhost:11434"
export FORGE_WORKSPACE="/path/to/workspace"  # default: ./workspace
python backend/server.py
```

## 📂 Project Structure

```
forge/
├── backend/
│   ├── server.py          # FastAPI server + chat endpoint
│   ├── agent.py           # Agent loop (think → call tools → repeat)
│   ├── tools.py           # File ops, shell, web search/fetch
│   └── requirements.txt
├── frontend/
│   └── index.html         # Single-file chat UI
├── workspace/             # Where the agent creates files
└── README.md
```

## 🧩 How it works

1. You type a request in the browser.
2. The frontend POSTs to `/api/chat`, which streams Server-Sent Events.
3. The backend calls Ollama with the conversation + tool schemas.
4. If the model returns tool calls, the agent executes them locally and feeds results back.
5. The loop repeats until the model produces a final answer (no more tool calls).
6. The UI shows each step: assistant text, tool calls, results, and the workspace file list updates live.

## 🛡 Safety notes

- File operations are sandboxed to `workspace/` — paths that try to escape (`../../etc/passwd`) are rejected.
- Shell commands run in `workspace/` as the user running the server. If you don't trust the model, run it in a container or VM. `run_shell` is powerful.
- Web requests go through your machine's network — review fetched content before acting on it.

## 🔧 Extending

Add new tools in `backend/tools.py`:

1. Write a function returning a `dict`.
2. Add a JSON schema entry to `TOOLS_SCHEMA`.
3. Register it in `TOOL_FUNCTIONS`.

The agent will discover and use it automatically.

## 📜 License

MIT — do whatever you want.

---

**Built with:** FastAPI · Ollama · Vanilla JS · Pure stubbornness 🔥
