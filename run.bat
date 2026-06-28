@echo off
REM Forge - one-shot launcher for Windows
cd /d "%~dp0"

echo.
echo  Forge AI Agent
echo  ----------------

where python >nul 2>nul
if errorlevel 1 (
  echo X Python not found. Install Python 3.9+
  exit /b 1
)

where ollama >nul 2>nul
if errorlevel 1 
  echo ! Ollama not found. Install from https://ollama.com
  echo   Then run:  ollama pull llama3.1
  exit /b 1
)

if not exist ".deps_installed" (
  echo - Installing Python dependencies...
  pip install -r backend\requirements.txt
  type nul > .deps_installed
)

echo - Starting Forge on http://localhost:8000
python backend\server.py
