@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo The .venv folder is missing. See README.md, "Setup on a new PC".
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" app.py
