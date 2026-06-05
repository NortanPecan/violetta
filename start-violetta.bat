@echo off
cd /d "%~dp0"
echo Starting Виолетта (custom HTML version)...
".\.venv\Scripts\python.exe" -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
pause
