# Launcher for Виолетта (custom HTML version - full control over avatar layout)
# Double-click or run in PowerShell
# Default: gemma4:31b-cloud (free cloud). Switch in .env to gemma4:e4b / e2b for local.
# The chat UI top pill shows the actual model at runtime (via /api/model).

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectDir

Write-Host "=== Виолетта — твой AI-даэмон (custom HTML) ===" -ForegroundColor Cyan
Write-Host "Project: $projectDir" -ForegroundColor Gray
Write-Host "Using pure custom frontend (no Chainlit) for perfect left-avatar + caption layout" -ForegroundColor DarkGray

# Check Ollama
$ollamaExe = "C:\Users\shalo\AppData\Local\Programs\Ollama\ollama.exe"
if (Test-Path $ollamaExe) {
    Write-Host "Ollama found. Models:" -ForegroundColor Green
    & $ollamaExe list
} else {
    Write-Host "WARNING: Ollama not found at expected path. Start the Ollama app first!" -ForegroundColor Yellow
}

# Show what model the app will use (from .env)
$envModel = (Get-Content .env | Select-String '^OLLAMA_MODEL=' | ForEach-Object { $_.ToString().Split('=')[1].Trim() })
if ($envModel) {
    Write-Host "App will use: $envModel (shown in UI top pill)" -ForegroundColor Cyan
}

# Activate venv and run custom FastAPI server
$venvPython = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "ERROR: Virtual environment not found. Run setup first." -ForegroundColor Red
    exit 1
}

Write-Host "`nStarting custom server on http://localhost:8000 ..." -ForegroundColor Green
Write-Host "The beautiful custom HTML chat (with proper avatar on left + why caption below) will open." -ForegroundColor Yellow
Write-Host "Press Ctrl+C in this window to stop." -ForegroundColor DarkGray

& $venvPython -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
