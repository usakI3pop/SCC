$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$backend = Join-Path $project 'backend'
$runtimeTemp = Join-Path $project 'data\temp'
New-Item -ItemType Directory -Force -Path $runtimeTemp | Out-Null
$env:TEMP = $runtimeTemp
$env:TMP = $runtimeTemp

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
$pythonPath = if ($pythonCommand -and $pythonCommand.Source -notlike '*\WindowsApps\*') { $pythonCommand.Source } else { $null }
if (-not $pythonPath) {
    $pythonPath = Get-ChildItem (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python*\python.exe'), 'C:\Program Files\Python*\python.exe' -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $pythonPath) { throw 'Python 3.10+ is required. Install Python, then rerun start.ps1.' }

# Start FastAPI backend server
Write-Host "Starting FastAPI Backend on http://127.0.0.1:8000..." -ForegroundColor Cyan
Start-Process -FilePath $pythonPath -ArgumentList @('-m', 'uvicorn', 'main:app', '--app-dir', $backend, '--host', '127.0.0.1', '--port', '8000', '--reload') -WorkingDirectory $backend

# Start Frontend static HTTP server
Write-Host "Starting Frontend Interface on http://127.0.0.1:4173..." -ForegroundColor Green
Push-Location (Join-Path $project 'frontend')
try { & node serve.mjs } finally { Pop-Location }
