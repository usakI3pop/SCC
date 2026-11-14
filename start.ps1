$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$backend = Join-Path $project 'backend'
$runtimeTemp = Join-Path (Split-Path -Parent (Split-Path -Parent $project)) 'work\python-temp'
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
$envDir = Join-Path $backend '.venv'
if (-not (Test-Path (Join-Path $envDir 'Scripts\python.exe'))) {
    & $pythonPath -m venv $envDir
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
}
$venvPython = Join-Path $envDir 'Scripts\python.exe'
& $venvPython -m pip install -r (Join-Path $backend 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Review the Python/PyTorch installation instructions in README.md.' }
$env:SENTINEL_DB = Join-Path $project 'data\experiments.sqlite3'
$env:HF_HOME = Join-Path $project 'data\models'
Start-Process -FilePath $venvPython -ArgumentList @('-m','uvicorn','sentinel.main:app','--app-dir',(Join-Path $backend ''),'--host','127.0.0.1','--port','8000') -WorkingDirectory $backend -WindowStyle Hidden
Push-Location (Join-Path $project 'frontend')
try { & node serve.mjs } finally { Pop-Location }
