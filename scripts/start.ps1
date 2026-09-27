$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    python -m venv .venv
    & ./.venv/Scripts/python.exe -m pip install -e '.[dev,research]'
}
if (-not (Test-Path -LiteralPath 'frontend/dist/index.html')) {
    Push-Location frontend
    npm.cmd ci
    npm.cmd run build
    Pop-Location
}
if (-not (Test-Path -LiteralPath '.env')) {
    Copy-Item -LiteralPath '.env.example' -Destination '.env'
}
& ./.venv/Scripts/python.exe -m uvicorn surveilx.api:app --host 127.0.0.1 --port 8000 --workers 1
