# Run tests locally in Windows PowerShell
# Creates venv, installs dev requirements, spins up docker compose and runs pytest

$ErrorActionPreference = 'Stop'

if (!(Test-Path -Path .venv)) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

Write-Host "Activating virtual environment..."
. .\.venv\Scripts\Activate.ps1

Write-Host "Installing dev dependencies..."
pip install --upgrade pip
pip install -r requirements-dev.txt

# Ensure reports folder exists
if (!(Test-Path -Path .\reports)) { New-Item -ItemType Directory -Path .\reports | Out-Null }

Write-Host "Bringing up docker-compose services..."
docker compose up -d

Write-Host "Waiting 10s for services to initialize..."
Start-Sleep -Seconds 10

Write-Host "Running pytest (all tests)..."
python -m pytest tests/ -v --junitxml=reports/junit.xml --cov=app --cov-report=html:reports/coverage

$exitCode = $LASTEXITCODE

Write-Host "Tearing down docker-compose services..."
docker compose down

Exit $exitCode
